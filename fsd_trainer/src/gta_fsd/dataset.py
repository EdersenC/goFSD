"""Helpers for loading processed trip datasets from one or more run directories."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor
from torch.utils.data import Dataset

from config import (
    DEFAULT_AUX_TARGET_NAMES,
    DEFAULT_CONTROL_TARGET_NAMES,
    DEFAULT_FUTURE_OFFSETS,
    DEFAULT_IMAGE_OFFSETS,
    DEFAULT_TELEMETRY_FEATURE_NAMES,
    DEFAULT_TELEMETRY_OFFSETS,
)
from image_io import load_rgb_tensor_from_path


DatasetImages = Tensor
DatasetTelemetry = Tensor
DatasetTargetControls = Tensor
DatasetTargetAux = Tensor
DatasetItem = tuple[DatasetImages, DatasetTelemetry, DatasetTargetControls, DatasetTargetAux]

TelemetryMap = dict[str, Any]


class Trip:
    def __init__(self, trip_dir: str | Path, *, run_path: str | Path):
        self.trip_dir = Path(trip_dir)
        self.run_path = Path(run_path)
        self.sample_indices: list[int] = []

    @property
    def trip_key(self) -> str:
        relative_trip_dir = self.trip_dir.relative_to(self.run_path)
        return (Path(self.run_path.name) / relative_trip_dir).as_posix()

    def load_data(self) -> dict[str, Any]:
        metadata_path = self.trip_dir / "metadata.json"
        video_path = self.trip_dir / "video.mkv"
        log_path = self.trip_dir / "video.log"
        dataset_path = self.trip_dir / "dataset.jsonl"
        processing_path = self.trip_dir / "processing.json"

        if not self.trip_dir.is_dir():
            raise FileNotFoundError(f"Trip directory does not exist: {self.trip_dir}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Trip metadata is missing: {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        samples = self.load_samples()
        return {
            "trip_dir": self.trip_dir,
            "trip_name": self.trip_dir.name,
            "trip_key": self.trip_key,
            "run_path": self.run_path,
            "run_name": self.run_path.name,
            "metadata": metadata,
            "metadata_path": metadata_path,
            "video_path": video_path,
            "log_path": log_path,
            "dataset_path": dataset_path,
            "processing_path": processing_path,
            "samples": samples,
            "sample_indices": list(self.sample_indices),
            "run_id": metadata.get("runId"),
            "scene_id": metadata.get("sceneId"),
            "scene_variant": metadata.get("sceneVariant"),
            "trip_index": metadata.get("tripIndex"),
        }

    def load_samples(self) -> list[dict[str, Any]]:
        dataset_path = self.trip_dir / "dataset.jsonl"
        if not dataset_path.is_file():
            return []

        samples: list[dict[str, Any]] = []
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                sample["trip_dir"] = self.trip_dir
                sample["run_path"] = self.run_path
                sample["run_name"] = self.run_path.name
                sample["trip_key"] = self.trip_key
                sample["frame_paths"] = [self.trip_dir / Path(path) for path in sample.get("frame_paths", [])]
                samples.append(sample)
        return samples


def _coerce_float(mapping: TelemetryMap, key: str) -> float:
    if key not in mapping:
        raise KeyError(f"missing telemetry key '{key}'")
    value = mapping[key]
    if isinstance(value, bool):
        raise TypeError(f"telemetry key '{key}' must be numeric, got bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"telemetry key '{key}' must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"telemetry key '{key}' must be finite")
    return result


def wrap_degrees_delta(current_yaw: float, future_yaw: float) -> float:
    return ((future_yaw - current_yaw + 180.0) % 360.0) - 180.0


def flatten_grouped_mapping(mapping: TelemetryMap) -> TelemetryMap:
    if not isinstance(mapping, dict):
        raise TypeError(f"expected grouped telemetry mapping, got {type(mapping).__name__}")
    if "control" not in mapping and "aux" not in mapping and "raw" not in mapping:
        return mapping

    flat: TelemetryMap = {}
    for section_name in ("control", "aux", "raw"):
        section = mapping.get(section_name)
        if section is None:
            continue
        if not isinstance(section, dict):
            raise TypeError(f"telemetry section '{section_name}' must be a dict")
        flat.update(section)
    return flat


def build_telemetry_features(telemetry: TelemetryMap, feature_names: tuple[str, ...]) -> Tensor:
    telemetry = flatten_grouped_mapping(telemetry)
    current_speed = _coerce_float(telemetry, "currentSpeed")
    yaw_degrees = _coerce_float(telemetry, "yaw")
    yaw_radians = math.radians(yaw_degrees)
    yaw_rate = _coerce_float(telemetry, "yawRate")
    steering = _coerce_float(telemetry, "Steering")
    acceleration = _coerce_float(telemetry, "acceleration")

    feature_map = {
        "current_speed": current_speed,
        "yaw_sin": math.sin(yaw_radians),
        "yaw_cos": math.cos(yaw_radians),
        "yaw_rate": yaw_rate,
        "steering": steering,
        "acceleration": acceleration,
    }
    return torch.tensor([feature_map[name] for name in feature_names], dtype=torch.float32)


def build_control_targets(telemetry: TelemetryMap, target_names: tuple[str, ...]) -> Tensor:
    telemetry = flatten_grouped_mapping(telemetry)
    target_map = {
        "steering": _coerce_float(telemetry, "Steering"),
        "acceleration": _coerce_float(telemetry, "acceleration"),
        "brakePressureAvg": _coerce_float(telemetry, "brakePressureAvg"),
    }
    return torch.tensor([target_map[name] for name in target_names], dtype=torch.float32)


def build_aux_targets(current_telemetry: TelemetryMap, future_telemetry: TelemetryMap, target_names: tuple[str, ...]) -> Tensor:
    current_telemetry = flatten_grouped_mapping(current_telemetry)
    future_telemetry = flatten_grouped_mapping(future_telemetry)
    current_yaw = _coerce_float(current_telemetry, "yaw")
    future_yaw = _coerce_float(future_telemetry, "yaw")
    target_map = {
        "future_speed": _coerce_float(future_telemetry, "currentSpeed"),
        "future_yaw_delta": wrap_degrees_delta(current_yaw, future_yaw),
        "future_yaw_rate": _coerce_float(future_telemetry, "yawRate"),
    }
    return torch.tensor([target_map[name] for name in target_names], dtype=torch.float32)


class FsdDataset(Dataset[DatasetItem]):
    def __init__(
        self,
        run_paths: Iterable[str | Path] | None = None,
        *,
        run_id: str | None = None,
        data_root: str | Path | None = None,
        image_size: tuple[int, int] = (480, 480),
        expected_window_size: int | None = None,
        image_offsets: tuple[int, ...] = DEFAULT_IMAGE_OFFSETS,
        telemetry_offsets: tuple[int, ...] = DEFAULT_TELEMETRY_OFFSETS,
        future_offsets: tuple[int, ...] = DEFAULT_FUTURE_OFFSETS,
        telemetry_feature_names: tuple[str, ...] = DEFAULT_TELEMETRY_FEATURE_NAMES,
        control_target_names: tuple[str, ...] = DEFAULT_CONTROL_TARGET_NAMES,
        aux_target_names: tuple[str, ...] = DEFAULT_AUX_TARGET_NAMES,
        state_input_config: Any | None = None,
        head_specs: Any | None = None,
    ):
        del state_input_config, head_specs
        if expected_window_size is not None and expected_window_size != len(image_offsets):
            raise ValueError(
                "expected_window_size must match the configured sparse image offset count: "
                f"expected_window_size={expected_window_size} image_offsets={len(image_offsets)}"
            )

        self.image_offsets = tuple(image_offsets)
        self.telemetry_offsets = tuple(telemetry_offsets)
        self.future_offsets = tuple(future_offsets)
        self.telemetry_feature_names = tuple(telemetry_feature_names)
        self.control_target_names = tuple(control_target_names)
        self.aux_target_names = tuple(aux_target_names)
        self.image_size = image_size
        self.data_root = None if data_root is None else Path(data_root)
        self.run_paths: list[Path] = self._resolve_run_paths(run_paths, run_id=run_id, data_root=data_root)
        self.trips: list[Trip] = self._load_trips()
        self.rejected_sample_summary: dict[str, Any] = {
            "rejected_samples": 0,
            "bad_frame_paths": 0,
            "bad_telemetry_history": 0,
            "bad_telemetry_future": 0,
            "non_dict_history_items": 0,
            "non_dict_future_items": 0,
            "observed_frame_path_lengths": {},
            "observed_telemetry_history_lengths": {},
            "observed_telemetry_future_lengths": {},
        }
        self.samples: list[dict[str, Any]] = self._load_samples()

    def _resolve_run_paths(
        self,
        run_paths: Iterable[str | Path] | None,
        *,
        run_id: str | None,
        data_root: str | Path | None,
    ) -> list[Path]:
        resolved: list[Path] = []
        if run_paths is not None:
            for raw_path in run_paths:
                path = Path(raw_path)
                if not str(path).strip():
                    raise ValueError("run_paths entries must be non-empty")
                resolved.append(path)

        if resolved:
            return resolved

        if run_id is None or data_root is None:
            raise ValueError("Provide run_paths or both run_id and data_root")
        return [Path(data_root) / "runs" / run_id]

    def _resolve_scene_dirs(self, run_path: Path) -> list[Path]:
        if not run_path.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {run_path}")

        scene_dirs = sorted(path for path in run_path.iterdir() if path.is_dir())
        if not scene_dirs:
            raise FileNotFoundError(f"No scene directory found in run: {run_path}")
        return scene_dirs

    def _load_trips(self) -> list[Trip]:
        trips: list[Trip] = []
        for run_path in self.run_paths:
            for scene_dir in self._resolve_scene_dirs(run_path):
                trip_dirs = sorted(
                    path
                    for path in scene_dir.iterdir()
                    if path.is_dir() and path.name.startswith("trip-")
                )
                for trip_dir in trip_dirs:
                    trips.append(Trip(trip_dir, run_path=run_path))
        return trips

    @staticmethod
    def _increment_count(bucket: dict[int, int], value: int) -> None:
        bucket[value] = int(bucket.get(value, 0)) + 1

    def _sample_has_required_offsets(self, sample: dict[str, Any]) -> bool:
        frame_paths = sample.get("frame_paths", [])
        history = sample.get("telemetry_history", [])
        future = sample.get("telemetry_future", [])

        ok = True
        self._increment_count(self.rejected_sample_summary["observed_frame_path_lengths"], len(frame_paths))
        self._increment_count(self.rejected_sample_summary["observed_telemetry_history_lengths"], len(history))
        self._increment_count(self.rejected_sample_summary["observed_telemetry_future_lengths"], len(future))

        if len(frame_paths) != len(self.image_offsets):
            self.rejected_sample_summary["bad_frame_paths"] += 1
            ok = False
        if len(history) < len(self.telemetry_offsets):
            self.rejected_sample_summary["bad_telemetry_history"] += 1
            ok = False
        if len(future) < len(self.future_offsets):
            self.rejected_sample_summary["bad_telemetry_future"] += 1
            ok = False

        if not all(isinstance(item, dict) for item in history):
            self.rejected_sample_summary["non_dict_history_items"] += 1
            ok = False
        if not all(isinstance(item, dict) for item in future):
            self.rejected_sample_summary["non_dict_future_items"] += 1
            ok = False
        if not ok:
            self.rejected_sample_summary["rejected_samples"] += 1
        return ok

    def _load_samples(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for trip in self.trips:
            trip_samples = trip.load_samples()
            trip.sample_indices = []
            for sample in trip_samples:
                if not self._sample_has_required_offsets(sample):
                    continue
                trip.sample_indices.append(len(samples))
                samples.append(sample)
        return samples

    def format_rejected_sample_summary(self) -> str:
        summary = self.rejected_sample_summary
        return (
            "rejected_samples_summary="
            f"rejected={summary['rejected_samples']} "
            f"bad_frame_paths={summary['bad_frame_paths']} "
            f"bad_telemetry_history={summary['bad_telemetry_history']} "
            f"bad_telemetry_future={summary['bad_telemetry_future']} "
            f"non_dict_history_items={summary['non_dict_history_items']} "
            f"non_dict_future_items={summary['non_dict_future_items']} "
            f"observed_frame_path_lengths={summary['observed_frame_path_lengths']} "
            f"observed_telemetry_history_lengths={summary['observed_telemetry_history_lengths']} "
            f"observed_telemetry_future_lengths={summary['observed_telemetry_future_lengths']} "
            f"expected_frame_paths={len(self.image_offsets)} "
            f"expected_telemetry_history>={len(self.telemetry_offsets)} "
            f"expected_telemetry_future>={len(self.future_offsets)}"
        )

    @property
    def trip_count(self) -> int:
        return len(self.trips)

    def trip_sample_indices(self) -> list[list[int]]:
        return [list(trip.sample_indices) for trip in self.trips]

    def load_trip_data(self, trip_index: int) -> dict[str, Any]:
        if trip_index < 0 or trip_index >= len(self.trips):
            raise IndexError(f"Trip index out of range: {trip_index}")
        return self.trips[trip_index].load_data()

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frame_tensor(self, frame_path: Path) -> Tensor:
        return load_rgb_tensor_from_path(frame_path, self.image_size)

    def _history_item_for_offset(self, history: list[TelemetryMap], offset: int) -> TelemetryMap:
        index = offset + len(history) - 1
        return history[index]

    def _future_item_for_offset(self, future: list[TelemetryMap], offset: int) -> TelemetryMap:
        return future[offset - 1]

    def __getitem__(self, index: int) -> DatasetItem:
        sample = self.samples[index]
        frame_paths = sample.get("frame_paths", [])
        if len(frame_paths) != len(self.image_offsets):
            raise ValueError(
                "Expected frame paths aligned to image_offsets, "
                f"found {len(frame_paths)} paths for {len(self.image_offsets)} offsets"
            )

        frame_tensors: list[Tensor] = []
        for frame_path in frame_paths:
            if not frame_path.is_file():
                raise FileNotFoundError(f"Frame image not found: {frame_path}")
            frame_tensors.append(self._load_frame_tensor(frame_path))

        history = sample.get("telemetry_history", [])
        future = sample.get("telemetry_future", [])
        current_telemetry = self._history_item_for_offset(history, 0)

        telemetry_tensors = [
            build_telemetry_features(self._history_item_for_offset(history, offset), self.telemetry_feature_names)
            for offset in self.telemetry_offsets
        ]
        control_targets = [
            build_control_targets(self._future_item_for_offset(future, offset), self.control_target_names)
            for offset in self.future_offsets
        ]
        aux_targets = [
            build_aux_targets(current_telemetry, self._future_item_for_offset(future, offset), self.aux_target_names)
            for offset in self.future_offsets
        ]

        images = torch.stack(frame_tensors, dim=0)
        telemetry = torch.stack(telemetry_tensors, dim=0)
        target_controls = torch.stack(control_targets, dim=0)
        target_aux = torch.stack(aux_targets, dim=0)
        return images, telemetry, target_controls, target_aux
