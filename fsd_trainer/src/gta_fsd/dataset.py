"""Helpers for loading processed trip datasets from one or more run directories."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from io import BufferedReader
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor
from torch.utils.data import Dataset, get_worker_info

from config import (
    DEFAULT_AUX_TARGET_NAMES,
    DEFAULT_CONTROL_TARGET_NAMES,
    DEFAULT_FUTURE_OFFSETS,
    DEFAULT_IMAGE_OFFSETS,
    DEFAULT_TELEMETRY_FEATURE_NAMES,
    DEFAULT_TELEMETRY_OFFSETS,
)
from control_contract import (
    DESIRED_SPEED_MPS,
    DESIRED_WHEEL_STEER_NORMALIZED,
    STOP_PROBABILITY,
)
from image_io import load_rgb_uint8_tensor_from_path
from target_transforms import (
    TargetTransform,
    build_target_transform_registry,
    normalize_target_tensor,
)
from state_inputs import build_state_input_vector_from_mapping, state_input_config_from_metadata


DatasetImages = Tensor
DatasetTelemetry = Tensor
DatasetStateInputs = Tensor
DatasetTargetControls = Tensor
DatasetTargetAux = Tensor
DatasetItem = tuple[DatasetImages, DatasetTelemetry, DatasetStateInputs, DatasetTargetControls, DatasetTargetAux]

TelemetryMap = dict[str, Any]
PARKING_SETTLING_MAX_SPEED_MPS = 0.15


def _attach_sample_context(
    sample: dict[str, Any],
    *,
    trip_dir: Path,
    run_path: Path,
    trip_key: str,
) -> dict[str, Any]:
    sample["trip_dir"] = trip_dir
    sample["run_path"] = run_path
    sample["run_name"] = run_path.name
    sample["trip_key"] = trip_key
    sample["frame_paths"] = [trip_dir / Path(path) for path in sample.get("frame_paths", [])]
    return sample


@dataclass(frozen=True)
class DatasetSampleRef:
    dataset_path: Path
    byte_offset: int
    trip_dir: Path
    run_path: Path
    trip_key: str

    def load(self) -> dict[str, Any]:
        with self.dataset_path.open("rb") as handle:
            return self.load_from_handle(handle)

    def load_from_handle(self, handle: BufferedReader) -> dict[str, Any]:
        handle.seek(self.byte_offset)
        raw_line = handle.readline()
        if not raw_line:
            raise IndexError(f"sample offset out of range: {self.dataset_path}:{self.byte_offset}")
        sample = json.loads(raw_line.decode("utf-8"))
        return _attach_sample_context(
            sample,
            trip_dir=self.trip_dir,
            run_path=self.run_path,
            trip_key=self.trip_key,
        )

    def get(self, key: str, default: Any = None) -> Any:
        return self.load().get(key, default)


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
                samples.append(_attach_sample_context(
                    sample,
                    trip_dir=self.trip_dir,
                    run_path=self.run_path,
                    trip_key=self.trip_key,
                ))
        return samples


def _load_trip_metadata_for_filtering(trip_dir: Path) -> dict[str, Any] | None:
    metadata_path = trip_dir / "metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read trip metadata: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"trip metadata must be a JSON object: {metadata_path}")
    return metadata


def _is_successful_parking_attempt(trip_dir: Path) -> bool:
    metadata = _load_trip_metadata_for_filtering(trip_dir)
    if metadata is None:
        return False
    parking_goal = metadata.get("parkingGoal")
    outcome = metadata.get("parkingOutcome")
    return (
        isinstance(parking_goal, Mapping)
        and str(parking_goal.get("task", "")).strip().lower() == "parking"
        and str(parking_goal.get("maneuver", "")).strip().lower() == "forward-bay"
        and isinstance(outcome, Mapping)
        and outcome.get("success") is True
        and str(outcome.get("status", "")).strip().lower() == "succeeded"
    )


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


def _coerce_bool(mapping: TelemetryMap, key: str) -> bool:
    if key not in mapping:
        raise KeyError(f"missing telemetry key '{key}'")
    value = mapping[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise TypeError(f"telemetry key '{key}' must be boolean")


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
    # Full-lock calibration is the capture-version marker that makes Steering's [-1, 1] meaning unambiguous.
    full_lock_radians = _coerce_float(telemetry, "wheelSteeringFullLock")
    if full_lock_radians <= 0.0:
        raise ValueError(
            "wheelSteeringFullLock must be positive; recapture parking data with the normalized steering contract"
        )
    desired_steer = _coerce_float(telemetry, "Steering")
    if desired_steer < -1.0 or desired_steer > 1.0:
        raise ValueError(
            f"Steering must be normalized to [-1, 1], got {desired_steer}; "
            "recapture parking data with the normalized steering contract"
        )
    desired_speed = _coerce_float(telemetry, "currentSpeed")
    if desired_speed < 0.0:
        raise ValueError(f"desired parking speed must be non-negative, got {desired_speed}")
    target_map = {
        DESIRED_WHEEL_STEER_NORMALIZED: desired_steer,
        DESIRED_SPEED_MPS: desired_speed,
        STOP_PROBABILITY: derive_stop_probability(telemetry),
    }
    return torch.tensor([target_map[name] for name in target_names], dtype=torch.float32)


def derive_stop_probability(telemetry: TelemetryMap) -> float:
    telemetry = flatten_grouped_mapping(telemetry)
    parked = _coerce_bool(telemetry, "parkingParked") if "parkingParked" in telemetry else False
    raw_phase = telemetry.get("parkingPhase")
    if raw_phase is None:
        if "parkingParked" not in telemetry:
            raise KeyError(
                "missing parking stop supervision: expected parkingPhase or parkingParked telemetry"
            )
        return 1.0 if parked else 0.0
    if not isinstance(raw_phase, str) or not raw_phase.strip():
        raise TypeError("telemetry key 'parkingPhase' must be a non-empty string")

    phase = raw_phase.strip().lower()
    known_phases = {
        "idle",
        "ready",
        "spawning",
        "recording",
        "parking",
        "settling",
        "succeeded",
        "failed",
        "stopping",
    }
    if phase not in known_phases:
        raise ValueError(f"unsupported parkingPhase for stop supervision: {raw_phase}")
    if phase in {"failed", "stopping"}:
        if parked:
            raise ValueError(
                f"contradictory parking stop supervision: phase={phase!r} cannot be parked"
            )
        return 0.0
    if parked:
        return 1.0
    if phase != "settling":
        return 0.0

    inside_bay = _coerce_bool(telemetry, "parkingInsideBay")
    aligned = _coerce_bool(telemetry, "parkingAligned")
    speed_mps = _coerce_float(telemetry, "currentSpeed")
    valid_settling = (
        inside_bay
        and aligned
        and abs(speed_mps) <= PARKING_SETTLING_MAX_SPEED_MPS
    )
    return 1.0 if valid_settling else 0.0


def build_aux_targets(current_telemetry: TelemetryMap, future_telemetry: TelemetryMap, target_names: tuple[str, ...]) -> Tensor:
    current_telemetry = flatten_grouped_mapping(current_telemetry)
    future_telemetry = flatten_grouped_mapping(future_telemetry)
    current_yaw = _coerce_float(current_telemetry, "yaw")
    future_yaw = _coerce_float(future_telemetry, "yaw")
    current_speed = _coerce_float(current_telemetry, "currentSpeed")
    future_speed = _coerce_float(future_telemetry, "currentSpeed")
    target_map = {
        "future_speed": future_speed,
        "future_speed_delta": future_speed - current_speed,
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
        target_transforms: Mapping[str, TargetTransform] | None = None,
        state_input_config: Any | None = None,
        include_failed_or_nonparking_trips: bool = False,
    ):
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
        self.target_transforms = build_target_transform_registry(
            tuple(self.control_target_names) + tuple(self.aux_target_names),
            target_transforms,
        )
        self.state_input_config = state_input_config_from_metadata(state_input_config)
        if not isinstance(include_failed_or_nonparking_trips, bool):
            raise TypeError("include_failed_or_nonparking_trips must be boolean")
        self.include_failed_or_nonparking_trips = include_failed_or_nonparking_trips
        self.excluded_failed_or_nonparking_trip_count = 0
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
        self.samples: list[DatasetSampleRef] = self._load_samples()
        self._sample_file_handles: dict[Path, BufferedReader] = {}

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
                    if (
                        not self.include_failed_or_nonparking_trips
                        and not _is_successful_parking_attempt(trip_dir)
                    ):
                        self.excluded_failed_or_nonparking_trip_count += 1
                        continue
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

    def _load_samples(self) -> list[DatasetSampleRef]:
        samples: list[DatasetSampleRef] = []
        for trip in self.trips:
            dataset_path = trip.trip_dir / "dataset.jsonl"
            trip.sample_indices = []
            if not dataset_path.is_file():
                continue
            with dataset_path.open("rb") as handle:
                while True:
                    byte_offset = handle.tell()
                    raw_line = handle.readline()
                    if not raw_line:
                        break
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    sample = json.loads(raw_line.decode("utf-8"))
                    if not self._sample_has_required_offsets(sample):
                        continue
                    trip.sample_indices.append(len(samples))
                    samples.append(DatasetSampleRef(
                        dataset_path=dataset_path,
                        byte_offset=byte_offset,
                        trip_dir=trip.trip_dir,
                        run_path=trip.run_path,
                        trip_key=trip.trip_key,
                    ))
        return samples

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_sample_file_handles"] = {}
        return state

    def close(self) -> None:
        for handle in self._sample_file_handles.values():
            handle.close()
        self._sample_file_handles.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def format_rejected_sample_summary(self) -> str:
        summary = self.rejected_sample_summary
        return (
            "rejected_samples_summary="
            f"excluded_failed_or_nonparking_trips={self.excluded_failed_or_nonparking_trip_count} "
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
        return load_rgb_uint8_tensor_from_path(frame_path, self.image_size)

    def _load_sample(self, index: int) -> dict[str, Any]:
        ref = self.samples[index]
        if get_worker_info() is None:
            return ref.load()
        handle = self._sample_file_handles.get(ref.dataset_path)
        if handle is None or handle.closed:
            handle = ref.dataset_path.open("rb")
            self._sample_file_handles[ref.dataset_path] = handle
        return ref.load_from_handle(handle)

    def _history_item_for_offset(self, history: list[TelemetryMap], offset: int) -> TelemetryMap:
        index = offset + len(history) - 1
        return history[index]

    def _future_item_for_offset(self, future: list[TelemetryMap], offset: int) -> TelemetryMap:
        return future[offset - 1]

    def __getitem__(self, index: int) -> DatasetItem:
        sample = self._load_sample(index)
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
        raw_target_controls = torch.stack(control_targets, dim=0)
        raw_target_aux = torch.stack(aux_targets, dim=0)
        target_controls = normalize_target_tensor(
            raw_target_controls,
            self.control_target_names,
            self.target_transforms,
        )
        target_aux = normalize_target_tensor(
            raw_target_aux,
            self.aux_target_names,
            self.target_transforms,
        )
        state_inputs = build_state_input_vector_from_mapping(flatten_grouped_mapping(current_telemetry), self.state_input_config)
        return images, telemetry, state_inputs, target_controls, target_aux
