"""Helpers for loading processed trip datasets from one or more run directories."""

from __future__ import annotations

import json
import math
import re
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
    STOP_SIGN_AUX_TARGET_NAMES,
)
from control_contract import (
    DEFAULT_TELEMETRY_SAMPLE_INTERVAL_MS,
    FUTURE_SPEED_MPS,
    MAX_DESIRED_SPEED_MPS,
    STOP_INTENT,
)
from image_io import load_rgb_uint8_tensor_from_path
from target_transforms import (
    TargetTransform,
    build_target_transform_registry,
    normalize_target_tensor,
)
from state_inputs import build_state_input_vector_from_mapping, state_input_config_from_metadata
from stop_sign_contract import (
    inverse_frequency_phase_weights,
    stop_location_key_from_metadata,
    stop_sign_phase_from_telemetry,
)


DatasetImages = Tensor
DatasetTelemetry = Tensor
DatasetStateInputs = Tensor
DatasetTargetControls = Tensor
DatasetTargetAux = Tensor
DatasetItem = tuple[DatasetImages, DatasetTelemetry, DatasetStateInputs, DatasetTargetControls, DatasetTargetAux]

TelemetryMap = dict[str, Any]
PROCESSING_FINGERPRINT_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
NUMBERED_JPEG_NAME_PATTERN = re.compile(r"[0-9]{6}\.jpg")


def _processing_integer_list(status: Mapping[str, Any], key: str, processing_path: Path) -> tuple[int, ...]:
    raw_values = status.get(key)
    if not isinstance(raw_values, list) or not raw_values:
        raise ValueError(f"processed dataset contract is missing {key}: {processing_path}")
    values: list[int] = []
    for index, value in enumerate(raw_values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"processed dataset contract {key}[{index}] must be an integer: {processing_path}")
        values.append(value)
    return tuple(values)


def _processing_count(
    status: Mapping[str, Any],
    key: str,
    processing_path: Path,
    *,
    minimum: int,
) -> int:
    value = status.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"processed dataset contract {key} must be an integer >= {minimum}: {processing_path}"
        )
    return value


def _validate_numbered_frame_set(trip_dir: Path, frame_count: int) -> None:
    frames_dir = trip_dir / "frames"
    try:
        regular_files = {
            entry.name: entry.stat().st_size
            for entry in frames_dir.iterdir()
            if not entry.is_symlink() and entry.is_file()
        }
    except OSError as exc:
        raise ValueError(f"failed to inspect processed frame set: {frames_dir}") from exc

    expected_files = {f"{index:06d}.jpg" for index in range(1, frame_count + 1)}
    if set(regular_files) != expected_files or any(size < 1 for size in regular_files.values()):
        raise ValueError(
            "processed frame set must contain exactly the contiguous numbered files "
            f"000001.jpg..{frame_count:06d}.jpg: {frames_dir}"
        )


def _dataset_frame_name(raw_path: Any, *, dataset_path: Path, line_number: int) -> str:
    if not isinstance(raw_path, str):
        raise ValueError(
            f"processed dataset frame_paths must contain strings at line {line_number}: {dataset_path}"
        )
    parts = raw_path.split("/")
    if (
        "\\" in raw_path
        or len(parts) != 2
        or parts[0] != "frames"
        or NUMBERED_JPEG_NAME_PATTERN.fullmatch(parts[1]) is None
        or parts[1] == "000000.jpg"
    ):
        raise ValueError(
            f"processed dataset has invalid frame path {raw_path!r} at line {line_number}: {dataset_path}"
        )
    return parts[1]


def _read_dataset_frame_references(dataset_path: Path) -> tuple[int, set[str]]:
    row_count = 0
    referenced_frames: set[str] = set()
    try:
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"processed dataset contains invalid JSON at line {line_number}: {dataset_path}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"processed dataset row must be a JSON object at line {line_number}: {dataset_path}"
                    )
                frame_paths = row.get("frame_paths")
                if not isinstance(frame_paths, list) or not frame_paths:
                    raise ValueError(
                        f"processed dataset row is missing non-empty frame_paths at line {line_number}: {dataset_path}"
                    )
                for frame_path in frame_paths:
                    referenced_frames.add(
                        _dataset_frame_name(
                            frame_path,
                            dataset_path=dataset_path,
                            line_number=line_number,
                        )
                    )
                row_count += 1
    except FileNotFoundError as exc:
        raise ValueError(f"processed dataset is missing: {dataset_path}") from exc
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"failed to read processed dataset: {dataset_path}") from exc
    return row_count, referenced_frames


def _validate_referenced_frame_set(
    trip_dir: Path,
    referenced_frames: set[str],
    frame_count: int,
) -> None:
    frames_dir = trip_dir / "frames"
    if len(referenced_frames) != frame_count:
        raise ValueError(
            "processed dataset referenced frame count mismatch: "
            f"expected={frame_count} actual={len(referenced_frames)} path={frames_dir}"
        )

    try:
        jpeg_entries = {
            entry.name: entry
            for entry in frames_dir.iterdir()
            if entry.name.endswith(".jpg")
        }
    except OSError as exc:
        raise ValueError(f"failed to inspect processed frame set: {frames_dir}") from exc

    if set(jpeg_entries) != referenced_frames:
        missing = sorted(referenced_frames - set(jpeg_entries))
        extra = sorted(set(jpeg_entries) - referenced_frames)
        raise ValueError(
            "processed frame set must contain exactly the dataset-referenced JPEG files: "
            f"missing={missing} extra={extra} path={frames_dir}"
        )

    for name, entry in jpeg_entries.items():
        try:
            if entry.is_symlink() or not entry.is_file() or entry.stat().st_size < 1:
                raise ValueError(
                    "processed frame set must contain only non-empty regular dataset-referenced JPEG files: "
                    f"{frames_dir / name}"
                )
        except OSError as exc:
            raise ValueError(f"failed to inspect processed frame: {frames_dir / name}") from exc


def _validate_processed_dataset_contract(
    trip_dir: Path,
    *,
    image_size: tuple[int, int],
    image_offsets: tuple[int, ...],
    telemetry_offsets: tuple[int, ...],
    future_offsets: tuple[int, ...],
    telemetry_sample_interval_ms: int,
) -> None:
    processing_path = trip_dir / "processing.json"
    try:
        status = json.loads(processing_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"processed dataset contract is missing: {processing_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read processed dataset contract: {processing_path}") from exc
    if not isinstance(status, dict):
        raise ValueError(f"processed dataset contract must be a JSON object: {processing_path}")

    state = str(status.get("state", "")).strip().lower()
    if state not in {"completed", "skipped"}:
        raise ValueError(f"processed dataset contract is not terminal: state={state or 'missing'} path={processing_path}")
    fingerprint = status.get("configFingerprint")
    if not isinstance(fingerprint, str) or PROCESSING_FINGERPRINT_PATTERN.fullmatch(fingerprint) is None:
        raise ValueError(f"processed dataset contract has no valid config fingerprint: {processing_path}")

    expected_contract = {
        "imageOffsets": image_offsets,
        "telemetryOffsets": telemetry_offsets,
        "futureOffsets": future_offsets,
    }
    for key, expected in expected_contract.items():
        actual = _processing_integer_list(status, key, processing_path)
        if actual != expected:
            raise ValueError(
                f"processed dataset contract {key} mismatch: expected={list(expected)} "
                f"actual={list(actual)} path={processing_path}"
            )

    raw_interval = status.get("telemetrySampleIntervalMs")
    if isinstance(raw_interval, bool) or not isinstance(raw_interval, int):
        raise ValueError(f"processed dataset contract telemetrySampleIntervalMs must be an integer: {processing_path}")
    if raw_interval != telemetry_sample_interval_ms:
        raise ValueError(
            "processed dataset contract telemetrySampleIntervalMs mismatch: "
            f"expected={telemetry_sample_interval_ms} actual={raw_interval} path={processing_path}"
        )

    for key, expected in (("imageWidth", image_size[0]), ("imageHeight", image_size[1])):
        actual = status.get(key)
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
            raise ValueError(
                f"processed dataset contract {key} mismatch: expected={expected} "
                f"actual={actual!r} path={processing_path}"
            )

    frame_count = _processing_count(status, "frameCount", processing_path, minimum=0)
    sample_count = _processing_count(status, "sampleCount", processing_path, minimum=0)
    dataset_path = trip_dir / "dataset.jsonl"
    actual_sample_count, referenced_frames = _read_dataset_frame_references(dataset_path)
    if actual_sample_count != sample_count:
        raise ValueError(
            "processed dataset sampleCount mismatch: "
            f"expected={sample_count} actual={actual_sample_count} path={dataset_path}"
        )
    if sample_count == 0:
        _validate_numbered_frame_set(trip_dir, frame_count)
        return
    _validate_referenced_frame_set(trip_dir, referenced_frames, frame_count)


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
    phase: str
    stop_location_key: str

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


def _is_successful_stop_sign_attempt(trip_dir: Path) -> bool:
    metadata = _load_trip_metadata_for_filtering(trip_dir)
    if metadata is None:
        return False
    goal = metadata.get("stopSignGoal")
    outcome = metadata.get("stopSignOutcome")
    return (
        str(metadata.get("sceneId", "")).strip().lower() == "stop-sign"
        and str(metadata.get("sceneVariant", "")).strip().lower() == "temporal-v1"
        and isinstance(goal, Mapping)
        and str(goal.get("task", "")).strip().lower() == "stop-sign"
        and str(goal.get("contract", "")).strip().lower() == "stop-sign-goal.v1"
        and all(
            _is_complete_stop_sign_pose(goal.get(key))
            for key in ("signPose", "stopLinePose", "egoStopPose", "startPose")
        )
        and isinstance(outcome, Mapping)
        and outcome.get("success") is True
        and str(outcome.get("status", "")).strip().lower() == "succeeded"
    )


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _is_complete_stop_sign_pose(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return all(_is_finite_number(value.get(key)) for key in ("x", "y", "z", "heading"))


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
    if feature_names != DEFAULT_TELEMETRY_FEATURE_NAMES:
        raise ValueError(
            "stop-sign telemetry features must be exactly ['current_speed']"
        )
    return torch.tensor(
        [_coerce_float(telemetry, "currentSpeed")],
        dtype=torch.float32,
    )


def build_control_targets(telemetry: TelemetryMap, target_names: tuple[str, ...]) -> Tensor:
    telemetry = flatten_grouped_mapping(telemetry)
    desired_speed = _coerce_float(telemetry, "expertDesiredSpeedMps")
    if desired_speed < 0.0 or desired_speed > MAX_DESIRED_SPEED_MPS:
        raise ValueError(
            "expertDesiredSpeedMps must be within "
            f"[0, {MAX_DESIRED_SPEED_MPS}], got {desired_speed}"
        )
    target_map = {
        FUTURE_SPEED_MPS: desired_speed,
        STOP_INTENT: derive_stop_intent(telemetry),
    }
    return torch.tensor([target_map[name] for name in target_names], dtype=torch.float32)


def derive_stop_intent(telemetry: TelemetryMap) -> float:
    telemetry = flatten_grouped_mapping(telemetry)
    stop_probability = _coerce_float(telemetry, "expertStopProbability")
    if stop_probability < 0.0 or stop_probability > 1.0:
        raise ValueError(
            "expertStopProbability must be within [0, 1], "
            f"got {stop_probability}"
        )
    return stop_probability


def build_aux_targets(current_telemetry: TelemetryMap, future_telemetry: TelemetryMap, target_names: tuple[str, ...]) -> Tensor:
    del current_telemetry
    future_telemetry = flatten_grouped_mapping(future_telemetry)
    target_builders = {
        "expert_throttle": lambda: _bounded_unit_interval(future_telemetry, "expertThrottle"),
        "expert_brake": lambda: _bounded_unit_interval(future_telemetry, "expertBrake"),
        "actual_brake_pressure": lambda: _bounded_unit_interval(future_telemetry, "brakePressureAvg"),
    }
    return torch.tensor([target_builders[name]() for name in target_names], dtype=torch.float32)


def _bounded_unit_interval(telemetry: TelemetryMap, key: str) -> float:
    value = _coerce_float(telemetry, key)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"telemetry key '{key}' must be within [0, 1], got {value}")
    return value


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
        telemetry_sample_interval_ms: int = DEFAULT_TELEMETRY_SAMPLE_INTERVAL_MS,
        telemetry_feature_names: tuple[str, ...] = DEFAULT_TELEMETRY_FEATURE_NAMES,
        control_target_names: tuple[str, ...] = DEFAULT_CONTROL_TARGET_NAMES,
        aux_target_names: tuple[str, ...] = DEFAULT_AUX_TARGET_NAMES,
        target_transforms: Mapping[str, TargetTransform] | None = None,
        state_input_config: Any | None = None,
        include_failed_or_non_stop_sign_trips: bool = False,
    ):
        if expected_window_size is not None and expected_window_size != len(image_offsets):
            raise ValueError(
                "expected_window_size must match the configured sparse image offset count: "
                f"expected_window_size={expected_window_size} image_offsets={len(image_offsets)}"
            )
        if (
            isinstance(telemetry_sample_interval_ms, bool)
            or not isinstance(telemetry_sample_interval_ms, int)
            or telemetry_sample_interval_ms <= 0
        ):
            raise ValueError("telemetry_sample_interval_ms must be a positive integer")

        self.image_offsets = tuple(image_offsets)
        self.telemetry_offsets = tuple(telemetry_offsets)
        self.future_offsets = tuple(future_offsets)
        self.telemetry_sample_interval_ms = telemetry_sample_interval_ms
        self.telemetry_feature_names = tuple(telemetry_feature_names)
        self.control_target_names = tuple(control_target_names)
        self.aux_target_names = tuple(aux_target_names)
        if self.telemetry_feature_names != DEFAULT_TELEMETRY_FEATURE_NAMES:
            raise ValueError(
                "stop-sign telemetry features must be exactly ['current_speed']; "
                "oracle geometry and generic driving state are not model inputs"
            )
        if self.control_target_names != DEFAULT_CONTROL_TARGET_NAMES:
            raise ValueError(
                "stop-sign control targets must be exactly "
                f"{list(DEFAULT_CONTROL_TARGET_NAMES)}"
            )
        if self.aux_target_names != STOP_SIGN_AUX_TARGET_NAMES:
            raise ValueError(
                "stop-sign auxiliary targets must be exactly "
                f"{list(STOP_SIGN_AUX_TARGET_NAMES)}"
            )
        self.target_transforms = build_target_transform_registry(
            tuple(self.control_target_names) + tuple(self.aux_target_names),
            target_transforms,
        )
        self.state_input_config = state_input_config_from_metadata(state_input_config)
        if not isinstance(include_failed_or_non_stop_sign_trips, bool):
            raise TypeError("include_failed_or_non_stop_sign_trips must be boolean")
        self.include_failed_or_non_stop_sign_trips = include_failed_or_non_stop_sign_trips
        self.excluded_failed_or_non_stop_sign_trip_count = 0
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
                        not self.include_failed_or_non_stop_sign_trips
                        and not _is_successful_stop_sign_attempt(trip_dir)
                    ):
                        self.excluded_failed_or_non_stop_sign_trip_count += 1
                        continue
                    _validate_processed_dataset_contract(
                        trip_dir,
                        image_size=self.image_size,
                        image_offsets=self.image_offsets,
                        telemetry_offsets=self.telemetry_offsets,
                        future_offsets=self.future_offsets,
                        telemetry_sample_interval_ms=self.telemetry_sample_interval_ms,
                    )
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
        required_history_length = -min(self.telemetry_offsets) + 1
        required_future_length = max(self.future_offsets)
        if len(history) < required_history_length:
            self.rejected_sample_summary["bad_telemetry_history"] += 1
            ok = False
        if len(future) < required_future_length:
            self.rejected_sample_summary["bad_telemetry_future"] += 1
            ok = False

        if not all(isinstance(item, dict) for item in history):
            self.rejected_sample_summary["non_dict_history_items"] += 1
            ok = False
        if not all(isinstance(item, dict) for item in future):
            self.rejected_sample_summary["non_dict_future_items"] += 1
            ok = False
        if ok:
            try:
                for point in (*history, *future):
                    stop_sign_phase_from_telemetry(point)
            except (TypeError, ValueError):
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
            metadata = _load_trip_metadata_for_filtering(trip.trip_dir)
            if metadata is None:
                raise ValueError(f"trip metadata is missing: {trip.trip_dir / 'metadata.json'}")
            try:
                location_key = stop_location_key_from_metadata(metadata)
            except ValueError:
                if not self.include_failed_or_non_stop_sign_trips:
                    raise
                location_key = f"unscored:{trip.trip_key}"
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
                    phase = stop_sign_phase_from_telemetry(sample["telemetry_history"][-1])
                    trip.sample_indices.append(len(samples))
                    samples.append(DatasetSampleRef(
                        dataset_path=dataset_path,
                        byte_offset=byte_offset,
                        trip_dir=trip.trip_dir,
                        run_path=trip.run_path,
                        trip_key=trip.trip_key,
                        phase=phase,
                        stop_location_key=location_key,
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
            f"excluded_failed_or_non_stop_sign_trips={self.excluded_failed_or_non_stop_sign_trip_count} "
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
            f"expected_telemetry_history>={-min(self.telemetry_offsets) + 1} "
            f"expected_telemetry_future>={max(self.future_offsets)}"
        )

    @property
    def trip_count(self) -> int:
        return len(self.trips)

    def trip_sample_indices(self) -> list[list[int]]:
        return [list(trip.sample_indices) for trip in self.trips]

    def sample_phases(self) -> tuple[str, ...]:
        return tuple(sample.phase for sample in self.samples)

    def phase_balanced_sample_weights(self) -> tuple[float, ...]:
        return inverse_frequency_phase_weights(self.sample_phases())

    def stop_location_keys(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(sample.stop_location_key for sample in self.samples))

    def trip_stop_location_keys(self) -> tuple[str, ...]:
        keys: list[str] = []
        for trip in self.trips:
            if not trip.sample_indices:
                raise ValueError(f"stop-sign trip contains no usable samples: {trip.trip_key}")
            trip_keys = {self.samples[index].stop_location_key for index in trip.sample_indices}
            if len(trip_keys) != 1:
                raise AssertionError(f"trip spans multiple stop locations: {trip.trip_key}")
            keys.append(next(iter(trip_keys)))
        return tuple(keys)

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
