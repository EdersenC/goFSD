from __future__ import annotations

import os
from typing import Any

from target_transforms import (
    DeltaSpeedTargetTransform,
    delta_speed_target_transform_from_config,
)


DEFAULT_IMAGE_WIDTH = 480
DEFAULT_IMAGE_HEIGHT = 480
DEFAULT_WINDOW_SIZE = 3
DEFAULT_FRAME_STRIDE = 2
DEFAULT_SAMPLE_STRIDE = 2

DEFAULT_IMAGE_OFFSETS = (-8, -6, -4, -2, 0)
DEFAULT_TELEMETRY_OFFSETS = (-8, -7, -6, -5, -4, -3, -2, -1, 0)
DEFAULT_FUTURE_OFFSETS = (1, 2, 3, 4, 5, 6)
DEFAULT_TELEMETRY_FEATURE_NAMES = (
    "current_speed",
    "yaw_sin",
    "yaw_cos",
    "yaw_rate",
    "steering",
    "acceleration",
)
DEFAULT_CONTROL_TARGET_NAMES = ("steering", "acceleration", "brakePressureAvg")
DEFAULT_AUX_TARGET_NAMES = ("future_speed", "future_yaw_delta", "future_yaw_rate")
DEFAULT_AUX_LOSS_WEIGHT = 0.3
DEFAULT_HORIZON_LOSS_WEIGHTS = (1.0, 0.9, 0.8, 0.65, 0.5, 0.4)
DEFAULT_TELEMETRY_HIDDEN_DIM = 128


def normalize_windows_drive_path(value: str) -> str:
    cleaned = value.strip().strip("\"'")
    if os.name == "nt" and len(cleaned) >= 2 and cleaned[1] == ":":
        if len(cleaned) == 2:
            cleaned += "\\"
        elif cleaned[2] not in ("\\", "/"):
            cleaned = f"{cleaned[:2]}\\{cleaned[2:]}"
    return cleaned


def validate_frame_window(window_size: int, frame_stride: int, sample_stride: int, *, prefix: str = "dataset") -> None:
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(f"{prefix}.window_size must be a positive odd number")
    if frame_stride < 1:
        raise ValueError(f"{prefix}.frame_stride must be > 0")
    if sample_stride < 1:
        raise ValueError(f"{prefix}.sample_stride must be > 0")


def parse_dataset_window(raw: dict[str, Any]) -> tuple[int, int, int]:
    dataset_raw = raw.get("dataset", {})
    if "window_stride" in dataset_raw:
        raise ValueError(
            "dataset.window_stride is no longer supported; "
            "use dataset.frame_stride and dataset.sample_stride"
        )
    window_size = int(dataset_raw.get("window_size", DEFAULT_WINDOW_SIZE))
    frame_stride_raw = dataset_raw.get("frame_stride")
    if frame_stride_raw is None:
        raise ValueError("dataset.frame_stride must be configured explicitly")
    sample_stride_raw = dataset_raw.get("sample_stride")
    if sample_stride_raw is None:
        raise ValueError("dataset.sample_stride must be configured explicitly")

    frame_stride = int(frame_stride_raw)
    sample_stride = int(sample_stride_raw)
    validate_frame_window(window_size, frame_stride, sample_stride)
    return window_size, frame_stride, sample_stride


def _parse_int_offset_list(raw_value: Any, *, key: str, allow_positive: bool, allow_negative: bool) -> tuple[int, ...]:
    if not isinstance(raw_value, list) or not raw_value:
        raise ValueError(f"{key} must be a non-empty TOML array")
    offsets = tuple(int(item) for item in raw_value)
    if len(set(offsets)) != len(offsets):
        raise ValueError(f"{key} must not contain duplicate offsets")
    if tuple(sorted(offsets)) != offsets:
        raise ValueError(f"{key} must be sorted in ascending order")
    if not allow_negative and any(item < 0 for item in offsets):
        raise ValueError(f"{key} must not contain negative offsets")
    if not allow_positive and any(item > 0 for item in offsets):
        raise ValueError(f"{key} must not contain positive offsets")
    return offsets


def _parse_name_list(raw_value: Any, *, key: str) -> tuple[str, ...]:
    if not isinstance(raw_value, list) or not raw_value:
        raise ValueError(f"{key} must be a non-empty TOML array")
    names = tuple(str(item).strip() for item in raw_value if str(item).strip())
    if not names:
        raise ValueError(f"{key} must contain at least one non-empty name")
    if len(set(names)) != len(names):
        raise ValueError(f"{key} must not contain duplicate names")
    return names


def parse_temporal_dataset_config(
    raw: dict[str, Any],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    dataset_raw = raw.get("dataset", {})
    image_offsets = _parse_int_offset_list(
        dataset_raw.get("image_offsets", list(DEFAULT_IMAGE_OFFSETS)),
        key="dataset.image_offsets",
        allow_positive=False,
        allow_negative=True,
    )
    telemetry_offsets = _parse_int_offset_list(
        dataset_raw.get("telemetry_offsets", list(DEFAULT_TELEMETRY_OFFSETS)),
        key="dataset.telemetry_offsets",
        allow_positive=False,
        allow_negative=True,
    )
    future_offsets = _parse_int_offset_list(
        dataset_raw.get("future_offsets", list(DEFAULT_FUTURE_OFFSETS)),
        key="dataset.future_offsets",
        allow_positive=True,
        allow_negative=False,
    )

    if image_offsets[-1] != 0:
        raise ValueError("dataset.image_offsets must end at 0")
    if telemetry_offsets[-1] != 0:
        raise ValueError("dataset.telemetry_offsets must end at 0")
    if future_offsets[0] <= 0:
        raise ValueError("dataset.future_offsets must start at a positive offset")

    telemetry_feature_names = _parse_name_list(
        dataset_raw.get("telemetry_feature_names", list(DEFAULT_TELEMETRY_FEATURE_NAMES)),
        key="dataset.telemetry_feature_names",
    )
    control_target_names = _parse_name_list(
        dataset_raw.get("control_target_names", list(DEFAULT_CONTROL_TARGET_NAMES)),
        key="dataset.control_target_names",
    )
    aux_target_names = _parse_name_list(
        dataset_raw.get("aux_target_names", list(DEFAULT_AUX_TARGET_NAMES)),
        key="dataset.aux_target_names",
    )
    return (
        image_offsets,
        telemetry_offsets,
        future_offsets,
        telemetry_feature_names,
        control_target_names,
        aux_target_names,
    )


def parse_delta_speed_target_transform(raw: dict[str, Any]) -> DeltaSpeedTargetTransform:
    dataset_raw = raw.get("dataset", {})
    return delta_speed_target_transform_from_config(dataset_raw)
