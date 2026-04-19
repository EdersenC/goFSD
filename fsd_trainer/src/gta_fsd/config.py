from __future__ import annotations

import os
from typing import Any

from target_transforms import (
    DeltaSpeedTargetTransform,
    delta_speed_target_transform_from_config,
)


DEFAULT_IMAGE_WIDTH = 224
DEFAULT_IMAGE_HEIGHT = 224
DEFAULT_WINDOW_SIZE = 3
DEFAULT_FRAME_STRIDE = 2
DEFAULT_SAMPLE_STRIDE = 2


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


def parse_delta_speed_target_transform(raw: dict[str, Any]) -> DeltaSpeedTargetTransform:
    dataset_raw = raw.get("dataset", {})
    return delta_speed_target_transform_from_config(dataset_raw)
