from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


DEFAULT_FUTURE_SPEED_DELTA_CLIP = 2.0
DEFAULT_FUTURE_SPEED_DELTA_NORMALIZE = True

TARGET_TRANSFORM_TYPE_SIGNED_CAP = "signed_cap"
TARGET_TRANSFORM_TYPE_POSITIVE_CAP = "positive_cap"
TARGET_TRANSFORM_TYPE_ANGLE_CAP = "angle_cap"
TARGET_TRANSFORM_TYPE_IDENTITY = "identity"

DEFAULT_TARGET_TRANSFORMS: dict[str, tuple[str, float, float]] = {
    "steering": (TARGET_TRANSFORM_TYPE_SIGNED_CAP, -1.0, 1.0),
    "acceleration": (TARGET_TRANSFORM_TYPE_SIGNED_CAP, -1.0, 1.0),
    "brakePressureAvg": (TARGET_TRANSFORM_TYPE_POSITIVE_CAP, 0.0, 1.0),
    "future_speed": (TARGET_TRANSFORM_TYPE_POSITIVE_CAP, 0.0, 80.0),
    "future_speed_delta": (TARGET_TRANSFORM_TYPE_SIGNED_CAP, -DEFAULT_FUTURE_SPEED_DELTA_CLIP, DEFAULT_FUTURE_SPEED_DELTA_CLIP),
    "future_yaw_delta": (TARGET_TRANSFORM_TYPE_ANGLE_CAP, -45.0, 45.0),
    "future_yaw_rate": (TARGET_TRANSFORM_TYPE_SIGNED_CAP, -10.0, 10.0),
}


@dataclass(frozen=True)
class TargetTransform:
    target_name: str
    transform_type: str
    range_min: float
    range_max: float
    normalize: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "transform_type", str(self.transform_type).strip().lower())
        if self.transform_type not in {
            TARGET_TRANSFORM_TYPE_SIGNED_CAP,
            TARGET_TRANSFORM_TYPE_POSITIVE_CAP,
            TARGET_TRANSFORM_TYPE_ANGLE_CAP,
            TARGET_TRANSFORM_TYPE_IDENTITY,
        }:
            raise ValueError(f"unsupported transform_type: {self.transform_type}")

        if not self._is_finite(self.range_min) or not self._is_finite(self.range_max):
            raise ValueError("target transform ranges must be finite")
        if self.range_min > self.range_max:
            raise ValueError("target transform range_min must be <= range_max")
        if self.transform_type == TARGET_TRANSFORM_TYPE_POSITIVE_CAP and self.range_min < 0:
            raise ValueError("positive_cap requires a non-negative range_min")
        if self.normalize:
            if self.range_scale() <= 0.0:
                raise ValueError("target transform normalization requires a non-zero range")

    @staticmethod
    def _is_finite(value: float) -> bool:
        return isinstance(value, (int, float)) and value == value and value != float("inf") and value != float("-inf")

    def is_identity(self) -> bool:
        return self.transform_type == TARGET_TRANSFORM_TYPE_IDENTITY or (
            self.range_min == 0.0 and self.range_max == 0.0
        )

    def is_positive_only(self) -> bool:
        return self.range_min >= 0.0 and self.transform_type != TARGET_TRANSFORM_TYPE_SIGNED_CAP

    def range_scale(self) -> float:
        return max(abs(self.range_min), abs(self.range_max))

    def metadata(self) -> dict[str, Any]:
        return {
            "target_name": self.target_name,
            "transform_type": self.transform_type,
            "range_min": float(self.range_min),
            "range_max": float(self.range_max),
            "normalize": bool(self.normalize),
            "positive_only": bool(self.is_positive_only()),
            "signed": bool(not self.is_positive_only()),
            "range_scale": float(self.range_scale()),
        }

    def _clamp(self, value: float) -> float:
        return max(self.range_min, min(self.range_max, float(value)))

    def normalize_value(self, value: float) -> float:
        if self.is_identity():
            return float(self._clamp(value))
        return self._clamp(value) / self.range_scale() if self.normalize else self._clamp(value)

    def denormalize_value(self, value: float) -> float:
        if self.is_identity():
            return float(self._clamp(value))
        raw = float(value) if self.normalize else float(value)
        return self._clamp(raw * self.range_scale()) if self.normalize else self._clamp(raw)

    def normalize_tensor(self, value: Tensor) -> Tensor:
        raw = value.float()
        if self.is_identity():
            return torch.clamp(raw, self.range_min, self.range_max)
        clamped = torch.clamp(raw, self.range_min, self.range_max)
        return clamped if not self.normalize else clamped / self.range_scale()

    def denormalize_tensor(self, value: Tensor) -> Tensor:
        raw = value.float()
        if self.is_identity():
            return torch.clamp(raw, self.range_min, self.range_max)
        clamped = torch.clamp(raw, -1.0, 1.0) if self.normalize else raw
        return torch.clamp(clamped * self.range_scale(), self.range_min, self.range_max)


def _parse_transform_type(raw_type: Any) -> str:
    resolved = str(raw_type).strip().lower()
    if not resolved:
        return ""
    if resolved in {
        "signed",
        "signed_cap",
        "signed-cap",
        "cap_signed",
    }:
        return TARGET_TRANSFORM_TYPE_SIGNED_CAP
    if resolved in {
        "positive",
        "positive_cap",
        "positive-cap",
        "cap_positive",
    }:
        return TARGET_TRANSFORM_TYPE_POSITIVE_CAP
    if resolved in {
        "angle",
        "angle_cap",
        "angle-cap",
        "wrap",
        "wrap_angle",
        "angle_wrap",
    }:
        return TARGET_TRANSFORM_TYPE_ANGLE_CAP
    if resolved in {
        "identity",
        "identity_transform",
    }:
        return TARGET_TRANSFORM_TYPE_IDENTITY
    return resolved


def _as_target_transform(
    target_name: str,
    raw_config: Any,
) -> TargetTransform:
    if isinstance(raw_config, TargetTransform):
        if raw_config.target_name != target_name:
            return TargetTransform(
                target_name=target_name,
                transform_type=raw_config.transform_type,
                range_min=raw_config.range_min,
                range_max=raw_config.range_max,
                normalize=raw_config.normalize,
            )
        return raw_config

    if raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, Mapping):
        raise ValueError(f"target transform config for '{target_name}' must be a mapping")

    default_type, default_min, default_max = DEFAULT_TARGET_TRANSFORMS.get(
        target_name,
        (TARGET_TRANSFORM_TYPE_SIGNED_CAP, -1.0, 1.0),
    )
    if "type" in raw_config:
        transform_type = _parse_transform_type(raw_config.get("type"))
        if not transform_type:
            transform_type = default_type
    else:
        transform_type = default_type
    if not transform_type:
        transform_type = default_type
    normalize = bool(raw_config.get("normalize", True))

    if transform_type == TARGET_TRANSFORM_TYPE_IDENTITY:
        return TargetTransform(
            target_name=target_name,
            transform_type=TARGET_TRANSFORM_TYPE_IDENTITY,
            range_min=0.0,
            range_max=0.0,
            normalize=False,
        )

    if transform_type == TARGET_TRANSFORM_TYPE_SIGNED_CAP or transform_type == TARGET_TRANSFORM_TYPE_ANGLE_CAP:
        if "clip" in raw_config:
            clip = float(raw_config["clip"])
            if clip <= 0.0:
                raise ValueError(f"transform clip for '{target_name}' must be > 0")
            range_min = -abs(clip)
            range_max = abs(clip)
        else:
            if "range_min" in raw_config or "range_max" in raw_config:
                range_min = float(raw_config.get("range_min", default_min))
                range_max = float(raw_config.get("range_max", default_max))
            elif transform_type == default_type:
                range_min = default_min
                range_max = default_max
            else:
                range_min = -1.0
                range_max = 1.0
        if transform_type == TARGET_TRANSFORM_TYPE_ANGLE_CAP and (range_min >= 0 or range_max <= 0):
            raise ValueError(f"angle_cap for '{target_name}' requires a signed range")
        return TargetTransform(
            target_name=target_name,
            transform_type=transform_type,
            range_min=range_min,
            range_max=range_max,
            normalize=normalize,
        )

    if transform_type == TARGET_TRANSFORM_TYPE_POSITIVE_CAP:
        if "clip" in raw_config:
            clip = float(raw_config["clip"])
            if clip <= 0.0:
                raise ValueError(f"transform clip for '{target_name}' must be > 0")
            range_min = 0.0
            range_max = abs(clip)
        else:
            if "range_min" in raw_config or "range_max" in raw_config:
                range_min = float(raw_config.get("range_min", 0.0))
                range_max = float(raw_config.get("range_max", default_max))
            elif transform_type == default_type:
                range_min, range_max = default_min, default_max
            else:
                range_min = 0.0
                range_max = 1.0
        return TargetTransform(
            target_name=target_name,
            transform_type=TARGET_TRANSFORM_TYPE_POSITIVE_CAP,
            range_min=range_min,
            range_max=range_max,
            normalize=normalize,
        )

    raise ValueError(f"unsupported transform type '{transform_type}' for target '{target_name}'")


def build_target_transform_registry(
    target_names: Sequence[str],
    raw_target_transforms: Mapping[str, Any] | Sequence[Any] | None = None,
) -> dict[str, TargetTransform]:
    names = tuple(str(name).strip() for name in target_names if str(name).strip())
    if not names:
        return {}

    raw_map: dict[str, Any] = {}
    if raw_target_transforms is not None:
        if isinstance(raw_target_transforms, Mapping):
            raw_map = {
                str(key).strip(): value
                for key, value in raw_target_transforms.items()
                if str(key).strip()
            }
            unknown = [
                key
                for key in raw_map
                if key not in names and not key.startswith("target_")
            ]
            if unknown:
                raise ValueError(f"unknown target transform names: {', '.join(sorted(unknown))}")
        elif isinstance(raw_target_transforms, Sequence):
            for item in raw_target_transforms:
                if not isinstance(item, Mapping):
                    raise ValueError("target_transforms list entries must be mapping objects")
                target_name = str(item.get("target_name", "")).strip()
                if not target_name:
                    raise ValueError("target_transforms list entries must include a non-empty target_name")
                raw_map[target_name] = item
            unknown = [
                key
                for key in raw_map
                if key not in names
            ]
            if unknown:
                raise ValueError(f"unknown target transform names: {', '.join(sorted(unknown))}")
        else:
            raise ValueError("target_transforms must be a mapping of target name to transform config")

    resolved: dict[str, TargetTransform] = {}
    for name in names:
        transform_raw = raw_map.get(name, {})
        if isinstance(transform_raw, TargetTransform):
            if transform_raw.target_name != name:
                transform_raw = TargetTransform(
                    target_name=name,
                    transform_type=transform_raw.transform_type,
                    range_min=transform_raw.range_min,
                    range_max=transform_raw.range_max,
                    normalize=transform_raw.normalize,
                )
        resolved[name] = _as_target_transform(name, transform_raw)
    return resolved


def target_transform_metadata(target_transforms: Mapping[str, TargetTransform] | None) -> list[dict[str, Any]]:
    if not target_transforms:
        return []
    return [
        target_transforms[name].metadata()
        for name in sorted(target_transforms)
    ]


def resolve_checkpoint_target_transforms(
    checkpoint: Mapping[str, Any],
    target_names: Sequence[str],
) -> dict[str, TargetTransform]:
    raw = checkpoint.get("target_transforms")
    if raw is not None:
        return build_target_transform_registry(target_names, raw)

    return build_target_transform_registry(target_names, {})


def resolve_target_transforms_for_model(checkpoint: Mapping[str, Any], target_names: Sequence[str]) -> dict[str, TargetTransform]:
    return resolve_checkpoint_target_transforms(checkpoint, target_names)


def normalize_target_tensor(
    values: Tensor,
    target_names: Sequence[str],
    target_transforms: Mapping[str, TargetTransform],
) -> Tensor:
    transformed = values.float().clone()
    for index, name in enumerate(target_names):
        transform = target_transforms.get(name)
        if transform is None:
            continue
        transformed[..., index] = transform.normalize_tensor(transformed[..., index])
    return transformed


def denormalize_target_tensor(
    values: Tensor,
    target_names: Sequence[str],
    target_transforms: Mapping[str, TargetTransform],
) -> Tensor:
    transformed = values.float().clone()
    for index, name in enumerate(target_names):
        transform = target_transforms.get(name)
        if transform is None:
            continue
        transformed[..., index] = transform.denormalize_tensor(transformed[..., index])
    return transformed


def round_trip_transform_check(target_names: Sequence[str] | None = None) -> dict[str, float]:
    names = tuple(
        str(name).strip()
        for name in (tuple(DEFAULT_TARGET_TRANSFORMS) if target_names is None else target_names)
        if str(name).strip()
    )
    if not names:
        raise ValueError("at least one target name is required for round trip check")

    registry = build_target_transform_registry(names)
    sample_values = {
        "steering": 0.42,
        "acceleration": -0.23,
        "brakePressureAvg": 0.68,
        "future_speed": 19.5,
        "future_speed_delta": -2.0,
        "future_yaw_delta": -15.0,
        "future_yaw_rate": 1.5,
    }

    max_round_trip_error = 0.0
    for name in names:
        transform = registry[name]
        raw_value = float(sample_values[name]) if name in sample_values else 0.25
        if name == "future_speed_delta" and raw_value >= 0.0:
            raw_value = -2.0
        value = torch.tensor([raw_value], dtype=torch.float32)
        restored = float(transform.denormalize_tensor(transform.normalize_tensor(value)).item())
        max_round_trip_error = max(max_round_trip_error, abs(restored - raw_value))
        if name == "future_speed_delta" and restored >= 0.0:
            raise AssertionError("future_speed_delta negative test case should remain negative after round trip")

    brake_transform = registry["brakePressureAvg"]
    brake_restored = float(brake_transform.denormalize_tensor(brake_transform.normalize_tensor(torch.tensor([-0.4]))).item())
    if brake_restored < 0.0:
        raise AssertionError("brakePressureAvg must remain non-negative after denormalization")

    return {"max_round_trip_error": float(max_round_trip_error)}
