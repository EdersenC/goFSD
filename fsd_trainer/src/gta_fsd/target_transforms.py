from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from torch import Tensor


DEFAULT_DELTA_SPEED_CLIP = 2.0
DEFAULT_DELTA_SPEED_NORMALIZE = True
DELTA_SPEED_TARGET_LABEL_KEY = "delta_speed_target"


@dataclass(frozen=True)
class DeltaSpeedTargetTransform:
    clip_value: float = DEFAULT_DELTA_SPEED_CLIP
    normalize: bool = DEFAULT_DELTA_SPEED_NORMALIZE

    def metadata(self) -> dict[str, Any]:
        return {
            "clip_value": float(self.clip_value),
            "normalize": bool(self.normalize),
        }


def default_delta_speed_target_transform() -> DeltaSpeedTargetTransform:
    return DeltaSpeedTargetTransform()


def legacy_delta_speed_target_transform() -> DeltaSpeedTargetTransform:
    return DeltaSpeedTargetTransform(
        clip_value=DEFAULT_DELTA_SPEED_CLIP,
        normalize=False,
    )


def validate_delta_speed_target_transform(transform: DeltaSpeedTargetTransform) -> DeltaSpeedTargetTransform:
    clip_value = float(transform.clip_value)
    if clip_value <= 0.0:
        raise ValueError("dataset.delta_speed_clip must be > 0")
    return DeltaSpeedTargetTransform(
        clip_value=clip_value,
        normalize=bool(transform.normalize),
    )


def delta_speed_target_transform_from_config(dataset_raw: Mapping[str, Any] | None) -> DeltaSpeedTargetTransform:
    source = {} if dataset_raw is None else dataset_raw
    return validate_delta_speed_target_transform(DeltaSpeedTargetTransform(
        clip_value=float(source.get("delta_speed_clip", DEFAULT_DELTA_SPEED_CLIP)),
        normalize=bool(source.get("delta_speed_normalize", DEFAULT_DELTA_SPEED_NORMALIZE)),
    ))


def delta_speed_target_transform_from_metadata(raw: Any) -> DeltaSpeedTargetTransform:
    if raw is None:
        return legacy_delta_speed_target_transform()
    if not isinstance(raw, Mapping):
        raise ValueError("delta_speed_target_transform metadata must be a mapping")
    return validate_delta_speed_target_transform(DeltaSpeedTargetTransform(
        clip_value=float(raw.get("clip_value", DEFAULT_DELTA_SPEED_CLIP)),
        normalize=bool(raw.get("normalize", DEFAULT_DELTA_SPEED_NORMALIZE)),
    ))


def resolve_checkpoint_delta_speed_target_transform(checkpoint: Mapping[str, Any]) -> DeltaSpeedTargetTransform:
    return delta_speed_target_transform_from_metadata(checkpoint.get("delta_speed_target_transform"))


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def clip_delta_speed_value(value: float, transform: DeltaSpeedTargetTransform) -> float:
    return clamp(float(value), -float(transform.clip_value), float(transform.clip_value))


def build_delta_speed_target_value(value: float, transform: DeltaSpeedTargetTransform) -> float:
    clipped = clip_delta_speed_value(value, transform)
    if transform.normalize:
        return clipped / float(transform.clip_value)
    return clipped


def denormalize_delta_speed_value(value: float, transform: DeltaSpeedTargetTransform) -> float:
    raw = float(value)
    if transform.normalize:
        raw = clamp(raw, -1.0, 1.0) * float(transform.clip_value)
    return raw


def denormalize_delta_speed_tensor(value: Tensor, transform: DeltaSpeedTargetTransform) -> Tensor:
    tensor = value.float()
    if transform.normalize:
        return tensor.clamp(-1.0, 1.0) * float(transform.clip_value)
    return tensor
