from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor


CURRENT_SPEED_KEY = "current_speed"
DEFAULT_CURRENT_SPEED_CAP = 25.0
DEFAULT_CURRENT_SPEED_FUSION = "delta_head_only"


@dataclass(frozen=True)
class StateInputConfig:
    current_speed_enabled: bool = False
    current_speed_cap: float = DEFAULT_CURRENT_SPEED_CAP
    current_speed_fusion: str = DEFAULT_CURRENT_SPEED_FUSION


def training_state_input_config() -> StateInputConfig:
    return StateInputConfig(current_speed_enabled=True)


def state_inputs_metadata(config: StateInputConfig) -> dict[str, Any]:
    return {
        "current_speed": {
            "enabled": bool(config.current_speed_enabled),
            "cap": float(config.current_speed_cap),
            "fusion": str(config.current_speed_fusion),
        }
    }


def state_input_config_from_metadata(raw_metadata: Any) -> StateInputConfig:
    if not isinstance(raw_metadata, Mapping):
        return StateInputConfig()

    current_speed = raw_metadata.get("current_speed")
    if not isinstance(current_speed, Mapping):
        return StateInputConfig()

    enabled = bool(current_speed.get("enabled", False))
    cap = _coerce_positive_float(current_speed.get("cap"), DEFAULT_CURRENT_SPEED_CAP)
    fusion = str(current_speed.get("fusion", DEFAULT_CURRENT_SPEED_FUSION)).strip() or DEFAULT_CURRENT_SPEED_FUSION
    return StateInputConfig(
        current_speed_enabled=enabled,
        current_speed_cap=cap,
        current_speed_fusion=fusion,
    )


def normalize_current_speed_value(raw_value: Any, cap: float = DEFAULT_CURRENT_SPEED_CAP) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"currentSpeed/current_speed must be numeric, got {raw_value!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"currentSpeed/current_speed must be finite, got {raw_value!r}")
    if cap <= 0:
        raise ValueError("current speed cap must be > 0")
    clamped = min(max(value, 0.0), float(cap))
    return clamped / float(cap)


def normalize_current_speed_tensor(value: Tensor, cap: float = DEFAULT_CURRENT_SPEED_CAP) -> Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"current_speed must be a torch.Tensor, got {type(value).__name__}")
    if cap <= 0:
        raise ValueError("current speed cap must be > 0")

    tensor = value.float()
    if tensor.ndim > 1 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    if tensor.ndim > 1:
        raise ValueError(f"current_speed expected a scalar tensor per sample, got shape {tuple(tensor.shape)}")
    return tensor.clamp(min=0.0, max=float(cap)) / float(cap)


def build_state_inputs_from_label(
    label: dict[str, Any],
    config: StateInputConfig,
) -> dict[str, Tensor]:
    if not config.current_speed_enabled:
        return {}
    if "currentSpeed" not in label:
        raise ValueError("failed to build state input 'current_speed': missing currentSpeed")
    normalized = normalize_current_speed_value(label["currentSpeed"], config.current_speed_cap)
    return {
        CURRENT_SPEED_KEY: torch.tensor(normalized, dtype=torch.float32),
    }


def _coerce_positive_float(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(parsed) or parsed <= 0:
        return fallback
    return parsed
