from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor


CURRENT_SPEED_KEY = "current_speed"
ROUTE_FORWARD_DELTA_KEY = "route_forward_delta"
DEFAULT_CURRENT_SPEED_CAP = 25.0
DEFAULT_CURRENT_SPEED_FUSION = "delta_head_only"
DEFAULT_TRAINING_CURRENT_SPEED_FUSION = "legacy"
DEFAULT_ROUTE_FORWARD_DELTA_CAP = 1.5
DEFAULT_ROUTE_FORWARD_DELTA_FUSION = "legacy"
_FUSION_TO_HEADS: dict[str, tuple[str, ...]] = {
    "delta_head_only": ("delta_speed",),
    "delta_speed_only": ("delta_speed",),
    "delta_and_yaw_heads": ("delta_speed", "future_yaw_delta"),
    "legacy": ("delta_speed", "future_speed", "move_intent", "future_yaw_delta"),
}


@dataclass(frozen=True)
class StateInputConfig:
    current_speed_enabled: bool = False
    current_speed_cap: float = DEFAULT_CURRENT_SPEED_CAP
    current_speed_fusion: str = DEFAULT_CURRENT_SPEED_FUSION
    route_forward_delta_enabled: bool = False
    route_forward_delta_cap: float = DEFAULT_ROUTE_FORWARD_DELTA_CAP
    route_forward_delta_fusion: str = DEFAULT_ROUTE_FORWARD_DELTA_FUSION


def training_state_input_config(raw_metadata: Any | None = None) -> StateInputConfig:
    if raw_metadata is None:
        return StateInputConfig(
            current_speed_enabled=True,
            current_speed_fusion=DEFAULT_TRAINING_CURRENT_SPEED_FUSION,
        )
    if not isinstance(raw_metadata, Mapping):
        return StateInputConfig(
            current_speed_enabled=True,
            current_speed_fusion=DEFAULT_TRAINING_CURRENT_SPEED_FUSION,
        )
    return state_input_config_from_metadata(raw_metadata)


def state_inputs_metadata(config: StateInputConfig) -> dict[str, Any]:
    return {
        "current_speed": {
            "enabled": bool(config.current_speed_enabled),
            "cap": float(config.current_speed_cap),
            "fusion": str(config.current_speed_fusion),
        },
        "route_forward_delta": {
            "enabled": bool(config.route_forward_delta_enabled),
            "cap": float(config.route_forward_delta_cap),
            "fusion": str(config.route_forward_delta_fusion),
        },
    }


def state_input_config_from_metadata(raw_metadata: Any) -> StateInputConfig:
    if not isinstance(raw_metadata, Mapping):
        return StateInputConfig()

    current_speed = raw_metadata.get("current_speed")
    route_forward_delta = raw_metadata.get("route_forward_delta")
    current_speed_enabled = False
    current_speed_cap = DEFAULT_CURRENT_SPEED_CAP
    current_speed_fusion = DEFAULT_CURRENT_SPEED_FUSION
    if isinstance(current_speed, Mapping):
        current_speed_enabled = bool(current_speed.get("enabled", False))
        current_speed_cap = _coerce_positive_float(current_speed.get("cap"), DEFAULT_CURRENT_SPEED_CAP)
        current_speed_fusion = str(current_speed.get("fusion", DEFAULT_CURRENT_SPEED_FUSION)).strip() or DEFAULT_CURRENT_SPEED_FUSION

    route_forward_delta_enabled = False
    route_forward_delta_cap = DEFAULT_ROUTE_FORWARD_DELTA_CAP
    route_forward_delta_fusion = DEFAULT_ROUTE_FORWARD_DELTA_FUSION
    if isinstance(route_forward_delta, Mapping):
        route_forward_delta_enabled = bool(route_forward_delta.get("enabled", False))
        route_forward_delta_cap = _coerce_positive_float(
            route_forward_delta.get("cap"),
            DEFAULT_ROUTE_FORWARD_DELTA_CAP,
        )
        route_forward_delta_fusion = (
            str(route_forward_delta.get("fusion", DEFAULT_ROUTE_FORWARD_DELTA_FUSION)).strip()
            or DEFAULT_ROUTE_FORWARD_DELTA_FUSION
        )

    return StateInputConfig(
        current_speed_enabled=current_speed_enabled,
        current_speed_cap=current_speed_cap,
        current_speed_fusion=current_speed_fusion,
        route_forward_delta_enabled=route_forward_delta_enabled,
        route_forward_delta_cap=route_forward_delta_cap,
        route_forward_delta_fusion=route_forward_delta_fusion,
    )


def current_speed_fused_head_names(config: StateInputConfig) -> tuple[str, ...]:
    return _resolve_fused_head_names(
        input_name=CURRENT_SPEED_KEY,
        enabled=config.current_speed_enabled,
        fusion=config.current_speed_fusion,
        default_fusion=DEFAULT_CURRENT_SPEED_FUSION,
    )


def route_forward_delta_fused_head_names(config: StateInputConfig) -> tuple[str, ...]:
    return _resolve_fused_head_names(
        input_name=ROUTE_FORWARD_DELTA_KEY,
        enabled=config.route_forward_delta_enabled,
        fusion=config.route_forward_delta_fusion,
        default_fusion=DEFAULT_ROUTE_FORWARD_DELTA_FUSION,
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


def normalize_route_forward_delta_value(raw_value: Any, cap: float = DEFAULT_ROUTE_FORWARD_DELTA_CAP) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"routeForwardDelta/route_forward_delta must be numeric, got {raw_value!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"routeForwardDelta/route_forward_delta must be finite, got {raw_value!r}")
    if cap <= 0:
        raise ValueError("route forward delta cap must be > 0")
    clamped = min(max(value, -float(cap)), float(cap))
    return clamped / float(cap)


def normalize_route_forward_delta_tensor(value: Tensor, cap: float = DEFAULT_ROUTE_FORWARD_DELTA_CAP) -> Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"route_forward_delta must be a torch.Tensor, got {type(value).__name__}")
    if cap <= 0:
        raise ValueError("route forward delta cap must be > 0")

    tensor = value.float()
    if tensor.ndim > 1 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    if tensor.ndim > 1:
        raise ValueError(f"route_forward_delta expected a scalar tensor per sample, got shape {tuple(tensor.shape)}")
    return tensor.clamp(min=-float(cap), max=float(cap)) / float(cap)


def build_state_inputs_from_label(
    label: dict[str, Any],
    config: StateInputConfig,
) -> dict[str, Tensor]:
    state_inputs: dict[str, Tensor] = {}
    if config.current_speed_enabled:
        if "currentSpeed" not in label:
            raise ValueError("failed to build state input 'current_speed': missing currentSpeed")
        normalized = normalize_current_speed_value(label["currentSpeed"], config.current_speed_cap)
        state_inputs[CURRENT_SPEED_KEY] = torch.tensor(normalized, dtype=torch.float32)
    if config.route_forward_delta_enabled:
        if "routeForwardDelta" not in label:
            raise ValueError("failed to build state input 'route_forward_delta': missing routeForwardDelta")
        normalized = normalize_route_forward_delta_value(
            label["routeForwardDelta"],
            config.route_forward_delta_cap,
        )
        state_inputs[ROUTE_FORWARD_DELTA_KEY] = torch.tensor(normalized, dtype=torch.float32)
    return state_inputs


def _resolve_fused_head_names(
    *,
    input_name: str,
    enabled: bool,
    fusion: str,
    default_fusion: str,
) -> tuple[str, ...]:
    if not enabled:
        return ()
    fusion_key = str(fusion).strip().lower() or default_fusion
    if fusion_key not in _FUSION_TO_HEADS:
        raise ValueError(
            f"unsupported {input_name} fusion mode: "
            f"{fusion!r}; expected one of {sorted(_FUSION_TO_HEADS)}"
        )
    return _FUSION_TO_HEADS[fusion_key]


def _coerce_positive_float(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(parsed) or parsed <= 0:
        return fallback
    return parsed
