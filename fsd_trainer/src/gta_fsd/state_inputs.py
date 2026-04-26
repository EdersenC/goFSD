from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor

from heads import HEAD_SPECS


CURRENT_SPEED_KEY = "current_speed"
ROUTE_FORWARD_DELTA_KEY = "route_forward_delta"
ROUTE_HEADING_ERROR_KEY = "route_heading_error"
ROUTE_DISTANCE_KEY = "route_distance"
LEAD_VEHICLE_DISTANCE_KEY = "lead_vehicle_distance"
HAS_LEAD_VEHICLE_KEY = "has_lead_vehicle"

DEFAULT_CURRENT_SPEED_CAP = 25.0
DEFAULT_ROUTE_FORWARD_DELTA_CAP = 1.5
DEFAULT_ROUTE_HEADING_ERROR_CAP = 180.0
DEFAULT_ROUTE_DISTANCE_CAP = 100.0
DEFAULT_LEAD_VEHICLE_DISTANCE_CAP = 100.0
DEFAULT_WIDTH_MULTIPLIER = 1.5

DEFAULT_LEGACY_HEADS = (
    "delta_speed",
    "future_speed",
    "move_intent",
    "future_yaw_delta",
)

NORMALIZATION_POSITIVE_CAP = "positive_cap"
NORMALIZATION_SIGNED_CAP = "signed_cap"
NORMALIZATION_BINARY = "binary"


@dataclass(frozen=True)
class StateInputDefinition:
    key: str
    camel_key: str
    label_key: str
    normalization: str
    default_cap: float | None
    default_enabled: bool
    default_heads: tuple[str, ...]


STATE_INPUT_DEFINITIONS: tuple[StateInputDefinition, ...] = (
    StateInputDefinition(
        key=CURRENT_SPEED_KEY,
        camel_key="currentSpeed",
        label_key="currentSpeed",
        normalization=NORMALIZATION_POSITIVE_CAP,
        default_cap=DEFAULT_CURRENT_SPEED_CAP,
        default_enabled=True,
        default_heads=DEFAULT_LEGACY_HEADS,
    ),
    StateInputDefinition(
        key=ROUTE_FORWARD_DELTA_KEY,
        camel_key="routeForwardDelta",
        label_key="routeForwardDelta",
        normalization=NORMALIZATION_SIGNED_CAP,
        default_cap=DEFAULT_ROUTE_FORWARD_DELTA_CAP,
        default_enabled=True,
        default_heads=DEFAULT_LEGACY_HEADS,
    ),
    StateInputDefinition(
        key=ROUTE_HEADING_ERROR_KEY,
        camel_key="routeHeadingError",
        label_key="routeHeadingError",
        normalization=NORMALIZATION_SIGNED_CAP,
        default_cap=DEFAULT_ROUTE_HEADING_ERROR_CAP,
        default_enabled=False,
        default_heads=(),
    ),
    StateInputDefinition(
        key=ROUTE_DISTANCE_KEY,
        camel_key="routeDistance",
        label_key="routeDistance",
        normalization=NORMALIZATION_POSITIVE_CAP,
        default_cap=DEFAULT_ROUTE_DISTANCE_CAP,
        default_enabled=False,
        default_heads=(),
    ),
    StateInputDefinition(
        key=LEAD_VEHICLE_DISTANCE_KEY,
        camel_key="leadVehicleDistance",
        label_key="leadVehicleDistance",
        normalization=NORMALIZATION_POSITIVE_CAP,
        default_cap=DEFAULT_LEAD_VEHICLE_DISTANCE_CAP,
        default_enabled=False,
        default_heads=(),
    ),
    StateInputDefinition(
        key=HAS_LEAD_VEHICLE_KEY,
        camel_key="hasLeadVehicle",
        label_key="hasLeadVehicle",
        normalization=NORMALIZATION_BINARY,
        default_cap=None,
        default_enabled=False,
        default_heads=(),
    ),
)

STATE_INPUT_DEFINITIONS_BY_KEY: dict[str, StateInputDefinition] = {
    definition.key: definition for definition in STATE_INPUT_DEFINITIONS
}
STATE_INPUT_DEFINITIONS_BY_CAMEL: dict[str, StateInputDefinition] = {
    definition.camel_key: definition for definition in STATE_INPUT_DEFINITIONS
}
ACTIVE_HEAD_NAMES: tuple[str, ...] = tuple(spec.name for spec in HEAD_SPECS)
ACTIVE_HEAD_NAME_SET = frozenset(ACTIVE_HEAD_NAMES)


@dataclass(frozen=True)
class StateInputSpec:
    enabled: bool = False
    heads: tuple[str, ...] = ()
    cap: float | None = None


@dataclass(frozen=True)
class StateInputConfig:
    specs: dict[str, StateInputSpec] = field(default_factory=dict)

    def spec(self, key: str) -> StateInputSpec:
        definition = state_input_definition(key)
        return self.specs.get(
            key,
            StateInputSpec(
                enabled=definition.default_enabled,
                heads=definition.default_heads,
                cap=definition.default_cap,
            ),
        )

    def is_enabled(self, key: str) -> bool:
        return self.spec(key).enabled

    def enabled_keys(self) -> tuple[str, ...]:
        return tuple(
            definition.key
            for definition in STATE_INPUT_DEFINITIONS
            if self.spec(definition.key).enabled
        )


def state_input_definition(key: str) -> StateInputDefinition:
    normalized_key = str(key).strip()
    definition = STATE_INPUT_DEFINITIONS_BY_KEY.get(normalized_key)
    if definition is None:
        raise KeyError(f"unknown state input: {key}")
    return definition


def training_state_input_config(raw_metadata: Any | None = None) -> StateInputConfig:
    if raw_metadata is None:
        return default_training_state_input_config()
    if not isinstance(raw_metadata, Mapping):
        return default_training_state_input_config()
    return state_input_config_from_metadata(raw_metadata, fallback_to_training_defaults=True)


def default_training_state_input_config() -> StateInputConfig:
    specs: dict[str, StateInputSpec] = {}
    for definition in STATE_INPUT_DEFINITIONS:
        specs[definition.key] = StateInputSpec(
            enabled=definition.default_enabled,
            heads=definition.default_heads,
            cap=definition.default_cap,
        )
    return StateInputConfig(specs=specs)


def default_inference_state_input_config() -> StateInputConfig:
    return StateInputConfig(specs={
        definition.key: StateInputSpec(
            enabled=False,
            heads=(),
            cap=definition.default_cap,
        )
        for definition in STATE_INPUT_DEFINITIONS
    })


def state_inputs_metadata(config: StateInputConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for definition in STATE_INPUT_DEFINITIONS:
        spec = config.spec(definition.key)
        item: dict[str, Any] = {
            "enabled": bool(spec.enabled),
            "heads": list(spec.heads),
        }
        if definition.default_cap is not None:
            item["cap"] = float(resolve_state_input_cap(config, definition.key))
        payload[definition.key] = item
    return payload


def state_input_config_from_metadata(
    raw_metadata: Any,
    *,
    fallback_to_training_defaults: bool = False,
) -> StateInputConfig:
    if not isinstance(raw_metadata, Mapping):
        return default_training_state_input_config() if fallback_to_training_defaults else default_inference_state_input_config()

    base = (
        default_training_state_input_config()
        if fallback_to_training_defaults
        else default_inference_state_input_config()
    )
    specs = dict(base.specs)
    for definition in STATE_INPUT_DEFINITIONS:
        raw_item = raw_metadata.get(definition.key)
        if not isinstance(raw_item, Mapping):
            continue
        enabled = bool(raw_item.get("enabled", specs[definition.key].enabled))
        cap = definition.default_cap
        if definition.default_cap is not None:
            cap = _coerce_positive_float(raw_item.get("cap"), resolve_state_input_cap(base, definition.key))
        raw_heads = raw_item.get("heads", specs[definition.key].heads)
        heads = normalize_state_input_heads(raw_heads, definition.key)
        specs[definition.key] = StateInputSpec(enabled=enabled, heads=heads, cap=cap)
    return StateInputConfig(specs=specs)


def resolve_state_input_cap(config: StateInputConfig, key: str) -> float:
    definition = state_input_definition(key)
    if definition.default_cap is None:
        raise ValueError(f"{key} does not define a numeric cap")
    cap = config.spec(key).cap
    if cap is None:
        return float(definition.default_cap)
    return _coerce_positive_float(cap, float(definition.default_cap))


def state_input_head_map(config: StateInputConfig) -> dict[str, list[str]]:
    return {
        definition.key: list(config.spec(definition.key).heads)
        for definition in STATE_INPUT_DEFINITIONS
        if config.spec(definition.key).enabled
    }


def current_speed_fused_head_names(config: StateInputConfig) -> tuple[str, ...]:
    return config.spec(CURRENT_SPEED_KEY).heads if config.is_enabled(CURRENT_SPEED_KEY) else ()


def route_forward_delta_fused_head_names(config: StateInputConfig) -> tuple[str, ...]:
    return config.spec(ROUTE_FORWARD_DELTA_KEY).heads if config.is_enabled(ROUTE_FORWARD_DELTA_KEY) else ()


def normalize_state_input_heads(raw_heads: Any, input_name: str) -> tuple[str, ...]:
    if raw_heads is None:
        return ()
    if not isinstance(raw_heads, (list, tuple)):
        raise ValueError(f"{input_name} heads must be a list")
    resolved: list[str] = []
    seen: set[str] = set()
    for raw_head in raw_heads:
        head_name = str(raw_head).strip()
        if not head_name:
            continue
        if head_name not in ACTIVE_HEAD_NAME_SET:
            raise ValueError(
                f"{input_name} head {head_name!r} is unsupported; "
                f"expected one of {list(ACTIVE_HEAD_NAMES)}"
            )
        if head_name in seen:
            continue
        seen.add(head_name)
        resolved.append(head_name)
    return tuple(resolved)


def build_state_inputs_from_label(
    label: dict[str, Any],
    config: StateInputConfig,
) -> dict[str, Tensor]:
    state_inputs: dict[str, Tensor] = {}
    for definition in STATE_INPUT_DEFINITIONS:
        if not config.is_enabled(definition.key):
            continue
        normalized = normalize_state_input_value_from_mapping(label, definition.key, config)
        state_inputs[definition.key] = torch.tensor(normalized, dtype=torch.float32)
    return state_inputs


def normalize_state_input_value_from_mapping(
    source: Mapping[str, Any],
    key: str,
    config: StateInputConfig,
) -> float:
    definition = state_input_definition(key)
    raw_value = resolve_state_input_raw_value(source, definition, config)
    return normalize_state_input_value(key, raw_value, config)


def resolve_state_input_raw_value(
    source: Mapping[str, Any],
    definition: StateInputDefinition,
    config: StateInputConfig,
) -> Any:
    if definition.key == LEAD_VEHICLE_DISTANCE_KEY:
        has_lead = resolve_has_lead_vehicle_raw_value(source)
        if not has_lead:
            return resolve_state_input_cap(config, LEAD_VEHICLE_DISTANCE_KEY)
    if definition.label_key not in source:
        raise ValueError(
            f"failed to build state input '{definition.key}': missing {definition.label_key}"
        )
    return source[definition.label_key]


def resolve_has_lead_vehicle_raw_value(source: Mapping[str, Any]) -> bool:
    definition = state_input_definition(HAS_LEAD_VEHICLE_KEY)
    if definition.label_key not in source:
        raise ValueError(f"failed to build state input '{HAS_LEAD_VEHICLE_KEY}': missing {definition.label_key}")
    return _coerce_bool(source[definition.label_key], definition.key)


def normalize_state_input_value(key: str, raw_value: Any, config: StateInputConfig) -> float:
    definition = state_input_definition(key)
    if definition.normalization == NORMALIZATION_BINARY:
        return 1.0 if _coerce_bool(raw_value, key) else 0.0
    value = _coerce_finite_float(raw_value, key)
    cap = resolve_state_input_cap(config, key)
    if definition.normalization == NORMALIZATION_POSITIVE_CAP:
        clamped = min(max(value, 0.0), cap)
        return clamped / cap
    if definition.normalization == NORMALIZATION_SIGNED_CAP:
        clamped = min(max(value, -cap), cap)
        return clamped / cap
    raise ValueError(f"unsupported normalization kind for {key}: {definition.normalization}")


def normalize_state_input_tensor(key: str, value: Tensor, config: StateInputConfig) -> Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{key} must be a torch.Tensor, got {type(value).__name__}")
    definition = state_input_definition(key)
    tensor = value.float()
    if tensor.ndim > 1 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    if tensor.ndim > 1:
        raise ValueError(f"{key} expected a scalar tensor per sample, got shape {tuple(tensor.shape)}")
    if definition.normalization == NORMALIZATION_BINARY:
        return (tensor > 0).float()
    cap = resolve_state_input_cap(config, key)
    if definition.normalization == NORMALIZATION_POSITIVE_CAP:
        return tensor.clamp(min=0.0, max=cap) / cap
    if definition.normalization == NORMALIZATION_SIGNED_CAP:
        return tensor.clamp(min=-cap, max=cap) / cap
    raise ValueError(f"unsupported normalization kind for {key}: {definition.normalization}")


def normalize_current_speed_value(raw_value: Any, cap: float = DEFAULT_CURRENT_SPEED_CAP) -> float:
    return normalize_state_input_value(
        CURRENT_SPEED_KEY,
        raw_value,
        StateInputConfig(specs={
            CURRENT_SPEED_KEY: StateInputSpec(enabled=True, heads=(), cap=cap),
        }),
    )


def normalize_current_speed_tensor(value: Tensor, cap: float = DEFAULT_CURRENT_SPEED_CAP) -> Tensor:
    return normalize_state_input_tensor(
        CURRENT_SPEED_KEY,
        value,
        StateInputConfig(specs={
            CURRENT_SPEED_KEY: StateInputSpec(enabled=True, heads=(), cap=cap),
        }),
    )


def normalize_route_forward_delta_value(raw_value: Any, cap: float = DEFAULT_ROUTE_FORWARD_DELTA_CAP) -> float:
    return normalize_state_input_value(
        ROUTE_FORWARD_DELTA_KEY,
        raw_value,
        StateInputConfig(specs={
            ROUTE_FORWARD_DELTA_KEY: StateInputSpec(enabled=True, heads=(), cap=cap),
        }),
    )


def normalize_route_forward_delta_tensor(value: Tensor, cap: float = DEFAULT_ROUTE_FORWARD_DELTA_CAP) -> Tensor:
    return normalize_state_input_tensor(
        ROUTE_FORWARD_DELTA_KEY,
        value,
        StateInputConfig(specs={
            ROUTE_FORWARD_DELTA_KEY: StateInputSpec(enabled=True, heads=(), cap=cap),
        }),
    )


def _coerce_positive_float(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(parsed) or parsed <= 0:
        return fallback
    return parsed


def _coerce_finite_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return parsed


def _coerce_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value) != 0.0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be boolean-like, got {value!r}")
