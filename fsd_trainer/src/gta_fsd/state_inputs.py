from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor


CURRENT_SPEED_KEY = "current_speed"
DEFAULT_CURRENT_SPEED_CAP_MPS = 15.0
DEFAULT_WIDTH_MULTIPLIER = 1.5
NORMALIZATION_POSITIVE_CAP = "positive_cap"


@dataclass(frozen=True)
class StateInputDefinition:
    key: str
    camel_key: str
    label_key: str
    normalization: str
    default_cap: float
    default_enabled: bool


# The stop-sign planner is RGB-first. Current speed is the only optional
# non-visual state input; target geometry and phase labels are supervision only.
STATE_INPUT_DEFINITIONS: tuple[StateInputDefinition, ...] = (
    StateInputDefinition(
        key=CURRENT_SPEED_KEY,
        camel_key="currentSpeed",
        label_key="currentSpeed",
        normalization=NORMALIZATION_POSITIVE_CAP,
        default_cap=DEFAULT_CURRENT_SPEED_CAP_MPS,
        default_enabled=False,
    ),
)
STATE_INPUT_DEFINITIONS_BY_KEY: dict[str, StateInputDefinition] = {
    definition.key: definition for definition in STATE_INPUT_DEFINITIONS
}
STATE_INPUT_DEFINITIONS_BY_CAMEL: dict[str, StateInputDefinition] = {
    definition.camel_key: definition for definition in STATE_INPUT_DEFINITIONS
}


@dataclass(frozen=True)
class StateInputSpec:
    enabled: bool = False
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
        raise KeyError(f"unsupported stop-sign state input: {key}")
    return definition


def training_state_input_config(raw_metadata: Any | None = None) -> StateInputConfig:
    if isinstance(raw_metadata, StateInputConfig):
        return raw_metadata
    if raw_metadata is None:
        return default_training_state_input_config()
    return state_input_config_from_metadata(raw_metadata)


def default_training_state_input_config() -> StateInputConfig:
    return default_inference_state_input_config()


def default_inference_state_input_config() -> StateInputConfig:
    return StateInputConfig(
        specs={
            definition.key: StateInputSpec(
                enabled=definition.default_enabled,
                cap=definition.default_cap,
            )
            for definition in STATE_INPUT_DEFINITIONS
        }
    )


def state_inputs_metadata(config: StateInputConfig) -> dict[str, Any]:
    return {
        definition.key: {
            "enabled": bool(config.spec(definition.key).enabled),
            "cap": float(resolve_state_input_cap(config, definition.key)),
        }
        for definition in STATE_INPUT_DEFINITIONS
    }


def state_input_definitions_metadata() -> list[dict[str, Any]]:
    return [
        {
            "key": definition.key,
            "camelKey": definition.camel_key,
            "labelKey": definition.label_key,
            "normalization": definition.normalization,
            "defaultEnabled": bool(definition.default_enabled),
            "defaultCap": float(definition.default_cap),
        }
        for definition in STATE_INPUT_DEFINITIONS
    ]


def state_input_config_from_metadata(
    raw_metadata: Any,
    *,
    fallback_to_training_defaults: bool = False,
) -> StateInputConfig:
    del fallback_to_training_defaults
    if isinstance(raw_metadata, StateInputConfig):
        return raw_metadata
    if raw_metadata is None:
        return default_inference_state_input_config()
    if not isinstance(raw_metadata, Mapping):
        raise ValueError("state_inputs must be a mapping")

    unknown_keys = sorted(
        str(key)
        for key in raw_metadata
        if str(key) not in STATE_INPUT_DEFINITIONS_BY_KEY
    )
    if unknown_keys:
        raise ValueError(
            "unsupported stop-sign state inputs; only current_speed is allowed: "
            + ", ".join(unknown_keys)
        )

    config = default_inference_state_input_config()
    specs = dict(config.specs)
    for definition in STATE_INPUT_DEFINITIONS:
        raw_item = raw_metadata.get(definition.key)
        if raw_item is None:
            continue
        if not isinstance(raw_item, Mapping):
            raise ValueError(f"state_inputs.{definition.key} must be a mapping")
        specs[definition.key] = StateInputSpec(
            enabled=bool(raw_item.get("enabled", False)),
            cap=_coerce_positive_float(
                raw_item.get("cap", definition.default_cap),
                field_name=f"state_inputs.{definition.key}.cap",
            ),
        )
    return StateInputConfig(specs=specs)


def resolve_state_input_cap(config: StateInputConfig, key: str) -> float:
    definition = state_input_definition(key)
    cap = config.spec(key).cap
    return (
        float(definition.default_cap)
        if cap is None
        else _coerce_positive_float(cap, field_name=f"state_inputs.{key}.cap")
    )


def build_state_inputs_from_label(
    label: dict[str, Any],
    config: StateInputConfig,
) -> dict[str, Tensor]:
    return {
        definition.key: torch.tensor(
            normalize_state_input_value_from_mapping(label, definition.key, config),
            dtype=torch.float32,
        )
        for definition in STATE_INPUT_DEFINITIONS
        if config.is_enabled(definition.key)
    }


def build_state_input_vector_from_mapping(
    source: Mapping[str, Any],
    config: StateInputConfig,
) -> Tensor:
    values = [
        normalize_state_input_value_from_mapping(source, definition.key, config)
        for definition in STATE_INPUT_DEFINITIONS
        if config.is_enabled(definition.key)
    ]
    return torch.tensor(values, dtype=torch.float32)


def normalize_state_input_value_from_mapping(
    source: Mapping[str, Any],
    key: str,
    config: StateInputConfig,
) -> float:
    definition = state_input_definition(key)
    raw_value = resolve_state_input_raw_value(source, definition)
    return normalize_state_input_value(key, raw_value, config)


def resolve_state_input_raw_value(
    source: Mapping[str, Any],
    definition: StateInputDefinition,
) -> Any:
    for key in (definition.label_key, definition.camel_key, definition.key):
        if key in source:
            return source[key]
    raise ValueError(
        f"failed to build state input '{definition.key}': missing {definition.label_key}"
    )


def normalize_state_input_value(
    key: str,
    raw_value: Any,
    config: StateInputConfig,
) -> float:
    value = _coerce_finite_float(raw_value, key)
    cap = resolve_state_input_cap(config, key)
    return min(max(value, 0.0), cap) / cap


def normalize_state_input_tensor(
    key: str,
    value: Tensor,
    config: StateInputConfig,
) -> Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{key} must be a torch.Tensor, got {type(value).__name__}")
    tensor = value.float()
    if tensor.ndim > 1 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    if tensor.ndim > 1:
        raise ValueError(
            f"{key} expected a scalar tensor per sample, got shape {tuple(tensor.shape)}"
        )
    cap = resolve_state_input_cap(config, key)
    return tensor.clamp(min=0.0, max=cap) / cap


def normalize_current_speed_value(
    raw_value: Any,
    cap: float = DEFAULT_CURRENT_SPEED_CAP_MPS,
) -> float:
    return normalize_state_input_value(
        CURRENT_SPEED_KEY,
        raw_value,
        StateInputConfig(
            specs={CURRENT_SPEED_KEY: StateInputSpec(enabled=True, cap=cap)}
        ),
    )


def normalize_current_speed_tensor(
    value: Tensor,
    cap: float = DEFAULT_CURRENT_SPEED_CAP_MPS,
) -> Tensor:
    return normalize_state_input_tensor(
        CURRENT_SPEED_KEY,
        value,
        StateInputConfig(
            specs={CURRENT_SPEED_KEY: StateInputSpec(enabled=True, cap=cap)}
        ),
    )


def _coerce_positive_float(value: Any, *, field_name: str) -> float:
    parsed = _coerce_finite_float(value, field_name)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return parsed


def _coerce_finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric, got bool")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return parsed
