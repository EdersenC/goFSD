from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from stop_sign_contract import RELEASE_POLICY_NAME


LEGACY_PLANNER_FORMAT = "temporal_telemetry_gru_v2"
PLANNER_FORMAT = "temporal_stop_sign_v1"
PLANNER_FORMAT_VERSION = 1

CONTROL_CONTRACT_NAME = "stop_sign_motion_plan_v1"
CONTROL_CONTRACT_VERSION = 1
CONTROL_HORIZON_DIRECTION = "forward"
MAX_DESIRED_SPEED_MPS = 8.0
FUTURE_SPEED_MPS = "future_speed_mps"
STOP_INTENT = "stop_intent"
STOP_SIGN_CONTROL_TARGET_NAMES = (
    FUTURE_SPEED_MPS,
    STOP_INTENT,
)
CONTROL_OUTPUT_ACTIVATIONS = {
    FUTURE_SPEED_MPS: "sigmoid",
    STOP_INTENT: "sigmoid",
}
CONTROL_OUTPUT_RANGES: dict[str, tuple[float, float]] = {
    FUTURE_SPEED_MPS: (0.0, MAX_DESIRED_SPEED_MPS),
    STOP_INTENT: (0.0, 1.0),
}
DEFAULT_TELEMETRY_SAMPLE_INTERVAL_MS = 50

# Transitional symbols keep older utility imports readable while old parking
# checkpoints are deliberately rejected by validate_checkpoint_control_contract.
DESIRED_WHEEL_STEER_NORMALIZED = "desired_wheel_steer_normalized"
DESIRED_SPEED_MPS = FUTURE_SPEED_MPS
STOP_PROBABILITY = STOP_INTENT
PARKING_CONTROL_TARGET_NAMES = STOP_SIGN_CONTROL_TARGET_NAMES

LEGACY_CONTROL_CONTRACT_ERROR = (
    "Checkpoint uses an incompatible legacy parking or actuator-output contract. "
    "The stop-sign planner requires (future_speed_mps, stop_intent) at fixed future "
    "horizons; legacy outputs cannot be safely adapted, so retraining is required."
)


def require_stop_sign_control_target_names(
    names: Sequence[Any],
    *,
    source: str,
) -> tuple[str, ...]:
    resolved = tuple(str(name).strip() for name in names)
    if resolved != STOP_SIGN_CONTROL_TARGET_NAMES:
        raise ValueError(
            f"{source} must exactly match the stop-sign motion-plan contract: "
            f"expected={list(STOP_SIGN_CONTROL_TARGET_NAMES)} actual={list(resolved)}"
        )
    return resolved


require_parking_control_target_names = require_stop_sign_control_target_names


def control_contract_metadata() -> dict[str, Any]:
    return {
        "name": CONTROL_CONTRACT_NAME,
        "version": CONTROL_CONTRACT_VERSION,
        "direction": CONTROL_HORIZON_DIRECTION,
        "targets": list(STOP_SIGN_CONTROL_TARGET_NAMES),
        "output_activations": dict(CONTROL_OUTPUT_ACTIVATIONS),
        "output_ranges": {
            name: list(value_range)
            for name, value_range in CONTROL_OUTPUT_RANGES.items()
        },
    }


def apply_stop_sign_control_activations(
    raw_controls: Tensor,
    control_target_names: Sequence[Any],
) -> Tensor:
    names = require_stop_sign_control_target_names(
        control_target_names,
        source="model control_target_names",
    )
    if raw_controls.ndim < 1 or raw_controls.shape[-1] != len(names):
        raise ValueError(
            "raw control output width must match the stop-sign motion-plan contract: "
            f"shape={tuple(raw_controls.shape)} targets={len(names)}"
        )
    return torch.stack(
        (
            torch.sigmoid(raw_controls[..., 0]),
            torch.sigmoid(raw_controls[..., 1]),
        ),
        dim=-1,
    )


apply_parking_control_activations = apply_stop_sign_control_activations


def derive_control_horizon_dt_ms(
    future_offsets: Sequence[Any],
    telemetry_sample_interval_ms: Any,
) -> tuple[int, ...]:
    interval_ms = _positive_int(
        telemetry_sample_interval_ms,
        source="telemetry_sample_interval_ms",
    )
    offsets = tuple(
        _positive_int(value, source=f"future_offsets[{index}]")
        for index, value in enumerate(future_offsets)
    )
    if not offsets:
        raise ValueError("future_offsets must contain at least one positive offset")
    if tuple(sorted(offsets)) != offsets or len(set(offsets)) != len(offsets):
        raise ValueError("future_offsets must be strictly increasing")
    return tuple(offset * interval_ms for offset in offsets)


def resolve_checkpoint_control_horizon_dt_ms(checkpoint: Mapping[str, Any]) -> tuple[int, ...]:
    future_offsets = checkpoint.get("future_offsets")
    if not isinstance(future_offsets, list) or not future_offsets:
        raise ValueError("stop-sign checkpoint must include non-empty future_offsets")
    interval_ms = checkpoint.get("telemetry_sample_interval_ms")
    expected = derive_control_horizon_dt_ms(future_offsets, interval_ms)

    raw_horizon = checkpoint.get("control_horizon_dt_ms")
    if not isinstance(raw_horizon, list) or not raw_horizon:
        raise ValueError("stop-sign checkpoint must include non-empty control_horizon_dt_ms")
    actual = tuple(
        _positive_int(value, source=f"control_horizon_dt_ms[{index}]")
        for index, value in enumerate(raw_horizon)
    )
    if len(actual) != len(expected):
        raise ValueError(
            "control_horizon_dt_ms length must match future_offsets: "
            f"timings={len(actual)} offsets={len(expected)}"
        )
    if tuple(sorted(actual)) != actual or len(set(actual)) != len(actual):
        raise ValueError("control_horizon_dt_ms must be strictly increasing")
    if actual != expected:
        raise ValueError(
            "control_horizon_dt_ms must be derived from future_offsets and "
            "telemetry_sample_interval_ms: "
            f"expected={list(expected)} actual={list(actual)}"
        )
    return actual


def validate_checkpoint_control_contract(checkpoint: Mapping[str, Any]) -> tuple[int, ...]:
    planner_format = str(checkpoint.get("planner_format", "")).strip()
    if planner_format != PLANNER_FORMAT:
        if planner_format == LEGACY_PLANNER_FORMAT or _looks_like_legacy_controls(checkpoint):
            raise ValueError(LEGACY_CONTROL_CONTRACT_ERROR)
        raise ValueError(
            f"unsupported planner_format={planner_format or '<missing>'}; "
            f"expected {PLANNER_FORMAT}"
        )

    version = _positive_int(
        checkpoint.get("planner_format_version"),
        source="planner_format_version",
    )
    if version != PLANNER_FORMAT_VERSION:
        raise ValueError(
            f"planner_format_version must be {PLANNER_FORMAT_VERSION}, got {version}"
        )

    raw_contract = checkpoint.get("control_contract")
    if not isinstance(raw_contract, Mapping):
        raise ValueError("stop-sign checkpoint must include control_contract metadata")
    contract_name = str(raw_contract.get("name", "")).strip()
    if contract_name != CONTROL_CONTRACT_NAME:
        raise ValueError(
            f"control_contract.name must be {CONTROL_CONTRACT_NAME}, got {contract_name or '<missing>'}"
        )
    contract_version = _positive_int(
        raw_contract.get("version"),
        source="control_contract.version",
    )
    if contract_version != CONTROL_CONTRACT_VERSION:
        raise ValueError(
            f"control_contract.version must be {CONTROL_CONTRACT_VERSION}, got {contract_version}"
        )
    contract_direction = str(raw_contract.get("direction", "")).strip().lower()
    if contract_direction != CONTROL_HORIZON_DIRECTION:
        raise ValueError(
            "control_contract.direction must be "
            f"{CONTROL_HORIZON_DIRECTION}, got {contract_direction or '<missing>'}"
        )

    checkpoint_names = checkpoint.get("control_target_names")
    if not isinstance(checkpoint_names, list):
        raise ValueError("stop-sign checkpoint must include control_target_names")
    names = require_stop_sign_control_target_names(
        checkpoint_names,
        source="checkpoint control_target_names",
    )
    contract_targets = raw_contract.get("targets")
    if not isinstance(contract_targets, list):
        raise ValueError("control_contract.targets must be a list")
    require_stop_sign_control_target_names(
        contract_targets,
        source="control_contract.targets",
    )

    raw_activations = raw_contract.get("output_activations")
    if not isinstance(raw_activations, Mapping):
        raise ValueError("control_contract.output_activations must be a mapping")
    activations = {str(key): str(value).strip().lower() for key, value in raw_activations.items()}
    if activations != CONTROL_OUTPUT_ACTIVATIONS:
        raise ValueError(
            "control_contract.output_activations do not match the required bounded outputs: "
            f"expected={CONTROL_OUTPUT_ACTIVATIONS} actual={activations}"
        )

    raw_ranges = raw_contract.get("output_ranges")
    if not isinstance(raw_ranges, Mapping):
        raise ValueError("control_contract.output_ranges must be a mapping")
    ranges = _normalize_output_ranges(raw_ranges)
    if ranges != CONTROL_OUTPUT_RANGES:
        raise ValueError(
            "control_contract.output_ranges do not match the required bounded outputs: "
            f"expected={CONTROL_OUTPUT_RANGES} actual={ranges}"
        )
    if names != tuple(contract_targets):
        raise AssertionError("validated checkpoint control target metadata diverged unexpectedly")

    release_policy = checkpoint.get("release_policy")
    if release_policy is not None and not isinstance(release_policy, Mapping):
        raise ValueError("stop-sign checkpoint release_policy metadata must be a mapping")
    if isinstance(release_policy, Mapping) and (
        release_policy.get("name") != RELEASE_POLICY_NAME
        or release_policy.get("learned") is not False
    ):
        raise ValueError(
            f"V0 release_policy must be non-learned {RELEASE_POLICY_NAME}"
        )

    return resolve_checkpoint_control_horizon_dt_ms(checkpoint)


def _looks_like_legacy_controls(checkpoint: Mapping[str, Any]) -> bool:
    raw_names = checkpoint.get("control_target_names")
    if not isinstance(raw_names, list):
        return False
    names = tuple(str(name).strip() for name in raw_names)
    return names != STOP_SIGN_CONTROL_TARGET_NAMES


def _normalize_output_ranges(raw_ranges: Mapping[Any, Any]) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for raw_name, raw_range in raw_ranges.items():
        name = str(raw_name).strip()
        if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
            raise ValueError(f"control_contract.output_ranges.{name} must contain [min, max]")
        values: list[float] = []
        for value in raw_range:
            if isinstance(value, bool):
                raise ValueError(f"control_contract.output_ranges.{name} values must be finite numbers")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"control_contract.output_ranges.{name} values must be finite numbers"
                ) from exc
            if not math.isfinite(numeric):
                raise ValueError(f"control_contract.output_ranges.{name} values must be finite numbers")
            values.append(numeric)
        ranges[name] = (values[0], values[1])
    return ranges


def _positive_int(value: Any, *, source: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{source} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be a positive integer") from exc
    if not numeric.is_integer() or numeric <= 0:
        raise ValueError(f"{source} must be a positive integer")
    return int(numeric)
