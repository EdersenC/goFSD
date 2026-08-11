"""Stop-sign temporal labels, split guards, and deterministic handoff policy."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


STOP_SIGN_PHASES = (
    "accelerate",
    "cruise_approach",
    "decelerate",
    "stop_hold",
    "release",
)
STOP_SIGN_PHASE_SET = frozenset(STOP_SIGN_PHASES)
RELEASE_POLICY_NAME = "scripted_dwell_release_v0"


def normalize_stop_sign_phase(value: Any) -> str:
    phase = str(value).strip().lower()
    if phase not in STOP_SIGN_PHASE_SET:
        raise ValueError(
            "stopSignPhase must be one of "
            f"{list(STOP_SIGN_PHASES)}, got {phase or '<missing>'}"
        )
    return phase


def stop_sign_phase_from_telemetry(telemetry: Mapping[str, Any]) -> str:
    flattened = _flatten_telemetry(telemetry)
    return normalize_stop_sign_phase(flattened.get("stopSignPhase"))


def inverse_frequency_phase_weights(phases: Sequence[str]) -> tuple[float, ...]:
    """Return weights whose total mass is equal for every observed phase."""
    normalized = tuple(normalize_stop_sign_phase(phase) for phase in phases)
    if not normalized:
        return ()
    counts = Counter(normalized)
    return tuple(1.0 / counts[phase] for phase in normalized)


def stop_location_key_from_metadata(metadata: Mapping[str, Any]) -> str:
    """Build a split key from physical sign location, never from a frame/sample id."""
    goal = metadata.get("stopSignGoal")
    if not isinstance(goal, Mapping):
        raise ValueError("stop-sign metadata must include stopSignGoal")
    sign_pose = goal.get("signPose")
    if not isinstance(sign_pose, Mapping):
        raise ValueError("stopSignGoal.signPose must be a mapping")
    x = _finite_number(sign_pose.get("x"), "stopSignGoal.signPose.x")
    y = _finite_number(sign_pose.get("y"), "stopSignGoal.signPose.y")
    z = _finite_number(sign_pose.get("z"), "stopSignGoal.signPose.z")
    heading = _finite_number(sign_pose.get("heading"), "stopSignGoal.signPose.heading") % 360.0
    # Centimeter/0.1-degree quantization absorbs harmless serialization noise while
    # keeping physically distinct stop locations in separate split groups.
    return f"sign:{x:.2f}:{y:.2f}:{z:.2f}:{heading:.1f}"


def require_disjoint_stop_locations(
    train_location_keys: Sequence[str],
    validation_location_keys: Sequence[str],
) -> None:
    overlap = sorted(set(train_location_keys) & set(validation_location_keys))
    if overlap:
        preview = ", ".join(overlap[:5])
        raise ValueError(
            "train and validation datasets overlap physical stop-sign locations; "
            f"move whole scenarios between splits: {preview}"
        )


@dataclass(frozen=True)
class StopSignMotionPlan:
    horizon_dt_ms: tuple[int, ...]
    future_speed_mps: tuple[float, ...]
    stop_intent: tuple[float, ...]
    release_policy: str = RELEASE_POLICY_NAME

    def __post_init__(self) -> None:
        width = len(self.horizon_dt_ms)
        if width < 1 or len(self.future_speed_mps) != width or len(self.stop_intent) != width:
            raise ValueError("motion-plan horizons, speeds, and stop intents must have equal non-zero length")
        if any(current >= following for current, following in zip(self.horizon_dt_ms, self.horizon_dt_ms[1:])):
            raise ValueError("motion-plan horizon_dt_ms must be strictly increasing")
        if any(not math.isfinite(speed) or speed < 0.0 for speed in self.future_speed_mps):
            raise ValueError("motion-plan speeds must be finite and non-negative")
        if any(not math.isfinite(intent) or not 0.0 <= intent <= 1.0 for intent in self.stop_intent):
            raise ValueError("motion-plan stop_intent values must be within [0, 1]")
        if self.release_policy != RELEASE_POLICY_NAME:
            raise ValueError(f"V0 release_policy must be {RELEASE_POLICY_NAME}")


@dataclass(frozen=True)
class LongitudinalCommand:
    throttle: float
    brake: float
    target_speed_mps: float
    hold_stop: bool
    release_policy: str = RELEASE_POLICY_NAME


def deterministic_longitudinal_command(
    plan: StopSignMotionPlan,
    *,
    current_speed_mps: float,
    stop_threshold: float = 0.65,
    speed_deadband_mps: float = 0.10,
    throttle_gain: float = 0.40,
    brake_gain: float = 0.55,
) -> LongitudinalCommand:
    """Translate the nearest learned setpoint into exclusive throttle or brake.

    V0 never learns the dwell-release decision. The runtime owns the five-second
    dwell state machine and only calls this translator after scripted release.
    """
    current_speed = _finite_number(current_speed_mps, "current_speed_mps")
    if current_speed < 0.0:
        raise ValueError("current_speed_mps must be non-negative")
    target_speed = plan.future_speed_mps[0]
    hold_stop = plan.stop_intent[0] >= stop_threshold
    if hold_stop:
        return LongitudinalCommand(0.0, 1.0, 0.0, True)
    error = target_speed - current_speed
    if abs(error) <= speed_deadband_mps:
        return LongitudinalCommand(0.0, 0.0, target_speed, False)
    if error > 0.0:
        return LongitudinalCommand(min(1.0, error * throttle_gain), 0.0, target_speed, False)
    return LongitudinalCommand(0.0, min(1.0, -error * brake_gain), target_speed, False)


def release_policy_metadata() -> dict[str, Any]:
    return {
        "name": RELEASE_POLICY_NAME,
        "version": 0,
        "learned": False,
        "description": "release occurs only after the runtime completes the configured dwell",
    }


def _flatten_telemetry(telemetry: Mapping[str, Any]) -> dict[str, Any]:
    if not any(key in telemetry for key in ("control", "aux", "raw")):
        return dict(telemetry)
    flattened: dict[str, Any] = {}
    for section_name in ("control", "aux", "raw"):
        section = telemetry.get(section_name)
        if section is None:
            continue
        if not isinstance(section, Mapping):
            raise ValueError(f"telemetry section {section_name} must be a mapping")
        flattened.update(section)
    return flattened


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number
