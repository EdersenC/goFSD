from __future__ import annotations

from typing import Any

try:
    from .control_client import INPUT_MODE_MODEL_RAW, INPUT_MODE_NORMALIZED
except ImportError:
    from control_client import INPUT_MODE_MODEL_RAW, INPUT_MODE_NORMALIZED

CONTROL_SEMANTICS_CONTROLLER_INPUT = "controller_input"
CONTROL_SEMANTICS_SPEED_DELTA = "speed_delta"
CONTROL_SEMANTICS_VEHICLE_STATE = "vehicle_state"

DEFAULT_VEHICLE_STATE_STEER_GAIN = 12.0
DEFAULT_VEHICLE_STATE_STEER_DEADBAND = 0.02
DEFAULT_VEHICLE_STATE_ACCEL_CENTER = 0.5
DEFAULT_VEHICLE_STATE_ACCEL_GAIN = 2.0
DEFAULT_VEHICLE_STATE_ACCEL_DEADBAND = 0.05
DEFAULT_SPEED_DELTA_GAIN = 1.0
DEFAULT_SPEED_DELTA_DEADBAND = 0.02


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def translate_control_prediction(
    steering: float,
    acceleration: float,
    control_semantics: str,
) -> dict[str, Any]:
    semantics = str(control_semantics or "").strip() or CONTROL_SEMANTICS_CONTROLLER_INPUT
    raw_steer = float(steering)
    raw_accel = float(acceleration)

    if semantics == CONTROL_SEMANTICS_SPEED_DELTA:
        translated_steer = clamp(raw_steer, -1.0, 1.0)
        signed_accel = clamp(raw_accel * DEFAULT_SPEED_DELTA_GAIN, -1.0, 1.0)
        if abs(signed_accel) < DEFAULT_SPEED_DELTA_DEADBAND:
            signed_accel = 0.0

        return {
            "input_mode": INPUT_MODE_NORMALIZED,
            "steering": translated_steer,
            "acceleration": signed_accel,
            "throttle": max(signed_accel, 0.0),
            "brake": max(-signed_accel, 0.0),
            "translation": {
                "mode": CONTROL_SEMANTICS_SPEED_DELTA,
                "delta_speed_gain": DEFAULT_SPEED_DELTA_GAIN,
                "delta_speed_deadband": DEFAULT_SPEED_DELTA_DEADBAND,
            },
        }

    if semantics == CONTROL_SEMANTICS_VEHICLE_STATE:
        translated_steer = clamp(raw_steer * DEFAULT_VEHICLE_STATE_STEER_GAIN, -1.0, 1.0)
        if abs(translated_steer) < DEFAULT_VEHICLE_STATE_STEER_DEADBAND:
            translated_steer = 0.0

        signed_accel = clamp(
            (raw_accel - DEFAULT_VEHICLE_STATE_ACCEL_CENTER) * DEFAULT_VEHICLE_STATE_ACCEL_GAIN,
            -1.0,
            1.0,
        )
        if abs(signed_accel) < DEFAULT_VEHICLE_STATE_ACCEL_DEADBAND:
            signed_accel = 0.0

        return {
            "input_mode": INPUT_MODE_NORMALIZED,
            "steering": translated_steer,
            "acceleration": signed_accel,
            "throttle": max(signed_accel, 0.0),
            "brake": max(-signed_accel, 0.0),
            "translation": {
                "mode": CONTROL_SEMANTICS_VEHICLE_STATE,
                "steer_gain": DEFAULT_VEHICLE_STATE_STEER_GAIN,
                "accel_center": DEFAULT_VEHICLE_STATE_ACCEL_CENTER,
                "accel_gain": DEFAULT_VEHICLE_STATE_ACCEL_GAIN,
            },
        }

    return {
        "input_mode": INPUT_MODE_MODEL_RAW,
        "steering": raw_steer,
        "acceleration": raw_accel,
        "throttle": max(raw_accel, 0.0),
        "brake": max(-raw_accel, 0.0),
        "translation": {
            "mode": semantics,
        },
    }
