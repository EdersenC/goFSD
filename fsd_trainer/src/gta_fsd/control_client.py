from __future__ import annotations

import json
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INPUT_MODE_MODEL_RAW = "model_raw"
INPUT_MODE_NORMALIZED = "normalized"


@dataclass(frozen=True)
class ActuatorConfig:
    url: str
    request_timeout_seconds: float


def load_actuator_config(config_path: Path) -> ActuatorConfig:
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    backend_raw = raw.get("backend", {})
    actuator_raw = backend_raw.get("actuator", {})

    url = str(actuator_raw.get("url", "http://127.0.0.1:8080")).strip().rstrip("/")
    if not url:
        raise ValueError(f"Missing backend.actuator.url in {config_path}")

    request_timeout_raw = str(actuator_raw.get("request_timeout", "500ms")).strip().lower()
    request_timeout_seconds = _parse_duration_seconds(request_timeout_raw)
    if request_timeout_seconds <= 0:
        raise ValueError(f"backend.actuator.request_timeout must be > 0 in {config_path}")

    return ActuatorConfig(url=url, request_timeout_seconds=request_timeout_seconds)


def build_control_command(
    *,
    steering: float,
    acceleration: float,
    input_mode: str = INPUT_MODE_MODEL_RAW,
    sequence: int | None = None,
    timestamp_ms: int | None = None,
    handbrake: bool = False,
    enabled: bool = True,
) -> dict[str, Any]:
    throttle = max(float(acceleration), 0.0)
    brake = max(-float(acceleration), 0.0)
    command: dict[str, Any] = {
        "steer": float(steering),
        "throttle": throttle,
        "brake": brake,
        "inputMode": str(input_mode).strip() or INPUT_MODE_MODEL_RAW,
        "handbrake": bool(handbrake),
        "enabled": bool(enabled),
        "timestampMs": int(timestamp_ms if timestamp_ms is not None else time.time() * 1000),
    }
    if sequence is not None:
        command["sequence"] = int(sequence)
    return command


def post_control_command(config: ActuatorConfig, command: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(command).encode("utf-8")
    request = urllib.request.Request(
        f"{config.url}/actuator/command",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=config.request_timeout_seconds) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"actuator command failed: status={exc.code} body={detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"actuator command failed: {exc.reason}") from exc

    parsed = json.loads(payload or "{}")
    if not isinstance(parsed, dict):
        raise RuntimeError("actuator command returned a non-object JSON response")
    return parsed


def _parse_duration_seconds(raw_value: str) -> float:
    if raw_value.endswith("ms"):
        return float(raw_value[:-2]) / 1000.0
    if raw_value.endswith("s"):
        return float(raw_value[:-1])
    return float(raw_value)
