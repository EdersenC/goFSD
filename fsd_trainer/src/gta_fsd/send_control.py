from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

try:
    from .control_client import (
        INPUT_MODE_NORMALIZED,
        ActuatorConfig,
        build_control_command,
        load_actuator_config,
        post_control_command,
    )
except ImportError:
    from control_client import (
        INPUT_MODE_NORMALIZED,
        ActuatorConfig,
        build_control_command,
        load_actuator_config,
        post_control_command,
    )


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "train_config.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a manual control command to the local Go actuator.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help=f"Path to config TOML. Default: {DEFAULT_CONFIG_PATH}")
    parser.add_argument("--steer", type=float, default=0.0, help="Normalized steer in [-1.0, 1.0].")
    parser.add_argument("--throttle", type=float, default=0.0, help="Normalized throttle in [0.0, 1.0].")
    parser.add_argument("--brake", type=float, default=0.0, help="Normalized brake in [0.0, 1.0].")
    parser.add_argument("--handbrake", action="store_true", help="Press the virtual handbrake button.")
    parser.add_argument("--disabled", action="store_true", help="Send the command with enabled=false.")
    parser.add_argument("--sequence", type=int, default=0, help="Optional sequence number.")
    parser.add_argument("--hold-ms", type=int, default=1000, help="How long to repeat a non-disabled command. Default: 1000.")
    parser.add_argument("--repeat-hz", type=float, default=15.0, help="How often to resend a held command. Default: 15.")
    return parser.parse_args()

def repeat_iterations(hold_ms: int, repeat_hz: float) -> int:
    if hold_ms <= 0 or repeat_hz <= 0:
        return 1
    return max(1, int(math.ceil((hold_ms / 1000.0) * repeat_hz)))

def dispatch_manual_command(
    config: ActuatorConfig,
    *,
    steering: float,
    acceleration: float,
    sequence: int,
    handbrake: bool,
    enabled: bool,
    hold_ms: int,
    repeat_hz: float,
    post_fn: Callable[[ActuatorConfig, dict[str, Any]], dict[str, Any]] = post_control_command,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    iterations = 1 if not enabled else repeat_iterations(hold_ms, repeat_hz)
    interval_seconds = 0.0 if repeat_hz <= 0 else 1.0 / repeat_hz
    response: dict[str, Any] = {}

    for index in range(iterations):
        command = build_control_command(
            steering=steering,
            acceleration=acceleration,
            input_mode=INPUT_MODE_NORMALIZED,
            sequence=sequence + index,
            handbrake=handbrake,
            enabled=enabled,
        )
        response = post_fn(config, command)
        if index+1 < iterations and interval_seconds > 0:
            sleep_fn(interval_seconds)

    return response


def main() -> None:
    args = parse_args()
    config = load_actuator_config(args.config)
    acceleration = max(args.throttle, 0.0) - max(args.brake, 0.0)
    response = dispatch_manual_command(
        config,
        steering=args.steer,
        acceleration=acceleration,
        sequence=args.sequence,
        handbrake=args.handbrake,
        enabled=not args.disabled,
        hold_ms=args.hold_ms,
        repeat_hz=args.repeat_hz,
    )
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
