from __future__ import annotations

import argparse
import json
from pathlib import Path

from control_client import build_control_command, load_actuator_config, post_control_command
from inference import DEFAULT_CONFIG_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a manual control command to the local Go actuator.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help=f"Path to config TOML. Default: {DEFAULT_CONFIG_PATH}")
    parser.add_argument("--steer", type=float, default=0.0, help="Normalized steer in [-1.0, 1.0].")
    parser.add_argument("--throttle", type=float, default=0.0, help="Normalized throttle in [0.0, 1.0].")
    parser.add_argument("--brake", type=float, default=0.0, help="Normalized brake in [0.0, 1.0].")
    parser.add_argument("--handbrake", action="store_true", help="Press the virtual handbrake button.")
    parser.add_argument("--disabled", action="store_true", help="Send the command with enabled=false.")
    parser.add_argument("--sequence", type=int, default=0, help="Optional sequence number.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_actuator_config(args.config)
    acceleration = max(args.throttle, 0.0) - max(args.brake, 0.0)
    command = build_control_command(
        steering=args.steer,
        acceleration=acceleration,
        sequence=args.sequence,
        handbrake=args.handbrake,
        enabled=not args.disabled,
    )
    response = post_control_command(config, command)
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
