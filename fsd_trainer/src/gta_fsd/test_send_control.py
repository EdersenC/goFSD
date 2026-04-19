from __future__ import annotations

import unittest

try:
    from .control_client import ActuatorConfig, INPUT_MODE_NORMALIZED
    from .send_control import dispatch_manual_command, repeat_iterations
except ImportError:
    from control_client import ActuatorConfig, INPUT_MODE_NORMALIZED
    from send_control import dispatch_manual_command, repeat_iterations


class SendControlTests(unittest.TestCase):
    def test_repeat_iterations_matches_hold_duration(self) -> None:
        self.assertEqual(repeat_iterations(1000, 15.0), 15)
        self.assertEqual(repeat_iterations(0, 15.0), 1)
        self.assertEqual(repeat_iterations(1000, 0.0), 1)

    def test_dispatch_manual_command_repeats_enabled_commands(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_post(_config: ActuatorConfig, command: dict[str, object]) -> dict[str, object]:
            calls.append(command)
            return {"status": "accepted", "state": {}}

        response = dispatch_manual_command(
            ActuatorConfig(url="http://127.0.0.1:8080", request_timeout_seconds=0.5),
            steering=0.2,
            acceleration=0.3,
            sequence=7,
            handbrake=False,
            enabled=True,
            hold_ms=1000,
            repeat_hz=15.0,
            post_fn=fake_post,
            sleep_fn=lambda _seconds: None,
        )

        self.assertEqual(response["status"], "accepted")
        self.assertEqual(len(calls), 15)
        self.assertEqual(calls[0]["inputMode"], INPUT_MODE_NORMALIZED)
        self.assertEqual(calls[0]["sequence"], 7)
        self.assertEqual(calls[-1]["sequence"], 21)

    def test_dispatch_manual_command_sends_disabled_release_once(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_post(_config: ActuatorConfig, command: dict[str, object]) -> dict[str, object]:
            calls.append(command)
            return {"status": "accepted"}

        dispatch_manual_command(
            ActuatorConfig(url="http://127.0.0.1:8080", request_timeout_seconds=0.5),
            steering=0.0,
            acceleration=0.0,
            sequence=3,
            handbrake=False,
            enabled=False,
            hold_ms=5000,
            repeat_hz=15.0,
            post_fn=fake_post,
            sleep_fn=lambda _seconds: None,
        )

        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0]["enabled"])


if __name__ == "__main__":
    unittest.main()
