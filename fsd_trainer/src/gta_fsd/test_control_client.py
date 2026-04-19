from __future__ import annotations

import unittest

try:
    from .control_client import INPUT_MODE_MODEL_RAW, INPUT_MODE_NORMALIZED, build_control_command
except ImportError:
    from control_client import INPUT_MODE_MODEL_RAW, INPUT_MODE_NORMALIZED, build_control_command


class ControlClientTests(unittest.TestCase):
    def test_build_control_command_defaults_to_raw_model_mode(self) -> None:
        command = build_control_command(steering=0.1, acceleration=-0.25, sequence=7, timestamp_ms=1234)

        self.assertEqual(command["inputMode"], INPUT_MODE_MODEL_RAW)
        self.assertAlmostEqual(command["steer"], 0.1, places=6)
        self.assertAlmostEqual(command["throttle"], 0.0, places=6)
        self.assertAlmostEqual(command["brake"], 0.25, places=6)
        self.assertEqual(command["sequence"], 7)
        self.assertEqual(command["timestampMs"], 1234)

    def test_build_control_command_allows_explicit_normalized_mode(self) -> None:
        command = build_control_command(
            steering=-0.2,
            acceleration=0.35,
            input_mode=INPUT_MODE_NORMALIZED,
            enabled=False,
            handbrake=True,
        )

        self.assertEqual(command["inputMode"], INPUT_MODE_NORMALIZED)
        self.assertAlmostEqual(command["throttle"], 0.35, places=6)
        self.assertAlmostEqual(command["brake"], 0.0, places=6)
        self.assertFalse(command["enabled"])
        self.assertTrue(command["handbrake"])


if __name__ == "__main__":
    unittest.main()
