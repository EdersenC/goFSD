from __future__ import annotations

import unittest

try:
    from .control_client import INPUT_MODE_MODEL_RAW, INPUT_MODE_NORMALIZED
    from .control_translation import (
        CONTROL_SEMANTICS_TARGET_SPEED,
        CONTROL_SEMANTICS_VEHICLE_STATE,
        CONTROL_SEMANTICS_SPEED_DELTA,
        translate_control_prediction,
    )
except ImportError:
    from control_client import INPUT_MODE_MODEL_RAW, INPUT_MODE_NORMALIZED
    from control_translation import (
        CONTROL_SEMANTICS_SPEED_DELTA,
        CONTROL_SEMANTICS_TARGET_SPEED,
        CONTROL_SEMANTICS_VEHICLE_STATE,
        translate_control_prediction,
    )


class ControlTranslationTests(unittest.TestCase):
    def test_speed_delta_translation_uses_signed_output_directly(self) -> None:
        translated = translate_control_prediction(
            steering=0.15,
            acceleration=-0.35,
            control_semantics=CONTROL_SEMANTICS_SPEED_DELTA,
        )

        self.assertEqual(translated["input_mode"], INPUT_MODE_NORMALIZED)
        self.assertAlmostEqual(translated["steering"], 0.15, places=6)
        self.assertAlmostEqual(translated["acceleration"], -0.35, places=6)
        self.assertAlmostEqual(translated["throttle"], 0.0, places=6)
        self.assertAlmostEqual(translated["brake"], 0.35, places=6)

    def test_vehicle_state_translation_centers_longitudinal_output(self) -> None:
        translated = translate_control_prediction(
            steering=0.023,
            acceleration=0.502,
            control_semantics=CONTROL_SEMANTICS_VEHICLE_STATE,
        )

        self.assertEqual(translated["input_mode"], INPUT_MODE_NORMALIZED)
        self.assertAlmostEqual(translated["steering"], 0.276, places=6)
        self.assertAlmostEqual(translated["acceleration"], 0.0, places=6)
        self.assertAlmostEqual(translated["throttle"], 0.0, places=6)
        self.assertAlmostEqual(translated["brake"], 0.0, places=6)

    def test_target_speed_translation_uses_speed_error_against_current_speed(self) -> None:
        translated = translate_control_prediction(
            steering=0.1,
            acceleration=8.0,
            control_semantics=CONTROL_SEMANTICS_TARGET_SPEED,
            current_speed=6.0,
        )

        self.assertEqual(translated["input_mode"], INPUT_MODE_NORMALIZED)
        self.assertAlmostEqual(translated["steering"], 0.1, places=6)
        self.assertAlmostEqual(translated["acceleration"], 1.0, places=6)
        self.assertAlmostEqual(translated["throttle"], 1.0, places=6)
        self.assertAlmostEqual(translated["brake"], 0.0, places=6)
        self.assertAlmostEqual(translated["translation"]["speed_error"], 2.0, places=6)

    def test_target_speed_translation_requires_current_speed(self) -> None:
        with self.assertRaisesRegex(ValueError, "current_speed"):
            translate_control_prediction(
                steering=0.0,
                acceleration=5.0,
                control_semantics=CONTROL_SEMANTICS_TARGET_SPEED,
            )

    def test_controller_input_semantics_pass_through_raw_values(self) -> None:
        translated = translate_control_prediction(
            steering=-0.2,
            acceleration=0.35,
            control_semantics="controller_input",
        )

        self.assertEqual(translated["input_mode"], INPUT_MODE_MODEL_RAW)
        self.assertAlmostEqual(translated["steering"], -0.2, places=6)
        self.assertAlmostEqual(translated["acceleration"], 0.35, places=6)
        self.assertAlmostEqual(translated["throttle"], 0.35, places=6)
        self.assertAlmostEqual(translated["brake"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
