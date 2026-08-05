from __future__ import annotations

import unittest

from server import ModelRuntime
from state_inputs import (
    CURRENT_SPEED_KEY,
    PARKING_DISTANCE_KEY,
    PARKING_HEADING_ERROR_KEY,
    PARKING_LATERAL_ERROR_KEY,
    PARKING_LONGITUDINAL_ERROR_KEY,
    PARKING_TARGET_CONFIGURED_KEY,
    build_state_input_vector_from_mapping,
    state_input_config_from_metadata,
    state_input_definitions_metadata,
)


class ParkingStateInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = state_input_config_from_metadata({
            CURRENT_SPEED_KEY: {"enabled": True, "cap": 5.0},
            PARKING_TARGET_CONFIGURED_KEY: {"enabled": True},
            PARKING_LONGITUDINAL_ERROR_KEY: {"enabled": True, "cap": 15.0},
            PARKING_LATERAL_ERROR_KEY: {"enabled": True, "cap": 8.0},
            PARKING_HEADING_ERROR_KEY: {"enabled": True, "cap": 90.0},
            PARKING_DISTANCE_KEY: {"enabled": True, "cap": 20.0},
        })

    def test_parking_inputs_accept_live_camel_case_telemetry(self) -> None:
        vector = build_state_input_vector_from_mapping({
            "currentSpeed": 2.5,
            "parkingTargetConfigured": True,
            "parkingLongitudinalError": -7.5,
            "parkingLateralError": 4.0,
            "parkingHeadingError": -45.0,
            "parkingDistance": 10.0,
        }, self.config)

        self.assertEqual(tuple(vector.shape), (6,))
        self.assertEqual(vector.tolist(), [0.5, 1.0, -0.5, 0.5, -0.5, 0.5])

    def test_parking_inputs_clamp_outliers(self) -> None:
        vector = build_state_input_vector_from_mapping({
            "current_speed": 50.0,
            "parking_target_configured": False,
            "parking_longitudinal_error": -50.0,
            "parking_lateral_error": 50.0,
            "parking_heading_error": -180.0,
            "parking_distance": -1.0,
        }, self.config)

        self.assertEqual(vector.tolist(), [1.0, 0.0, -1.0, 1.0, -1.0, 0.0])

    def test_metadata_exposes_backend_field_names(self) -> None:
        definitions = {
            item["key"]: item
            for item in state_input_definitions_metadata()
        }

        self.assertEqual(
            definitions[PARKING_TARGET_CONFIGURED_KEY]["camelKey"],
            "parkingTargetConfigured",
        )
        self.assertEqual(
            definitions[PARKING_LONGITUDINAL_ERROR_KEY]["camelKey"],
            "parkingLongitudinalError",
        )
        self.assertEqual(
            definitions[PARKING_LATERAL_ERROR_KEY]["camelKey"],
            "parkingLateralError",
        )
        self.assertEqual(
            definitions[PARKING_HEADING_ERROR_KEY]["camelKey"],
            "parkingHeadingError",
        )
        self.assertEqual(
            definitions[PARKING_DISTANCE_KEY]["camelKey"],
            "parkingDistance",
        )

    def test_parking_checkpoint_rejects_inference_without_a_target(self) -> None:
        target_only_config = state_input_config_from_metadata({
            PARKING_TARGET_CONFIGURED_KEY: {"enabled": True},
        })

        with self.assertRaisesRegex(ValueError, "parking target must be configured"):
            ModelRuntime()._extract_state_inputs(
                {"parkingTargetConfigured": False},
                target_only_config,
            )


if __name__ == "__main__":
    unittest.main()
