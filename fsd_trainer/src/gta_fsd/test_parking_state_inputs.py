from __future__ import annotations

import unittest

from server import ModelRuntime
from state_inputs import CURRENT_SPEED_KEY, build_state_input_vector_from_mapping, state_input_config_from_metadata


class StopSignStateInputTests(unittest.TestCase):
    def test_current_speed_can_be_normalized_without_oracle_geometry(self) -> None:
        config = state_input_config_from_metadata({
            CURRENT_SPEED_KEY: {"enabled": True, "cap": 8.0},
        })
        vector = build_state_input_vector_from_mapping({"currentSpeed": 4.0}, config)
        self.assertEqual(vector.tolist(), [0.5])

    def test_inference_requires_only_enabled_non_oracle_inputs(self) -> None:
        config = state_input_config_from_metadata({
            CURRENT_SPEED_KEY: {"enabled": True, "cap": 8.0},
        })
        raw, normalized = ModelRuntime()._extract_state_inputs({"currentSpeed": 2.0}, config)
        self.assertEqual(raw, {CURRENT_SPEED_KEY: 2.0})
        self.assertAlmostEqual(float(normalized[CURRENT_SPEED_KEY].item()), 0.25)


if __name__ == "__main__":
    unittest.main()
