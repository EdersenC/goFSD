from __future__ import annotations

import unittest

from stop_sign_contract import (
    RELEASE_POLICY_NAME,
    STOP_SIGN_PHASES,
    StopSignMotionPlan,
    deterministic_longitudinal_command,
    inverse_frequency_phase_weights,
    normalize_stop_sign_phase,
    require_disjoint_stop_locations,
    stop_location_key_from_metadata,
)


class StopSignTemporalContractTests(unittest.TestCase):
    def test_fine_phase_contract_is_closed_and_explicit(self) -> None:
        self.assertEqual(
            STOP_SIGN_PHASES,
            ("accelerate", "cruise_approach", "decelerate", "stop_hold", "release"),
        )
        for phase in STOP_SIGN_PHASES:
            self.assertEqual(normalize_stop_sign_phase(phase), phase)
        with self.assertRaisesRegex(ValueError, "stopSignPhase"):
            normalize_stop_sign_phase("approach")

    def test_inverse_frequency_weights_equalize_phase_mass(self) -> None:
        phases = ("accelerate", "accelerate", "accelerate", "stop_hold", "release", "release")
        weights = inverse_frequency_phase_weights(phases)
        mass = {
            phase: sum(weight for item, weight in zip(phases, weights) if item == phase)
            for phase in set(phases)
        }
        self.assertEqual(mass, {"accelerate": 1.0, "stop_hold": 1.0, "release": 1.0})

    def test_split_key_is_physical_stop_location_not_attempt_or_variant(self) -> None:
        goal = {
            "signPose": {"x": 10.001, "y": 20.0, "z": 30.0, "heading": 90.04},
            "variationId": "sunny-red-car",
            "attemptIndex": 1,
        }
        first = stop_location_key_from_metadata({"stopSignGoal": goal})
        second = stop_location_key_from_metadata({
            "stopSignGoal": {**goal, "variationId": "rain-blue-car", "attemptIndex": 8},
        })
        self.assertEqual(first, second)
        with self.assertRaisesRegex(ValueError, "overlap physical stop-sign locations"):
            require_disjoint_stop_locations((first,), (second,))

    def test_deterministic_handoff_never_applies_throttle_and_brake_together(self) -> None:
        accelerate = StopSignMotionPlan((250, 500), (4.0, 3.5), (0.0, 0.0))
        accelerate_command = deterministic_longitudinal_command(accelerate, current_speed_mps=1.0)
        self.assertGreater(accelerate_command.throttle, 0.0)
        self.assertEqual(accelerate_command.brake, 0.0)

        brake_command = deterministic_longitudinal_command(accelerate, current_speed_mps=6.0)
        self.assertEqual(brake_command.throttle, 0.0)
        self.assertGreater(brake_command.brake, 0.0)

        stop = StopSignMotionPlan((250, 500), (0.0, 0.0), (0.9, 1.0))
        stop_command = deterministic_longitudinal_command(stop, current_speed_mps=0.05)
        self.assertTrue(stop_command.hold_stop)
        self.assertEqual(stop_command.release_policy, RELEASE_POLICY_NAME)
        self.assertEqual((stop_command.throttle, stop_command.brake), (0.0, 1.0))

    def test_v0_rejects_a_learned_release_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "V0 release_policy"):
            StopSignMotionPlan((250,), (0.0,), (1.0,), release_policy="learned_release")


if __name__ == "__main__":
    unittest.main()
