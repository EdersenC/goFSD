from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from config import DEFAULT_AUX_TARGET_NAMES, parse_temporal_dataset_config
from control_contract import (
    LEGACY_CONTROL_CONTRACT_ERROR,
    PARKING_CONTROL_TARGET_NAMES,
    PLANNER_FORMAT,
    PLANNER_FORMAT_VERSION,
    apply_parking_control_activations,
    control_contract_metadata,
    derive_control_horizon_dt_ms,
    validate_checkpoint_control_contract,
)
from dataset import build_control_targets, derive_stop_probability
from inference import load_checkpoint
from models.planner import DrivingCNN
from server import ModelRuntime, parse_sampled_at_s, require_predict_control_contract
from target_transforms import build_target_transform_registry, target_transform_metadata
from train import compute_planner_losses


def parking_checkpoint_metadata() -> dict[str, object]:
    future_offsets = [1, 2, 3, 4, 5, 6]
    transforms = build_target_transform_registry(
        PARKING_CONTROL_TARGET_NAMES + DEFAULT_AUX_TARGET_NAMES
    )
    return {
        "planner_format": PLANNER_FORMAT,
        "planner_format_version": PLANNER_FORMAT_VERSION,
        "control_contract": control_contract_metadata(),
        "control_target_names": list(PARKING_CONTROL_TARGET_NAMES),
        "aux_target_names": list(DEFAULT_AUX_TARGET_NAMES),
        "future_offsets": future_offsets,
        "telemetry_sample_interval_ms": 50,
        "control_horizon_dt_ms": list(derive_control_horizon_dt_ms(future_offsets, 50)),
        "target_transforms": target_transform_metadata(transforms),
        "model_state_dict": {},
    }


class ParkingControlContractTests(unittest.TestCase):
    def test_horizon_timing_is_derived_and_strictly_validated(self) -> None:
        self.assertEqual(
            derive_control_horizon_dt_ms([1, 2, 3, 4, 5, 6], 50),
            (50, 100, 150, 200, 250, 300),
        )
        checkpoint = parking_checkpoint_metadata()
        self.assertEqual(
            validate_checkpoint_control_contract(checkpoint),
            (50, 100, 150, 200, 250, 300),
        )

        checkpoint["control_horizon_dt_ms"] = [50, 100, 90, 200, 250, 300]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_checkpoint_control_contract(checkpoint)

        checkpoint = parking_checkpoint_metadata()
        checkpoint["control_horizon_dt_ms"] = [50, 100, 150]
        with self.assertRaisesRegex(ValueError, "length must match"):
            validate_checkpoint_control_contract(checkpoint)

        checkpoint = parking_checkpoint_metadata()
        checkpoint["control_horizon_dt_ms"] = [50, 100, 150, 200, 250, 301]
        with self.assertRaisesRegex(ValueError, "must be derived"):
            validate_checkpoint_control_contract(checkpoint)

    def test_contract_output_ranges_are_exact_and_validated(self) -> None:
        checkpoint = parking_checkpoint_metadata()
        self.assertEqual(
            checkpoint["control_contract"]["output_ranges"]["desired_speed_mps"],
            [0.0, 2.22],
        )

        checkpoint["control_contract"]["output_ranges"]["desired_speed_mps"] = [0.0, 2.2222222222]
        with self.assertRaisesRegex(ValueError, "output_ranges"):
            validate_checkpoint_control_contract(checkpoint)

    def test_bounded_control_activations_match_semantic_ranges(self) -> None:
        raw = torch.tensor([[[-20.0, -20.0, -20.0], [20.0, 20.0, 20.0]]])
        activated = apply_parking_control_activations(raw, PARKING_CONTROL_TARGET_NAMES)

        self.assertTrue(torch.all(activated[..., 0] >= -1.0))
        self.assertTrue(torch.all(activated[..., 0] <= 1.0))
        self.assertTrue(torch.all(activated[..., 1:] >= 0.0))
        self.assertTrue(torch.all(activated[..., 1:] <= 1.0))
        self.assertLess(float(activated[0, 0, 0]), 0.0)
        self.assertGreater(float(activated[0, 1, 0]), 0.0)

    def test_stop_loss_uses_logits_when_the_model_provides_them(self) -> None:
        predictions = torch.tensor([[[0.0, 0.5, 0.99]]])
        logits = torch.tensor([[[0.0, 0.0, -2.0]]])
        targets = torch.tensor([[[0.0, 0.5, 1.0]]])
        aux = torch.zeros((1, 1, len(DEFAULT_AUX_TARGET_NAMES)))

        losses = compute_planner_losses(
            predictions,
            targets,
            aux,
            aux,
            aux_loss_weight=0.3,
            horizon_loss_weights=(1.0,),
            control_target_names=PARKING_CONTROL_TARGET_NAMES,
            aux_target_names=DEFAULT_AUX_TARGET_NAMES,
            pred_control_logits=logits,
        )

        self.assertAlmostEqual(float(losses["stop_probability_loss"]), 2.126928, places=5)

    def test_parking_model_composes_bounded_outputs_with_trainable_logits(self) -> None:
        model = DrivingCNN(
            frame_count=3,
            telemetry_feature_dim=6,
            telemetry_hidden_dim=16,
            telemetry_sequence_length=9,
            horizon=2,
            control_dim=3,
            control_target_names=PARKING_CONTROL_TARGET_NAMES,
            aux_dim=len(DEFAULT_AUX_TARGET_NAMES),
            width_multiplier=0.25,
        )
        output = model(
            torch.zeros((1, 3, 3, 32, 32)),
            torch.zeros((1, 9, 6)),
        )

        self.assertEqual(tuple(output["pred_controls"].shape), (1, 2, 3))
        self.assertEqual(tuple(output["pred_control_logits"].shape), (1, 2, 3))
        self.assertTrue(torch.all(output["pred_controls"][..., 0].abs() <= 1.0))
        self.assertTrue(torch.all(output["pred_controls"][..., 1:] >= 0.0))
        self.assertTrue(torch.all(output["pred_controls"][..., 1:] <= 1.0))

        target_controls = torch.tensor([[[0.0, 0.25, 1.0], [0.0, 0.0, 1.0]]])
        target_aux = torch.zeros((1, 2, len(DEFAULT_AUX_TARGET_NAMES)))
        losses = compute_planner_losses(
            output["pred_controls"],
            target_controls,
            output["pred_aux"],
            target_aux,
            aux_loss_weight=0.3,
            horizon_loss_weights=(1.0, 0.5),
            control_target_names=PARKING_CONTROL_TARGET_NAMES,
            aux_target_names=DEFAULT_AUX_TARGET_NAMES,
            pred_control_logits=output["pred_control_logits"],
        )
        losses["loss"].backward()
        self.assertIsNotNone(model.control_decoder[-1].weight.grad)

    def test_dataset_derives_setpoints_from_normalized_parking_telemetry(self) -> None:
        telemetry = {
            "Steering": -0.25,
            "wheelSteeringFullLock": 0.733038306,
            "currentSpeed": 0.1,
            "parkingPhase": "settling",
            "parkingInsideBay": True,
            "parkingAligned": True,
            "parkingParked": False,
        }

        targets = build_control_targets(telemetry, PARKING_CONTROL_TARGET_NAMES)

        self.assertAlmostEqual(float(targets[0]), -0.25)
        self.assertAlmostEqual(float(targets[1]), 0.1)
        self.assertEqual(float(targets[2]), 1.0)
        self.assertEqual(derive_stop_probability({**telemetry, "parkingPhase": "parking"}), 0.0)
        self.assertEqual(derive_stop_probability({**telemetry, "parkingParked": True}), 1.0)

    def test_stop_probability_rejects_false_terminal_and_invalid_settling_signals(self) -> None:
        valid_settling = {
            "currentSpeed": 0.15,
            "parkingPhase": "settling",
            "parkingInsideBay": True,
            "parkingAligned": True,
            "parkingParked": False,
        }
        self.assertEqual(derive_stop_probability(valid_settling), 1.0)
        self.assertEqual(derive_stop_probability({**valid_settling, "currentSpeed": 0.151}), 0.0)
        self.assertEqual(derive_stop_probability({**valid_settling, "parkingInsideBay": False}), 0.0)
        self.assertEqual(derive_stop_probability({**valid_settling, "parkingAligned": False}), 0.0)
        self.assertEqual(derive_stop_probability({**valid_settling, "parkingPhase": "failed"}), 0.0)
        self.assertEqual(derive_stop_probability({**valid_settling, "parkingPhase": "stopping"}), 0.0)
        for phase in ("failed", "stopping"):
            with self.assertRaisesRegex(ValueError, "contradictory parking stop supervision"):
                derive_stop_probability({**valid_settling, "parkingPhase": phase, "parkingParked": True})

    def test_dataset_rejects_ambiguous_legacy_steering_labels(self) -> None:
        telemetry = {
            "Steering": 0.25,
            "currentSpeed": 1.0,
            "parkingPhase": "parking",
            "parkingParked": False,
        }
        with self.assertRaisesRegex(KeyError, "wheelSteeringFullLock"):
            build_control_targets(telemetry, PARKING_CONTROL_TARGET_NAMES)

    def test_config_rejects_legacy_actuator_targets(self) -> None:
        with self.assertRaisesRegex(ValueError, "parking control contract"):
            parse_temporal_dataset_config({
                "dataset": {
                    "control_target_names": ["steering", "acceleration", "brakePressureAvg"],
                },
            })

    def test_v1_checkpoint_fails_with_retrain_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "legacy.pt"
            torch.save({
                "planner_format": "temporal_telemetry_gru_v1",
                "control_target_names": ["steering", "acceleration", "brakePressureAvg"],
                "model_state_dict": {},
            }, checkpoint_path)

            with self.assertRaisesRegex(ValueError, "retraining is required"):
                load_checkpoint(checkpoint_path, torch.device("cpu"))
        self.assertIn("cannot be safely adapted", LEGACY_CONTROL_CONTRACT_ERROR)

    def test_v2_checkpoint_loads_only_with_explicit_contract_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "parking-v2.pt"
            torch.save(parking_checkpoint_metadata(), checkpoint_path)

            checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))

        self.assertEqual(checkpoint["planner_format"], PLANNER_FORMAT)
        self.assertEqual(checkpoint["control_contract"]["name"], "parking_setpoint_v1")
        self.assertEqual(len(checkpoint["target_transforms"]), 7)

    def test_checkpoint_rejects_speed_transform_outside_exact_contract_range(self) -> None:
        checkpoint = parking_checkpoint_metadata()
        speed_transform = next(
            item
            for item in checkpoint["target_transforms"]
            if item["target_name"] == "desired_speed_mps"
        )
        speed_transform["range_max"] = 2.2222222222

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "bad-speed-range.pt"
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(ValueError, r"\[0, 2\.22\]"):
                load_checkpoint(checkpoint_path, torch.device("cpu"))

    def test_model_status_exposes_contract_and_horizon_fields(self) -> None:
        status = ModelRuntime().status()
        self.assertEqual(status["control_contract"]["name"], "parking_setpoint_v1")
        self.assertEqual(status["control_contract"]["direction"], "forward")
        self.assertEqual(status["direction"], "forward")
        self.assertEqual(status["control_horizon_dt_ms"], [])
        self.assertEqual(status["telemetry_sample_interval_ms"], 0)

    def test_predict_sample_time_must_be_finite_and_positive(self) -> None:
        self.assertEqual(parse_sampled_at_s({"sampled_at_s": 123.5}), 123.5)
        for invalid in (None, True, 0, -1, float("inf"), float("nan"), "123.5"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "sampled_at_s"):
                    parse_sampled_at_s({"sampled_at_s": invalid})

    def test_predict_request_requires_exact_contract_ownership(self) -> None:
        self.assertIsNone(require_predict_control_contract({
            "control_contract": "parking_setpoint_v1",
        }))
        with self.assertRaisesRegex(ValueError, "parking_setpoint_v1"):
            require_predict_control_contract({})
        for invalid in (None, 1, {}, "", "parking_setpoint_v1 ", "legacy_actuator_v1"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "parking_setpoint_v1"):
                    require_predict_control_contract({"control_contract": invalid})

    def test_predict_echoes_sample_time_and_forward_direction(self) -> None:
        class StubModel:
            def __call__(
                self,
                images: torch.Tensor,
                telemetry: torch.Tensor,
                state_inputs: object,
            ) -> dict[str, torch.Tensor]:
                return {
                    "pred_controls": torch.tensor([[[0.25, 0.5, 0.75]]]),
                    "pred_aux": torch.zeros((1, 1, len(DEFAULT_AUX_TARGET_NAMES))),
                }

        runtime = ModelRuntime()
        runtime._model = StubModel()
        runtime._device = torch.device("cpu")
        runtime._checkpoint_path = Path("parking-v2.pt")
        runtime._planner_format = PLANNER_FORMAT
        runtime._image_offsets = [0]
        runtime._telemetry_offsets = [0]
        runtime._future_offsets = [1]
        runtime._telemetry_sample_interval_ms = 50
        runtime._control_horizon_dt_ms = [50]
        runtime._telemetry_feature_names = ["current_speed"]
        runtime._control_target_names = list(PARKING_CONTROL_TARGET_NAMES)
        runtime._aux_target_names = list(DEFAULT_AUX_TARGET_NAMES)
        runtime._target_transforms = build_target_transform_registry(
            PARKING_CONTROL_TARGET_NAMES + DEFAULT_AUX_TARGET_NAMES
        )
        runtime._extract_frames = lambda payload: [torch.zeros((3, 4, 4))]
        runtime._extract_planner_telemetry = lambda payload: torch.zeros((1, 1, 1))
        runtime._extract_state_inputs = lambda payload, config: ({}, {})

        model_status = runtime.status()
        response = runtime.predict({
            "control_contract": "parking_setpoint_v1",
            "sampled_at_s": 123.5,
        })

        self.assertTrue(model_status["loaded"])
        self.assertEqual(model_status["direction"], "forward")
        self.assertEqual(model_status["control_horizon_dt_ms"], [50])
        self.assertEqual(response["sampled_at_s"], 123.5)
        self.assertEqual(response["direction"], "forward")
        self.assertEqual(response["control_horizon_dt_ms"], [50])
        self.assertEqual(
            response["control_contract"]["output_ranges"]["desired_speed_mps"],
            [0.0, 2.22],
        )
        self.assertAlmostEqual(response["pred_controls"][0][0][1], 1.11)


if __name__ == "__main__":
    unittest.main()
