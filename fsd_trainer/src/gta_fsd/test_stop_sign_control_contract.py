from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from config import DEFAULT_AUX_TARGET_NAMES, parse_temporal_dataset_config
from control_contract import (
    CONTROL_CONTRACT_NAME,
    FUTURE_SPEED_MPS,
    MAX_DESIRED_SPEED_MPS,
    PLANNER_FORMAT,
    PLANNER_FORMAT_VERSION,
    STOP_INTENT,
    STOP_SIGN_CONTROL_TARGET_NAMES,
    apply_stop_sign_control_activations,
    control_contract_metadata,
    derive_control_horizon_dt_ms,
    validate_checkpoint_control_contract,
)
from dataset import build_control_targets
from inference import load_checkpoint
from models.planner import StopSignTemporalPlanner
from server import ModelRuntime, require_predict_control_contract
from stop_sign_contract import release_policy_metadata
from target_transforms import build_target_transform_registry, target_transform_metadata
from train import compute_planner_losses


def stop_sign_checkpoint_metadata() -> dict[str, object]:
    future_offsets = [2, 5, 10, 20]
    transforms = build_target_transform_registry(
        STOP_SIGN_CONTROL_TARGET_NAMES + DEFAULT_AUX_TARGET_NAMES
    )
    return {
        "planner_format": PLANNER_FORMAT,
        "planner_format_version": PLANNER_FORMAT_VERSION,
        "control_contract": control_contract_metadata(),
        "release_policy": release_policy_metadata(),
        "control_target_names": list(STOP_SIGN_CONTROL_TARGET_NAMES),
        "aux_target_names": list(DEFAULT_AUX_TARGET_NAMES),
        "telemetry_feature_names": ["current_speed"],
        "frame_window_size": 5,
        "image_size": {"width": 480, "height": 480},
        "image_offsets": [-20, -15, -10, -5, 0],
        "telemetry_offsets": [-20, -15, -10, -5, 0],
        "future_offsets": future_offsets,
        "telemetry_sample_interval_ms": 50,
        "control_horizon_dt_ms": list(derive_control_horizon_dt_ms(future_offsets, 50)),
        "target_transforms": target_transform_metadata(transforms),
        "model_state_dict": {},
    }


class StopSignControlContractTests(unittest.TestCase):
    def test_contract_is_future_speed_profile_plus_stop_intent(self) -> None:
        self.assertEqual(STOP_SIGN_CONTROL_TARGET_NAMES, (FUTURE_SPEED_MPS, STOP_INTENT))
        metadata = control_contract_metadata()
        self.assertEqual(metadata["name"], "stop_sign_motion_plan_v1")
        self.assertEqual(metadata["output_ranges"][FUTURE_SPEED_MPS], [0.0, MAX_DESIRED_SPEED_MPS])
        self.assertEqual(metadata["output_ranges"][STOP_INTENT], [0.0, 1.0])

    def test_horizon_timing_and_scripted_release_are_checkpoint_contracts(self) -> None:
        checkpoint = stop_sign_checkpoint_metadata()
        self.assertEqual(
            validate_checkpoint_control_contract(checkpoint),
            (100, 250, 500, 1000),
        )
        checkpoint["release_policy"] = {"name": "learned_release", "learned": True}
        with self.assertRaisesRegex(ValueError, "non-learned"):
            validate_checkpoint_control_contract(checkpoint)

    def test_model_outputs_bounded_temporal_motion_plan(self) -> None:
        model = StopSignTemporalPlanner(
            frame_count=3,
            telemetry_feature_dim=1,
            telemetry_hidden_dim=16,
            telemetry_sequence_length=3,
            horizon=2,
            control_dim=2,
            control_target_names=STOP_SIGN_CONTROL_TARGET_NAMES,
            aux_dim=len(DEFAULT_AUX_TARGET_NAMES),
            width_multiplier=0.25,
        )
        output = model(torch.zeros((1, 3, 3, 32, 32)), torch.zeros((1, 3, 1)))
        self.assertEqual(tuple(output["pred_motion_plan"].shape), (1, 2, 2))
        self.assertTrue(torch.equal(output["pred_motion_plan"], output["pred_controls"]))
        self.assertTrue(torch.all(output["pred_motion_plan"] >= 0.0))
        self.assertTrue(torch.all(output["pred_motion_plan"] <= 1.0))

    def test_stop_intent_loss_uses_logits(self) -> None:
        predictions = torch.tensor([[[0.5, 0.99]]])
        logits = torch.tensor([[[0.0, -2.0]]])
        targets = torch.tensor([[[0.5, 1.0]]])
        aux = torch.zeros((1, 1, len(DEFAULT_AUX_TARGET_NAMES)))
        losses = compute_planner_losses(
            predictions,
            targets,
            aux,
            aux,
            aux_loss_weight=0.3,
            horizon_loss_weights=(1.0,),
            control_target_names=STOP_SIGN_CONTROL_TARGET_NAMES,
            aux_target_names=DEFAULT_AUX_TARGET_NAMES,
            pred_control_logits=logits,
        )
        self.assertAlmostEqual(float(losses["stop_intent_loss"]), 2.126928, places=5)

    def test_dataset_targets_ignore_unmodeled_raw_actuation(self) -> None:
        targets = build_control_targets({
            "unusedRawActuation": -0.9,
            "expertDesiredSpeedMps": 4.25,
            "expertStopProbability": 0.75,
            "expertThrottle": 1.0,
            "expertBrake": 0.0,
        }, STOP_SIGN_CONTROL_TARGET_NAMES)
        self.assertEqual(targets.tolist(), [4.25, 0.75])

    def test_config_rejects_obsolete_actuator_outputs(self) -> None:
        for names in (
            ["desired_speed", "stop_probability"],
            ["throttle", "brake"],
        ):
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, "stop-sign"):
                parse_temporal_dataset_config({"dataset": {"control_target_names": names}})

    def test_checkpoint_and_predict_require_exact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "stop-sign.pt"
            torch.save(stop_sign_checkpoint_metadata(), checkpoint_path)
            checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
        self.assertEqual(checkpoint["control_contract"]["name"], CONTROL_CONTRACT_NAME)
        self.assertIsNone(require_predict_control_contract({"control_contract": CONTROL_CONTRACT_NAME}))
        with self.assertRaisesRegex(ValueError, CONTROL_CONTRACT_NAME):
            require_predict_control_contract({"control_contract": "direct_actuator_v0"})

    def test_activation_width_is_exact(self) -> None:
        raw = torch.tensor([[[-20.0, -20.0], [20.0, 20.0]]])
        activated = apply_stop_sign_control_activations(raw, STOP_SIGN_CONTROL_TARGET_NAMES)
        self.assertTrue(torch.all(activated >= 0.0))
        self.assertTrue(torch.all(activated <= 1.0))
        with self.assertRaisesRegex(ValueError, "width"):
            apply_stop_sign_control_activations(torch.zeros((1, 1, 3)), STOP_SIGN_CONTROL_TARGET_NAMES)

    def test_model_status_exposes_new_contract(self) -> None:
        status = ModelRuntime().status()
        self.assertEqual(status["control_contract"]["name"], CONTROL_CONTRACT_NAME)
        self.assertFalse(status["release_policy"]["learned"])


if __name__ == "__main__":
    unittest.main()
