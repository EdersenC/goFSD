from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from config import (
    DEFAULT_AUX_TARGET_NAMES,
    DEFAULT_CONTROL_TARGET_NAMES,
    DEFAULT_FUTURE_OFFSETS,
    DEFAULT_IMAGE_OFFSETS,
    DEFAULT_TELEMETRY_FEATURE_NAMES,
    DEFAULT_TELEMETRY_OFFSETS,
    parse_temporal_dataset_config,
)
from control_contract import STOP_SIGN_CONTROL_TARGET_NAMES
from dataset import FsdDataset
from models.planner import StopSignTemporalPlanner
from target_transforms import (
    build_target_transform_registry,
    denormalize_target_tensor,
    round_trip_transform_check,
)
from train import (
    EarlyStoppingState,
    EpochResult,
    check_early_stopping,
    compute_planner_losses,
    compute_stop_sign_score,
    finalize_metrics,
    format_first_batch_predictions,
    initialize_metric_totals,
    update_metric_totals,
)


def stop_sign_goal() -> dict[str, object]:
    return {
        "task": "stop-sign",
        "contract": "stop-sign-goal.v1",
        "signPose": {"x": 10.0, "y": 20.0, "z": 30.0, "heading": 90.0},
        "stopLinePose": {"x": 8.0, "y": 20.0, "z": 30.0, "heading": 90.0},
        "egoStopPose": {"x": 5.5, "y": 20.0, "z": 30.0, "heading": 90.0},
        "startPose": {"x": -10.0, "y": 20.0, "z": 30.0, "heading": 90.0},
    }


def telemetry_point(
    speed: float,
    *,
    desired_speed: float,
    stop_intent: float,
    phase: str,
) -> dict[str, object]:
    return {
        "control": {},
        "aux": {"currentSpeed": speed},
        "raw": {
            "expertDesiredSpeedMps": desired_speed,
            "expertStopProbability": stop_intent,
            "expertThrottle": 0.4 if desired_speed > 0 else 0.0,
            "expertBrake": 0.5 if stop_intent > 0 else 0.0,
            "brakePressureAvg": 0.35 if stop_intent > 0 else 0.0,
            "stopSignPhase": phase,
        },
    }


def create_trip(root: Path) -> Path:
    trip_dir = root / "runs" / "run-a" / "stop-sign_temporal-v1" / "trip-000"
    trip_dir.mkdir(parents=True)
    metadata = {
        "runId": "run-a",
        "sceneId": "stop-sign",
        "sceneVariant": "temporal-v1",
        "tripIndex": 0,
        "stopSignGoal": stop_sign_goal(),
        "stopSignOutcome": {"success": True, "status": "succeeded"},
    }
    (trip_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    frames_dir = trip_dir / "frames"
    frames_dir.mkdir()
    frame_paths: list[str] = []
    for index in range(1, len(DEFAULT_IMAGE_OFFSETS) + 1):
        relative = f"frames/{index:06d}.jpg"
        Image.new("RGB", (32, 32), color=(120, 160, 200)).save(
            trip_dir / relative,
            format="JPEG",
        )
        frame_paths.append(relative)

    history = [
        telemetry_point(5.0, desired_speed=5.0, stop_intent=0.0, phase="cruise_approach")
        for _ in DEFAULT_TELEMETRY_OFFSETS
    ]
    future = [
        telemetry_point(
            max(0.0, 6.0 - index),
            desired_speed=max(0.0, 6.0 - index),
            stop_intent=1.0 if index >= 4 else 0.0,
            phase="stop_hold" if index >= 4 else "decelerate",
        )
        for index in range(len(DEFAULT_FUTURE_OFFSETS))
    ]
    row = {
        "frame_paths": frame_paths,
        "telemetry_history": history,
        "telemetry_future": future,
        "label": {"control": {}, "aux": {}},
    }
    (trip_dir / "dataset.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    fingerprint = "sha256:" + ("a" * 64)
    processing = {
        "state": "completed",
        "configFingerprint": fingerprint,
        "imageWidth": 32,
        "imageHeight": 32,
        "imageOffsets": list(DEFAULT_IMAGE_OFFSETS),
        "telemetryOffsets": list(DEFAULT_TELEMETRY_OFFSETS),
        "futureOffsets": list(DEFAULT_FUTURE_OFFSETS),
        "telemetrySampleIntervalMs": 50,
        "frameCount": len(DEFAULT_IMAGE_OFFSETS),
        "sampleCount": 1,
        "datasetRowCount": 1,
        "referencedFrameCount": len(DEFAULT_IMAGE_OFFSETS),
    }
    (trip_dir / "processing.json").write_text(json.dumps(processing), encoding="utf-8")
    return trip_dir


class StopSignTemporalPlannerTests(unittest.TestCase):
    def test_config_accepts_only_current_speed_motion_plan_and_diagnostics(self) -> None:
        raw = {
            "dataset": {
                "image_offsets": list(DEFAULT_IMAGE_OFFSETS),
                "telemetry_offsets": list(DEFAULT_TELEMETRY_OFFSETS),
                "future_offsets": list(DEFAULT_FUTURE_OFFSETS),
                "telemetry_feature_names": list(DEFAULT_TELEMETRY_FEATURE_NAMES),
                "control_target_names": list(DEFAULT_CONTROL_TARGET_NAMES),
                "aux_target_names": list(DEFAULT_AUX_TARGET_NAMES),
            }
        }
        parsed = parse_temporal_dataset_config(raw)
        self.assertEqual(parsed[3], ("current_speed",))
        self.assertEqual(parsed[4], ("future_speed_mps", "stop_intent"))
        self.assertEqual(
            parsed[5],
            ("expert_throttle", "expert_brake", "actual_brake_pressure"),
        )

    def test_config_rejects_oracle_or_generic_driving_inputs(self) -> None:
        raw = {
            "dataset": {
                "image_offsets": list(DEFAULT_IMAGE_OFFSETS),
                "telemetry_offsets": list(DEFAULT_TELEMETRY_OFFSETS),
                "future_offsets": list(DEFAULT_FUTURE_OFFSETS),
                "telemetry_feature_names": ["current_speed", "stop_line_distance"],
                "control_target_names": list(DEFAULT_CONTROL_TARGET_NAMES),
                "aux_target_names": list(DEFAULT_AUX_TARGET_NAMES),
            }
        }
        with self.assertRaisesRegex(ValueError, "RGB-first stop-sign input contract"):
            parse_temporal_dataset_config(raw)

    def test_dataset_builds_current_contract_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_trip(root)
            dataset = FsdDataset(
                run_paths=[root / "runs" / "run-a"],
                image_size=(32, 32),
            )
            images, telemetry, state_inputs, controls, diagnostics = dataset[0]
        self.assertEqual(tuple(images.shape), (5, 3, 32, 32))
        self.assertEqual(tuple(telemetry.shape), (9, 1))
        self.assertEqual(tuple(state_inputs.shape), (0,))
        self.assertEqual(tuple(controls.shape), (6, 2))
        self.assertEqual(tuple(diagnostics.shape), (6, 3))
        denormalized = denormalize_target_tensor(
            controls,
            dataset.control_target_names,
            dataset.target_transforms,
        )
        self.assertAlmostEqual(float(denormalized[0, 0]), 6.0)
        self.assertEqual(float(denormalized[-1, 1]), 1.0)

    def test_dataset_excludes_failed_attempt_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trip_dir = create_trip(root)
            metadata = json.loads((trip_dir / "metadata.json").read_text(encoding="utf-8"))
            metadata["stopSignOutcome"] = {"success": False, "status": "failed"}
            (trip_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            dataset = FsdDataset(run_paths=[root / "runs" / "run-a"], image_size=(32, 32))
        self.assertEqual(len(dataset), 0)
        self.assertEqual(dataset.excluded_failed_or_non_stop_sign_trip_count, 1)

    def test_model_outputs_bounded_stop_sign_motion_plan(self) -> None:
        model = StopSignTemporalPlanner(
            frame_count=5,
            telemetry_feature_dim=1,
            telemetry_sequence_length=9,
            horizon=6,
            control_target_names=STOP_SIGN_CONTROL_TARGET_NAMES,
        )
        output = model(
            torch.zeros((2, 5, 3, 32, 32)),
            torch.zeros((2, 9, 1)),
        )
        self.assertEqual(tuple(output["pred_controls"].shape), (2, 6, 2))
        self.assertEqual(tuple(output["pred_aux"].shape), (2, 6, 3))
        self.assertTrue(torch.all(output["pred_controls"] >= 0.0))
        self.assertTrue(torch.all(output["pred_controls"] <= 1.0))

    def test_loss_weights_apply_to_named_motion_plan_targets(self) -> None:
        losses = compute_planner_losses(
            torch.zeros((1, 1, 2)),
            torch.tensor([[[1.0, 1.0]]]),
            torch.zeros((1, 1, 3)),
            torch.zeros((1, 1, 3)),
            aux_loss_weight=0.4,
            horizon_loss_weights=(1.0,),
            target_loss_weights={"future_speed_mps": 2.0, "stop_intent": 0.0},
            control_target_names=STOP_SIGN_CONTROL_TARGET_NAMES,
            smooth_l1_beta=1.0,
        )
        self.assertAlmostEqual(float(losses["control_loss"]), 0.5)
        self.assertAlmostEqual(float(losses["future_speed_mps_loss"]), 0.5)
        self.assertGreater(float(losses["stop_intent_loss"]), 0.0)

    def test_metrics_are_named_for_motion_plan_and_diagnostics(self) -> None:
        totals = initialize_metric_totals(
            2,
            control_target_names=STOP_SIGN_CONTROL_TARGET_NAMES,
            aux_target_names=DEFAULT_AUX_TARGET_NAMES,
        )
        controls = torch.zeros((1, 2, 2))
        target_controls = torch.ones((1, 2, 2))
        diagnostics = torch.zeros((1, 2, 3))
        target_diagnostics = torch.ones((1, 2, 3))
        losses = compute_planner_losses(
            controls,
            target_controls,
            diagnostics,
            target_diagnostics,
            aux_loss_weight=0.4,
            horizon_loss_weights=(1.0, 1.0),
            control_target_names=STOP_SIGN_CONTROL_TARGET_NAMES,
        )
        update_metric_totals(
            totals,
            pred_controls=controls,
            target_controls=target_controls,
            pred_aux=diagnostics,
            target_aux=target_diagnostics,
            losses=losses,
            control_target_names=STOP_SIGN_CONTROL_TARGET_NAMES,
            aux_target_names=DEFAULT_AUX_TARGET_NAMES,
        )
        metrics = finalize_metrics(
            totals,
            (1, 2),
            control_target_names=STOP_SIGN_CONTROL_TARGET_NAMES,
            aux_target_names=DEFAULT_AUX_TARGET_NAMES,
        )
        self.assertIn("future_speed_mps_mae", metrics)
        self.assertIn("stop_intent_loss", metrics)
        self.assertIn("actual_brake_pressure_mae", metrics)

    def test_stop_sign_score_penalizes_validation_gap(self) -> None:
        score = compute_stop_sign_score(
            {"control_mae_overall": 0.1},
            {"control_loss": 0.2, "control_mae_overall": 0.3},
        )
        self.assertAlmostEqual(score, 0.295)

    def test_early_stopping_uses_stop_sign_score(self) -> None:
        state = EarlyStoppingState(
            metric_name="stop_sign_score",
            patience=2,
            min_delta=0.0,
        )
        result = EpochResult(
            epoch_index=1,
            train_metrics={},
            val_metrics={"stop_sign_score": 0.2},
            train_epoch_time=0.0,
            val_epoch_time=0.0,
            avg_batch_time=0.0,
            avg_loader_wait_time=0.0,
            avg_h2d_time=0.0,
            avg_forward_backward_time=0.0,
            avg_optimizer_time=0.0,
            avg_iteration_time=0.0,
            avg_grad_norm=None,
            avg_learning_rate=0.0,
            memory_snapshots={},
        )
        improved, stopped, value = check_early_stopping(result, state)
        self.assertTrue(improved)
        self.assertFalse(stopped)
        self.assertEqual(value, 0.2)

    def test_prediction_debug_uses_current_contract_names(self) -> None:
        message = format_first_batch_predictions(
            torch.zeros((1, 0)),
            torch.tensor([[[0.5, 0.75]]]),
            torch.tensor([[[0.25, 1.0]]]),
            torch.tensor([[[0.4, 0.2, 0.1]]]),
            torch.tensor([[[0.3, 0.5, 0.4]]]),
            state_input_names=(),
            control_target_names=STOP_SIGN_CONTROL_TARGET_NAMES,
            aux_target_names=DEFAULT_AUX_TARGET_NAMES,
            future_offsets=(5,),
        )
        self.assertIn("future_speed_mps=0.5 stop_intent=0.75", message)
        self.assertIn("expert_throttle=0.4", message)
        self.assertNotIn("steering", message)

    def test_target_transform_round_trip(self) -> None:
        transforms = build_target_transform_registry(
            STOP_SIGN_CONTROL_TARGET_NAMES + DEFAULT_AUX_TARGET_NAMES
        )
        self.assertEqual(set(transforms), set(STOP_SIGN_CONTROL_TARGET_NAMES + DEFAULT_AUX_TARGET_NAMES))
        self.assertLess(round_trip_transform_check()["max_round_trip_error"], 1e-6)


if __name__ == "__main__":
    unittest.main()
