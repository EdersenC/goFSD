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
    DEFAULT_HORIZON_LOSS_WEIGHTS,
    DEFAULT_IMAGE_OFFSETS,
    DEFAULT_TELEMETRY_FEATURE_NAMES,
    DEFAULT_TELEMETRY_OFFSETS,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    parse_temporal_dataset_config,
)
from dataset import FsdDataset, wrap_degrees_delta
from inference import load_checkpoint
from models.planner import DrivingCNN
from train import (
    DatasetConfig,
    LoaderConfig,
    ModelConfig,
    OutputConfig,
    TrainConfig,
    TrainingConfig,
    build_training_context,
    finalize_metrics,
    format_first_batch_debug,
    format_first_batch_predictions,
    initialize_metric_totals,
    update_metric_totals,
    compute_planner_losses,
)


def write_jpeg(path: Path, *, image_size: tuple[int, int]) -> None:
    width, height = image_size
    image = Image.new("RGB", (width, height), color=(120, 160, 200))
    image.save(path, format="JPEG")


def telemetry_point(
    *,
    speed: float,
    yaw: float,
    yaw_rate: float,
    steering: float,
    acceleration: float,
    brake_pressure_avg: float = 0.0,
) -> dict[str, float]:
    return {
        "control": {
            "Steering": steering,
            "acceleration": acceleration,
            "brakePressureAvg": brake_pressure_avg,
        },
        "aux": {
            "currentSpeed": speed,
            "yaw": yaw,
            "yawRate": yaw_rate,
        },
        "raw": {
            "time": speed * 10.0,
        },
    }


def create_temporal_trip_fixture(
    root: Path,
    *,
    run_name: str,
    trip_name: str,
    include_invalid_sample: bool = False,
    image_size: tuple[int, int] = (32, 32),
) -> None:
    trip_dir = root / "runs" / run_name / "inner-city-driving_default" / trip_name
    trip_dir.mkdir(parents=True, exist_ok=True)
    (trip_dir / "metadata.json").write_text(json.dumps({
        "runId": run_name,
        "sceneId": "inner-city-driving",
        "sceneVariant": "default",
        "tripIndex": int(trip_name.split("-")[-1]),
    }), encoding="utf-8")

    frames_dir = trip_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    frame_paths = [f"frames/{index:06d}.jpg" for index in range(len(DEFAULT_IMAGE_OFFSETS))]
    for frame_path in frame_paths:
        write_jpeg(trip_dir / frame_path, image_size=image_size)

    history = [
        telemetry_point(speed=5.0 + index, yaw=10.0 * (index + 1), yaw_rate=0.1 * index, steering=0.05 * index, acceleration=0.2 + (0.01 * index))
        for index in range(len(DEFAULT_TELEMETRY_OFFSETS) - 1)
    ]
    history.append(telemetry_point(speed=13.0, yaw=179.0, yaw_rate=0.8, steering=0.4, acceleration=0.6))

    future = [
        telemetry_point(speed=14.0 + index, yaw=(-179.0 if index == 0 else -170.0 + index), yaw_rate=1.0 + index, steering=0.5 + (0.05 * index), acceleration=0.7 + (0.1 * index))
        for index in range(len(DEFAULT_FUTURE_OFFSETS))
    ]

    samples = [{
        "frame_paths": frame_paths,
        "telemetry_history": history,
        "telemetry_future": future,
        "label": {
            "control": {},
            "aux": {},
        },
    }]
    if include_invalid_sample:
        samples.append({
            "frame_paths": frame_paths,
            "telemetry_history": history[:-1],
            "telemetry_future": future,
            "label": {
                "control": {},
                "aux": {},
            },
        })

    (trip_dir / "dataset.jsonl").write_text(
        "\n".join(json.dumps(sample) for sample in samples) + "\n",
        encoding="utf-8",
    )


class TemporalPlannerUpgradeTests(unittest.TestCase):
    def test_parse_temporal_dataset_config_reads_explicit_sequence_fields(self) -> None:
        raw = {
            "dataset": {
                "image_offsets": [-8, -6, -4, -2, 0],
                "telemetry_offsets": [-8, -7, -6, -5, -4, -3, -2, -1, 0],
                "future_offsets": [1, 2, 3, 4, 5, 6],
                "telemetry_feature_names": ["current_speed", "yaw_sin", "yaw_cos", "yaw_rate", "steering", "acceleration"],
        "control_target_names": ["steering", "acceleration", "brakePressureAvg"],
                "aux_target_names": ["future_speed", "future_yaw_delta", "future_yaw_rate"],
            }
        }

        (
            image_offsets,
            telemetry_offsets,
            future_offsets,
            telemetry_feature_names,
            control_target_names,
            aux_target_names,
        ) = parse_temporal_dataset_config(raw)

        self.assertEqual(image_offsets, DEFAULT_IMAGE_OFFSETS)
        self.assertEqual(telemetry_offsets, DEFAULT_TELEMETRY_OFFSETS)
        self.assertEqual(future_offsets, DEFAULT_FUTURE_OFFSETS)
        self.assertEqual(telemetry_feature_names, DEFAULT_TELEMETRY_FEATURE_NAMES)
        self.assertEqual(control_target_names, DEFAULT_CONTROL_TARGET_NAMES)
        self.assertEqual(aux_target_names, DEFAULT_AUX_TARGET_NAMES)

    def test_wrap_degrees_delta_handles_angle_wraparound(self) -> None:
        self.assertAlmostEqual(wrap_degrees_delta(179.0, -179.0), 2.0)
        self.assertAlmostEqual(wrap_degrees_delta(-179.0, 179.0), -2.0)

    def test_dataset_builds_temporal_tensors_and_filters_rows_missing_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_temporal_trip_fixture(root, run_name="run-a", trip_name="trip-000", include_invalid_sample=True)

            dataset = FsdDataset(
                run_paths=[root / "runs" / "run-a"],
                image_size=(32, 32),
                image_offsets=DEFAULT_IMAGE_OFFSETS,
                telemetry_offsets=DEFAULT_TELEMETRY_OFFSETS,
                future_offsets=DEFAULT_FUTURE_OFFSETS,
                telemetry_feature_names=DEFAULT_TELEMETRY_FEATURE_NAMES,
                control_target_names=DEFAULT_CONTROL_TARGET_NAMES,
                aux_target_names=DEFAULT_AUX_TARGET_NAMES,
            )

            self.assertEqual(dataset.trip_count, 1)
            self.assertEqual(len(dataset), 1)

            images, telemetry, target_controls, target_aux = dataset[0]
            self.assertEqual(tuple(images.shape), (5, 3, 32, 32))
            self.assertEqual(tuple(telemetry.shape), (9, 6))
            self.assertEqual(tuple(target_controls.shape), (6, 3))
            self.assertEqual(tuple(target_aux.shape), (6, 3))

            first_history = telemetry[0]
            self.assertAlmostEqual(float(first_history[0].item()), 5.0, places=6)
            self.assertAlmostEqual(float(first_history[1].item()), 0.173648, places=5)
            self.assertAlmostEqual(float(first_history[2].item()), 0.984807, places=5)
            self.assertAlmostEqual(float(target_controls[0, 0].item()), 0.5, places=6)
            self.assertAlmostEqual(float(target_controls[0, 1].item()), 0.7, places=6)
            self.assertAlmostEqual(float(target_controls[0, 2].item()), 0.0, places=6)
            self.assertAlmostEqual(float(target_aux[0, 0].item()), 14.0, places=6)
            self.assertAlmostEqual(float(target_aux[0, 1].item()), 2.0, places=6)
            self.assertAlmostEqual(float(target_aux[0, 2].item()), 1.0, places=6)

    def test_driving_cnn_returns_horizon_predictions_and_rejects_bad_shapes(self) -> None:
        model = DrivingCNN(
            frame_count=5,
            telemetry_feature_dim=6,
            telemetry_hidden_dim=32,
            telemetry_sequence_length=9,
            horizon=6,
        )
        images = torch.zeros((2, 5, 3, 32, 32), dtype=torch.float32)
        telemetry = torch.zeros((2, 9, 6), dtype=torch.float32)

        output = model(images, telemetry)

        self.assertEqual(tuple(output["pred_controls"].shape), (2, 6, 2))
        self.assertEqual(tuple(output["pred_aux"].shape), (2, 6, 3))

        with self.assertRaisesRegex(ValueError, "images expected T_img=5"):
            model(torch.zeros((2, 4, 3, 32, 32)), telemetry)
        with self.assertRaisesRegex(ValueError, "telemetry feature count mismatch"):
            model(images, torch.zeros((2, 9, 5)))

    def test_compute_planner_losses_applies_horizon_weights(self) -> None:
        pred_controls = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float32)
        target_controls = torch.tensor([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], dtype=torch.float32)
        pred_aux = torch.zeros((1, 2, 3), dtype=torch.float32)
        target_aux = torch.zeros((1, 2, 3), dtype=torch.float32)

        losses = compute_planner_losses(
            pred_controls,
            target_controls,
            pred_aux,
            target_aux,
            aux_loss_weight=0.3,
            horizon_loss_weights=(1.0, 0.5),
        )

        self.assertAlmostEqual(float(losses["control_loss"].item()), 5.0 / 6.0, places=6)
        self.assertAlmostEqual(float(losses["aux_loss"].item()), 0.0, places=6)
        self.assertAlmostEqual(float(losses["loss"].item()), 5.0 / 6.0, places=6)

    def test_finalize_metrics_reports_new_overall_and_per_horizon_maes(self) -> None:
        totals = initialize_metric_totals(
            2,
            control_target_names=DEFAULT_CONTROL_TARGET_NAMES,
            aux_target_names=DEFAULT_AUX_TARGET_NAMES,
        )
        pred_controls = torch.zeros((1, 2, 3), dtype=torch.float32)
        target_controls = torch.ones((1, 2, 3), dtype=torch.float32)
        pred_aux = torch.zeros((1, 2, 3), dtype=torch.float32)
        target_aux = torch.full((1, 2, 3), 2.0, dtype=torch.float32)
        losses = compute_planner_losses(
            pred_controls,
            target_controls,
            pred_aux,
            target_aux,
            aux_loss_weight=0.3,
            horizon_loss_weights=(1.0, 1.0),
        )

        update_metric_totals(
            totals,
            pred_controls=pred_controls,
            target_controls=target_controls,
            pred_aux=pred_aux,
            target_aux=target_aux,
            losses=losses,
            control_target_names=DEFAULT_CONTROL_TARGET_NAMES,
            aux_target_names=DEFAULT_AUX_TARGET_NAMES,
        )
        metrics = finalize_metrics(
            totals,
            (1, 2),
            control_target_names=DEFAULT_CONTROL_TARGET_NAMES,
            aux_target_names=DEFAULT_AUX_TARGET_NAMES,
        )

        self.assertEqual(metrics["control_mae_overall"], 1.0)
        self.assertEqual(metrics["aux_mae_overall"], 2.0)
        self.assertEqual(metrics["steering_mae"], 1.0)
        self.assertEqual(metrics["acceleration_mae"], 1.0)
        self.assertEqual(metrics["brakePressureAvg_mae"], 1.0)
        self.assertEqual(metrics["future_speed_mae"], 2.0)
        self.assertEqual(metrics["future_yaw_delta_mae"], 2.0)
        self.assertEqual(metrics["future_yaw_rate_mae"], 2.0)
        self.assertEqual(metrics["control_mae_t+1"], 1.0)
        self.assertEqual(metrics["control_mae_t+2"], 1.0)
        self.assertEqual(metrics["aux_mae_t+1"], 2.0)
        self.assertEqual(metrics["aux_mae_t+2"], 2.0)
        self.assertIn("val_loss", metrics)

    def test_format_first_batch_debug_prints_temporal_shapes(self) -> None:
        message = format_first_batch_debug(
            torch.zeros((4, 5, 3, 480, 480), dtype=torch.float32),
            torch.zeros((4, 9, 6), dtype=torch.float32),
            torch.zeros((4, 6, 3), dtype=torch.float32),
            torch.zeros((4, 6, 3), dtype=torch.float32),
        )

        self.assertIn("images_shape=(4, 5, 3, 480, 480)", message)
        self.assertIn("telemetry_shape=(4, 9, 6)", message)
        self.assertIn("target_controls_shape=(4, 6, 3)", message)
        self.assertIn("target_aux_shape=(4, 6, 3)", message)

    def test_format_first_batch_predictions_prints_raw_temporal_outputs(self) -> None:
        message = format_first_batch_predictions(
            torch.tensor([[[0.123456, -0.987654, 0.3333], [1.25, 2.5, 0.6666]]], dtype=torch.float32),
            torch.tensor([[[0.0, 0.5, 0.25], [1.0, 2.0, 0.75]]], dtype=torch.float32),
            torch.tensor([[[3.33333, -4.44444, 5.55555], [6.0, 7.0, 8.0]]], dtype=torch.float32),
            torch.tensor([[[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]]], dtype=torch.float32),
            control_target_names=("steering", "acceleration", "brakePressureAvg"),
            aux_target_names=("future_speed", "future_yaw_delta", "future_yaw_rate"),
            future_offsets=(1, 2),
            batch_index=30,
        )

        self.assertIn("train_batch_predictions sample=0 batch=30", message)
        self.assertIn("\npred_controls:\n", message)
        self.assertIn("  t+1: steering=0.1235 acceleration=-0.9877 brakePressureAvg=0.3333", message)
        self.assertIn("  t+2: steering=1.25 acceleration=2.5 brakePressureAvg=0.6666", message)
        self.assertIn("\npred_aux:\n", message)
        self.assertIn("  t+1: future_speed=3.3333 future_yaw_delta=-4.4444 future_yaw_rate=5.5555", message)
        self.assertIn("  t+2: future_speed=6.0 future_yaw_delta=7.0 future_yaw_rate=8.0", message)
        self.assertIn("\ntarget_aux:\n", message)
        self.assertIn("  t+2: future_speed=12.0 future_yaw_delta=13.0 future_yaw_rate=14.0", message)

    def test_build_training_context_rejects_empty_validation_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_temporal_trip_fixture(root, run_name="train-a", trip_name="trip-000")
            trip_dir = root / "runs" / "val-a" / "inner-city-driving_default" / "trip-000"
            trip_dir.mkdir(parents=True, exist_ok=True)
            (trip_dir / "metadata.json").write_text(json.dumps({
                "runId": "val-a",
                "sceneId": "inner-city-driving",
                "sceneVariant": "default",
                "tripIndex": 0,
            }), encoding="utf-8")
            (trip_dir / "dataset.jsonl").write_text("", encoding="utf-8")

            config = TrainConfig(
                dataset=DatasetConfig(
                    data_root=str(root),
                    train_run_ids=("train-a",),
                    val_run_ids=("val-a",),
                    train_run_paths=(str(root / "runs" / "train-a"),),
                    val_run_paths=(str(root / "runs" / "val-a"),),
                    image_width=32,
                    image_height=32,
                    image_offsets=DEFAULT_IMAGE_OFFSETS,
                    telemetry_offsets=DEFAULT_TELEMETRY_OFFSETS,
                    future_offsets=DEFAULT_FUTURE_OFFSETS,
                    telemetry_feature_names=DEFAULT_TELEMETRY_FEATURE_NAMES,
                    control_target_names=DEFAULT_CONTROL_TARGET_NAMES,
                    aux_target_names=DEFAULT_AUX_TARGET_NAMES,
                    window_size=len(DEFAULT_IMAGE_OFFSETS),
                    frame_stride=2,
                    sample_stride=10,
                ),
                output=OutputConfig(base_dir=str(root / "training_runs")),
                model=ModelConfig(width_multiplier=1.0, telemetry_hidden_dim=32),
                training=TrainingConfig(
                    device="cpu",
                    epochs=1,
                    learning_rate=1e-3,
                    early_stopping_metric="val_loss",
                    aux_loss_weight=0.3,
                    horizon_loss_weights=DEFAULT_HORIZON_LOSS_WEIGHTS,
                ),
                loader=LoaderConfig(
                    train_batch_size=1,
                    train_num_workers=0,
                    train_pin_memory=False,
                    train_prefetch_factor=1,
                    train_persistent_workers=False,
                    val_batch_size=1,
                    val_num_workers=0,
                    val_pin_memory=False,
                    val_prefetch_factor=1,
                    val_persistent_workers=False,
                    log_every_n_batches=1,
                    val_split=1.0,
                    cpu_batch_size=1,
                ),
            )

            with self.assertRaisesRegex(ValueError, "Validation dataset resolved to zero samples"):
                build_training_context(config, root / "train_config.toml")

    def test_inference_load_checkpoint_fails_fast_for_temporal_planner_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "epoch-001.pt"
            torch.save({
                "planner_format": "temporal_telemetry_gru_v1",
                "model_state_dict": {},
            }, checkpoint_path)

            with self.assertRaisesRegex(ValueError, "not supported by inference.py"):
                load_checkpoint(checkpoint_path, torch.device("cpu"))

    def test_temporal_defaults_keep_training_at_480_square(self) -> None:
        self.assertEqual(DEFAULT_IMAGE_WIDTH, 480)
        self.assertEqual(DEFAULT_IMAGE_HEIGHT, 480)


if __name__ == "__main__":
    unittest.main()
