from __future__ import annotations

import argparse
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
from control_contract import (
    PARKING_CONTROL_TARGET_NAMES,
    PLANNER_FORMAT,
    PLANNER_FORMAT_VERSION,
    control_contract_metadata,
    derive_control_horizon_dt_ms,
)
from dataset import FsdDataset, wrap_degrees_delta
from inference import (
    load_checkpoint,
    load_config as load_inference_config,
    resolve_config as resolve_inference_config,
)
from models.planner import DrivingCNN
from train import (
    DatasetConfig,
    EarlyStoppingState,
    EpochResult,
    LoaderConfig,
    ModelConfig,
    OutputConfig,
    TrainConfig,
    TrainingConfig,
    build_training_context,
    check_early_stopping,
    compute_drive_score,
    finalize_metrics,
    format_first_batch_debug,
    format_first_batch_predictions,
    initialize_metric_totals,
    update_metric_totals,
    compute_planner_losses,
)
from target_transforms import (
    build_target_transform_registry,
    round_trip_transform_check,
    target_transform_metadata,
)


GENERIC_CONTROL_TARGET_NAMES = ("steering", "acceleration", "brakePressureAvg")


def stop_sign_goal(*, x: float = 10.0) -> dict[str, object]:
    return {
        "task": "stop-sign",
        "contract": "stop-sign-goal.v1",
        "signPose": {"x": x, "y": 20.0, "z": 30.0, "heading": 90.0},
        "stopLinePose": {"x": x - 2.0, "y": 20.0, "z": 30.0, "heading": 90.0},
        "egoStopPose": {"x": x - 4.5, "y": 20.0, "z": 30.0, "heading": 90.0},
        "startPose": {"x": x - 20.0, "y": 20.0, "z": 30.0, "heading": 90.0},
    }


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
    parking_phase: str = "accelerate",
    parking_parked: bool = False,
) -> dict[str, object]:
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
            "expertDesiredWheelSteerNormalized": steering,
            "expertDesiredSpeedMps": 0.0 if parking_parked else 2.22,
            "expertStopProbability": 1.0 if parking_parked else 0.0,
            "stopSignPhase": "stop_hold" if parking_parked else parking_phase,
        },
    }


def create_temporal_trip_fixture(
    root: Path,
    *,
    run_name: str,
    trip_name: str,
    include_invalid_sample: bool = False,
    image_size: tuple[int, int] = (32, 32),
    metadata_overrides: dict[str, object] | None = None,
    frame_numbers: tuple[int, ...] | None = None,
) -> None:
    trip_dir = root / "runs" / run_name / "stop-sign_temporal-v1" / trip_name
    trip_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "runId": run_name,
        "sceneId": "stop-sign",
        "sceneVariant": "temporal-v1",
        "tripIndex": int(trip_name.split("-")[-1]),
        "stopSignGoal": stop_sign_goal(x=10.0 + float(sum(map(ord, run_name)) % 100)),
        "stopSignOutcome": {"success": True, "status": "succeeded"},
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)
    (trip_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    frames_dir = trip_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    resolved_frame_numbers = frame_numbers or tuple(range(1, len(DEFAULT_IMAGE_OFFSETS) + 1))
    if len(resolved_frame_numbers) != len(DEFAULT_IMAGE_OFFSETS):
        raise ValueError("frame_numbers must match the configured image offset count")
    frame_paths = [f"frames/{index:06d}.jpg" for index in resolved_frame_numbers]
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
    (trip_dir / "processing.json").write_text(
        json.dumps({
            "state": "completed",
            "configFingerprint": "sha256:" + ("0" * 64),
            "imageWidth": image_size[0],
            "imageHeight": image_size[1],
            "imageOffsets": list(DEFAULT_IMAGE_OFFSETS),
            "telemetryOffsets": list(DEFAULT_TELEMETRY_OFFSETS),
            "futureOffsets": list(DEFAULT_FUTURE_OFFSETS),
            "telemetrySampleIntervalMs": 50,
            "frameCount": len(frame_paths),
            "sampleCount": len(samples),
        }),
        encoding="utf-8",
    )


class TemporalPlannerUpgradeTests(unittest.TestCase):
    def test_model_server_can_read_clean_config_without_a_default_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "train_config.toml"
            config_path.write_text(
                """
[dataset]
window_size = 5
frame_stride = 2
sample_stride = 4

[inference]
checkpoint = ''
""".strip()
                + "\n",
                encoding="utf-8",
            )

            config = load_inference_config(config_path, require_checkpoint=False)

            self.assertEqual(config.checkpoint, "")
            self.assertIsNone(config.run_id)
            with self.assertRaisesRegex(ValueError, "Missing inference.checkpoint"):
                load_inference_config(config_path)

            resolved = resolve_inference_config(argparse.Namespace(
                config=config_path,
                checkpoint=Path("fresh-parking-model.pt"),
                device=None,
                data_root=None,
                run_id=None,
                sample_index=None,
                output_json=False,
                metadata_only=False,
            ))
            self.assertEqual(resolved.checkpoint, "fresh-parking-model.pt")
            self.assertIsNone(resolved.run_id)

    def test_dataset_requires_successful_complete_stop_sign_attempts_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            goal = stop_sign_goal()
            create_temporal_trip_fixture(
                root,
                run_name="run-a",
                trip_name="trip-000",
                metadata_overrides={
                    "stopSignGoal": goal,
                    "stopSignOutcome": {"success": True, "status": "succeeded"},
                },
            )
            create_temporal_trip_fixture(
                root,
                run_name="run-a",
                trip_name="trip-001",
                metadata_overrides={
                    "stopSignGoal": goal,
                    "stopSignOutcome": {"success": False, "status": "collision"},
                },
            )
            create_temporal_trip_fixture(
                root,
                run_name="run-a",
                trip_name="trip-002",
                metadata_overrides={
                    "stopSignGoal": None,
                    "stopSignOutcome": {"success": True, "status": "incomplete-metadata"},
                },
            )
            create_temporal_trip_fixture(
                root,
                run_name="run-a",
                trip_name="trip-003",
                metadata_overrides={"stopSignGoal": None, "stopSignOutcome": None},
            )
            create_temporal_trip_fixture(
                root,
                run_name="run-a",
                trip_name="trip-004",
                metadata_overrides={
                    "stopSignGoal": {"task": "parking"},
                    "stopSignOutcome": {"success": True, "status": "succeeded"},
                },
            )
            create_temporal_trip_fixture(
                root,
                run_name="run-a",
                trip_name="trip-005",
                metadata_overrides={
                    "sceneVariant": "temporal-v0",
                    "stopSignGoal": goal,
                    "stopSignOutcome": {"success": True, "status": "succeeded"},
                },
            )
            incomplete_goal = stop_sign_goal()
            incomplete_goal.pop("egoStopPose")
            create_temporal_trip_fixture(
                root,
                run_name="run-a",
                trip_name="trip-006",
                metadata_overrides={
                    "stopSignGoal": incomplete_goal,
                    "stopSignOutcome": {"success": True, "status": "succeeded"},
                },
            )

            successful_only = FsdDataset(
                run_paths=[root / "runs" / "run-a"],
                image_size=(32, 32),
            )
            with_failures = FsdDataset(
                run_paths=[root / "runs" / "run-a"],
                image_size=(32, 32),
                include_failed_or_non_stop_sign_trips=True,
            )

            self.assertEqual(successful_only.trip_count, 1)
            self.assertEqual(successful_only.excluded_failed_or_non_stop_sign_trip_count, 6)
            self.assertEqual(with_failures.trip_count, 7)
            self.assertEqual(with_failures.excluded_failed_or_non_stop_sign_trip_count, 0)

    def test_dataset_rejects_stale_processed_timeline_contract(self) -> None:
        cases = (
            ("imageOffsets", [-10, -6, -4, -2, 0], "imageOffsets mismatch"),
            ("telemetrySampleIntervalMs", 40, "telemetrySampleIntervalMs mismatch"),
            ("imageWidth", 64, "imageWidth mismatch"),
        )
        for key, value, message in cases:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                create_temporal_trip_fixture(root, run_name="run-a", trip_name="trip-000")
                processing_path = (
                    root
                    / "runs"
                    / "run-a"
                    / "stop-sign_temporal-v1"
                    / "trip-000"
                    / "processing.json"
                )
                status = json.loads(processing_path.read_text(encoding="utf-8"))
                status[key] = value
                processing_path.write_text(json.dumps(status), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, message):
                    FsdDataset(
                        run_paths=[root / "runs" / "run-a"],
                        image_size=(32, 32),
                    )

    def test_dataset_rejects_incomplete_published_outputs(self) -> None:
        cases = (
            (
                "truncated valid dataset",
                lambda trip_dir, status: (trip_dir / "dataset.jsonl").write_text("", encoding="utf-8"),
                "sampleCount mismatch",
            ),
            (
                "invalid dataset row",
                lambda trip_dir, status: (trip_dir / "dataset.jsonl").write_text("not-json\n", encoding="utf-8"),
                "invalid JSON",
            ),
            (
                "gapped frame set",
                lambda trip_dir, status: (trip_dir / "frames" / "000001.jpg").rename(
                    trip_dir / "frames" / "000006.jpg"
                ),
                "exactly the dataset-referenced JPEG files",
            ),
            (
                "zero byte frame",
                lambda trip_dir, status: (trip_dir / "frames" / "000001.jpg").write_bytes(b""),
                "only non-empty regular dataset-referenced JPEG files",
            ),
            (
                "missing sample count",
                lambda trip_dir, status: status.pop("sampleCount"),
                "sampleCount must be an integer",
            ),
        )
        for name, mutate, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                create_temporal_trip_fixture(root, run_name="run-a", trip_name="trip-000")
                trip_dir = root / "runs" / "run-a" / "stop-sign_temporal-v1" / "trip-000"
                processing_path = trip_dir / "processing.json"
                status = json.loads(processing_path.read_text(encoding="utf-8"))
                mutate(trip_dir, status)
                processing_path.write_text(json.dumps(status), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, message):
                    FsdDataset(run_paths=[root / "runs" / "run-a"], image_size=(32, 32))

    def test_dataset_accepts_sparse_processor_referenced_frame_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame_numbers = (77, 79, 81, 83, 85)
            create_temporal_trip_fixture(
                root,
                run_name="run-a",
                trip_name="trip-000",
                frame_numbers=frame_numbers,
            )

            dataset = FsdDataset(
                run_paths=[root / "runs" / "run-a"],
                image_size=(32, 32),
            )

            self.assertEqual(len(dataset), 1)
            sample = dataset._load_sample(0)
            self.assertEqual(
                [path.name for path in sample["frame_paths"]],
                [f"{frame_number:06d}.jpg" for frame_number in frame_numbers],
            )

    def test_dataset_rejects_missing_and_extra_sparse_referenced_frames(self) -> None:
        cases = (
            (
                "missing",
                lambda trip_dir: (trip_dir / "frames" / "000081.jpg").unlink(),
                "missing=\\['000081.jpg'\\]",
            ),
            (
                "extra",
                lambda trip_dir: write_jpeg(
                    trip_dir / "frames" / "000999.jpg",
                    image_size=(32, 32),
                ),
                "extra=\\['000999.jpg'\\]",
            ),
        )
        for name, mutate, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                create_temporal_trip_fixture(
                    root,
                    run_name="run-a",
                    trip_name="trip-000",
                    frame_numbers=(77, 79, 81, 83, 85),
                )
                trip_dir = root / "runs" / "run-a" / "stop-sign_temporal-v1" / "trip-000"
                mutate(trip_dir)

                with self.assertRaisesRegex(ValueError, message):
                    FsdDataset(run_paths=[root / "runs" / "run-a"], image_size=(32, 32))

    def test_dataset_rejects_missing_or_malformed_frame_references(self) -> None:
        cases = (
            ("missing", None, "missing non-empty frame_paths"),
            ("not a list", "frames/000001.jpg", "missing non-empty frame_paths"),
            ("parent traversal", ["frames/../000001.jpg"], "invalid frame path"),
            ("windows separator", [r"frames\\000001.jpg"], "invalid frame path"),
            ("zero index", ["frames/000000.jpg"], "invalid frame path"),
            ("non-string", [1], "must contain strings"),
        )
        for name, frame_paths, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                create_temporal_trip_fixture(root, run_name="run-a", trip_name="trip-000")
                trip_dir = root / "runs" / "run-a" / "stop-sign_temporal-v1" / "trip-000"
                dataset_path = trip_dir / "dataset.jsonl"
                row = json.loads(dataset_path.read_text(encoding="utf-8"))
                if frame_paths is None:
                    row.pop("frame_paths")
                else:
                    row["frame_paths"] = frame_paths
                dataset_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

                with self.assertRaisesRegex(ValueError, message):
                    FsdDataset(run_paths=[root / "runs" / "run-a"], image_size=(32, 32))

    def test_zero_sample_dataset_keeps_contiguous_or_empty_frame_rule(self) -> None:
        cases = (
            ("empty", 0, (), False),
            ("contiguous", len(DEFAULT_IMAGE_OFFSETS), tuple(range(1, len(DEFAULT_IMAGE_OFFSETS) + 1)), False),
            ("sparse rejected", len(DEFAULT_IMAGE_OFFSETS), (77, 79, 81, 83, 85), True),
        )
        for name, frame_count, frame_numbers, should_fail in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                create_temporal_trip_fixture(root, run_name="run-a", trip_name="trip-000")
                trip_dir = root / "runs" / "run-a" / "stop-sign_temporal-v1" / "trip-000"
                frames_dir = trip_dir / "frames"
                for frame_path in frames_dir.iterdir():
                    frame_path.unlink()
                for frame_number in frame_numbers:
                    write_jpeg(frames_dir / f"{frame_number:06d}.jpg", image_size=(32, 32))
                (trip_dir / "dataset.jsonl").write_text("", encoding="utf-8")
                processing_path = trip_dir / "processing.json"
                status = json.loads(processing_path.read_text(encoding="utf-8"))
                status["sampleCount"] = 0
                status["frameCount"] = frame_count
                processing_path.write_text(json.dumps(status), encoding="utf-8")

                if should_fail:
                    with self.assertRaisesRegex(ValueError, "exactly the contiguous numbered files"):
                        FsdDataset(run_paths=[root / "runs" / "run-a"], image_size=(32, 32))
                    continue
                dataset = FsdDataset(run_paths=[root / "runs" / "run-a"], image_size=(32, 32))
                self.assertEqual(len(dataset), 0)
                self.assertEqual(dataset.trip_count, 1)

    def test_dataset_rejects_missing_processed_outputs_in_selected_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_temporal_trip_fixture(root, run_name="run-a", trip_name="trip-000")
            trip_dir = root / "runs" / "run-a" / "stop-sign_temporal-v1" / "trip-000"
            (trip_dir / "dataset.jsonl").unlink()

            with self.assertRaisesRegex(ValueError, "processed dataset is missing"):
                FsdDataset(run_paths=[root / "runs" / "run-a"], image_size=(32, 32))

    def test_dataset_filters_samples_without_dense_sparse_offset_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_temporal_trip_fixture(root, run_name="run-a", trip_name="trip-000")
            trip_dir = root / "runs" / "run-a" / "stop-sign_temporal-v1" / "trip-000"
            dataset_path = trip_dir / "dataset.jsonl"
            sample = json.loads(dataset_path.read_text(encoding="utf-8"))
            sample["telemetry_history"] = sample["telemetry_history"][:2]
            sample["telemetry_future"] = sample["telemetry_future"][:2]
            dataset_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            processing_path = trip_dir / "processing.json"
            status = json.loads(processing_path.read_text(encoding="utf-8"))
            status["telemetryOffsets"] = [-8, 0]
            status["futureOffsets"] = [1, 6]
            processing_path.write_text(json.dumps(status), encoding="utf-8")

            dataset = FsdDataset(
                run_paths=[root / "runs" / "run-a"],
                image_size=(32, 32),
                telemetry_offsets=(-8, 0),
                future_offsets=(1, 6),
            )

            self.assertEqual(len(dataset), 0)
            self.assertEqual(dataset.rejected_sample_summary["bad_telemetry_history"], 1)
            self.assertEqual(dataset.rejected_sample_summary["bad_telemetry_future"], 1)

    def test_parse_temporal_dataset_config_reads_explicit_sequence_fields(self) -> None:
        raw = {
            "dataset": {
                "image_offsets": [-8, -6, -4, -2, 0],
                "telemetry_offsets": [-8, -7, -6, -5, -4, -3, -2, -1, 0],
                "future_offsets": [1, 2, 3, 4, 5, 6],
                "telemetry_feature_names": ["current_speed", "yaw_sin", "yaw_cos", "yaw_rate", "steering", "acceleration"],
                "control_target_names": list(PARKING_CONTROL_TARGET_NAMES),
                "aux_target_names": ["future_speed", "future_speed_delta", "future_yaw_delta", "future_yaw_rate"],
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

    def test_build_target_transform_registry_rejects_unknown_transform_without_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown target transform names"):
            build_target_transform_registry(
                ("steering", "acceleration"),
                {"future_speed_delta": {"type": "signed_cap", "clip": 1.5}},
            )

    def test_build_target_transform_registry_configures_future_speed_delta(self) -> None:
        registry = build_target_transform_registry(
            ("future_speed_delta", "steering"),
            {"future_speed_delta": {"type": "signed_cap", "clip": 1.5}},
        )
        self.assertIn("future_speed_delta", registry)
        self.assertAlmostEqual(float(registry["future_speed_delta"].range_max), 1.5)

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
            self.assertFalse(isinstance(dataset.samples[0], dict))

            images, telemetry, state_inputs, target_controls, target_aux = dataset[0]
            self.assertEqual(tuple(images.shape), (5, 3, 32, 32))
            self.assertEqual(images.dtype, torch.uint8)
            self.assertEqual(tuple(telemetry.shape), (9, 6))
            self.assertEqual(tuple(state_inputs.shape), (0,))
            self.assertEqual(tuple(target_controls.shape), (6, 2))
            self.assertEqual(tuple(target_aux.shape), (6, 4))

            first_history = telemetry[0]
            self.assertAlmostEqual(float(first_history[0].item()), 5.0, places=6)
            self.assertAlmostEqual(float(first_history[1].item()), 0.173648, places=5)
            self.assertAlmostEqual(float(first_history[2].item()), 0.984807, places=5)
            self.assertAlmostEqual(float(target_controls[0, 0].item()), 2.22 / 8.0, places=6)
            self.assertAlmostEqual(float(target_controls[0, 1].item()), 0.0, places=6)

            aux_transform_names = dataset.aux_target_names
            expected_aux = torch.tensor([14.0, 1.0, 2.0, 1.0], dtype=torch.float32)
            transform_targets = dataset.target_transforms
            normalized_expected = torch.tensor([
                transform_targets[name].normalize_value(float(expected_aux[index]))
                for index, name in enumerate(aux_transform_names)
            ], dtype=torch.float32)
            self.assertTrue(torch.allclose(target_aux[0], normalized_expected, atol=1e-6))

            from target_transforms import denormalize_target_tensor

            denorm_aux = denormalize_target_tensor(
                target_aux,
                aux_transform_names,
                transform_targets,
            )[0]
            self.assertAlmostEqual(float(denorm_aux[0].item()), 14.0, places=6)
            self.assertAlmostEqual(float(denorm_aux[1].item()), 1.0, places=6)
            self.assertAlmostEqual(float(denorm_aux[2].item()), 2.0, places=6)
            self.assertAlmostEqual(float(denorm_aux[3].item()), 1.0, places=6)

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
        uint8_output = model(torch.zeros((2, 5, 3, 32, 32), dtype=torch.uint8), telemetry)

        self.assertEqual(tuple(output["pred_controls"].shape), (2, 6, 2))
        self.assertEqual(tuple(output["pred_aux"].shape), (2, 6, 4))
        self.assertEqual(tuple(uint8_output["pred_controls"].shape), (2, 6, 2))

        with self.assertRaisesRegex(ValueError, "images expected T_img=5"):
            model(torch.zeros((2, 4, 3, 32, 32)), telemetry)
        with self.assertRaisesRegex(ValueError, "telemetry feature count mismatch"):
            model(images, torch.zeros((2, 9, 5)))

    def test_compute_planner_losses_applies_horizon_weights(self) -> None:
        pred_controls = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float32)
        target_controls = torch.tensor([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], dtype=torch.float32)
        pred_aux = torch.zeros((1, 2, 4), dtype=torch.float32)
        target_aux = torch.zeros((1, 2, 4), dtype=torch.float32)

        losses = compute_planner_losses(
            pred_controls,
            target_controls,
            pred_aux,
            target_aux,
            aux_loss_weight=0.3,
            horizon_loss_weights=(1.0, 0.5),
            control_target_names=GENERIC_CONTROL_TARGET_NAMES,
            smooth_l1_beta=1.0,
        )

        self.assertAlmostEqual(float(losses["control_loss"].item()), 5.0 / 6.0, places=6)
        self.assertAlmostEqual(float(losses["aux_loss"].item()), 0.0, places=6)
        self.assertAlmostEqual(float(losses["loss"].item()), 5.0 / 6.0, places=6)
        self.assertAlmostEqual(float(losses["steering_loss"].item()), 5.0 / 6.0, places=6)

    def test_compute_planner_losses_applies_target_weights_to_matching_targets(self) -> None:
        pred_controls = torch.zeros((1, 1, 3), dtype=torch.float32)
        target_controls = torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float32)
        pred_aux = torch.zeros((1, 1, 4), dtype=torch.float32)
        target_aux = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)

        base = compute_planner_losses(
            pred_controls,
            target_controls,
            pred_aux,
            target_aux,
            aux_loss_weight=0.3,
            horizon_loss_weights=(1.0,),
            control_target_names=GENERIC_CONTROL_TARGET_NAMES,
            smooth_l1_beta=1.0,
        )
        weighted = compute_planner_losses(
            pred_controls,
            target_controls,
            pred_aux,
            target_aux,
            aux_loss_weight=0.3,
            horizon_loss_weights=(1.0,),
            target_loss_weights={"acceleration": 3.0, "future_speed": 5.0},
            control_target_names=GENERIC_CONTROL_TARGET_NAMES,
            smooth_l1_beta=1.0,
        )

        self.assertAlmostEqual(float(base["control_loss"].item()), 0.5 / 3.0, places=6)
        self.assertAlmostEqual(float(weighted["control_loss"].item()), 0.5 * 3.0 / 5.0, places=6)
        self.assertAlmostEqual(float(base["aux_loss"].item()), 0.5 / 4.0, places=6)
        self.assertAlmostEqual(float(weighted["aux_loss"].item()), 0.5 * 5.0 / 8.0, places=6)
        self.assertAlmostEqual(float(weighted["steering_loss"].item()), 0.0, places=6)
        self.assertAlmostEqual(float(weighted["acceleration_loss"].item()), 0.5, places=6)
        self.assertAlmostEqual(float(weighted["future_speed_loss"].item()), 0.5, places=6)

    def test_compute_planner_losses_zero_target_weight_disables_target(self) -> None:
        pred_controls = torch.zeros((1, 1, 3), dtype=torch.float32)
        target_controls = torch.tensor([[[10.0, 0.0, 0.0]]], dtype=torch.float32)
        pred_aux = torch.zeros((1, 1, 4), dtype=torch.float32)
        target_aux = torch.zeros((1, 1, 4), dtype=torch.float32)

        losses = compute_planner_losses(
            pred_controls,
            target_controls,
            pred_aux,
            target_aux,
            aux_loss_weight=0.3,
            horizon_loss_weights=(1.0,),
            target_loss_weights={"steering": 0.0},
            control_target_names=GENERIC_CONTROL_TARGET_NAMES,
            smooth_l1_beta=1.0,
        )

        self.assertAlmostEqual(float(losses["control_loss"].item()), 0.0, places=6)
        self.assertGreater(float(losses["steering_loss"].item()), 0.0)
        self.assertAlmostEqual(float(losses["loss"].item()), 0.0, places=6)

    def test_compute_planner_losses_missing_target_weights_default_to_one(self) -> None:
        pred_controls = torch.zeros((1, 1, 3), dtype=torch.float32)
        target_controls = torch.ones((1, 1, 3), dtype=torch.float32)
        pred_aux = torch.zeros((1, 1, 4), dtype=torch.float32)
        target_aux = torch.ones((1, 1, 4), dtype=torch.float32)

        partial = compute_planner_losses(
            pred_controls,
            target_controls,
            pred_aux,
            target_aux,
            aux_loss_weight=0.3,
            horizon_loss_weights=(1.0,),
            target_loss_weights={"steering": 2.0},
            control_target_names=GENERIC_CONTROL_TARGET_NAMES,
            smooth_l1_beta=1.0,
        )
        explicit = compute_planner_losses(
            pred_controls,
            target_controls,
            pred_aux,
            target_aux,
            aux_loss_weight=0.3,
            horizon_loss_weights=(1.0,),
            target_loss_weights={
                "steering": 2.0,
                "acceleration": 1.0,
                "brakePressureAvg": 1.0,
                "future_speed": 1.0,
                "future_speed_delta": 1.0,
                "future_yaw_delta": 1.0,
                "future_yaw_rate": 1.0,
            },
            control_target_names=GENERIC_CONTROL_TARGET_NAMES,
            smooth_l1_beta=1.0,
        )

        self.assertAlmostEqual(float(partial["control_loss"].item()), float(explicit["control_loss"].item()), places=6)
        self.assertAlmostEqual(float(partial["aux_loss"].item()), float(explicit["aux_loss"].item()), places=6)

    def test_compute_planner_losses_uses_configured_smooth_l1_beta(self) -> None:
        pred_controls = torch.zeros((1, 1, 3), dtype=torch.float32)
        target_controls = torch.tensor([[[0.05, 0.0, 0.0]]], dtype=torch.float32)
        pred_aux = torch.zeros((1, 1, 4), dtype=torch.float32)
        target_aux = torch.zeros((1, 1, 4), dtype=torch.float32)

        losses = compute_planner_losses(
            pred_controls,
            target_controls,
            pred_aux,
            target_aux,
            aux_loss_weight=0.3,
            horizon_loss_weights=(1.0,),
            target_loss_weights={"steering": 1.0, "acceleration": 0.0, "brakePressureAvg": 0.0},
            control_target_names=GENERIC_CONTROL_TARGET_NAMES,
            smooth_l1_beta=0.1,
        )

        self.assertAlmostEqual(float(losses["control_loss"].item()), 0.0125, places=6)

    def test_finalize_metrics_reports_new_overall_and_per_horizon_maes(self) -> None:
        totals = initialize_metric_totals(
            2,
            control_target_names=GENERIC_CONTROL_TARGET_NAMES,
            aux_target_names=DEFAULT_AUX_TARGET_NAMES,
        )
        pred_controls = torch.zeros((1, 2, 3), dtype=torch.float32)
        target_controls = torch.ones((1, 2, 3), dtype=torch.float32)
        pred_aux = torch.zeros((1, 2, 4), dtype=torch.float32)
        target_aux = torch.full((1, 2, 4), 2.0, dtype=torch.float32)
        losses = compute_planner_losses(
            pred_controls,
            target_controls,
            pred_aux,
            target_aux,
            aux_loss_weight=0.3,
            horizon_loss_weights=(1.0, 1.0),
            control_target_names=GENERIC_CONTROL_TARGET_NAMES,
        )

        update_metric_totals(
            totals,
            pred_controls=pred_controls,
            target_controls=target_controls,
            pred_aux=pred_aux,
            target_aux=target_aux,
            losses=losses,
            control_target_names=GENERIC_CONTROL_TARGET_NAMES,
            aux_target_names=DEFAULT_AUX_TARGET_NAMES,
        )
        metrics = finalize_metrics(
            totals,
            (1, 2),
            control_target_names=GENERIC_CONTROL_TARGET_NAMES,
            aux_target_names=DEFAULT_AUX_TARGET_NAMES,
        )

        self.assertEqual(metrics["control_mae_overall"], 1.0)
        self.assertEqual(metrics["aux_mae_overall"], 2.0)
        self.assertEqual(metrics["steering_mae"], 1.0)
        self.assertEqual(metrics["acceleration_mae"], 1.0)
        self.assertEqual(metrics["brakePressureAvg_mae"], 1.0)
        self.assertEqual(metrics["future_speed_mae"], 2.0)
        self.assertEqual(metrics["future_speed_delta_mae"], 2.0)
        self.assertEqual(metrics["future_yaw_delta_mae"], 2.0)
        self.assertEqual(metrics["future_yaw_rate_mae"], 2.0)
        self.assertIn("steering_loss", metrics)
        self.assertIn("acceleration_loss", metrics)
        self.assertIn("brakePressureAvg_loss", metrics)
        self.assertIn("future_speed_loss", metrics)
        self.assertIn("future_speed_delta_loss", metrics)
        self.assertIn("future_yaw_delta_loss", metrics)
        self.assertIn("future_yaw_rate_loss", metrics)
        self.assertEqual(metrics["control_mae_t+1"], 1.0)
        self.assertEqual(metrics["control_mae_t+2"], 1.0)
        self.assertEqual(metrics["aux_mae_t+1"], 2.0)
        self.assertEqual(metrics["aux_mae_t+2"], 2.0)
        self.assertIn("val_loss", metrics)

    def test_round_trip_target_transforms(self) -> None:
        payload = round_trip_transform_check()
        self.assertLess(payload["max_round_trip_error"], 1e-6)

    def test_format_first_batch_debug_prints_temporal_shapes(self) -> None:
        message = format_first_batch_debug(
            torch.zeros((4, 5, 3, 480, 480), dtype=torch.uint8),
            torch.zeros((4, 9, 6), dtype=torch.float32),
            torch.zeros((4, 0), dtype=torch.float32),
            torch.zeros((4, 6, 3), dtype=torch.float32),
            torch.zeros((4, 6, 4), dtype=torch.float32),
        )

        self.assertIn("images_shape=(4, 5, 3, 480, 480)", message)
        self.assertIn("dtype=torch.uint8", message)
        self.assertIn("telemetry_shape=(4, 9, 6)", message)
        self.assertIn("state_inputs_shape=(4, 0)", message)
        self.assertIn("target_controls_shape=(4, 6, 3)", message)
        self.assertIn("target_aux_shape=(4, 6, 4)", message)

    def test_format_first_batch_predictions_prints_raw_temporal_outputs(self) -> None:
        message = format_first_batch_predictions(
            torch.zeros((1, 0), dtype=torch.float32),
            torch.tensor([[[0.123456, -0.987654, 0.3333], [1.25, 2.5, 0.6666]]], dtype=torch.float32),
            torch.tensor([[[0.0, 0.5, 0.25], [1.0, 2.0, 0.75]]], dtype=torch.float32),
            torch.tensor([[[3.33333, 1.1111, -4.44444, 5.55555], [6.0, 2.0, 7.0, 8.0]]], dtype=torch.float32),
            torch.tensor([[[9.0, 0.1, 10.0, 11.0], [12.0, 0.2, 13.0, 14.0]]], dtype=torch.float32),
            state_input_names=(),
            control_target_names=("steering", "acceleration", "brakePressureAvg"),
            aux_target_names=("future_speed", "future_speed_delta", "future_yaw_delta", "future_yaw_rate"),
            future_offsets=(1, 2),
            batch_index=30,
        )

        self.assertIn("train_batch_predictions sample=0 batch=30", message)
        self.assertIn("state_inputs_sample0: <none>", message)
        self.assertIn("\npred_controls:\n", message)
        self.assertIn("  t+1: steering=0.1235 acceleration=-0.9877 brakePressureAvg=0.3333", message)
        self.assertIn("  t+2: steering=1.25 acceleration=2.5 brakePressureAvg=0.6666", message)
        self.assertIn("\npred_aux:\n", message)
        self.assertIn("  t+1: future_speed=3.3333 future_speed_delta=1.1111 future_yaw_delta=-4.4444 future_yaw_rate=5.5556", message)
        self.assertIn("  t+2: future_speed=6.0 future_speed_delta=2.0 future_yaw_delta=7.0 future_yaw_rate=8.0", message)
        self.assertIn("\ntarget_aux:\n", message)
        self.assertIn("  t+1: future_speed=9.0 future_speed_delta=0.1 future_yaw_delta=10.0 future_yaw_rate=11.0", message)
        self.assertIn("  t+2: future_speed=12.0 future_speed_delta=0.2 future_yaw_delta=13.0 future_yaw_rate=14.0", message)

    def test_compute_drive_score_penalizes_validation_control_and_gap(self) -> None:
        score = compute_drive_score(
            {"control_mae_overall": 0.10},
            {"control_loss": 0.20, "control_mae_overall": 0.15},
        )

        self.assertAlmostEqual(score, 0.2425, places=6)

    def test_drive_score_best_epoch_selection_prefers_lower_drive_score(self) -> None:
        state = EarlyStoppingState(metric_name="drive_score", patience=3, min_delta=0.0)

        first = EpochResult(
            epoch_index=1,
            train_metrics={"loss": 0.1, "control_mae_overall": 0.1, "aux_mae_overall": 0.1},
            val_metrics={"val_loss": 0.10, "drive_score": 0.20, "control_mae_overall": 0.1, "aux_mae_overall": 0.1},
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
        second = EpochResult(
            epoch_index=2,
            train_metrics={"loss": 0.05, "control_mae_overall": 0.1, "aux_mae_overall": 0.1},
            val_metrics={"val_loss": 0.05, "drive_score": 0.30, "control_mae_overall": 0.1, "aux_mae_overall": 0.1},
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

        first_is_best, _, _ = check_early_stopping(first, state)
        second_is_best, _, _ = check_early_stopping(second, state)

        self.assertTrue(first_is_best)
        self.assertFalse(second_is_best)
        self.assertEqual(state.best_epoch, 1)
        self.assertAlmostEqual(state.best_value, 0.20, places=6)

    def test_build_training_context_rejects_empty_validation_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_temporal_trip_fixture(root, run_name="train-a", trip_name="trip-000")
            trip_dir = root / "runs" / "val-a" / "stop-sign_temporal-v1" / "trip-000"
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

    def test_build_training_context_records_stats_without_retaining_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_temporal_trip_fixture(root, run_name="train-a", trip_name="trip-000")
            create_temporal_trip_fixture(root, run_name="val-a", trip_name="trip-000")

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

            context = build_training_context(config, root / "train_config.toml")

            self.assertEqual(context.train_stats.sample_count, 1)
            self.assertEqual(context.val_stats.sample_count, 1)
            self.assertFalse(hasattr(context, "train_dataset"))
            self.assertFalse(hasattr(context, "val_dataset"))
            self.assertFalse(hasattr(context, "val_subset"))

    def test_inference_load_checkpoint_accepts_temporal_planner_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "epoch-001.pt"
            future_offsets = list(DEFAULT_FUTURE_OFFSETS)
            transforms = build_target_transform_registry(PARKING_CONTROL_TARGET_NAMES)
            torch.save({
                "planner_format": PLANNER_FORMAT,
                "planner_format_version": PLANNER_FORMAT_VERSION,
                "control_contract": control_contract_metadata(),
                "control_target_names": list(PARKING_CONTROL_TARGET_NAMES),
                "frame_window_size": len(DEFAULT_IMAGE_OFFSETS),
                "image_size": {"width": DEFAULT_IMAGE_WIDTH, "height": DEFAULT_IMAGE_HEIGHT},
                "image_offsets": list(DEFAULT_IMAGE_OFFSETS),
                "telemetry_offsets": list(DEFAULT_TELEMETRY_OFFSETS),
                "future_offsets": future_offsets,
                "telemetry_sample_interval_ms": 50,
                "control_horizon_dt_ms": list(derive_control_horizon_dt_ms(future_offsets, 50)),
                "target_transforms": target_transform_metadata(transforms),
                "model_state_dict": {},
            }, checkpoint_path)

            checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))

            self.assertEqual(checkpoint["planner_format"], PLANNER_FORMAT)

    def test_temporal_defaults_keep_training_at_480_square(self) -> None:
        self.assertEqual(DEFAULT_IMAGE_WIDTH, 480)
        self.assertEqual(DEFAULT_IMAGE_HEIGHT, 480)


if __name__ == "__main__":
    unittest.main()
