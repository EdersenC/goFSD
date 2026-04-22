from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from config import parse_dataset_window
from dataset import FsdDataset, build_targets
from heads import (
    ALL_HEAD_SPECS,
    ALL_HEAD_SPECS_BY_NAME,
    CURRENT_SPEED_TARGET_KEY,
    HEAD_SPECS,
    apply_loss_weight_overrides,
    build_targets_from_label,
    get_control_outputs,
    head_layout_metadata,
    inactive_loss_weight_override_names,
    resolve_checkpoint_head_specs,
    validate_checkpoint_head_layout,
)
from inference import InferenceConfig, build_output, remap_legacy_state_dict_keys, resolve_checkpoint_frame_stride
from model_output import (
    control_tensor_from_output,
    single_control_prediction_from_output,
    single_prediction_from_output,
)
from models.planner import DrivingCNN
from state_inputs import (
    CURRENT_SPEED_KEY,
    ROUTE_FORWARD_DELTA_KEY,
    StateInputConfig,
    current_speed_fused_head_names,
    normalize_current_speed_tensor,
    normalize_current_speed_value,
    normalize_route_forward_delta_tensor,
    normalize_route_forward_delta_value,
    route_forward_delta_fused_head_names,
    state_input_config_from_metadata,
    state_inputs_metadata,
    training_state_input_config,
)
from target_transforms import DeltaSpeedTargetTransform
from server import ModelRuntime
from train import (
    _compute_weighted_future_yaw_delta_loss,
    _load_turn_oversampling_config,
    _load_yaw_loss_weighting_config,
    _turn_oversampling_sampler,
    _compute_speed_consistency_loss,
    build_turn_oversampling_weights,
    DatasetConfig,
    LoaderConfig,
    OutputConfig,
    TrainConfig,
    TrainingConfig,
    TurnOversamplingConfig,
    YawLossWeightingConfig,
    build_training_context,
    build_validation_subset,
)


def create_trip_fixture(
    root: Path,
    *,
    run_name: str,
    scene_name: str = "inner-city-driving_default",
    trip_name: str,
    sample_count: int = 1,
    write_frames: bool = False,
    image_size: tuple[int, int] = (480, 480),
) -> Path:
    trip_dir = root / "runs" / run_name / scene_name / trip_name
    trip_dir.mkdir(parents=True, exist_ok=True)
    scene_id, scene_variant = scene_name.split("_", 1)
    (trip_dir / "metadata.json").write_text(json.dumps({
        "runId": run_name,
        "sceneId": scene_id,
        "sceneVariant": scene_variant,
        "tripIndex": int(trip_name.split("-")[-1]),
    }), encoding="utf-8")

    samples: list[str] = []
    for sample_index in range(sample_count):
        samples.append(json.dumps({
            "frame_paths": [
                f"frames/{sample_index:06d}.jpg",
                f"frames/{sample_index + 1:06d}.jpg",
                f"frames/{sample_index + 2:06d}.jpg",
            ],
            "label": {
                "Steering": float(sample_index),
                "future_yaw_delta": float(sample_index) + 0.25,
                "future_horizon_seconds": 0.2,
                "delta_speed": 0.0,
                "delta_speed_target": 0.0,
                "future_speed": 4.0,
                "future_speed_target": 4.0,
                "yaw_rate": 0.1,
                "routeForwardDelta": 0.5,
                "gps": [1.0, 2.0, 3.0],
                "currentSpeed": 4.0,
                "isStopped": 0,
            },
        }))
    (trip_dir / "dataset.jsonl").write_text("\n".join(samples) + "\n", encoding="utf-8")
    if write_frames:
        frames_dir = trip_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        for frame_index in range(sample_count + 2):
            write_jpeg(frames_dir / f"{frame_index:06d}.jpg", image_size=image_size)
    return trip_dir


def write_jpeg(path: Path, *, image_size: tuple[int, int]) -> None:
    width, height = image_size
    image = Image.new("RGB", (width, height), color=(120, 160, 200))
    image.save(path, format="JPEG")


def encode_jpeg_base64(*, image_size: tuple[int, int]) -> str:
    width, height = image_size
    image = Image.new("RGB", (width, height), color=(120, 160, 200))
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class FrameWindowConfigTests(unittest.TestCase):
    def test_parse_dataset_window_reads_dataset_section(self) -> None:
        window_size, frame_stride, sample_stride = parse_dataset_window({
            "dataset": {
                "window_size": 5,
                "frame_stride": 2,
                "sample_stride": 10,
            }
        })
        self.assertEqual(window_size, 5)
        self.assertEqual(frame_stride, 2)
        self.assertEqual(sample_stride, 10)

    def test_parse_dataset_window_rejects_legacy_window_stride(self) -> None:
        with self.assertRaisesRegex(ValueError, "window_stride is no longer supported"):
            parse_dataset_window({
                "dataset": {
                    "window_size": 5,
                    "window_stride": 2,
                }
            })

    def test_driving_cnn_scales_input_channels_and_head_outputs(self) -> None:
        model = DrivingCNN(frame_count=5, state_input_config=training_state_input_config())
        self.assertEqual(model.conv1.in_channels, 15)

        x = torch.zeros((2, 15, 32, 32), dtype=torch.float32)
        current_speed = torch.zeros((2,), dtype=torch.float32)
        output = model(x, current_speed=current_speed)

        self.assertEqual(set(output.keys()), {spec.name for spec in HEAD_SPECS})
        self.assertEqual(tuple(output["future_yaw_delta"].shape), (2,))
        self.assertEqual(tuple(output["future_speed"].shape), (2,))
        self.assertEqual(tuple(output["delta_speed"].shape), (2,))

    def test_driving_cnn_runs_without_current_speed_when_disabled(self) -> None:
        model = DrivingCNN(frame_count=5, state_input_config=StateInputConfig())
        self.assertEqual(model.conv1.in_channels, 15)

        x = torch.zeros((2, 15, 32, 32), dtype=torch.float32)
        output = model(x)

        self.assertEqual(set(output.keys()), {spec.name for spec in HEAD_SPECS})
        self.assertEqual(tuple(output["future_yaw_delta"].shape), (2,))
        self.assertEqual(tuple(output["future_speed"].shape), (2,))
        self.assertEqual(tuple(output["delta_speed"].shape), (2,))

    def test_driving_cnn_route_forward_delta_legacy_fusion_targets_legacy_heads(self) -> None:
        model = DrivingCNN(
            frame_count=5,
            state_input_config=StateInputConfig(
                route_forward_delta_enabled=True,
                route_forward_delta_fusion="legacy",
            ),
        )

        self.assertEqual(model.heads["future_yaw_delta"].fc1.in_features, 272)
        self.assertEqual(model.heads["delta_speed"].fc1.in_features, 272)
        self.assertEqual(model.heads["future_speed"].fc1.in_features, 272)
        self.assertEqual(model.heads["move_intent"].fc1.in_features, 272)
        self.assertEqual(model.heads["yaw_rate"].fc1.in_features, 256)

    def test_driving_cnn_current_speed_fusion_can_target_delta_and_yaw_heads(self) -> None:
        model = DrivingCNN(
            frame_count=5,
            state_input_config=StateInputConfig(
                current_speed_enabled=True,
                current_speed_fusion="delta_and_yaw_heads",
            ),
        )

        self.assertEqual(model.heads["future_yaw_delta"].fc1.in_features, 272)
        self.assertEqual(model.heads["delta_speed"].fc1.in_features, 272)
        self.assertEqual(model.heads["future_speed"].fc1.in_features, 256)
        self.assertEqual(model.heads["move_intent"].fc1.in_features, 256)

    def test_dataset_rejects_mismatched_frame_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trip_dir = Path(tmp) / "runs" / "run-a" / "inner-city-driving_default" / "trip-000"
            trip_dir.mkdir(parents=True)
            (trip_dir / "metadata.json").write_text(json.dumps({
                "runId": "run-a",
                "sceneId": "inner-city-driving",
                "sceneVariant": "default",
                "tripIndex": 0,
            }), encoding="utf-8")
            (trip_dir / "dataset.jsonl").write_text(json.dumps({
                "frame_paths": [
                    "frames/000001.jpg",
                    "frames/000003.jpg",
                    "frames/000005.jpg",
                ],
                "label": {
                    "Steering": 0.0,
                    "future_yaw_delta": 0.0,
                    "future_horizon_seconds": 0.2,
                    "delta_speed": 0.0,
                    "delta_speed_target": 0.0,
                    "future_speed": 4.0,
                    "future_speed_target": 4.0,
                    "yaw_rate": 0.0,
                    "routeForwardDelta": 0.5,
                    "gps": [1.0, 2.0, 3.0],
                    "currentSpeed": 4.0,
                    "isStopped": 1,
                },
            }) + "\n", encoding="utf-8")

            dataset = FsdDataset(run_id="run-a", data_root=tmp, expected_window_size=5)
            with self.assertRaisesRegex(ValueError, "Expected 5 frame paths"):
                _ = dataset[0]

    def test_dataset_accumulates_samples_from_multiple_run_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_trip_fixture(root, run_name="run-a", trip_name="trip-000", sample_count=2)
            create_trip_fixture(root, run_name="run-b", trip_name="trip-000", sample_count=3)

            dataset = FsdDataset(
                run_paths=[
                    root / "runs" / "run-a",
                    root / "runs" / "run-b",
                ],
                expected_window_size=3,
            )

            self.assertEqual(dataset.trip_count, 2)
            self.assertEqual(len(dataset), 5)
            self.assertEqual(len(dataset.trip_sample_indices()), 2)
            self.assertEqual(dataset.trip_sample_indices()[0], [0, 1])
            self.assertEqual(dataset.trip_sample_indices()[1], [2, 3, 4])
            self.assertEqual(dataset.samples[0]["trip_key"], "run-a/inner-city-driving_default/trip-000")
            self.assertEqual(dataset.samples[2]["trip_key"], "run-b/inner-city-driving_default/trip-000")

    def test_route_forward_delta_normalization_clamps_signed_range(self) -> None:
        self.assertAlmostEqual(normalize_route_forward_delta_value(0.0, 1.5), 0.0)
        self.assertAlmostEqual(normalize_route_forward_delta_value(1.5, 1.5), 1.0)
        self.assertAlmostEqual(normalize_route_forward_delta_value(-1.5, 1.5), -1.0)
        self.assertAlmostEqual(normalize_route_forward_delta_value(5.0, 1.5), 1.0)
        self.assertAlmostEqual(normalize_route_forward_delta_value(-5.0, 1.5), -1.0)

        tensor = torch.tensor([-3.0, 0.0, 3.0], dtype=torch.float32)
        normalized = normalize_route_forward_delta_tensor(tensor, 1.5)
        self.assertTrue(torch.equal(normalized, torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)))

    def test_state_input_metadata_round_trip_includes_route_forward_delta(self) -> None:
        config = StateInputConfig(
            current_speed_enabled=True,
            current_speed_cap=25.0,
            current_speed_fusion="legacy",
            route_forward_delta_enabled=True,
            route_forward_delta_cap=1.5,
            route_forward_delta_fusion="legacy",
        )
        payload = state_inputs_metadata(config)
        restored = state_input_config_from_metadata(payload)
        self.assertEqual(restored, config)
        self.assertEqual(route_forward_delta_fused_head_names(restored), ("delta_speed", "future_speed", "move_intent", "future_yaw_delta"))

    def test_dataset_builds_route_forward_delta_state_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_trip_fixture(root, run_name="run-a", trip_name="trip-000", sample_count=1, write_frames=True, image_size=(32, 32))
            dataset = FsdDataset(
                run_id="run-a",
                data_root=tmp,
                image_size=(32, 32),
                expected_window_size=3,
                state_input_config=StateInputConfig(
                    current_speed_enabled=True,
                    current_speed_fusion="delta_and_yaw_heads",
                    route_forward_delta_enabled=True,
                    route_forward_delta_cap=1.5,
                    route_forward_delta_fusion="legacy",
                ),
            )
            _, state_inputs, _ = dataset[0]
            self.assertIn(CURRENT_SPEED_KEY, state_inputs)
            self.assertIn(ROUTE_FORWARD_DELTA_KEY, state_inputs)
            self.assertAlmostEqual(float(state_inputs[ROUTE_FORWARD_DELTA_KEY].item()), 0.5 / 1.5)

    def test_dataset_loads_multiple_scenes_from_one_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_trip_fixture(
                root,
                run_name="run-a",
                scene_name="inner-city-driving_default",
                trip_name="trip-000",
                sample_count=2,
            )
            create_trip_fixture(
                root,
                run_name="run-a",
                scene_name="coastal-drive_default",
                trip_name="trip-000",
                sample_count=3,
            )

            dataset = FsdDataset(run_paths=[root / "runs" / "run-a"], expected_window_size=3)

            self.assertEqual(dataset.trip_count, 2)
            self.assertEqual(len(dataset), 5)
            self.assertEqual(dataset.trip_sample_indices()[0], [0, 1])
            self.assertEqual(dataset.trip_sample_indices()[1], [2, 3, 4])
            self.assertEqual(dataset.samples[0]["trip_key"], "run-a/coastal-drive_default/trip-000")
            self.assertEqual(dataset.samples[2]["trip_key"], "run-a/inner-city-driving_default/trip-000")

    def test_build_training_context_rejects_empty_validation_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_trip_fixture(root, run_name="train-a", trip_name="trip-000", sample_count=1)
            create_trip_fixture(root, run_name="val-a", trip_name="trip-000", sample_count=0)

            config = TrainConfig(
                dataset=DatasetConfig(
                    data_root=str(root),
                    train_run_ids=("train-a",),
                    val_run_ids=("val-a",),
                    train_run_paths=(str(root / "runs" / "train-a"),),
                    val_run_paths=(str(root / "runs" / "val-a"),),
                    image_width=32,
                    image_height=32,
                    window_size=3,
                    frame_stride=2,
                    sample_stride=2,
                ),
                output=OutputConfig(base_dir=str(root / "training_runs")),
                training=TrainingConfig(
                    device="cpu",
                    epochs=1,
                    learning_rate=1e-3,
                    early_stopping_metric="control_overall_mae",
                    early_stopping_patience=1,
                    early_stopping_min_delta=0.0,
                    head_loss_weights={},
                    yaw_consistency_weight=0.0,
                    yaw_rate_scale_to_degrees=57.29577951308232,
                    speed_consistency_weight=0.0,
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

    def test_dataset_loads_pre_sized_frames_without_resize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_trip_fixture(
                root,
                run_name="run-a",
                trip_name="trip-000",
                sample_count=1,
                write_frames=True,
                image_size=(16, 12),
            )

            dataset = FsdDataset(
                run_paths=[root / "runs" / "run-a"],
                image_size=(16, 12),
                expected_window_size=3,
            )

            sample_x, state_inputs, _ = dataset[0]
            self.assertEqual(tuple(sample_x.shape), (9, 12, 16))
            self.assertIn(CURRENT_SPEED_KEY, state_inputs)

    def test_dataset_rejects_mismatched_image_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_trip_fixture(
                root,
                run_name="run-a",
                trip_name="trip-000",
                sample_count=1,
                write_frames=True,
                image_size=(20, 10),
            )

            dataset = FsdDataset(
                run_paths=[root / "runs" / "run-a"],
                image_size=(16, 12),
                expected_window_size=3,
            )

            with self.assertRaisesRegex(ValueError, "unexpected size"):
                _ = dataset[0]

    def test_validation_subset_selects_whole_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_trip_fixture(root, run_name="run-a", trip_name="trip-000", sample_count=2)
            create_trip_fixture(root, run_name="run-a", trip_name="trip-001", sample_count=2)
            create_trip_fixture(root, run_name="run-a", trip_name="trip-002", sample_count=2)
            create_trip_fixture(root, run_name="run-a", trip_name="trip-003", sample_count=2)

            dataset = FsdDataset(run_paths=[root / "runs" / "run-a"], expected_window_size=3)
            subset, selected_trip_count, total_trip_count = build_validation_subset(dataset, 0.50, seed=123)

            self.assertEqual(selected_trip_count, 2)
            self.assertEqual(total_trip_count, 4)
            assert hasattr(subset, "indices")
            subset_indices = list(subset.indices)
            subset_trip_keys = {dataset.samples[index]["trip_key"] for index in subset_indices}
            self.assertEqual(len(subset_trip_keys), 2)

            selected_trip_index_sets = [
                trip_indices
                for trip_indices in dataset.trip_sample_indices()
                if dataset.samples[trip_indices[0]]["trip_key"] in subset_trip_keys
            ]
            self.assertEqual(len(selected_trip_index_sets), 2)
            self.assertEqual(sorted(subset_indices), sorted(index for trip in selected_trip_index_sets for index in trip))

    def test_model_runtime_rejects_wrong_request_frame_count(self) -> None:
        runtime = ModelRuntime()
        runtime._frame_count = 5
        with self.assertRaisesRegex(ValueError, "expected 5 frame_paths entries"):
            runtime._extract_frames({"frame_paths": ["a", "b", "c"]})

    def test_model_runtime_rejects_mismatched_frame_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "frame.jpg"
            write_jpeg(image_path, image_size=(20, 10))

            runtime = ModelRuntime()
            runtime._frame_count = 1
            runtime._image_size = (16, 12)

            with self.assertRaisesRegex(ValueError, "unexpected size"):
                runtime._extract_frames({"frame_paths": [str(image_path)]})

    def test_model_runtime_accepts_matching_base64_frame_size(self) -> None:
        runtime = ModelRuntime()
        runtime._frame_count = 1
        runtime._image_size = (16, 12)

        frames = runtime._extract_frames({"frames_base64": [encode_jpeg_base64(image_size=(16, 12))]})

        self.assertEqual(len(frames), 1)
        self.assertEqual(tuple(frames[0].shape), (3, 12, 16))

    def test_model_runtime_requires_current_speed_for_speed_enabled_checkpoint(self) -> None:
        runtime = ModelRuntime()
        runtime._state_input_config = training_state_input_config()
        with self.assertRaisesRegex(ValueError, CURRENT_SPEED_KEY):
            runtime._extract_current_speed({}, runtime._state_input_config)

    def test_model_runtime_normalizes_current_speed(self) -> None:
        runtime = ModelRuntime()
        runtime._state_input_config = training_state_input_config()

        raw, tensor = runtime._extract_current_speed({CURRENT_SPEED_KEY: 12.5}, runtime._state_input_config)

        self.assertEqual(raw, 12.5)
        assert tensor is not None
        self.assertAlmostEqual(float(tensor.item()), 0.5, places=6)

    def test_resolve_checkpoint_frame_stride_accepts_legacy_metadata(self) -> None:
        config = InferenceConfig(
            checkpoint="model.pt",
            device="cpu",
            data_root=None,
            run_id=None,
            sample_index=0,
            output_json=False,
            image_width=480,
            image_height=480,
            window_size=5,
            frame_stride=2,
            sample_stride=10,
        )
        stride = resolve_checkpoint_frame_stride({"frame_window_stride": 2}, config)
        self.assertEqual(stride, 2)

    def test_build_output_reports_new_stride_metadata(self) -> None:
        output = build_output(
            Path("/tmp/model.pt"),
            {
                "epoch": 4,
                "frame_window_size": 5,
                "frame_stride": 2,
                "sample_stride": 10,
                "input_channels": 15,
                "state_inputs": state_inputs_metadata(training_state_input_config()),
            },
            torch.device("cpu"),
            sample_result=None,
        )
        self.assertEqual(output["frame_stride"], 2)
        self.assertEqual(output["sample_stride"], 10)
        self.assertTrue(output["state_inputs"]["current_speed"]["enabled"])

    def test_model_output_helpers_unpack_named_predictions(self) -> None:
        output = {
            "future_yaw_delta": torch.tensor([0.25], dtype=torch.float32),
            "delta_speed": torch.tensor([-0.1], dtype=torch.float32),
            "future_speed": torch.tensor([2.9], dtype=torch.float32),
            "move_intent": torch.tensor([1.25], dtype=torch.float32),
            "route_xy": torch.tensor([[1.5, -2.0]], dtype=torch.float32),
            "speed": torch.tensor([3.0], dtype=torch.float32),
            "is_stopped": torch.tensor([0.8], dtype=torch.float32),
        }

        control = control_tensor_from_output(output)
        prediction = single_prediction_from_output(output)
        control_prediction = single_control_prediction_from_output(output)

        self.assertEqual(tuple(control.shape), (1, 2))
        self.assertAlmostEqual(float(control[0, 0].item()), 0.25, places=6)
        self.assertAlmostEqual(float(control[0, 1].item()), 2.9, places=6)
        self.assertEqual(
            prediction,
            {"future_yaw_delta": 0.25, "future_speed": 2.9000000953674316, "move_intent": 1.25, "delta_speed": -0.10000000149011612},
        )
        self.assertEqual(control_prediction, {"future_yaw_delta": 0.25, "future_speed": 2.9000000953674316})

    def test_model_output_helpers_denormalize_delta_speed_when_requested(self) -> None:
        output = {
            "future_yaw_delta": torch.tensor([0.25], dtype=torch.float32),
            "delta_speed": torch.tensor([-0.5], dtype=torch.float32),
            "future_speed": torch.tensor([2.9], dtype=torch.float32),
            "move_intent": torch.tensor([0.0], dtype=torch.float32),
        }

        prediction = single_prediction_from_output(
            output,
            delta_speed_transform=DeltaSpeedTargetTransform(clip_value=2.0, normalize=True),
        )
        control_prediction = single_control_prediction_from_output(
            output,
            delta_speed_transform=DeltaSpeedTargetTransform(clip_value=2.0, normalize=True),
        )

        self.assertEqual(
            prediction,
            {"future_yaw_delta": 0.25, "future_speed": 2.9000000953674316, "move_intent": 0.0, "delta_speed": -1.0},
        )
        self.assertEqual(control_prediction, {"future_yaw_delta": 0.25, "future_speed": 2.9000000953674316})

    def test_dataset_builds_spec_driven_targets(self) -> None:
        label = {
            "Steering": 0.25,
            "future_yaw_delta": 0.35,
            "future_horizon_seconds": 0.2,
            "delta_speed": -0.1,
            "delta_speed_target": -0.05,
            "future_speed": 2.9,
            "future_speed_target": 2.8,
            "yaw_rate": 0.15,
            "move_intent": True,
            "gps": [190.0, -815.0, 31.0],
            "currentSpeed": 3.0,
            "isStopped": 1,
            "time": 123.0,
        }

        targets = build_targets(label)
        self.assertEqual(
            set(targets.keys()),
            {
                "future_yaw_delta",
                "delta_speed",
                "future_speed",
                "move_intent",
                "yaw_rate",
                "future_horizon_seconds",
                CURRENT_SPEED_TARGET_KEY,
            },
        )
        self.assertAlmostEqual(float(targets["future_yaw_delta"].item()), 0.35, places=6)
        self.assertAlmostEqual(float(targets["delta_speed"].item()), -0.05, places=6)
        self.assertAlmostEqual(float(targets["future_speed"].item()), 2.8, places=6)
        self.assertAlmostEqual(float(targets["move_intent"].item()), 1.0, places=6)
        self.assertAlmostEqual(float(targets["yaw_rate"].item()), 0.15, places=6)
        self.assertAlmostEqual(float(targets["future_horizon_seconds"].item()), 0.2, places=6)
        self.assertAlmostEqual(float(targets[CURRENT_SPEED_TARGET_KEY].item()), 3.0, places=6)

    def test_dataset_returns_normalized_current_speed_state_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_trip_fixture(
                root,
                run_name="run-a",
                trip_name="trip-000",
                sample_count=1,
                write_frames=True,
            )

            dataset = FsdDataset(run_paths=[root / "runs" / "run-a"], expected_window_size=3)
            _, state_inputs, _ = dataset[0]

            self.assertAlmostEqual(float(state_inputs[CURRENT_SPEED_KEY].item()), 4.0 / 25.0, places=6)

    def test_current_speed_normalization_clamps_to_cap(self) -> None:
        self.assertAlmostEqual(normalize_current_speed_value(50.0), 1.0, places=6)
        tensor = normalize_current_speed_tensor(torch.tensor([50.0], dtype=torch.float32))
        self.assertAlmostEqual(float(tensor.item()), 1.0, places=6)

    def test_state_input_config_defaults_to_legacy_disabled_when_missing(self) -> None:
        config = state_input_config_from_metadata(None)
        self.assertFalse(config.current_speed_enabled)

    def test_state_input_metadata_round_trips(self) -> None:
        config = StateInputConfig(current_speed_enabled=True, current_speed_cap=25.0, current_speed_fusion="delta_head_only")
        restored = state_input_config_from_metadata(state_inputs_metadata(config))
        self.assertEqual(restored, config)

    def test_training_state_input_config_reads_current_speed_table(self) -> None:
        config = training_state_input_config({
            "current_speed": {
                "enabled": True,
                "cap": 30.0,
                "fusion": "delta_and_yaw_heads",
            }
        })

        self.assertTrue(config.current_speed_enabled)
        self.assertEqual(config.current_speed_cap, 30.0)
        self.assertEqual(config.current_speed_fusion, "delta_and_yaw_heads")

    def test_current_speed_fused_head_names_resolve_modes(self) -> None:
        self.assertEqual(
            current_speed_fused_head_names(
                StateInputConfig(current_speed_enabled=True, current_speed_fusion="delta_and_yaw_heads")
            ),
            ("delta_speed", "future_yaw_delta"),
        )
        self.assertEqual(
            current_speed_fused_head_names(
                StateInputConfig(current_speed_enabled=True, current_speed_fusion="legacy")
            ),
            ("delta_speed", "future_speed", "move_intent", "future_yaw_delta"),
        )

    def test_dataset_scalar_target_values_reads_labels_without_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_trip_fixture(root, run_name="run-a", trip_name="trip-000", sample_count=3, write_frames=False)

            dataset = FsdDataset(run_paths=[root / "runs" / "run-a"], expected_window_size=3)
            values = dataset.scalar_target_values("future_yaw_delta")

            self.assertEqual(values, (0.25, 1.25, 2.25))

    def test_weighted_future_yaw_delta_loss_matches_unweighted_when_disabled(self) -> None:
        prediction = torch.tensor([0.0, 0.2], dtype=torch.float32)
        target = torch.tensor([0.0, 0.3], dtype=torch.float32)

        unweighted, weighted = _compute_weighted_future_yaw_delta_loss(
            prediction,
            target,
            YawLossWeightingConfig(enabled=False),
        )

        self.assertAlmostEqual(float(unweighted.item()), float(weighted.item()), places=6)

    def test_weighted_future_yaw_delta_loss_upweights_turn_heavy_samples(self) -> None:
        prediction = torch.tensor([0.0, 0.0], dtype=torch.float32)
        target = torch.tensor([0.01, 0.30], dtype=torch.float32)

        unweighted, weighted = _compute_weighted_future_yaw_delta_loss(
            prediction,
            target,
            YawLossWeightingConfig(enabled=True, base_weight=1.0, alpha=2.0, tau=0.25, max_scale=3.0),
        )

        self.assertGreater(float(weighted.item()), float(unweighted.item()))

    def test_turn_oversampling_weights_bucket_future_yaw_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_trip_fixture(root, run_name="run-a", trip_name="trip-000", sample_count=4, write_frames=False)

            dataset = FsdDataset(run_paths=[root / "runs" / "run-a"], expected_window_size=3)
            weights, summary = build_turn_oversampling_weights(
                dataset,
                TurnOversamplingConfig(
                    enabled=True,
                    straight_weight=1.0,
                    light_turn_weight=1.5,
                    medium_turn_weight=2.5,
                    sharp_turn_weight=4.0,
                    light_turn_threshold=0.5,
                    medium_turn_threshold=1.5,
                    sharp_turn_threshold=2.5,
                ),
            )

            assert weights is not None
            self.assertEqual(weights.tolist(), [1.0, 1.5, 2.5, 4.0])
            self.assertEqual(summary["bucket_counts"], {"straight": 1, "light_turn": 1, "medium_turn": 1, "sharp_turn": 1})

    def test_turn_oversampling_sampler_is_disabled_when_weights_missing(self) -> None:
        self.assertIsNone(_turn_oversampling_sampler(None))

    def test_loader_and_yaw_weighting_config_helpers_parse_defaults(self) -> None:
        yaw_config = _load_yaw_loss_weighting_config({"yaw_loss_weighting": {"enabled": True}})
        sampler_config = _load_turn_oversampling_config({"turn_oversampling": {"enabled": True}})

        self.assertTrue(yaw_config.enabled)
        self.assertTrue(sampler_config.enabled)

    def test_head_helpers_keep_control_and_aux_separate(self) -> None:
        outputs = {
            "future_yaw_delta": torch.tensor([0.2], dtype=torch.float32),
            "delta_speed": torch.tensor([0.4], dtype=torch.float32),
            "future_speed": torch.tensor([3.4], dtype=torch.float32),
            "move_intent": torch.tensor([0.1], dtype=torch.float32),
            "route_xy": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            "speed": torch.tensor([3.0], dtype=torch.float32),
            "is_stopped": torch.tensor([0.0], dtype=torch.float32),
        }

        control_outputs = get_control_outputs(outputs)
        self.assertEqual(set(control_outputs.keys()), {"future_yaw_delta", "future_speed"})

    def test_head_layout_metadata_matches_registry_order(self) -> None:
        self.assertEqual(
            [item["name"] for item in head_layout_metadata()],
            [spec.name for spec in HEAD_SPECS],
        )

    def test_all_head_specs_preserve_auxiliary_heads_for_future_reactivation(self) -> None:
        self.assertEqual(
            [spec.name for spec in ALL_HEAD_SPECS],
            ["future_yaw_delta", "future_speed", "move_intent", "delta_speed", "yaw_rate", "route_xy", "speed", "is_stopped"],
        )

    def test_build_targets_from_label_includes_trained_auxiliary_heads(self) -> None:
        label = {
            "Steering": 0.02145645022392273,
            "future_yaw_delta": 0.03145645022392273,
            "future_horizon_seconds": 0.2,
            "delta_speed": 0.04291290044784546,
            "delta_speed_target": 0.02145645022392273,
            "future_speed": 0.05235547572374344,
            "future_speed_target": 0.06235547572374344,
            "yaw_rate": 0.01235547572374344,
            "move_intent": True,
            "coords": [191.87716674804688, -814.9187622070312, 31.073318481445312],
            "currentSpeed": 0.00944257527589798,
            "drivingStyle": 526523,
            "gps": [190.74496459960938, -815.3811645507812, 31.052885055541992],
            "isStopped": 1,
            "time": 8982154,
            "yaw": 70.4722900390625,
        }

        targets = build_targets_from_label(label)
        self.assertEqual(set(targets.keys()), {"future_yaw_delta", "delta_speed", "future_speed", "move_intent", "yaw_rate"})

    def test_auxiliary_target_builders_are_still_available(self) -> None:
        label = {
            "Steering": 0.02145645022392273,
            "future_yaw_delta": 0.03145645022392273,
            "future_horizon_seconds": 0.2,
            "delta_speed": 0.04291290044784546,
            "delta_speed_target": 0.02145645022392273,
            "future_speed": 0.05235547572374344,
            "future_speed_target": 0.06235547572374344,
            "yaw_rate": 0.01235547572374344,
            "move_intent": True,
            "coords": [191.87716674804688, -814.9187622070312, 31.073318481445312],
            "currentSpeed": 0.00944257527589798,
            "drivingStyle": 526523,
            "gps": [190.74496459960938, -815.3811645507812, 31.052885055541992],
            "isStopped": 1,
            "time": 8982154,
            "yaw": 70.4722900390625,
        }

        future_yaw_delta = ALL_HEAD_SPECS_BY_NAME["future_yaw_delta"].target_builder(label)
        future_speed = ALL_HEAD_SPECS_BY_NAME["future_speed"].target_builder(label)
        move_intent = ALL_HEAD_SPECS_BY_NAME["move_intent"].target_builder(label)
        yaw_rate = ALL_HEAD_SPECS_BY_NAME["yaw_rate"].target_builder(label)
        route_xy = ALL_HEAD_SPECS_BY_NAME["route_xy"].target_builder(label)
        speed = ALL_HEAD_SPECS_BY_NAME["speed"].target_builder(label)
        is_stopped = ALL_HEAD_SPECS_BY_NAME["is_stopped"].target_builder(label)

        self.assertAlmostEqual(float(future_yaw_delta.item()), 0.03145645022392273, places=6)
        self.assertAlmostEqual(float(future_speed.item()), 0.06235547572374344, places=6)
        self.assertAlmostEqual(float(move_intent.item()), 1.0, places=6)
        self.assertAlmostEqual(float(yaw_rate.item()), 0.01235547572374344, places=6)
        self.assertEqual(route_xy.tolist(), [190.74496459960938, -815.3811645507812])
        self.assertAlmostEqual(float(speed.item()), 0.00944257527589798, places=6)
        self.assertAlmostEqual(float(is_stopped.item()), 1.0, places=6)

    def test_speed_consistency_loss_is_zero_for_consistent_predictions(self) -> None:
        outputs = {
            "future_speed": torch.tensor([3.5], dtype=torch.float32),
            "delta_speed": torch.tensor([0.25], dtype=torch.float32),
        }
        targets = {
            CURRENT_SPEED_TARGET_KEY: torch.tensor([3.0], dtype=torch.float32),
        }

        loss = _compute_speed_consistency_loss(
            outputs,
            targets,
            delta_speed_transform=DeltaSpeedTargetTransform(clip_value=2.0, normalize=True),
            speed_consistency_weight=0.1,
        )

        self.assertIsNotNone(loss)
        assert loss is not None
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)

    def test_is_stopped_target_accepts_boolean_false(self) -> None:
        is_stopped = ALL_HEAD_SPECS_BY_NAME["is_stopped"].target_builder({
            "Steering": 0.0,
            "future_yaw_delta": 0.0,
            "future_horizon_seconds": 0.2,
            "delta_speed": 0.0,
            "delta_speed_target": 0.0,
            "future_speed": 0.0,
            "future_speed_target": 0.0,
            "yaw_rate": 0.0,
            "gps": [0.0, 0.0, 0.0],
            "currentSpeed": 0.0,
            "isStopped": False,
        })
        self.assertAlmostEqual(float(is_stopped.item()), 0.0, places=6)

    def test_inactive_auxiliary_loss_weight_overrides_are_ignored(self) -> None:
        overrides = {
            "future_yaw_delta": 1.5,
            "move_intent": 0.2,
            "route_xy": 0.25,
            "speed": 0.5,
        }

        resolved = apply_loss_weight_overrides(overrides)

        self.assertEqual([spec.name for spec in resolved], ["future_yaw_delta", "future_speed", "move_intent", "delta_speed", "yaw_rate"])
        self.assertAlmostEqual(resolved[0].loss_weight, 1.5, places=6)
        self.assertAlmostEqual(resolved[2].loss_weight, 0.2, places=6)
        self.assertEqual(inactive_loss_weight_override_names(overrides), ("route_xy", "speed"))

    def test_negative_inactive_auxiliary_loss_weight_override_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "route_xy"):
            apply_loss_weight_overrides({"route_xy": -0.1})

    def test_checkpoint_layout_validation_accepts_current_supported_layout(self) -> None:
        validate_checkpoint_head_layout({
            "head_layout": [
                ALL_HEAD_SPECS_BY_NAME["future_yaw_delta"].layout_metadata(),
                {
                    **ALL_HEAD_SPECS_BY_NAME["delta_speed"].layout_metadata(),
                    "kind": "control",
                    "used_for_control": True,
                },
            ],
        })

    def test_checkpoint_head_resolution_preserves_delta_speed_control(self) -> None:
        resolved = resolve_checkpoint_head_specs({
            "head_layout": [
                ALL_HEAD_SPECS_BY_NAME["future_yaw_delta"].layout_metadata(),
                {
                    **ALL_HEAD_SPECS_BY_NAME["delta_speed"].layout_metadata(),
                    "kind": "control",
                    "used_for_control": True,
                },
            ],
        })

        self.assertEqual([spec.name for spec in resolved], ["future_yaw_delta", "delta_speed"])
        self.assertTrue(resolved[1].used_for_control)
        self.assertEqual(resolved[1].kind, "control")

    def test_remap_legacy_state_dict_keys_preserves_current_state_dict(self) -> None:
        state_dict = {
            "heads.future_yaw_delta.fc1.weight": torch.tensor([1.0], dtype=torch.float32),
            "heads.future_yaw_delta.out.bias": torch.tensor([2.0], dtype=torch.float32),
            "heads.future_speed.out.bias": torch.tensor([3.0], dtype=torch.float32),
        }

        remapped = remap_legacy_state_dict_keys(state_dict)

        self.assertIn("heads.future_yaw_delta.fc1.weight", remapped)
        self.assertIn("heads.future_yaw_delta.out.bias", remapped)
        self.assertTrue(torch.equal(remapped["heads.future_yaw_delta.fc1.weight"], torch.tensor([1.0], dtype=torch.float32)))
        self.assertTrue(torch.equal(remapped["heads.future_yaw_delta.out.bias"], torch.tensor([2.0], dtype=torch.float32)))

    def test_remap_legacy_state_dict_keys_maps_legacy_steer_head(self) -> None:
        state_dict = {
            "heads.steer.fc1.weight": torch.tensor([1.0], dtype=torch.float32),
            "heads.steer.out.bias": torch.tensor([2.0], dtype=torch.float32),
            "heads.future_speed.out.bias": torch.tensor([3.0], dtype=torch.float32),
        }

        remapped = remap_legacy_state_dict_keys(state_dict)

        self.assertIn("heads.future_yaw_delta.fc1.weight", remapped)
        self.assertIn("heads.future_yaw_delta.out.bias", remapped)
        self.assertNotIn("heads.steer.fc1.weight", remapped)
        self.assertNotIn("heads.steer.out.bias", remapped)
        self.assertTrue(torch.equal(remapped["heads.future_yaw_delta.fc1.weight"], torch.tensor([1.0], dtype=torch.float32)))
        self.assertTrue(torch.equal(remapped["heads.future_yaw_delta.out.bias"], torch.tensor([2.0], dtype=torch.float32)))

    def test_resolve_checkpoint_head_specs_accepts_legacy_steer_alias(self) -> None:
        resolved = resolve_checkpoint_head_specs({
            "head_layout": [
                {
                    "name": "steer",
                    "kind": "control",
                    "output_dim": 1,
                    "used_for_control": True,
                },
                ALL_HEAD_SPECS_BY_NAME["future_speed"].layout_metadata(),
            ],
        })

        self.assertEqual([spec.name for spec in resolved], ["future_yaw_delta", "future_speed"])


if __name__ == "__main__":
    unittest.main()
