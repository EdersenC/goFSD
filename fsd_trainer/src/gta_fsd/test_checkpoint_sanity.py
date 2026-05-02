from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from checkpoint_sanity import (
    build_debug_windows,
    resolve_dataset_run_id,
    resolve_debug_dir,
    resolve_latest_debug_dir,
    summarize_control_ranges,
)
from train import DatasetConfig, LoaderConfig, ModelConfig, OutputConfig, TrainConfig, TrainingConfig


def make_train_config() -> TrainConfig:
    return TrainConfig(
        dataset=DatasetConfig(
            data_root="/tmp/data",
            train_run_ids=("train-a", "train-b"),
            val_run_ids=("val-a",),
            train_run_paths=("/tmp/data/runs/train-a", "/tmp/data/runs/train-b"),
            val_run_paths=("/tmp/data/runs/val-a",),
            image_width=480,
            image_height=480,
            window_size=5,
            frame_stride=2,
            sample_stride=10,
        ),
        output=OutputConfig(base_dir="/tmp/out"),
        model=ModelConfig(width_multiplier=1.0, telemetry_hidden_dim=128),
        training=TrainingConfig(
            device="cpu",
            epochs=1,
            learning_rate=1e-3,
            early_stopping_metric="val_loss",
            early_stopping_patience=1,
            early_stopping_min_delta=0.0,
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


class CheckpointSanityTests(unittest.TestCase):
    def test_resolve_dataset_run_id_prefers_override_then_inference_then_val(self) -> None:
        config = make_train_config()

        self.assertEqual(resolve_dataset_run_id(config, "infer-a", "manual-a"), "manual-a")
        self.assertEqual(resolve_dataset_run_id(config, "infer-a", None), "infer-a")
        self.assertEqual(resolve_dataset_run_id(config, None, None), "val-a")

    def test_build_debug_windows_uses_frame_stride(self) -> None:
        frame_paths = [Path(f"frame-{index:03d}.jpg") for index in range(12)]

        windows = build_debug_windows(frame_paths, window_size=3, frame_stride=2, limit=2)

        self.assertEqual(
            windows,
            [
                [frame_paths[0], frame_paths[2], frame_paths[4]],
                [frame_paths[1], frame_paths[3], frame_paths[5]],
            ],
        )

    def test_build_debug_windows_rejects_too_few_frames(self) -> None:
        frame_paths = [Path(f"frame-{index:03d}.jpg") for index in range(4)]

        with self.assertRaisesRegex(ValueError, "Not enough debug frames"):
            build_debug_windows(frame_paths, window_size=3, frame_stride=2, limit=1)

    def test_summarize_control_ranges_marks_flat_outputs(self) -> None:
        summary = summarize_control_ranges([
            {
                "control_target_names": ["steering", "acceleration"],
                "pred_controls": [[[0.1, 0.5], [0.1, 0.50001]]],
            },
            {
                "control_target_names": ["steering", "acceleration"],
                "pred_controls": [[[0.1, 0.50002], [0.1, 0.50003]]],
            },
        ])

        self.assertTrue(bool(summary["steering"]["flat"]))
        self.assertTrue(bool(summary["acceleration"]["flat"]))

    def test_resolve_debug_dir_accepts_explicit_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            debug_dir = Path(tmp)

            resolved = resolve_debug_dir(debug_dir)

            self.assertEqual(resolved, debug_dir.resolve())

    def test_resolve_latest_debug_dir_selects_latest_dump(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            older = root / "older"
            newer = root / "newer"
            older.mkdir()
            newer.mkdir()
            now = time.time()
            os.utime(older, (now - 10, now - 10))
            os.utime(newer, (now, now))

            resolved = resolve_latest_debug_dir(root)
            self.assertEqual(resolved, newer)


if __name__ == "__main__":
    unittest.main()
