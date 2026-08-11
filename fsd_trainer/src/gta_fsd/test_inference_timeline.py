from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from config import DEFAULT_AUX_TARGET_NAMES
from control_contract import (
    STOP_SIGN_CONTROL_TARGET_NAMES,
    PLANNER_FORMAT,
    PLANNER_FORMAT_VERSION,
    control_contract_metadata,
    derive_control_horizon_dt_ms,
)
from inference import (
    load_checkpoint,
    require_checkpoint_image_size,
    resolve_checkpoint_timeline,
    run_sample_inference,
)
from target_transforms import build_target_transform_registry, target_transform_metadata


def timeline_checkpoint() -> dict[str, object]:
    future_offsets = [1, 3, 5]
    sample_interval_ms = 40
    transforms = build_target_transform_registry(
        STOP_SIGN_CONTROL_TARGET_NAMES + DEFAULT_AUX_TARGET_NAMES
    )
    return {
        "planner_format": PLANNER_FORMAT,
        "planner_format_version": PLANNER_FORMAT_VERSION,
        "control_contract": control_contract_metadata(),
        "frame_window_size": 3,
        "image_size": {"width": 96, "height": 64},
        "image_offsets": [-6, -2, 0],
        "telemetry_offsets": [-5, -2, 0],
        "future_offsets": future_offsets,
        "telemetry_sample_interval_ms": sample_interval_ms,
        "control_horizon_dt_ms": list(
            derive_control_horizon_dt_ms(future_offsets, sample_interval_ms)
        ),
        "control_target_names": list(STOP_SIGN_CONTROL_TARGET_NAMES),
        "aux_target_names": list(DEFAULT_AUX_TARGET_NAMES),
        "telemetry_feature_names": ["current_speed"],
        "target_transforms": target_transform_metadata(transforms),
        "model_state_dict": {},
    }


class InferenceTimelineTests(unittest.TestCase):
    def test_sample_inference_constructs_dataset_from_non_default_checkpoint_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "non-default-timeline.pt"
            torch.save(timeline_checkpoint(), checkpoint_path)
            checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
            timeline = resolve_checkpoint_timeline(checkpoint)

            with patch("inference.FsdDataset") as dataset_type:
                dataset_type.return_value.__len__.return_value = 0

                with self.assertRaisesRegex(IndexError, "sample_index out of range"):
                    run_sample_inference(
                        object(),
                        torch.device("cpu"),
                        data_root="dataset-root",
                        run_id="stop-sign-run",
                        sample_index=0,
                        image_size=(96, 64),
                        expected_window_size=len(timeline.image_offsets),
                        timeline=timeline,
                        control_target_names=STOP_SIGN_CONTROL_TARGET_NAMES,
                        aux_target_names=DEFAULT_AUX_TARGET_NAMES,
                        target_transforms=None,
                    )

            dataset_type.assert_called_once()
            dataset_args = dataset_type.call_args.kwargs
        self.assertEqual(dataset_args["image_offsets"], (-6, -2, 0))
        self.assertEqual(dataset_args["telemetry_offsets"], (-5, -2, 0))
        self.assertEqual(dataset_args["future_offsets"], (1, 3, 5))
        self.assertEqual(dataset_args["telemetry_sample_interval_ms"], 40)

    def test_load_checkpoint_rejects_missing_timeline_metadata(self) -> None:
        for missing_key in (
            "image_offsets",
            "image_size",
            "telemetry_offsets",
            "future_offsets",
            "telemetry_sample_interval_ms",
        ):
            with self.subTest(missing_key=missing_key), tempfile.TemporaryDirectory() as tmp:
                checkpoint = copy.deepcopy(timeline_checkpoint())
                checkpoint.pop(missing_key)
                checkpoint_path = Path(tmp) / "missing-timeline.pt"
                torch.save(checkpoint, checkpoint_path)

                with self.assertRaisesRegex(ValueError, missing_key):
                    load_checkpoint(checkpoint_path, torch.device("cpu"))

    def test_load_checkpoint_rejects_malformed_timeline_metadata(self) -> None:
        cases = (
            ("image_offsets", [-6, 0, -2], "image_offsets must be strictly increasing"),
            ("image_size", {"width": 96, "height": 0}, "image_size.height must be a positive integer"),
            ("telemetry_offsets", [-5, -2], "telemetry_offsets must end at 0"),
            ("telemetry_sample_interval_ms", 40.0, "telemetry_sample_interval_ms must be a positive integer"),
            ("frame_window_size", 5, "frame_window_size must match image_offsets"),
        )
        for key, value, message in cases:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                checkpoint = copy.deepcopy(timeline_checkpoint())
                checkpoint[key] = value
                checkpoint_path = Path(tmp) / "malformed-timeline.pt"
                torch.save(checkpoint, checkpoint_path)

                with self.assertRaisesRegex(ValueError, message):
                    load_checkpoint(checkpoint_path, torch.device("cpu"))

    def test_runtime_image_size_must_match_checkpoint_truth(self) -> None:
        checkpoint = timeline_checkpoint()
        self.assertEqual(require_checkpoint_image_size(checkpoint, (96, 64)), (96, 64))
        with self.assertRaisesRegex(ValueError, "Config/checkpoint image-size mismatch"):
            require_checkpoint_image_size(checkpoint, (480, 480))


if __name__ == "__main__":
    unittest.main()
