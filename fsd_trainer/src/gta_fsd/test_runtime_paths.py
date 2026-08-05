from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import resolve_data_root, resolve_data_root_child
from inference import DEFAULT_CONFIG_PATH, load_config as load_inference_config
from server import discover_models, load_training_runs_dir
from state_inputs import (
    PARKING_HEADING_ERROR_KEY,
    PARKING_LATERAL_ERROR_KEY,
    PARKING_LONGITUDINAL_ERROR_KEY,
)
from train import load_config as load_train_config
from training_runtime import _resolve_jobs_dir


def write_runnable_parking_config(directory: Path) -> Path:
    config_path = directory / "train_config.toml"
    config_text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    config_text = config_text.replace("train_run_ids = []", "train_run_ids = ['parking-train']")
    config_text = config_text.replace("val_run_ids = []", "val_run_ids = ['parking-val']")
    config_path.write_text(config_text, encoding="utf-8")
    return config_path


class RuntimePathTests(unittest.TestCase):
    def test_data_root_child_rejects_missing_config_without_environment_override(self) -> None:
        with patch.dict(os.environ, {"FSD_DATA_ROOT": ""}):
            with self.assertRaisesRegex(
                ValueError,
                "path must be configured or FSD_DATA_ROOT must be set for training_runs",
            ):
                resolve_data_root_child(None, "training_runs")

    def test_data_root_environment_keeps_runtime_artifacts_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "parking-data"
            config_path = write_runnable_parking_config(Path(tmp))
            with patch.dict(os.environ, {"FSD_DATA_ROOT": str(data_root)}):
                self.assertEqual(resolve_data_root(r"S:\fsd_fivem_data"), str(data_root))
                self.assertEqual(
                    resolve_data_root_child(r"S:\fsd_fivem_data\training_runs", "training_runs"),
                    str(data_root / "training_runs"),
                )

                train_config = load_train_config(config_path)
                inference_config = load_inference_config(config_path, require_checkpoint=False)
                self.assertEqual(train_config.dataset.data_root, str(data_root))
                self.assertEqual(train_config.output.base_dir, str(data_root / "training_runs"))
                self.assertEqual(inference_config.data_root, str(data_root))
                self.assertEqual(
                    _resolve_jobs_dir(config_path),
                    data_root / "training_jobs",
                )
                self.assertEqual(
                    load_training_runs_dir(config_path),
                    (data_root / "training_runs").resolve(),
                )

    def test_empty_training_runs_directory_has_no_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "parking-data"
            with patch.dict(os.environ, {"FSD_DATA_ROOT": str(data_root)}):
                self.assertEqual(discover_models(DEFAULT_CONFIG_PATH), [])

    def test_checked_in_parking_caps_cover_the_complete_curriculum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = write_runnable_parking_config(Path(tmp))
            with patch.dict(os.environ, {"FSD_DATA_ROOT": tmp}):
                config = load_train_config(config_path)

        self.assertEqual(config.state_inputs.spec(PARKING_LONGITUDINAL_ERROR_KEY).cap, 17.0)
        self.assertEqual(config.state_inputs.spec(PARKING_LATERAL_ERROR_KEY).cap, 3.0)
        self.assertEqual(config.state_inputs.spec(PARKING_HEADING_ERROR_KEY).cap, 20.0)


if __name__ == "__main__":
    unittest.main()
