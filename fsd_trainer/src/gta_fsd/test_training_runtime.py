from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training_runtime import (
    ALLOWED_EARLY_STOPPING_METRICS,
    ALLOWED_LOSS_WEIGHT_KEYS,
    TrainingManager,
    _parse_job_specs,
)


def write_training_runtime_config(root: Path) -> Path:
    project_root = root / "project"
    config_dir = project_root / "fsd_trainer"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "train_config.toml"
    config_path.write_text(
        """
[backend]
[backend.training]
jobs_dir = "training_jobs"

[dataset]
data_root = "training_data"
window_size = 5
frame_stride = 2
sample_stride = 10

[training]
epochs = 15
learning_rate = 0.001
early_stopping_metric = "drive_score"
smooth_l1_beta = 0.1

[training.target_loss_weights]
steering = 2.2
future_yaw_rate = 1.5

[training.consistency]
yaw_delta_vs_yaw_rate_weight = 1.5
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path


class TrainingRuntimeTests(unittest.TestCase):
    def test_parse_job_specs_accepts_single_array_and_wrapped(self) -> None:
        self.assertEqual(len(_parse_job_specs({"name": "a"})), 1)
        self.assertEqual(len(_parse_job_specs([{"name": "a"}, {"name": "b"}])), 2)
        self.assertEqual(len(_parse_job_specs({"jobs": [{"name": "a"}, {"name": "b"}]})), 2)

    def test_training_manager_page_config_exposes_expected_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = write_training_runtime_config(Path(tmp))
            manager = TrainingManager(config_path, start_worker=False)
            page = manager.page_config()

            self.assertEqual(page["epochs"], 15)
            self.assertEqual(page["learningRate"], 0.001)
            self.assertEqual(page["smoothL1Beta"], 0.1)
            self.assertEqual(page["earlyStoppingMetric"], "drive_score")
            self.assertEqual(page["lossWeights"]["steering"], 2.2)
            self.assertEqual(page["lossWeights"]["future_yaw_rate"], 1.5)
            self.assertEqual(page["lossWeights"]["acceleration"], 1.0)
            self.assertEqual(page["allowedLossWeightKeys"], ALLOWED_LOSS_WEIGHT_KEYS)
            self.assertEqual(page["allowedEarlyStoppingMetrics"], ALLOWED_EARLY_STOPPING_METRICS)
            self.assertNotIn("consistency", page)
            self.assertNotIn("allowedConsistencyKeys", page)
            self.assertNotIn("yawLossWeighting", page)
            self.assertIn("turnOversampling", page)
            self.assertIn("stateInputs", page)
            self.assertIn("trainRunIds", page)
            self.assertIn("valRunIds", page)
            self.assertNotIn("allowedStateInputHeads", page)
            self.assertTrue(page["pythonBin"])
            self.assertTrue(page["trainScript"].endswith("train.py"))
            self.assertEqual(page["historyLimit"], 100)

    def test_training_manager_page_config_reads_legacy_loss_weight_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = write_training_runtime_config(Path(tmp))
            text = config_path.read_text(encoding="utf-8")
            text = text.replace("[training.target_loss_weights]", "[training.loss_weights]")
            config_path.write_text(text, encoding="utf-8")

            manager = TrainingManager(config_path, start_worker=False)
            page = manager.page_config()

            self.assertEqual(page["lossWeights"]["steering"], 2.2)
            self.assertEqual(page["lossWeights"]["future_yaw_rate"], 1.5)

    def test_delete_job_removes_terminal_job_and_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = write_training_runtime_config(Path(tmp))
            manager = TrainingManager(config_path, start_worker=False)
            job_dir = manager._jobs_dir / "train-terminal-delete"
            job_dir.mkdir(parents=True, exist_ok=True)
            job = {
                "id": "train-terminal-delete",
                "name": "delete-me",
                "notes": "",
                "status": "failed",
                "epochs": None,
                "learningRate": None,
                "lossWeights": {},
                "createdAt": "2026-04-21T00:00:00.000000000Z",
                "lastUpdatedAt": "2026-04-21T00:00:00.000000000Z",
                "finishedAt": "2026-04-21T00:00:00.000000000Z",
                "configPath": str(job_dir / "derived_train_config.toml"),
                "logPath": str(job_dir / "train.log"),
                "jobDir": str(job_dir),
                "runDir": "",
                "runMetricsPath": "",
                "exitCode": 1,
                "error": "boom",
                "command": [],
                "cancelRequested": False,
                "stopRequested": False,
            }
            (job_dir / "job.json").write_text(json.dumps({"job": job}) + "\n", encoding="utf-8")
            with manager._lock:
                manager._jobs[job["id"]] = dict(job)

            deleted = manager.delete_job(job["id"])

            self.assertEqual(deleted["id"], job["id"])
            self.assertFalse(job_dir.exists())
            with manager._lock:
                self.assertNotIn(job["id"], manager._jobs)

    def test_clear_history_removes_only_terminal_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = write_training_runtime_config(Path(tmp))
            manager = TrainingManager(config_path, start_worker=False)

            def make_job(job_id: str, status: str) -> dict[str, object]:
                job_dir = manager._jobs_dir / job_id
                job_dir.mkdir(parents=True, exist_ok=True)
                job = {
                    "id": job_id,
                    "name": job_id,
                    "notes": "",
                    "status": status,
                    "epochs": None,
                    "learningRate": None,
                    "lossWeights": {},
                    "createdAt": "2026-04-21T00:00:00.000000000Z",
                    "lastUpdatedAt": "2026-04-21T00:00:00.000000000Z",
                    "finishedAt": "2026-04-21T00:00:00.000000000Z" if status in {"completed", "failed", "canceled", "stopped"} else "",
                    "configPath": str(job_dir / "derived_train_config.toml"),
                    "logPath": str(job_dir / "train.log"),
                    "jobDir": str(job_dir),
                    "runDir": "",
                    "runMetricsPath": "",
                    "exitCode": 0 if status == "completed" else None,
                    "error": "",
                    "command": [],
                    "cancelRequested": False,
                    "stopRequested": False,
                }
                (job_dir / "job.json").write_text(json.dumps({"job": job}) + "\n", encoding="utf-8")
                return job

            terminal_job = make_job("train-terminal-clear", "completed")
            running_job = make_job("train-running-keep", "running")
            with manager._lock:
                manager._jobs[terminal_job["id"]] = dict(terminal_job)
                manager._jobs[running_job["id"]] = dict(running_job)

            result = manager.clear_history()

            self.assertEqual(result["status"], "cleared")
            self.assertEqual(result["deletedCount"], 1)
            self.assertEqual(result["jobs"][0]["id"], terminal_job["id"])
            self.assertFalse((manager._jobs_dir / terminal_job["id"]).exists())
            self.assertTrue((manager._jobs_dir / running_job["id"]).exists())
            with manager._lock:
                self.assertNotIn(terminal_job["id"], manager._jobs)
                self.assertIn(running_job["id"], manager._jobs)

    def test_enqueue_uses_provided_name_for_job_id_with_numeric_suffix_on_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = write_training_runtime_config(Path(tmp))
            manager = TrainingManager(config_path, start_worker=False)

            created = manager.enqueue([
                {"name": "YAW", "learningRate": 0.001},
                {"name": "YAW", "learningRate": 0.002},
            ])

            self.assertEqual(created[0]["id"], "yaw")
            self.assertEqual(created[1]["id"], "yaw-2")
            self.assertTrue((manager._jobs_dir / "yaw").is_dir())
            self.assertTrue((manager._jobs_dir / "yaw-2").is_dir())

    def test_requeue_failed_job_clones_spec_into_new_queued_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = write_training_runtime_config(Path(tmp))
            manager = TrainingManager(config_path, start_worker=False)
            job_dir = manager._jobs_dir / "yaw-failed"
            job_dir.mkdir(parents=True, exist_ok=True)
            failed_job = {
                "id": "yaw-failed",
                "name": "Yaw",
                "notes": "retry me",
                "status": "failed",
                "epochs": 12,
                "learningRate": 0.001,
                "widthMultiplier": 1.5,
                "smoothL1Beta": 0.1,
                "earlyStoppingMetric": "drive_score",
                "trainRunIds": ["run-a"],
                "valRunIds": ["run-b"],
                "lossWeights": {"steering": 2.5},
                "consistency": {"yaw_delta_vs_yaw_rate_weight": 1.25},
                "turnOversampling": {"enabled": True},
                "yawLossWeighting": {"enabled": False},
                "stateInputs": {"currentSpeed": {"enabled": True, "cap": 25.0}},
                "createdAt": "2026-04-21T00:00:00.000000000Z",
                "lastUpdatedAt": "2026-04-21T00:01:00.000000000Z",
                "finishedAt": "2026-04-21T00:01:00.000000000Z",
                "configPath": str(job_dir / "derived_train_config.toml"),
                "logPath": str(job_dir / "train.log"),
                "jobDir": str(job_dir),
                "runDir": "old-run",
                "runMetricsPath": "old-metrics",
                "exitCode": 1,
                "error": "boom",
                "command": ["python", "train.py"],
                "cancelRequested": False,
                "stopRequested": False,
            }
            (job_dir / "job.json").write_text(json.dumps({"job": failed_job}) + "\n", encoding="utf-8")
            with manager._lock:
                manager._jobs[failed_job["id"]] = dict(failed_job)

            created = manager.requeue(failed_job["id"])

            self.assertEqual(created["status"], "queued")
            self.assertNotEqual(created["id"], failed_job["id"])
            self.assertTrue(created["id"].startswith("yaw"))
            self.assertEqual(created["name"], failed_job["name"])
            self.assertEqual(created["notes"], failed_job["notes"])
            self.assertEqual(created["epochs"], failed_job["epochs"])
            self.assertEqual(created["learningRate"], failed_job["learningRate"])
            self.assertEqual(created["widthMultiplier"], failed_job["widthMultiplier"])
            self.assertEqual(created["smoothL1Beta"], failed_job["smoothL1Beta"])
            self.assertEqual(created["earlyStoppingMetric"], failed_job["earlyStoppingMetric"])
            self.assertEqual(created["trainRunIds"], failed_job["trainRunIds"])
            self.assertEqual(created["valRunIds"], failed_job["valRunIds"])
            self.assertEqual(created["lossWeights"], failed_job["lossWeights"])
            self.assertEqual(created["turnOversampling"], failed_job["turnOversampling"])
            self.assertEqual(created["stateInputs"], failed_job["stateInputs"])
            self.assertNotIn("consistency", created)
            self.assertNotIn("yawLossWeighting", created)
            self.assertEqual(created["runDir"], "")
            self.assertEqual(created["runMetricsPath"], "")
            self.assertEqual(created["error"], "")
            self.assertIsNone(created["exitCode"])
            original = manager.get_job(failed_job["id"])
            self.assertEqual(original["status"], "failed")
            self.assertEqual(original["runDir"], "old-run")

    def test_requeue_rejects_completed_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = write_training_runtime_config(Path(tmp))
            manager = TrainingManager(config_path, start_worker=False)
            job_dir = manager._jobs_dir / "yaw-complete"
            job_dir.mkdir(parents=True, exist_ok=True)
            completed_job = {
                "id": "yaw-complete",
                "name": "Yaw",
                "notes": "",
                "status": "completed",
                "epochs": 12,
                "learningRate": 0.001,
                "widthMultiplier": 1.5,
                "trainRunIds": None,
                "valRunIds": None,
                "lossWeights": {},
                "turnOversampling": {},
                "stateInputs": {},
                "createdAt": "2026-04-21T00:00:00.000000000Z",
                "lastUpdatedAt": "2026-04-21T00:01:00.000000000Z",
                "finishedAt": "2026-04-21T00:01:00.000000000Z",
                "configPath": str(job_dir / "derived_train_config.toml"),
                "logPath": str(job_dir / "train.log"),
                "jobDir": str(job_dir),
                "runDir": "",
                "runMetricsPath": "",
                "exitCode": 0,
                "error": "",
                "command": [],
                "cancelRequested": False,
                "stopRequested": False,
            }
            (job_dir / "job.json").write_text(json.dumps({"job": completed_job}) + "\n", encoding="utf-8")
            with manager._lock:
                manager._jobs[completed_job["id"]] = dict(completed_job)

            with self.assertRaisesRegex(RuntimeError, "failed or stopped"):
                manager.requeue(completed_job["id"])

    def test_parse_job_specs_accepts_sampling_routes_and_run_selection(self) -> None:
        jobs = _parse_job_specs({
            "name": "YAW",
            "epochs": 15,
            "learningRate": 0.001,
            "widthMultiplier": 1.5,
            "smoothL1Beta": 0.2,
            "earlyStoppingMetric": "control_loss",
            "trainRunIds": ["run-a", "run-b"],
            "valRunIds": ["run-c"],
            "lossWeights": {
                "steering": 2.0,
                "future_yaw_rate": 0.5,
            },
            "turnOversampling": {
                "enabled": True,
                "sharp_turn_weight": 3.0,
            },
            "stateInputs": {
                "currentSpeed": {
                    "enabled": True,
                    "cap": 25.0,
                },
                "hasLeadVehicle": {
                    "enabled": True,
                },
            },
        })

        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual(job["epochs"], 15)
        self.assertEqual(job["trainRunIds"], ["run-a", "run-b"])
        self.assertEqual(job["valRunIds"], ["run-c"])
        self.assertEqual(job["widthMultiplier"], 1.5)
        self.assertEqual(job["smoothL1Beta"], 0.2)
        self.assertEqual(job["earlyStoppingMetric"], "control_loss")
        self.assertEqual(job["lossWeights"]["steering"], 2.0)
        self.assertEqual(job["lossWeights"]["future_yaw_rate"], 0.5)
        self.assertTrue(job["turnOversampling"]["enabled"])
        self.assertNotIn("heads", job["stateInputs"]["currentSpeed"])
        self.assertNotIn("heads", job["stateInputs"]["hasLeadVehicle"])

    def test_parse_job_specs_rejects_removed_training_fields(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "removed training job field"):
            _parse_job_specs({
                "name": "stale",
                "consistency": {
                    "yaw_delta_vs_yaw_rate_weight": 1.25,
                },
            })

    def test_parse_job_specs_rejects_legacy_state_input_heads(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "heads has been removed"):
            _parse_job_specs({
                "stateInputs": {
                    "currentSpeed": {
                        "enabled": True,
                        "heads": ["future_speed"],
                    },
                },
            })

    def test_derived_config_drops_removed_training_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = write_training_runtime_config(Path(tmp))
            manager = TrainingManager(config_path, start_worker=False)
            job_dir = manager._jobs_dir / "clean-derived"
            job_dir.mkdir(parents=True, exist_ok=True)
            job = {
                "configPath": str(job_dir / "derived_train_config.toml"),
                "learningRate": 0.0007,
                "lossWeights": {"future_yaw_rate": 0.5},
                "consistency": {"yaw_delta_vs_yaw_rate_weight": 9.0},
                "yawLossWeighting": {"enabled": True, "tau": 0.5},
                "turnOversampling": {"enabled": True, "sharp_turn_threshold": 0.15, "medium_turn_threshold": 0.30},
            }

            manager._write_derived_config(job)

            derived = Path(job["configPath"]).read_text(encoding="utf-8")
            self.assertIn("learning_rate = 0.0007", derived)
            self.assertIn("future_yaw_rate = 0.5", derived)
            self.assertIn("[training.target_loss_weights]", derived)
            self.assertNotIn("[training.loss_weights]", derived)
            self.assertIn("[loader.turn_oversampling]", derived)
            self.assertNotIn("[training.consistency]", derived)
            self.assertNotIn("yaw_delta_vs_yaw_rate_weight", derived)
            self.assertNotIn("yaw_loss_weighting", derived)

    def test_parse_job_specs_normalizes_turn_threshold_order(self) -> None:
        jobs = _parse_job_specs({
            "name": "YAW",
            "turnOversampling": {
                "enabled": True,
                "light_turn_threshold": 0.05,
                "medium_turn_threshold": 0.30,
                "sharp_turn_threshold": 0.15,
            },
        })

        job = jobs[0]
        self.assertEqual(job["turnOversampling"]["medium_turn_threshold"], 0.30)
        self.assertEqual(job["turnOversampling"]["sharp_turn_threshold"], 0.30)

    def test_parse_job_specs_rejects_stale_loss_weight_names(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unknown: yaw_rate"):
            _parse_job_specs({
                "name": "stale",
                "lossWeights": {"yaw_rate": 1.0},
            })


if __name__ == "__main__":
    unittest.main()
