from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training_runtime import (
    ALLOWED_STATE_INPUT_HEADS,
    ALLOWED_CONSISTENCY_KEYS,
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

[training.loss_weights]
future_yaw_delta = 2.2
future_speed = 1.5

[training.consistency]
yaw_delta_vs_yaw_rate_weight = 1.5
future_speed_vs_delta_speed_weight = 0.5
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
            manager = TrainingManager(config_path)
            page = manager.page_config()

            self.assertEqual(page["epochs"], 15)
            self.assertEqual(page["learningRate"], 0.001)
            self.assertEqual(page["lossWeights"]["future_yaw_delta"], 2.2)
            self.assertEqual(page["consistency"]["future_speed_vs_delta_speed_weight"], 0.5)
            self.assertEqual(page["allowedLossWeightKeys"], ALLOWED_LOSS_WEIGHT_KEYS)
            self.assertEqual(page["allowedConsistencyKeys"], ALLOWED_CONSISTENCY_KEYS)
            self.assertIn("turnOversampling", page)
            self.assertIn("yawLossWeighting", page)
            self.assertIn("stateInputs", page)
            self.assertIn("trainRunIds", page)
            self.assertIn("valRunIds", page)
            self.assertIn("allowedStateInputHeads", page)
            self.assertEqual(page["allowedStateInputHeads"], ALLOWED_STATE_INPUT_HEADS)
            self.assertTrue(page["pythonBin"])
            self.assertTrue(page["trainScript"].endswith("train.py"))
            self.assertEqual(page["historyLimit"], 100)

    def test_delete_job_removes_terminal_job_and_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = write_training_runtime_config(Path(tmp))
            manager = TrainingManager(config_path)
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
                "consistency": {},
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
            manager = TrainingManager(config_path)

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
                    "consistency": {},
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
            manager = TrainingManager(config_path)

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
            manager = TrainingManager(config_path)
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
                "trainRunIds": ["run-a"],
                "valRunIds": ["run-b"],
                "lossWeights": {"future_yaw_delta": 2.5},
                "consistency": {"yaw_delta_vs_yaw_rate_weight": 1.25},
                "turnOversampling": {"enabled": True},
                "yawLossWeighting": {"enabled": False},
                "stateInputs": {"currentSpeed": {"enabled": True, "cap": 25.0, "heads": ["future_speed"]}},
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
            self.assertEqual(created["trainRunIds"], failed_job["trainRunIds"])
            self.assertEqual(created["valRunIds"], failed_job["valRunIds"])
            self.assertEqual(created["lossWeights"], failed_job["lossWeights"])
            self.assertEqual(created["consistency"], failed_job["consistency"])
            self.assertEqual(created["turnOversampling"], failed_job["turnOversampling"])
            self.assertEqual(created["yawLossWeighting"], failed_job["yawLossWeighting"])
            self.assertEqual(created["stateInputs"], failed_job["stateInputs"])
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
            manager = TrainingManager(config_path)
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
                "consistency": {},
                "turnOversampling": {},
                "yawLossWeighting": {},
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
            "trainRunIds": ["run-a", "run-b"],
            "valRunIds": ["run-c"],
            "turnOversampling": {
                "enabled": True,
                "sharp_turn_weight": 3.0,
            },
            "yawLossWeighting": {
                "enabled": True,
                "tau": 0.5,
            },
            "stateInputs": {
                "currentSpeed": {
                    "enabled": True,
                    "cap": 25.0,
                    "heads": ["future_speed", "delta_speed"],
                },
                "hasLeadVehicle": {
                    "enabled": True,
                    "heads": ["move_intent"],
                },
            },
        })

        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual(job["epochs"], 15)
        self.assertEqual(job["trainRunIds"], ["run-a", "run-b"])
        self.assertEqual(job["valRunIds"], ["run-c"])
        self.assertEqual(job["widthMultiplier"], 1.5)
        self.assertTrue(job["turnOversampling"]["enabled"])
        self.assertEqual(job["yawLossWeighting"]["tau"], 0.5)
        self.assertEqual(job["stateInputs"]["currentSpeed"]["heads"], ["future_speed", "delta_speed"])
        self.assertEqual(job["stateInputs"]["hasLeadVehicle"]["heads"], ["move_intent"])

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


if __name__ == "__main__":
    unittest.main()
