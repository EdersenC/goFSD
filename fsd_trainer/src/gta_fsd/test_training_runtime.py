from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from train import _resolve_training_run_ids
from training_runtime import (
    ALLOWED_EARLY_STOPPING_METRICS,
    ALLOWED_LOSS_WEIGHT_KEYS,
    TRAINING_EVENT_PREFIX,
    TrainingJobDuplicateError,
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

[output]
base_dir = "training_runs"

[training]
epochs = 15
learning_rate = 0.001
early_stopping_metric = "drive_score"
smooth_l1_beta = 0.1

[training.target_loss_weights]
future_speed_mps = 2.2
expert_brake = 1.5

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

    def test_parse_job_specs_rejects_fractional_and_boolean_epoch_counts(self) -> None:
        for value in (1.5, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(RuntimeError, "integer >= 1"):
                    _parse_job_specs({"epochs": value})

    def test_training_manager_page_config_exposes_expected_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = write_training_runtime_config(Path(tmp))
            manager = TrainingManager(config_path, start_worker=False)
            page = manager.page_config()

            self.assertEqual(page["epochs"], 15)
            self.assertEqual(page["learningRate"], 0.001)
            self.assertEqual(page["smoothL1Beta"], 0.1)
            self.assertEqual(page["earlyStoppingMetric"], "drive_score")
            self.assertEqual(page["lossWeights"]["future_speed_mps"], 2.2)
            self.assertEqual(page["lossWeights"]["expert_brake"], 1.5)
            self.assertEqual(page["lossWeights"]["stop_intent"], 1.0)
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

            self.assertEqual(page["lossWeights"]["future_speed_mps"], 2.2)
            self.assertEqual(page["lossWeights"]["expert_brake"], 1.5)

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
                "lossWeights": {"future_speed_mps": 2.5},
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
                "future_speed_mps": 2.0,
                "expert_brake": 0.5,
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
        self.assertEqual(job["lossWeights"]["future_speed_mps"], 2.0)
        self.assertEqual(job["lossWeights"]["expert_brake"], 0.5)
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
                "lossWeights": {"expert_brake": 0.5},
                "consistency": {"yaw_delta_vs_yaw_rate_weight": 9.0},
                "yawLossWeighting": {"enabled": True, "tau": 0.5},
                "turnOversampling": {"enabled": True, "sharp_turn_threshold": 0.15, "medium_turn_threshold": 0.30},
            }

            manager._write_derived_config(job)

            derived = Path(job["configPath"]).read_text(encoding="utf-8")
            self.assertIn("learning_rate = 0.0007", derived)
            self.assertIn("expert_brake = 0.5", derived)
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

    def test_run_selection_rejects_paths_duplicates_and_train_validation_overlap(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "run folder name"):
            _parse_job_specs({"trainRunIds": ["../outside"]})
        with self.assertRaisesRegex(RuntimeError, "duplicate run ids"):
            _parse_job_specs({"trainRunIds": ["run-a", "run-a"]})
        with self.assertRaisesRegex(RuntimeError, "must be separate"):
            _parse_job_specs({
                "trainRunIds": ["run-a", "run-b"],
                "valRunIds": ["run-b"],
            })

        with self.assertRaisesRegex(ValueError, "run folder names"):
            _resolve_training_run_ids(
                {"train_run_ids": ["..\\outside"]},
                "train_run_ids",
                fallback_key="run_id",
            )

    def test_enqueue_rejects_identical_active_jobs_without_partial_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingManager(write_training_runtime_config(Path(tmp)), start_worker=False)
            spec = {
                "name": "Parking baseline",
                "epochs": 2,
                "trainRunIds": ["run-a"],
                "valRunIds": ["run-b"],
            }

            first = manager.enqueue(spec)[0]

            with self.assertRaisesRegex(TrainingJobDuplicateError, "already queued or running"):
                manager.enqueue(spec)

            self.assertEqual([job["id"] for job in manager.list_jobs()], [first["id"]])
            second = manager.enqueue({**spec, "epochs": 3})[0]
            self.assertNotEqual(first["id"], second["id"])

    def test_enqueue_rolls_back_the_batch_when_job_persistence_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingManager(write_training_runtime_config(Path(tmp)), start_worker=False)
            persist_job = manager._persist_job

            def fail_second_job(job: dict[str, object]) -> None:
                if job["name"] == "second":
                    raise OSError("disk full")
                persist_job(job)

            manager._persist_job = fail_second_job  # type: ignore[method-assign]

            with self.assertRaisesRegex(OSError, "disk full"):
                manager.enqueue([{"name": "first"}, {"name": "second"}])

            self.assertEqual(manager.list_jobs(), [])
            self.assertEqual(list(manager._jobs_dir.iterdir()), [])

    def test_restart_restores_queued_jobs_and_marks_active_jobs_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = write_training_runtime_config(root)
            manager = TrainingManager(config_path, start_worker=False)
            queued, interrupted = manager.enqueue([
                {"name": "wait-for-restart", "epochs": 2},
                {"name": "active-at-restart", "epochs": 3},
            ])

            with manager._lock:
                active_job = manager._jobs[interrupted["id"]]
                active_job["status"] = "running"
                active_job["phase"] = "training"
                active_job["processId"] = 12345
                active_job["jobDir"] = str(root / "outside")
                active_job["configPath"] = str(root / "outside" / "config.toml")
                active_job["logPath"] = str(root / "outside" / "secret.log")
                manager._persist_job(active_job)
            active_job_path = manager._jobs_dir / interrupted["id"] / "job.json"
            persisted_payload = json.loads(active_job_path.read_text(encoding="utf-8"))
            persisted_payload["job"]["jobDir"] = str(root / "outside")
            persisted_payload["job"]["configPath"] = str(root / "outside" / "config.toml")
            persisted_payload["job"]["logPath"] = str(root / "outside" / "secret.log")
            active_job_path.write_text(json.dumps(persisted_payload), encoding="utf-8")

            restored = TrainingManager(config_path, start_worker=False)
            state = restored.state()

            self.assertEqual(state["queuedCount"], 1)
            self.assertEqual(state["queuedJobs"][0]["id"], queued["id"])
            self.assertIn("Restored after", state["queuedJobs"][0]["recoveryMessage"])
            interrupted_after_restart = restored.get_job(interrupted["id"])
            self.assertEqual(interrupted_after_restart["status"], "failed")
            self.assertEqual(interrupted_after_restart["phase"], "finished")
            self.assertIsNone(interrupted_after_restart["processId"])
            self.assertIn("Try again", interrupted_after_restart["error"])
            expected_job_dir = restored._jobs_dir / interrupted["id"]
            self.assertEqual(interrupted_after_restart["jobDir"], str(expected_job_dir))
            self.assertEqual(interrupted_after_restart["logPath"], str(expected_job_dir / "train.log"))

    def test_structured_training_events_persist_artifacts_and_epoch_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = write_training_runtime_config(Path(tmp))
            manager = TrainingManager(config_path, start_worker=False)
            job = manager.enqueue({"name": "progress", "epochs": 4})[0]
            run_dir = manager._runs_dir / "run-with-spaces"
            manager._record_training_output(
                job["id"],
                TRAINING_EVENT_PREFIX + json.dumps({
                    "type": "artifacts",
                    "runDir": str(run_dir),
                    "runMetricsPath": str(run_dir / "run_metrics.json"),
                }),
            )
            manager._record_training_output(
                job["id"],
                TRAINING_EVENT_PREFIX + json.dumps({
                    "type": "epoch",
                    "epoch": 2,
                    "totalEpochs": 4,
                    "bestEpoch": 1,
                    "elapsedSeconds": 12.5,
                    "metrics": {
                        "trainLoss": 0.4,
                        "valLoss": 0.5,
                        "ignored": "not numeric",
                    },
                }),
            )

            current = manager.get_job(job["id"])
            self.assertEqual(current["runDir"], str(run_dir))
            self.assertEqual(current["runMetricsPath"], str(run_dir / "run_metrics.json"))
            self.assertEqual(current["currentEpoch"], 2)
            self.assertEqual(current["totalEpochs"], 4)
            self.assertEqual(current["progressPercent"], 50.0)
            self.assertEqual(current["latestMetrics"], {"trainLoss": 0.4, "valLoss": 0.5})
            self.assertEqual(current["bestEpoch"], 1)
            self.assertEqual(current["elapsedSeconds"], 12.5)

            restored = TrainingManager(config_path, start_worker=False).get_job(job["id"])
            self.assertEqual(restored["currentEpoch"], 2)
            self.assertEqual(restored["progressPercent"], 50.0)
            self.assertEqual(restored["runMetricsPath"], str(run_dir / "run_metrics.json"))

    def test_subprocess_lifecycle_uses_unbuffered_events_and_surfaces_failure_detail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = write_training_runtime_config(root)
            manager = TrainingManager(config_path, start_worker=False)
            run_dir = manager._runs_dir / "run-subprocess"
            successful_script = root / "successful_training.py"
            successful_script.write_text(
                "import json\n"
                f"run_dir = {str(run_dir)!r}\n"
                "print('training_event=' + json.dumps({"
                "'type': 'artifacts', 'runDir': run_dir, "
                "'runMetricsPath': run_dir + '/run_metrics.json'}), flush=True)\n"
                "print('training_event=' + json.dumps({"
                "'type': 'epoch', 'epoch': 1, 'totalEpochs': 2, 'bestEpoch': 1, "
                "'elapsedSeconds': 1.25, 'metrics': {'valLoss': 0.25}}), flush=True)\n",
                encoding="utf-8",
            )
            manager._train_script = successful_script
            successful_job = manager.enqueue({"name": "successful", "epochs": 2})[0]
            manager._active_job_id = successful_job["id"]

            manager._run_job(successful_job["id"])

            completed = manager.get_job(successful_job["id"])
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["phase"], "finished")
            self.assertEqual(completed["currentEpoch"], 1)
            self.assertEqual(completed["progressPercent"], 100.0)
            self.assertEqual(completed["runMetricsPath"], str(run_dir / "run_metrics.json"))
            self.assertIn("-u", completed["command"])

            failing_script = root / "failing_training.py"
            failing_script.write_text(
                "import sys\n"
                "print('ValueError: validation data has no usable samples', file=sys.stderr, flush=True)\n"
                "raise SystemExit(7)\n",
                encoding="utf-8",
            )
            manager._train_script = failing_script
            failed_job = manager.enqueue({"name": "failed", "epochs": 2})[0]
            manager._active_job_id = failed_job["id"]

            manager._run_job(failed_job["id"])

            failed = manager.get_job(failed_job["id"])
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["exitCode"], 7)
            self.assertIn("validation data has no usable samples", failed["error"])

    def test_stop_terminates_an_active_training_process_and_persists_retry_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = TrainingManager(write_training_runtime_config(root), start_worker=False)
            training_script = root / "long_training.py"
            training_script.write_text(
                "import time\n"
                "print('training process ready', flush=True)\n"
                "while True:\n"
                "    time.sleep(0.1)\n",
                encoding="utf-8",
            )
            manager._train_script = training_script
            job = manager.enqueue({"name": "stop-me", "epochs": 2})[0]
            manager._active_job_id = job["id"]
            worker = threading.Thread(target=manager._run_job, args=(job["id"],), daemon=True)
            worker.start()

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                with manager._lock:
                    if manager._active_process is not None:
                        break
                time.sleep(0.01)
            else:
                self.fail("training subprocess did not start")

            stopping = manager.stop(job["id"])
            worker.join(timeout=5.0)

            self.assertFalse(worker.is_alive())
            self.assertTrue(stopping["stopRequested"])
            stopped = manager.get_job(job["id"])
            self.assertEqual(stopped["status"], "stopped")
            self.assertEqual(stopped["phase"], "finished")
            self.assertIn("Try again", stopped["error"])

    def test_close_stops_the_queue_worker_and_rejects_new_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingManager(write_training_runtime_config(Path(tmp)))

            manager.close()

            self.assertIsNotNone(manager._thread)
            self.assertFalse(manager._thread.is_alive())
            with self.assertRaisesRegex(RuntimeError, "shutting down"):
                manager.enqueue({"name": "too-late"})


if __name__ == "__main__":
    unittest.main()
