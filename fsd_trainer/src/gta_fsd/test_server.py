from __future__ import annotations

import json
import tempfile
import unittest
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from server import RequestHandler, discover_models
from training_runtime import (
    TrainingJobDuplicateError,
    TrainingJobError,
    TrainingJobNotActiveError,
    TrainingJobNotFoundError,
    TrainingJobNotPendingError,
    TrainingJobNotRequeueableError,
    TrainingJobNotTerminalError,
    TrainingJobRequestError,
)


class RaisingTrainingManager:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def get_job(self, job_id: str) -> dict[str, Any]:
        raise self.error

    def cancel(self, job_id: str) -> dict[str, Any]:
        raise self.error


def create_handler(
    method: str,
    error: Exception,
) -> tuple[RequestHandler, list[tuple[HTTPStatus, dict[str, Any]]]]:
    handler = object.__new__(RequestHandler)
    handler.path = "/training/jobs/job-1" if method == "GET" else "/training/jobs/job-1/cancel"
    handler.server = SimpleNamespace(training=RaisingTrainingManager(error))
    responses: list[tuple[HTTPStatus, dict[str, Any]]] = []
    handler._write_json = lambda status, payload: responses.append((status, payload))
    handler._read_json_body = lambda: {}
    return handler, responses


def write_catalog_config(root: Path, runs_dir: Path) -> Path:
    config_path = root / "train_config.toml"
    config_path.write_text(
        f"[output]\nbase_dir = {json.dumps(str(runs_dir))}\n",
        encoding="utf-8",
    )
    return config_path


def write_run_metrics(run_dir: Path, *, best_epoch: int, eval_model: str) -> None:
    (run_dir / "run_metrics.json").write_text(
        json.dumps({
            "best_epoch": best_epoch,
            "epochs": [{
                "epoch": best_epoch,
                "val_metrics": {"eval_model": eval_model},
            }],
        }),
        encoding="utf-8",
    )


class RequestHandlerErrorTests(unittest.TestCase):
    def test_health_identifies_the_stop_sign_lab_model_service(self) -> None:
        handler = object.__new__(RequestHandler)
        handler.path = "/healthz"
        responses: list[tuple[HTTPStatus, dict[str, Any]]] = []
        handler._write_json = lambda status, payload: responses.append((status, payload))

        handler.do_GET()

        self.assertEqual(responses, [(
            HTTPStatus.OK,
            {"status": "ok", "service": "stop-sign-lab-model"},
        )])

    def test_get_and_post_share_training_job_error_mapping(self) -> None:
        cases = (
            (TrainingJobRequestError, HTTPStatus.BAD_REQUEST),
            (TrainingJobDuplicateError, HTTPStatus.CONFLICT),
            (TrainingJobNotFoundError, HTTPStatus.NOT_FOUND),
            (TrainingJobNotPendingError, HTTPStatus.CONFLICT),
            (TrainingJobNotActiveError, HTTPStatus.CONFLICT),
            (TrainingJobNotRequeueableError, HTTPStatus.CONFLICT),
            (TrainingJobNotTerminalError, HTTPStatus.CONFLICT),
            (TrainingJobError, HTTPStatus.INTERNAL_SERVER_ERROR),
            (FileNotFoundError, HTTPStatus.NOT_FOUND),
            (ValueError, HTTPStatus.BAD_REQUEST),
            (RuntimeError, HTTPStatus.CONFLICT),
        )
        for method in ("GET", "POST"):
            for error_type, expected_status in cases:
                with self.subTest(method=method, error=error_type.__name__):
                    handler, responses = create_handler(method, error_type("expected error"))

                    getattr(handler, f"do_{method}")()

                    self.assertEqual(
                        responses,
                        [(expected_status, {"error": "expected error"})],
                    )


class ModelCatalogTests(unittest.TestCase):
    def test_ema_validated_best_epoch_prefers_ema_and_keeps_raw_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_dir = root / "training_runs"
            run_dir = runs_dir / "run-ema"
            run_dir.mkdir(parents=True)
            for name in (
                "epoch-001.pt",
                "epoch-001-ema.pt",
                "epoch-002.pt",
                "epoch-002-ema.pt",
                "epoch-final.pt",
            ):
                (run_dir / name).write_bytes(b"checkpoint")
            write_run_metrics(run_dir, best_epoch=2, eval_model="ema")

            models = discover_models(write_catalog_config(root, runs_dir))

            self.assertEqual(len(models), 4)
            self.assertEqual(Path(models[0]["path"]).name, "epoch-002-ema.pt")
            self.assertEqual(models[0]["variant"], "ema")
            self.assertTrue(models[0]["isBest"])
            self.assertEqual(models[0]["label"], "run-ema - epoch 002 EMA (best)")

            raw_best_epoch = next(
                model for model in models if Path(model["path"]).name == "epoch-002.pt"
            )
            self.assertEqual(raw_best_epoch["variant"], "model")
            self.assertFalse(raw_best_epoch["isBest"])
            self.assertEqual(raw_best_epoch["label"], "run-ema - epoch 002")

    def test_raw_validated_best_epoch_prefers_raw_and_keeps_ema_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_dir = root / "training_runs"
            run_dir = runs_dir / "run-model"
            run_dir.mkdir(parents=True)
            (run_dir / "epoch-003.pt").write_bytes(b"checkpoint")
            (run_dir / "epoch-003-ema.pt").write_bytes(b"checkpoint")
            write_run_metrics(run_dir, best_epoch=3, eval_model="model")

            models = discover_models(write_catalog_config(root, runs_dir))

            self.assertEqual(len(models), 2)
            self.assertEqual(Path(models[0]["path"]).name, "epoch-003.pt")
            self.assertEqual(models[0]["variant"], "model")
            self.assertTrue(models[0]["isBest"])
            ema_checkpoint = next(model for model in models if model["variant"] == "ema")
            self.assertFalse(ema_checkpoint["isBest"])

    def test_legacy_ema_enabled_metrics_prefer_ema_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_dir = root / "training_runs"
            run_dir = runs_dir / "run-legacy-ema"
            run_dir.mkdir(parents=True)
            (run_dir / "epoch-004.pt").write_bytes(b"checkpoint")
            (run_dir / "epoch-004-ema.pt").write_bytes(b"checkpoint")
            (run_dir / "run_metrics.json").write_text(
                json.dumps({
                    "best_epoch": 4,
                    "training": {"ema": {"enabled": True}},
                }),
                encoding="utf-8",
            )

            models = discover_models(write_catalog_config(root, runs_dir))

            self.assertEqual(Path(models[0]["path"]).name, "epoch-004-ema.pt")
            self.assertTrue(models[0]["isBest"])


if __name__ == "__main__":
    unittest.main()
