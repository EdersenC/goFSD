from __future__ import annotations

import argparse
import base64
import json
import math
import re
import threading
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import torch

from config import DEFAULT_FUTURE_OFFSETS, resolve_data_root_child
from control_contract import (
    CONTROL_CONTRACT_NAME,
    CONTROL_HORIZON_DIRECTION,
    PLANNER_FORMAT,
    PLANNER_FORMAT_VERSION,
    control_contract_metadata,
    resolve_checkpoint_control_horizon_dt_ms,
)
from image_io import load_rgb_tensor_from_bytes, load_rgb_tensor_from_path
from inference import (
    DEFAULT_CONFIG_PATH,
    LEGACY_SCALAR_HEAD_ERROR,
    build_model,
    load_checkpoint,
    load_config,
    normalize_windows_drive_path,
    resolve_checkpoint_control_target_names,
    resolve_checkpoint_aux_target_names,
    resolve_checkpoint_width_multiplier,
    resolve_checkpoint_frame_count,
    resolve_checkpoint_frame_stride,
    require_checkpoint_image_size,
    resolve_existing_path,
    select_device,
)
from state_inputs import (
    DEFAULT_WIDTH_MULTIPLIER,
    STATE_INPUT_DEFINITIONS,
    StateInputConfig,
    default_inference_state_input_config,
    normalize_state_input_value,
    state_input_config_from_metadata,
    state_inputs_metadata,
)
from stop_sign_contract import release_policy_metadata
from target_transforms import (
    TargetTransform,
    denormalize_target_tensor,
    resolve_checkpoint_target_transforms,
    target_transform_metadata,
)
from training_runtime import (
    TrainingJobDuplicateError,
    TrainingJobError,
    TrainingJobNotActiveError,
    TrainingJobNotFoundError,
    TrainingJobNotPendingError,
    TrainingJobNotRequeueableError,
    TrainingJobNotTerminalError,
    TrainingJobRequestError,
    TrainingManager,
)


def _http_status_for_exception(exc: Exception) -> HTTPStatus:
    if isinstance(exc, TrainingJobRequestError):
        return HTTPStatus.BAD_REQUEST
    if isinstance(exc, TrainingJobNotFoundError):
        return HTTPStatus.NOT_FOUND
    if isinstance(
        exc,
        (
            TrainingJobDuplicateError,
            TrainingJobNotPendingError,
            TrainingJobNotActiveError,
            TrainingJobNotRequeueableError,
            TrainingJobNotTerminalError,
        ),
    ):
        return HTTPStatus.CONFLICT
    if isinstance(exc, TrainingJobError):
        return HTTPStatus.INTERNAL_SERVER_ERROR
    if isinstance(exc, FileNotFoundError):
        return HTTPStatus.NOT_FOUND
    if isinstance(exc, ValueError):
        return HTTPStatus.BAD_REQUEST
    if isinstance(exc, RuntimeError):
        return HTTPStatus.CONFLICT
    return HTTPStatus.INTERNAL_SERVER_ERROR


@dataclass(frozen=True)
class ModelOption:
    label: str
    path: str
    run_id: str
    epoch: int
    variant: str
    is_best: bool
    updated_at: str


def load_training_runs_dir(config_path: Path) -> Path:
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    output_raw = raw.get("output", {})
    base_dir_raw = resolve_data_root_child(output_raw.get("base_dir"), "training_runs")
    if not base_dir_raw:
        raise ValueError(f"Missing output.base_dir in {config_path}")

    candidate = Path(base_dir_raw)
    config_dir = config_path.resolve().parent
    project_root = config_dir.parent
    script_dir = Path(__file__).resolve().parent

    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        # Prefer config/project-relative resolution so model discovery stays
        # stable even if the server process starts from src/gta_fsd or another
        # cwd that also contains an older mirrored training_runs directory.
        candidates.extend([
            project_root / candidate,
            config_dir / candidate,
            Path.cwd() / candidate,
            script_dir / candidate,
        ])

    seen: set[Path] = set()
    ordered_candidates: list[Path] = []
    for resolved in candidates:
        normalized = resolved.resolve(strict=False)
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered_candidates.append(normalized)

    for resolved in ordered_candidates:
        if resolved.is_dir():
            return resolved

    if ordered_candidates:
        return ordered_candidates[0]
    raise ValueError(f"Unable to resolve output.base_dir from {config_path}")


CHECKPOINT_NAME_PATTERN = re.compile(r"epoch-(?P<epoch>[0-9]+)(?P<ema>-ema)?\.pt")


def _configured_validation_variant(metrics_payload: dict[str, Any]) -> str:
    training = metrics_payload.get("training")
    ema = training.get("ema") if isinstance(training, dict) else None
    if not isinstance(ema, dict):
        return "model"
    eval_model = str(ema.get("eval_model", "")).strip().lower()
    if eval_model in {"ema", "model"}:
        return eval_model
    return "ema" if ema.get("enabled") is True else "model"


def _best_checkpoint_selection(metrics_payload: Any) -> tuple[int, str]:
    if not isinstance(metrics_payload, dict):
        return 0, "model"

    raw_best_epoch = metrics_payload.get("best_epoch")
    if isinstance(raw_best_epoch, bool):
        return 0, "model"
    try:
        best_epoch = int(raw_best_epoch or 0)
    except (TypeError, ValueError):
        return 0, "model"
    if best_epoch < 1:
        return 0, "model"

    configured_variant = _configured_validation_variant(metrics_payload)
    epochs = metrics_payload.get("epochs")
    if not isinstance(epochs, list):
        return best_epoch, configured_variant
    for epoch_payload in epochs:
        if not isinstance(epoch_payload, dict):
            continue
        raw_epoch = epoch_payload.get("epoch")
        if isinstance(raw_epoch, bool):
            continue
        try:
            epoch = int(raw_epoch)
        except (TypeError, ValueError):
            continue
        if epoch != best_epoch:
            continue

        val_metrics = epoch_payload.get("val_metrics")
        if isinstance(val_metrics, dict):
            eval_model = str(val_metrics.get("eval_model", "")).strip().lower()
            if eval_model in {"ema", "model"}:
                return best_epoch, eval_model

        training = epoch_payload.get("training")
        ema = training.get("ema") if isinstance(training, dict) else None
        if isinstance(ema, dict):
            eval_model = str(ema.get("eval_model", "")).strip().lower()
            if eval_model in {"ema", "model"}:
                return best_epoch, eval_model
        return best_epoch, configured_variant
    return best_epoch, configured_variant


def _load_best_checkpoint_selection(run_dir: Path) -> tuple[int, str]:
    metrics_path = run_dir / "run_metrics.json"
    try:
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return 0, "model"
    return _best_checkpoint_selection(metrics_payload)


def _checkpoint_identity(checkpoint_path: Path) -> tuple[int, str] | None:
    match = CHECKPOINT_NAME_PATTERN.fullmatch(checkpoint_path.name)
    if match is None:
        return None
    epoch = int(match.group("epoch"))
    variant = "ema" if match.group("ema") else "model"
    return epoch, variant


def discover_models(config_path: Path) -> list[dict[str, Any]]:
    runs_dir = load_training_runs_dir(config_path)
    if not runs_dir.is_dir():
        return []
    models: list[ModelOption] = []

    for run_dir in sorted((path for path in runs_dir.iterdir() if path.is_dir()), reverse=True):
        best_epoch, best_variant = _load_best_checkpoint_selection(run_dir)

        checkpoints = sorted(run_dir.glob("epoch-*.pt"), reverse=True)
        for checkpoint_path in checkpoints:
            identity = _checkpoint_identity(checkpoint_path)
            if identity is None:
                continue
            epoch, variant = identity

            is_best = best_epoch > 0 and epoch == best_epoch and variant == best_variant
            label = f"{run_dir.name} - epoch {epoch:03d}"
            if variant == "ema":
                label += " EMA"
            if is_best:
                label += " (best)"

            models.append(ModelOption(
                label=label,
                path=str(checkpoint_path.resolve()),
                run_id=run_dir.name,
                epoch=epoch,
                variant=variant,
                is_best=is_best,
                updated_at=datetime.fromtimestamp(
                    checkpoint_path.stat().st_mtime, tz=UTC
                ).isoformat().replace("+00:00", "Z"),
            ))

    models.sort(
        key=lambda item: (
            item.run_id,
            item.is_best,
            item.epoch,
            item.variant == "ema",
            item.path,
        ),
        reverse=True,
    )
    return [
        {
            "label": item.label,
            "path": item.path,
            "runId": item.run_id,
            "epoch": item.epoch,
            "variant": item.variant,
            "isBest": item.is_best,
            "updatedAt": item.updated_at,
        }
        for item in models
    ]


@dataclass(frozen=True)
class ServerArgs:
    host: str
    port: int
    config: Path


class ModelRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: Any = None
        self._device: torch.device | None = None
        self._checkpoint_path: Path | None = None
        self._planner_format = ""
        self._state_input_config = default_inference_state_input_config()
        self._image_size = (224, 224)
        self._frame_count = 3
        self._frame_stride = 2
        self._width_multiplier = DEFAULT_WIDTH_MULTIPLIER
        self._image_offsets: list[int] = []
        self._telemetry_offsets: list[int] = []
        self._future_offsets: list[int] = []
        self._telemetry_sample_interval_ms = 0
        self._control_horizon_dt_ms: list[int] = []
        self._telemetry_feature_names: list[str] = []
        self._control_target_names: list[str] = []
        self._aux_target_names: list[str] = []
        self._target_transforms: dict[str, TargetTransform] = {}

    def status(self) -> dict[str, Any]:
        with self._lock:
            loaded = self._model is not None
            return {
                "loaded": loaded,
                "direction": CONTROL_HORIZON_DIRECTION,
                "device": None if self._device is None else str(self._device),
                "checkpoint": None if self._checkpoint_path is None else str(self._checkpoint_path),
                "planner_format": self._planner_format or PLANNER_FORMAT,
                "planner_format_version": PLANNER_FORMAT_VERSION,
                "image_size": {
                    "width": self._image_size[0],
                    "height": self._image_size[1],
                },
                "frame_window": {
                    "size": self._frame_count,
                    "frame_stride": self._frame_stride,
                    "input_channels": self._frame_count * 3,
                },
                "model": {
                    "width_multiplier": self._width_multiplier,
                },
                "image_offsets": list(self._image_offsets),
                "telemetry_offsets": list(self._telemetry_offsets),
                "future_offsets": list(self._future_offsets),
                "telemetry_sample_interval_ms": self._telemetry_sample_interval_ms,
                "control_horizon_dt_ms": list(self._control_horizon_dt_ms),
                "control_contract": control_contract_metadata(),
                "release_policy": release_policy_metadata(),
                "telemetry_feature_names": list(self._telemetry_feature_names),
                "control_target_names": list(self._control_target_names),
                "aux_target_names": list(self._aux_target_names),
                "target_transforms": target_transform_metadata(self._target_transforms),
                "state_inputs": state_inputs_metadata(self._state_input_config),
            }

    def load_model(
        self,
        checkpoint_path: Path,
        device_name: str,
        image_size: tuple[int, int],
        frame_count: int,
        frame_stride: int,
    ) -> dict[str, Any]:
        device = select_device(device_name)
        checkpoint = load_checkpoint(checkpoint_path, device)
        checkpoint_image_size = require_checkpoint_image_size(checkpoint, image_size)
        model = build_model(checkpoint, device, frame_count)
        planner_format = str(checkpoint.get("planner_format", "")).strip()
        if planner_format != PLANNER_FORMAT:
            raise ValueError(LEGACY_SCALAR_HEAD_ERROR)
        width_multiplier = resolve_checkpoint_width_multiplier(checkpoint)
        state_input_config = state_input_config_from_metadata(checkpoint.get("state_inputs"))
        future_offsets = list(checkpoint.get("future_offsets") or DEFAULT_FUTURE_OFFSETS)
        control_horizon_dt_ms = list(resolve_checkpoint_control_horizon_dt_ms(checkpoint))
        control_target_names = resolve_checkpoint_control_target_names(
            checkpoint,
            future_steps=len(future_offsets),
        )
        aux_target_names = resolve_checkpoint_aux_target_names(
            checkpoint,
            future_steps=len(future_offsets),
        )
        target_transforms = resolve_checkpoint_target_transforms(
            checkpoint,
            tuple(control_target_names) + tuple(aux_target_names),
        )

        with self._lock:
            self._model = model
            self._device = device
            self._checkpoint_path = checkpoint_path
            self._planner_format = planner_format
            self._state_input_config = state_input_config
            self._image_size = checkpoint_image_size
            self._frame_count = frame_count
            self._frame_stride = frame_stride
            self._width_multiplier = width_multiplier
            self._image_offsets = list(checkpoint.get("image_offsets", []))
            self._telemetry_offsets = list(checkpoint.get("telemetry_offsets", []))
            self._future_offsets = future_offsets
            self._telemetry_sample_interval_ms = int(checkpoint["telemetry_sample_interval_ms"])
            self._control_horizon_dt_ms = control_horizon_dt_ms
            self._telemetry_feature_names = list(checkpoint.get("telemetry_feature_names", []))
            self._control_target_names = control_target_names
            self._aux_target_names = aux_target_names
            self._target_transforms = target_transforms

        return {
            "status": "loaded",
            "direction": CONTROL_HORIZON_DIRECTION,
            "checkpoint": str(checkpoint_path),
            "device": str(device),
            "planner_format": planner_format,
            "planner_format_version": PLANNER_FORMAT_VERSION,
            "image_size": {
                "width": checkpoint_image_size[0],
                "height": checkpoint_image_size[1],
            },
            "frame_window": {
                "size": frame_count,
                "frame_stride": frame_stride,
                "input_channels": frame_count * 3,
            },
            "model": {
                "width_multiplier": width_multiplier,
                "telemetry_hidden_dim": (checkpoint.get("model", {}) or {}).get("telemetry_hidden_dim"),
            },
            "image_offsets": checkpoint.get("image_offsets", []),
            "telemetry_offsets": checkpoint.get("telemetry_offsets", []),
            "future_offsets": checkpoint.get("future_offsets", []),
            "telemetry_sample_interval_ms": int(checkpoint["telemetry_sample_interval_ms"]),
            "control_horizon_dt_ms": control_horizon_dt_ms,
            "control_contract": control_contract_metadata(),
            "release_policy": release_policy_metadata(),
            "telemetry_feature_names": checkpoint.get("telemetry_feature_names", []),
            "control_target_names": control_target_names,
            "aux_target_names": aux_target_names,
            "state_inputs": checkpoint.get("state_inputs", state_inputs_metadata(state_input_config)),
            "target_transforms": target_transform_metadata(target_transforms),
            "epoch": int(checkpoint.get("epoch", 0)),
            "train_metrics": checkpoint.get("train_metrics"),
            "val_metrics": checkpoint.get("val_metrics"),
        }

    def unload_model(self) -> dict[str, Any]:
        with self._lock:
            previous = self._checkpoint_path
            self._model = None
            self._device = None
            self._checkpoint_path = None
            self._planner_format = ""
            self._state_input_config = default_inference_state_input_config()
            self._width_multiplier = DEFAULT_WIDTH_MULTIPLIER
            self._image_offsets = []
            self._telemetry_offsets = []
            self._future_offsets = []
            self._telemetry_sample_interval_ms = 0
            self._control_horizon_dt_ms = []
            self._telemetry_feature_names = []
            self._control_target_names = []
            self._aux_target_names = []
            self._target_transforms = {}
        return {
            "status": "unloaded",
            "checkpoint": None if previous is None else str(previous),
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._model is None or self._device is None or self._checkpoint_path is None:
                raise RuntimeError("model is not loaded")
            model = self._model
            device = self._device
            checkpoint_path = self._checkpoint_path
            planner_format = self._planner_format
            state_input_config = self._state_input_config
            target_transforms = self._target_transforms
            telemetry_offsets = list(self._telemetry_offsets)
            future_offsets = list(self._future_offsets)
            telemetry_sample_interval_ms = self._telemetry_sample_interval_ms
            control_horizon_dt_ms = list(self._control_horizon_dt_ms)
            telemetry_feature_names = list(self._telemetry_feature_names)
            control_target_names = list(self._control_target_names)
            aux_target_names = list(self._aux_target_names)
            image_offsets = list(self._image_offsets)

        if planner_format != PLANNER_FORMAT:
            raise RuntimeError(LEGACY_SCALAR_HEAD_ERROR)
        require_predict_control_contract(payload)
        sampled_at_s = parse_sampled_at_s(payload)
        frames = self._extract_frames(payload)

        images = torch.stack(frames, dim=0).unsqueeze(0).to(device, non_blocking=device.type == "cuda")
        telemetry = self._extract_planner_telemetry(payload).to(device, non_blocking=device.type == "cuda")
        raw_state_inputs, model_state_inputs = self._extract_state_inputs(payload, state_input_config)
        planner_state_inputs = None
        if model_state_inputs:
            ordered = [
                model_state_inputs[definition.key]
                for definition in STATE_INPUT_DEFINITIONS
                if state_input_config.is_enabled(definition.key)
            ]
            planner_state_inputs = torch.cat(ordered, dim=0).unsqueeze(0).to(
                device,
                non_blocking=device.type == "cuda",
            )
        with torch.no_grad():
            output = model(images, telemetry, planner_state_inputs)
        pred_controls = output["pred_controls"].detach().cpu()
        pred_aux = output["pred_aux"].detach().cpu()
        pred_controls_denorm = denormalize_target_tensor(
            pred_controls,
            control_target_names,
            target_transforms,
        )
        pred_aux_denorm = denormalize_target_tensor(
            pred_aux,
            aux_target_names,
            target_transforms,
        )
        response = {
            "checkpoint": str(checkpoint_path),
            "device": str(device),
            "sampled_at_s": sampled_at_s,
            "direction": CONTROL_HORIZON_DIRECTION,
            "planner_format": planner_format,
            "planner_format_version": PLANNER_FORMAT_VERSION,
            "pred_controls": pred_controls_denorm.tolist(),
            "motion_plan": {
                "horizon_dt_ms": control_horizon_dt_ms,
                "future_speed_mps": pred_controls_denorm[..., 0].tolist(),
                "stop_intent": pred_controls_denorm[..., 1].tolist(),
                "release_policy": release_policy_metadata(),
            },
            "pred_aux": pred_aux_denorm.tolist(),
            "pred_controls_normalized": pred_controls.tolist(),
            "pred_aux_normalized": pred_aux.tolist(),
            "target_transforms": target_transform_metadata(target_transforms),
            "image_shape": list(images.shape),
            "telemetry_shape": list(telemetry.shape),
            "pred_controls_shape": list(pred_controls.shape),
            "pred_aux_shape": list(pred_aux.shape),
            "image_offsets": image_offsets,
            "telemetry_offsets": telemetry_offsets,
            "future_offsets": future_offsets,
            "telemetry_sample_interval_ms": telemetry_sample_interval_ms,
            "control_horizon_dt_ms": control_horizon_dt_ms,
            "control_contract": control_contract_metadata(),
            "telemetry_feature_names": telemetry_feature_names,
            "control_target_names": control_target_names,
            "aux_target_names": aux_target_names,
            "state_inputs": state_inputs_metadata(state_input_config),
        }
        if raw_state_inputs:
            response["raw_state_inputs"] = raw_state_inputs
            response["normalized_state_inputs"] = {
                definition.camel_key: float(model_state_inputs[definition.key].item())
                for definition in STATE_INPUT_DEFINITIONS
                if definition.key in model_state_inputs
            }
        return response

    def _extract_planner_telemetry(self, payload: dict[str, Any]) -> torch.Tensor:
        raw = payload.get("telemetry")
        if not isinstance(raw, list) or len(raw) != 1:
            raise ValueError("telemetry must have shape [1, T, F]")
        batch_item = raw[0]
        if not isinstance(batch_item, list) or not batch_item:
            raise ValueError("telemetry batch item must be a non-empty list")
        telemetry_rows: list[list[float]] = []
        expected_width = len(self._telemetry_feature_names)
        for row in batch_item:
            if not isinstance(row, list) or len(row) != expected_width:
                raise ValueError(
                    f"telemetry row width mismatch: expected {expected_width}, got "
                    f"{0 if not isinstance(row, list) else len(row)}"
                )
            telemetry_rows.append([float(value) for value in row])
        expected_length = len(self._telemetry_offsets)
        if len(telemetry_rows) != expected_length:
            raise ValueError(f"telemetry length mismatch: expected {expected_length}, got {len(telemetry_rows)}")
        return torch.tensor([telemetry_rows], dtype=torch.float32)

    def _extract_frames(self, payload: dict[str, Any]) -> list[torch.Tensor]:
        if "frame_paths" in payload:
            raw_paths = payload["frame_paths"]
            if not isinstance(raw_paths, list):
                raise ValueError("frame_paths must be a list of strings")
            if len(raw_paths) != self._frame_count:
                raise ValueError(f"expected {self._frame_count} frame_paths entries, got {len(raw_paths)}")
            return [self._load_image_from_path(path) for path in raw_paths]

        if "frames_base64" in payload:
            raw_frames = payload["frames_base64"]
            if not isinstance(raw_frames, list):
                raise ValueError("frames_base64 must be a list of base64 strings")
            if len(raw_frames) != self._frame_count:
                raise ValueError(f"expected {self._frame_count} frames_base64 entries, got {len(raw_frames)}")
            return [self._load_image_from_base64(item) for item in raw_frames]

        raise ValueError("request must include frame_paths or frames_base64")

    def _load_image_from_path(self, raw_path: Any) -> torch.Tensor:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("frame path entries must be non-empty strings")
        image_path = Path(normalize_windows_drive_path(raw_path))
        if not image_path.is_file():
            raise FileNotFoundError(f"frame image not found: {image_path}")
        return load_rgb_tensor_from_path(image_path, self._image_size)

    def _load_image_from_base64(self, raw_value: Any) -> torch.Tensor:
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError("frames_base64 entries must be non-empty strings")
        payload = raw_value.split(",", 1)[-1]
        image_bytes = base64.b64decode(payload)
        return load_rgb_tensor_from_bytes(image_bytes, self._image_size, source="frames_base64")

    def _extract_state_inputs(
        self,
        payload: dict[str, Any],
        config: StateInputConfig,
    ) -> tuple[dict[str, float | bool], dict[str, torch.Tensor]]:
        raw_state_inputs: dict[str, float | bool] = {}
        normalized_state_inputs: dict[str, torch.Tensor] = {}
        for definition in STATE_INPUT_DEFINITIONS:
            if not config.is_enabled(definition.key):
                continue
            raw_value = None
            if definition.key in payload:
                raw_value = payload[definition.key]
            elif definition.camel_key in payload:
                raw_value = payload[definition.camel_key]
            if raw_value is None:
                raise ValueError(f"request must include {definition.key} for this checkpoint")
            normalized = normalize_state_input_value(definition.key, raw_value, config)
            raw_state_inputs[definition.key] = float(raw_value)
            normalized_state_inputs[definition.key] = torch.tensor([normalized], dtype=torch.float32)
        return raw_state_inputs, normalized_state_inputs


def parse_sampled_at_s(payload: dict[str, Any]) -> float:
    raw_value = payload.get("sampled_at_s")
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ValueError("sampled_at_s must be a finite number greater than 0")
    sampled_at_s = float(raw_value)
    if not math.isfinite(sampled_at_s) or sampled_at_s <= 0:
        raise ValueError("sampled_at_s must be a finite number greater than 0")
    return sampled_at_s


def require_predict_control_contract(payload: dict[str, Any]) -> None:
    raw_contract = payload.get("control_contract")
    if raw_contract != CONTROL_CONTRACT_NAME:
        raise ValueError(
            f"predict request control_contract must equal '{CONTROL_CONTRACT_NAME}'"
        )


class ModelServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler_cls: type[BaseHTTPRequestHandler], *, config_path: Path):
        super().__init__(server_address, handler_cls)
        self.runtime = ModelRuntime()
        self.training = TrainingManager(config_path)
        self.config_path = config_path

    def server_close(self) -> None:
        try:
            self.training.close()
        finally:
            super().server_close()


class RequestHandler(BaseHTTPRequestHandler):
    server: ModelServer

    def do_GET(self) -> None:
        self._handle_request(self._dispatch_get)

    def _dispatch_get(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/healthz":
            self._write_json(HTTPStatus.OK, {
                "status": "ok",
                "service": "stop-sign-lab-model",
            })
            return
        if path == "/model":
            self._write_json(HTTPStatus.OK, self.server.runtime.status())
            return
        if path == "/models":
            self._write_json(HTTPStatus.OK, {"models": discover_models(self.server.config_path)})
            return
        if path == "/training/config":
            self._write_json(HTTPStatus.OK, self.server.training.page_config())
            return
        if path == "/training/state":
            self._write_json(HTTPStatus.OK, self.server.training.state())
            return
        if path == "/training/jobs":
            self._write_json(HTTPStatus.OK, {"jobs": self.server.training.list_jobs()})
            return
        if path == "/training/history":
            self._write_json(HTTPStatus.OK, {
                "jobs": self.server.training.state().get("recentJobs", []),
            })
            return
        if path.startswith("/training/jobs/"):
            self._handle_training_get(path, parse_qs(parsed.query))
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        self._handle_request(self._dispatch_post)

    def _dispatch_post(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        payload = self._read_json_body() if path != "/training/jobs" else self._read_json_value()
        if path == "/model/load":
            self._handle_model_load(payload)
            return
        if path == "/model/unload":
            self._write_json(HTTPStatus.OK, self.server.runtime.unload_model())
            return
        if path == "/predict":
            result = self.server.runtime.predict(payload)
            self._write_json(HTTPStatus.OK, result)
            return
        if path == "/training/jobs":
            jobs = self.server.training.enqueue(payload)
            self._write_json(HTTPStatus.OK, {"status": "queued", "jobs": jobs})
            return
        if path == "/training/history/clear":
            self._write_json(HTTPStatus.OK, self.server.training.clear_history())
            return
        if path.startswith("/training/jobs/"):
            self._handle_training_post(path)
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def _handle_request(self, action: Callable[[], None]) -> None:
        try:
            action()
        except Exception as exc:
            self._write_json(_http_status_for_exception(exc), {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _handle_model_load(self, payload: dict[str, Any]) -> None:
        config_path = self.server.config_path
        if "config" in payload:
            raw_config = payload["config"]
            if not isinstance(raw_config, str) or not raw_config.strip():
                raise ValueError("config must be a non-empty string when provided")
            config_path = Path(raw_config)

        file_config = load_config(config_path, require_checkpoint=False)
        raw_checkpoint = payload.get("checkpoint", file_config.checkpoint)
        raw_device = payload.get("device") or file_config.device or "cuda"

        if not isinstance(raw_checkpoint, str) or not raw_checkpoint.strip():
            raise ValueError("checkpoint must be a non-empty string")
        if not isinstance(raw_device, str) or not raw_device.strip():
            raise ValueError("device must be a non-empty string")

        checkpoint_path = resolve_existing_path(raw_checkpoint, config_path)
        image_size = (file_config.image_width, file_config.image_height)
        checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
        frame_count = resolve_checkpoint_frame_count(checkpoint, file_config)
        frame_stride = resolve_checkpoint_frame_stride(checkpoint, file_config)
        result = self.server.runtime.load_model(
            checkpoint_path,
            raw_device.strip().lower(),
            image_size,
            frame_count,
            frame_stride,
        )
        self._write_json(HTTPStatus.OK, result)

    def _read_json_body(self) -> dict[str, Any]:
        payload = self._read_json_value()
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _read_json_value(self) -> Any:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length == 0:
            return {}
        raw = self.rfile.read(content_length)
        return json.loads(raw.decode("utf-8"))

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_text(self, status: HTTPStatus, payload: str) -> None:
        body = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_training_get(self, path: str, query: dict[str, list[str]]) -> None:
        prefix = "/training/jobs/"
        suffix = path[len(prefix):].strip("/")
        parts = [part for part in suffix.split("/") if part]
        if not parts:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        job_id = parts[0]
        if len(parts) == 1:
            self._write_json(HTTPStatus.OK, self.server.training.get_job(job_id))
            return
        if len(parts) == 2 and parts[1] == "log":
            tail_lines = 0
            if "tailLines" in query and query["tailLines"]:
                try:
                    tail_lines = int(query["tailLines"][0])
                except ValueError as exc:
                    raise TrainingJobRequestError("tailLines must be a non-negative integer") from exc
                if tail_lines < 0:
                    raise TrainingJobRequestError("tailLines must be a non-negative integer")
            self._write_text(HTTPStatus.OK, self.server.training.read_log(job_id, tail_lines))
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def _handle_training_post(self, path: str) -> None:
        prefix = "/training/jobs/"
        suffix = path[len(prefix):].strip("/")
        parts = [part for part in suffix.split("/") if part]
        if len(parts) != 2:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        job_id, action = parts
        if action == "cancel":
            job = self.server.training.cancel(job_id)
            self._write_json(HTTPStatus.OK, {"status": "canceled", "job": job})
            return
        if action == "stop":
            job = self.server.training.stop(job_id)
            self._write_json(HTTPStatus.OK, {"status": "stopping", "job": job})
            return
        if action == "delete":
            job = self.server.training.delete_job(job_id)
            self._write_json(HTTPStatus.OK, {"status": "deleted", "job": job})
            return
        if action == "requeue":
            job = self.server.training.requeue(job_id)
            self._write_json(HTTPStatus.OK, {"status": "queued", "job": job})
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def _read_int(self, value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("sequence and timestamp_ms must be integers when provided") from exc


def parse_args() -> ServerArgs:
    parser = argparse.ArgumentParser(description="Serve a GTA FSD planner model over HTTP.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the HTTP server to.")
    parser.add_argument("--port", type=int, default=8090, help="Port to bind the HTTP server to.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the TOML config. Default: {DEFAULT_CONFIG_PATH}",
    )
    args = parser.parse_args()
    return ServerArgs(host=args.host, port=args.port, config=args.config)


def main() -> None:
    args = parse_args()
    server = ModelServer((args.host, args.port), RequestHandler, config_path=args.config)
    print(f"Serving model API on http://{args.host}:{args.port}")
    print("POST /model/load to load a checkpoint into memory")
    print("POST /predict with frame_paths or frames_base64 to run inference")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
