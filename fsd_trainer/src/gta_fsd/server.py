from __future__ import annotations

import argparse
import base64
import json
import math
import threading
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse

import torch

from control_translation import (
    CONTROL_SEMANTICS_CONTROLLER_INPUT,
    CONTROL_SEMANTICS_SPEED_DELTA,
    CONTROL_SEMANTICS_TARGET_SPEED,
)
from heads import (
    FUTURE_YAW_DELTA_HEAD_NAME,
    HeadSpec,
    control_head_specs,
    head_layout_metadata,
    head_specs_metadata,
    resolve_checkpoint_head_specs,
)
from image_io import load_rgb_tensor_from_bytes, load_rgb_tensor_from_path
from inference import (
    DEFAULT_CONFIG_PATH,
    build_model,
    load_checkpoint,
    load_config,
    normalize_windows_drive_path,
    resolve_checkpoint_control_target_names,
    resolve_checkpoint_width_multiplier,
    resolve_checkpoint_frame_count,
    resolve_checkpoint_frame_stride,
    resolve_existing_path,
    select_device,
)
from model_output import single_control_prediction_from_output, single_prediction_from_output
from state_inputs import (
    DEFAULT_WIDTH_MULTIPLIER,
    STATE_INPUT_DEFINITIONS,
    StateInputConfig,
    default_inference_state_input_config,
    normalize_state_input_value,
    resolve_state_input_cap,
    state_input_config_from_metadata,
    state_inputs_metadata,
)
from target_transforms import (
    DeltaSpeedTargetTransform,
    legacy_delta_speed_target_transform,
    resolve_checkpoint_delta_speed_target_transform,
)
from training_runtime import (
    TrainingJobError,
    TrainingJobNotActiveError,
    TrainingJobNotFoundError,
    TrainingJobNotPendingError,
    TrainingJobNotRequeueableError,
    TrainingJobNotTerminalError,
    TrainingJobRequestError,
    TrainingManager,
)


def sigmoid_probability(value: float) -> float:
    if value >= 0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


@dataclass(frozen=True)
class ModelOption:
    label: str
    path: str
    run_id: str
    epoch: int
    is_best: bool
    updated_at: str


def load_training_runs_dir(config_path: Path) -> Path:
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    output_raw = raw.get("output", {})
    base_dir_raw = str(output_raw.get("base_dir", "")).strip()
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

    tried = "\n".join(f"- {path}" for path in ordered_candidates)
    raise FileNotFoundError(f"Training runs directory does not exist: {base_dir_raw}\nTried:\n{tried}")


def discover_models(config_path: Path) -> list[dict[str, Any]]:
    runs_dir = load_training_runs_dir(config_path)
    models: list[ModelOption] = []

    for run_dir in sorted((path for path in runs_dir.iterdir() if path.is_dir()), reverse=True):
        best_epoch = 0
        metrics_path = run_dir / "run_metrics.json"
        if metrics_path.is_file():
            try:
                metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                best_epoch = int(metrics_payload.get("best_epoch", 0) or 0)
            except Exception:
                best_epoch = 0

        checkpoints = sorted(run_dir.glob("epoch-*.pt"), reverse=True)
        for checkpoint_path in checkpoints:
            stem = checkpoint_path.stem
            try:
                epoch = int(stem.split("-", 1)[1])
            except Exception:
                continue

            is_best = best_epoch > 0 and epoch == best_epoch
            label = f"{run_dir.name} - epoch {epoch:03d}"
            if is_best:
                label += " (best)"

            models.append(ModelOption(
                label=label,
                path=str(checkpoint_path.resolve()),
                run_id=run_dir.name,
                epoch=epoch,
                is_best=is_best,
                updated_at=datetime.fromtimestamp(
                    checkpoint_path.stat().st_mtime, tz=UTC
                ).isoformat().replace("+00:00", "Z"),
            ))

    models.sort(key=lambda item: (item.run_id, item.is_best, item.epoch, item.path), reverse=True)
    return [
        {
            "label": item.label,
            "path": item.path,
            "runId": item.run_id,
            "epoch": item.epoch,
            "isBest": item.is_best,
            "updatedAt": item.updated_at,
        }
        for item in models
    ]


def control_target_sources(head_specs: list[dict[str, Any]] | None) -> dict[str, str]:
    if not isinstance(head_specs, list):
        return {}

    sources: dict[str, str] = {}
    for item in head_specs:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        if not bool(item.get("used_for_control", False)):
            continue
        target_source = str(item.get("target_source", "")).strip()
        if target_source:
            sources[name] = target_source
    return sources


def control_semantics_from_sources(sources: Mapping[str, str]) -> str:
    lateral_source = str(sources.get(FUTURE_YAW_DELTA_HEAD_NAME, "")).lower()
    future_speed_source = str(sources.get("future_speed", "")).lower()
    delta_speed_source = str(sources.get("delta_speed", "")).lower()
    accel_source = str(sources.get("accel", "")).lower()
    if "steerinput" in lateral_source and ("accelinput" in accel_source or "deltaspeedinput" in delta_speed_source):
        return CONTROL_SEMANTICS_CONTROLLER_INPUT
    if "label.future_speed" in future_speed_source or "future_speed" in future_speed_source:
        return CONTROL_SEMANTICS_TARGET_SPEED
    if "label.delta_speed" in delta_speed_source or "delta_speed" in delta_speed_source:
        return CONTROL_SEMANTICS_SPEED_DELTA
    if lateral_source or accel_source or delta_speed_source or future_speed_source:
        return "vehicle_state"
    return CONTROL_SEMANTICS_CONTROLLER_INPUT


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def build_semantic_intent(
    outputs: Mapping[str, Any],
    *,
    current_speed: float | None,
) -> dict[str, Any]:
    heading_error = float(outputs.get("future_yaw_delta", 0.0) or 0.0)
    future_speed = outputs.get("future_speed")
    delta_speed = outputs.get("delta_speed")
    move_intent_prob = float(outputs.get("move_intent_prob", 0.0) or 0.0)

    target_speed = 0.0
    if future_speed is not None:
        target_speed = max(float(future_speed), 0.0)
    elif delta_speed is not None and current_speed is not None:
        target_speed = max(float(current_speed) + float(delta_speed), 0.0)

    target_accel = 0.0
    if delta_speed is not None:
        target_accel = float(delta_speed)
    elif future_speed is not None and current_speed is not None:
        target_accel = float(future_speed) - float(current_speed)

    if move_intent_prob >= 0.55 or target_speed > 0.35:
        motion_intent = "move"
    elif target_speed <= 0.15 and target_accel <= 0.0:
        motion_intent = "stop"
    else:
        motion_intent = "hold"

    return {
        "lateral": {
            "heading_error_deg": heading_error,
            "confidence": 1.0,
        },
        "longitudinal": {
            "target_speed_mps": max(target_speed, 0.0),
            "target_accel_mps2": clamp(target_accel, -4.0, 4.0),
            "confidence": 1.0,
        },
        "motion": {
            "intent": motion_intent,
            "confidence": clamp(move_intent_prob if move_intent_prob > 0 else 1.0, 0.0, 1.0),
        },
    }


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
        self._head_specs: tuple[HeadSpec, ...] = resolve_checkpoint_head_specs({
            "head_specs": head_specs_metadata(),
        })
        self._head_specs_metadata: list[dict[str, Any]] = head_specs_metadata(self._head_specs)
        self._state_input_config = default_inference_state_input_config()
        self._delta_speed_transform: DeltaSpeedTargetTransform = legacy_delta_speed_target_transform()
        self._image_size = (224, 224)
        self._frame_count = 3
        self._frame_stride = 2
        self._width_multiplier = DEFAULT_WIDTH_MULTIPLIER
        self._image_offsets: list[int] = []
        self._telemetry_offsets: list[int] = []
        self._future_offsets: list[int] = []
        self._telemetry_feature_names: list[str] = []
        self._control_target_names: list[str] = []
        self._aux_target_names: list[str] = []

    def status(self) -> dict[str, Any]:
        with self._lock:
            loaded = self._model is not None
            head_specs = self._head_specs
            target_sources = control_target_sources(self._head_specs_metadata)
            if self._planner_format == "temporal_telemetry_gru_v1":
                return {
                    "loaded": loaded,
                    "device": None if self._device is None else str(self._device),
                    "checkpoint": None if self._checkpoint_path is None else str(self._checkpoint_path),
                    "planner_format": self._planner_format,
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
                    "telemetry_feature_names": list(self._telemetry_feature_names),
                    "control_target_names": list(self._control_target_names),
                    "aux_target_names": list(self._aux_target_names),
                }
            return {
                "loaded": loaded,
                "device": None if self._device is None else str(self._device),
                "checkpoint": None if self._checkpoint_path is None else str(self._checkpoint_path),
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
                "state_inputs": state_inputs_metadata(self._state_input_config),
                "delta_speed_target_transform": self._delta_speed_transform.metadata(),
                "head_specs": self._head_specs_metadata,
                "head_layout": head_layout_metadata(head_specs),
                "control_target_sources": target_sources,
                "control_semantics": control_semantics_from_sources(target_sources),
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
        model = build_model(checkpoint, device, frame_count)
        planner_format = str(checkpoint.get("planner_format", "")).strip()
        width_multiplier = resolve_checkpoint_width_multiplier(checkpoint)
        state_input_config = default_inference_state_input_config()
        head_specs = resolve_checkpoint_head_specs({
            "head_specs": head_specs_metadata(),
        })
        head_specs_metadata_payload = head_specs_metadata(head_specs)
        delta_speed_transform = legacy_delta_speed_target_transform()
        if planner_format != "temporal_telemetry_gru_v1":
            state_input_config = state_input_config_from_metadata(checkpoint.get("state_inputs"))
            head_specs = resolve_checkpoint_head_specs(checkpoint)
            head_specs_metadata_payload = checkpoint.get("head_specs", head_specs_metadata(head_specs))
            delta_speed_transform = resolve_checkpoint_delta_speed_target_transform(checkpoint)

        with self._lock:
            self._model = model
            self._device = device
            self._checkpoint_path = checkpoint_path
            self._planner_format = planner_format
            self._head_specs = head_specs
            self._head_specs_metadata = head_specs_metadata_payload
            self._state_input_config = state_input_config
            self._delta_speed_transform = delta_speed_transform
            self._image_size = image_size
            self._frame_count = frame_count
            self._frame_stride = frame_stride
            self._width_multiplier = width_multiplier
            self._image_offsets = list(checkpoint.get("image_offsets", []))
            self._telemetry_offsets = list(checkpoint.get("telemetry_offsets", []))
            self._future_offsets = list(checkpoint.get("future_offsets", []))
            self._telemetry_feature_names = list(checkpoint.get("telemetry_feature_names", []))
            self._control_target_names = resolve_checkpoint_control_target_names(
                checkpoint,
                future_steps=len(self._future_offsets) or 6,
            )
            self._aux_target_names = list(checkpoint.get("aux_target_names", []))

        target_sources = control_target_sources(head_specs_metadata_payload)
        if planner_format == "temporal_telemetry_gru_v1":
            return {
                "status": "loaded",
                "checkpoint": str(checkpoint_path),
                "device": str(device),
                "planner_format": planner_format,
                "image_size": {
                    "width": image_size[0],
                    "height": image_size[1],
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
                "telemetry_feature_names": checkpoint.get("telemetry_feature_names", []),
                "control_target_names": resolve_checkpoint_control_target_names(
                    checkpoint,
                    future_steps=len(checkpoint.get("future_offsets", [])) or 6,
                ),
                "aux_target_names": checkpoint.get("aux_target_names", []),
                "epoch": int(checkpoint.get("epoch", 0)),
                "train_metrics": checkpoint.get("train_metrics"),
                "val_metrics": checkpoint.get("val_metrics"),
            }
        return {
            "status": "loaded",
            "checkpoint": str(checkpoint_path),
            "device": str(device),
            "image_size": {
                "width": image_size[0],
                "height": image_size[1],
            },
            "frame_window": {
                "size": frame_count,
                "frame_stride": frame_stride,
                "input_channels": frame_count * 3,
            },
            "model": {
                "width_multiplier": width_multiplier,
            },
            "state_inputs": checkpoint.get("state_inputs", state_inputs_metadata(state_input_config)),
            "delta_speed_target_transform": delta_speed_transform.metadata(),
            "head_specs": head_specs_metadata_payload,
            "head_layout": checkpoint.get("head_layout", head_layout_metadata(head_specs)),
            "control_target_sources": target_sources,
            "control_semantics": control_semantics_from_sources(target_sources),
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
            self._head_specs = resolve_checkpoint_head_specs({
                "head_specs": head_specs_metadata(),
            })
            self._head_specs_metadata = head_specs_metadata(self._head_specs)
            self._state_input_config = default_inference_state_input_config()
            self._delta_speed_transform = legacy_delta_speed_target_transform()
            self._width_multiplier = DEFAULT_WIDTH_MULTIPLIER
            self._image_offsets = []
            self._telemetry_offsets = []
            self._future_offsets = []
            self._telemetry_feature_names = []
            self._control_target_names = []
            self._aux_target_names = []
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
            head_specs = self._head_specs
            head_specs_metadata_payload = self._head_specs_metadata
            state_input_config = self._state_input_config
            delta_speed_transform = self._delta_speed_transform
            frame_count = self._frame_count
            frame_stride = self._frame_stride
            telemetry_offsets = list(self._telemetry_offsets)
            future_offsets = list(self._future_offsets)
            telemetry_feature_names = list(self._telemetry_feature_names)
            control_target_names = list(self._control_target_names)
            aux_target_names = list(self._aux_target_names)
            image_offsets = list(self._image_offsets)

        frames = self._extract_frames(payload)
        if planner_format == "temporal_telemetry_gru_v1":
            images = torch.stack(frames, dim=0).unsqueeze(0).to(device, non_blocking=device.type == "cuda")
            telemetry = self._extract_planner_telemetry(payload).to(device, non_blocking=device.type == "cuda")
            with torch.no_grad():
                output = model(images, telemetry)
            pred_controls = output["pred_controls"].detach().cpu()
            pred_aux = output["pred_aux"].detach().cpu()
            return {
                "checkpoint": str(checkpoint_path),
                "device": str(device),
                "planner_format": planner_format,
                "pred_controls": pred_controls.tolist(),
                "pred_aux": pred_aux.tolist(),
                "image_shape": list(images.shape),
                "telemetry_shape": list(telemetry.shape),
                "pred_controls_shape": list(pred_controls.shape),
                "pred_aux_shape": list(pred_aux.shape),
                "image_offsets": image_offsets,
                "telemetry_offsets": telemetry_offsets,
                "future_offsets": future_offsets,
                "telemetry_feature_names": telemetry_feature_names,
                "control_target_names": control_target_names,
                "aux_target_names": aux_target_names,
            }

        x = torch.cat(frames, dim=0).unsqueeze(0).to(device, non_blocking=device.type == "cuda")
        raw_state_inputs, model_state_inputs = self._extract_state_inputs(payload, state_input_config)
        model_state_inputs = {
            key: value.to(device, non_blocking=device.type == "cuda")
            for key, value in model_state_inputs.items()
        }

        with torch.no_grad():
            output = model(x, state_inputs=model_state_inputs)

        outputs = single_prediction_from_output(
            output,
            head_specs=head_specs,
            delta_speed_transform=delta_speed_transform,
        )
        control_outputs = single_control_prediction_from_output(
            output,
            head_specs=head_specs,
            delta_speed_transform=delta_speed_transform,
        )
        for spec in control_head_specs(head_specs):
            if spec.name in outputs:
                control_outputs[spec.name] = outputs[spec.name]
        if "delta_speed" in outputs:
            control_outputs["delta_speed"] = outputs["delta_speed"]
        if "move_intent" in outputs:
            move_intent_logit = float(outputs["move_intent"])
            control_outputs["move_intent_prob"] = sigmoid_probability(move_intent_logit)

        target_sources = control_target_sources(head_specs_metadata_payload)
        response = {
            "checkpoint": str(checkpoint_path),
            "device": str(device),
            "intent": build_semantic_intent(control_outputs, current_speed=raw_state_inputs.get("current_speed")),
            "outputs": outputs,
            "control_outputs": control_outputs,
            "model": {
                "width_multiplier": self._width_multiplier,
            },
            "state_inputs": state_inputs_metadata(state_input_config),
            "delta_speed_target_transform": delta_speed_transform.metadata(),
            "control_target_sources": target_sources,
            "control_semantics": control_semantics_from_sources(target_sources),
            "frame_window": {
                "size": frame_count,
                "frame_stride": frame_stride,
                "input_channels": frame_count * 3,
            },
        }
        normalized_state_inputs = {
            key: float(value.squeeze(0).item())
            for key, value in model_state_inputs.items()
        }
        if raw_state_inputs:
            response["raw_state_inputs"] = raw_state_inputs
            response["normalized_state_inputs"] = normalized_state_inputs
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
            if definition.key == "lead_vehicle_distance" and definition.key not in payload:
                has_lead = payload.get("has_lead_vehicle")
                if has_lead in (False, 0, 0.0):
                    raw_value = resolve_state_input_cap(config, definition.key)
                else:
                    raise ValueError(f"request must include {definition.key} for this checkpoint")
            else:
                if definition.key not in payload:
                    raise ValueError(f"request must include {definition.key} for this checkpoint")
                raw_value = payload[definition.key]
            if definition.key == "lead_vehicle_distance":
                has_lead_raw = payload.get("has_lead_vehicle")
                if has_lead_raw in (False, 0, 0.0):
                    raw_value = resolve_state_input_cap(config, definition.key)
            normalized = normalize_state_input_value(definition.key, raw_value, config)
            raw_state_inputs[definition.key] = (normalized >= 0.5) if definition.key == "has_lead_vehicle" else float(raw_value)
            normalized_state_inputs[definition.key] = torch.tensor([normalized], dtype=torch.float32)
        return raw_state_inputs, normalized_state_inputs


class ModelServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler_cls: type[BaseHTTPRequestHandler], *, config_path: Path):
        super().__init__(server_address, handler_cls)
        self.runtime = ModelRuntime()
        self.training = TrainingManager(config_path)
        self.config_path = config_path


class RequestHandler(BaseHTTPRequestHandler):
    server: ModelServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/healthz":
            self._write_json(HTTPStatus.OK, {"status": "ok"})
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
        try:
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
        except FileNotFoundError as exc:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except ValueError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except RuntimeError as exc:
            self._write_json(HTTPStatus.CONFLICT, {"error": str(exc)})
        except TrainingJobNotFoundError as exc:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except (TrainingJobNotPendingError, TrainingJobNotActiveError, TrainingJobNotRequeueableError) as exc:
            self._write_json(HTTPStatus.CONFLICT, {"error": str(exc)})
        except TrainingJobNotTerminalError as exc:
            self._write_json(HTTPStatus.CONFLICT, {"error": str(exc)})
        except TrainingJobRequestError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except TrainingJobError as exc:
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
        except Exception as exc:
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _handle_model_load(self, payload: dict[str, Any]) -> None:
        config_path = self.server.config_path
        if "config" in payload:
            raw_config = payload["config"]
            if not isinstance(raw_config, str) or not raw_config.strip():
                raise ValueError("config must be a non-empty string when provided")
            config_path = Path(raw_config)

        file_config = load_config(config_path)
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
    server.serve_forever()


if __name__ == "__main__":
    main()
