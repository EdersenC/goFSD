from __future__ import annotations

import argparse
import base64
import json
import threading
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

import torch

from control_client import (
    INPUT_MODE_MODEL_RAW,
    load_actuator_config,
    post_control_command,
)
from control_translation import (
    CONTROL_SEMANTICS_CONTROLLER_INPUT,
    CONTROL_SEMANTICS_SPEED_DELTA,
    CONTROL_SEMANTICS_TARGET_SPEED,
    translate_control_prediction,
)
from control_client import build_control_command
from heads import HeadSpec, head_layout_metadata, head_specs_metadata, resolve_checkpoint_head_specs
from image_io import load_rgb_tensor_from_bytes, load_rgb_tensor_from_path
from inference import (
    DEFAULT_CONFIG_PATH,
    build_model,
    load_checkpoint,
    load_config,
    normalize_windows_drive_path,
    resolve_checkpoint_frame_count,
    resolve_checkpoint_frame_stride,
    resolve_existing_path,
    select_device,
)
from model_output import single_control_prediction_from_output, single_prediction_from_output
from state_inputs import (
    CURRENT_SPEED_KEY,
    StateInputConfig,
    normalize_current_speed_value,
    state_input_config_from_metadata,
    state_inputs_metadata,
)
from target_transforms import (
    DeltaSpeedTargetTransform,
    legacy_delta_speed_target_transform,
    resolve_checkpoint_delta_speed_target_transform,
)


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
        candidates.extend([
            Path.cwd() / candidate,
            config_dir / candidate,
            project_root / candidate,
            script_dir / candidate,
        ])

    for resolved in candidates:
        if resolved.is_dir():
            return resolved.resolve()

    tried = "\n".join(f"- {path}" for path in candidates)
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
    steer_source = str(sources.get("steer", "")).lower()
    future_speed_source = str(sources.get("future_speed", "")).lower()
    delta_speed_source = str(sources.get("delta_speed", "")).lower()
    accel_source = str(sources.get("accel", "")).lower()
    if "steerinput" in steer_source and ("accelinput" in accel_source or "deltaspeedinput" in delta_speed_source):
        return CONTROL_SEMANTICS_CONTROLLER_INPUT
    if "label.future_speed" in future_speed_source or "future_speed" in future_speed_source:
        return CONTROL_SEMANTICS_TARGET_SPEED
    if "label.delta_speed" in delta_speed_source or "delta_speed" in delta_speed_source:
        return CONTROL_SEMANTICS_SPEED_DELTA
    if steer_source or accel_source or delta_speed_source or future_speed_source:
        return "vehicle_state"
    return CONTROL_SEMANTICS_CONTROLLER_INPUT


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
        self._head_specs: tuple[HeadSpec, ...] = resolve_checkpoint_head_specs({
            "head_specs": head_specs_metadata(),
        })
        self._head_specs_metadata: list[dict[str, Any]] = head_specs_metadata(self._head_specs)
        self._state_input_config = StateInputConfig()
        self._delta_speed_transform: DeltaSpeedTargetTransform = legacy_delta_speed_target_transform()
        self._image_size = (224, 224)
        self._frame_count = 3
        self._frame_stride = 2

    def status(self) -> dict[str, Any]:
        with self._lock:
            loaded = self._model is not None
            head_specs = self._head_specs
            target_sources = control_target_sources(self._head_specs_metadata)
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
        state_input_config = state_input_config_from_metadata(checkpoint.get("state_inputs"))
        head_specs = resolve_checkpoint_head_specs(checkpoint)
        head_specs_metadata_payload = checkpoint.get("head_specs", head_specs_metadata(head_specs))
        delta_speed_transform = resolve_checkpoint_delta_speed_target_transform(checkpoint)

        with self._lock:
            self._model = model
            self._device = device
            self._checkpoint_path = checkpoint_path
            self._head_specs = head_specs
            self._head_specs_metadata = head_specs_metadata_payload
            self._state_input_config = state_input_config
            self._delta_speed_transform = delta_speed_transform
            self._image_size = image_size
            self._frame_count = frame_count
            self._frame_stride = frame_stride
            self._transform = self._build_transform(image_size)

        target_sources = control_target_sources(head_specs_metadata_payload)
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
            self._head_specs = resolve_checkpoint_head_specs({
                "head_specs": head_specs_metadata(),
            })
            self._head_specs_metadata = head_specs_metadata(self._head_specs)
            self._state_input_config = StateInputConfig()
            self._delta_speed_transform = legacy_delta_speed_target_transform()
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
            head_specs = self._head_specs
            head_specs_metadata_payload = self._head_specs_metadata
            state_input_config = self._state_input_config
            delta_speed_transform = self._delta_speed_transform
            frame_count = self._frame_count
            frame_stride = self._frame_stride

        frames = self._extract_frames(payload)
        x = torch.cat(frames, dim=0).unsqueeze(0).to(device, non_blocking=device.type == "cuda")
        current_speed_raw, current_speed = self._extract_current_speed(payload, state_input_config)
        if current_speed is not None:
            current_speed = current_speed.to(device, non_blocking=device.type == "cuda")

        with torch.no_grad():
            output = model(x, current_speed=current_speed)

        target_sources = control_target_sources(head_specs_metadata_payload)
        response = {
            "checkpoint": str(checkpoint_path),
            "device": str(device),
            "outputs": single_prediction_from_output(
                output,
                head_specs=head_specs,
                delta_speed_transform=delta_speed_transform,
            ),
            "control_outputs": single_control_prediction_from_output(
                output,
                head_specs=head_specs,
                delta_speed_transform=delta_speed_transform,
            ),
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
        if current_speed_raw is not None and current_speed is not None:
            response["raw_state_inputs"] = {CURRENT_SPEED_KEY: current_speed_raw}
            response["normalized_state_inputs"] = {
                CURRENT_SPEED_KEY: float(current_speed.squeeze(0).item()),
            }
        return response

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

    def _extract_current_speed(
        self,
        payload: dict[str, Any],
        config: StateInputConfig,
    ) -> tuple[float | None, torch.Tensor | None]:
        if not config.current_speed_enabled:
            return None, None
        if CURRENT_SPEED_KEY not in payload:
            raise ValueError(f"request must include {CURRENT_SPEED_KEY} for this checkpoint")
        raw_value = payload[CURRENT_SPEED_KEY]
        normalized = normalize_current_speed_value(raw_value, config.current_speed_cap)
        return float(raw_value), torch.tensor([normalized], dtype=torch.float32)


class ModelServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler_cls: type[BaseHTTPRequestHandler], *, config_path: Path):
        super().__init__(server_address, handler_cls)
        self.runtime = ModelRuntime()
        self.config_path = config_path
        self.actuator = load_actuator_config(config_path)


class RequestHandler(BaseHTTPRequestHandler):
    server: ModelServer

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._write_json(HTTPStatus.OK, {"status": "ok"})
            return
        if self.path == "/model":
            self._write_json(HTTPStatus.OK, self.server.runtime.status())
            return
        if self.path == "/models":
            self._write_json(HTTPStatus.OK, {"models": discover_models(self.server.config_path)})
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            payload = self._read_json_body()
            if self.path == "/model/load":
                self._handle_model_load(payload)
                return
            if self.path == "/model/unload":
                self._write_json(HTTPStatus.OK, self.server.runtime.unload_model())
                return
            if self.path == "/predict":
                result = self.server.runtime.predict(payload)
                control_outputs = result.get("control_outputs", {})
                steer = float(control_outputs.get("steer", 0.0))
                control_semantics = str(result.get("control_semantics") or CONTROL_SEMANTICS_CONTROLLER_INPUT)
                longitudinal_output = float(
                    control_outputs.get("future_speed", control_outputs.get("delta_speed", 0.0))
                )
                raw_state_inputs = result.get("raw_state_inputs", {})
                current_speed = None
                if isinstance(raw_state_inputs, dict) and CURRENT_SPEED_KEY in raw_state_inputs:
                    current_speed = float(raw_state_inputs[CURRENT_SPEED_KEY])
                translated = translate_control_prediction(
                    steer,
                    longitudinal_output,
                    control_semantics,
                    current_speed=current_speed,
                )
                sequence = self._read_int(payload.get("sequence"))
                timestamp_ms = self._read_int(payload.get("timestamp_ms"))
                command = build_control_command(
                    steering=float(translated["steering"]),
                    acceleration=float(translated["acceleration"]),
                    input_mode=str(translated["input_mode"] or INPUT_MODE_MODEL_RAW),
                    sequence=sequence,
                    timestamp_ms=timestamp_ms,
                )
                result["translated_control"] = translated
                print(json.dumps({
                    "event": "predict_to_actuator",
                    "sequence": sequence,
                    "timestamp_ms": timestamp_ms,
                    "checkpoint": result.get("checkpoint"),
                    "control_semantics": control_semantics,
                    "outputs": result.get("outputs", {}),
                    "control_outputs": control_outputs,
                    "translated_control": translated,
                    "actuator_command": command,
                }, separators=(",", ":")), flush=True)
                post_control_command(self.server.actuator, command)
                self._write_json(HTTPStatus.OK, result)
                return
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except FileNotFoundError as exc:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except ValueError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except RuntimeError as exc:
            self._write_json(HTTPStatus.CONFLICT, {"error": str(exc)})
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
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length == 0:
            return {}
        raw = self.rfile.read(content_length)
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
