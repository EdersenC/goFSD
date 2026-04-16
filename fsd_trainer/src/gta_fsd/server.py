from __future__ import annotations

import argparse
import base64
import io
import json
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from inference import (
    DEFAULT_CONFIG_PATH,
    build_model,
    load_checkpoint,
    load_config,
    normalize_windows_drive_path,
    resolve_existing_path,
    select_device,
)


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
        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def status(self) -> dict[str, Any]:
        with self._lock:
            loaded = self._model is not None
            return {
                "loaded": loaded,
                "device": None if self._device is None else str(self._device),
                "checkpoint": None if self._checkpoint_path is None else str(self._checkpoint_path),
            }

    def load_model(self, checkpoint_path: Path, device_name: str) -> dict[str, Any]:
        device = select_device(device_name)
        checkpoint = load_checkpoint(checkpoint_path, device)
        model = build_model(checkpoint, device)

        with self._lock:
            self._model = model
            self._device = device
            self._checkpoint_path = checkpoint_path

        return {
            "status": "loaded",
            "checkpoint": str(checkpoint_path),
            "device": str(device),
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

        frames = self._extract_frames(payload)
        x = torch.cat(frames, dim=0).unsqueeze(0).to(device, non_blocking=device.type == "cuda")

        with torch.no_grad():
            pred = model(x).squeeze(0).detach().cpu()

        return {
            "checkpoint": str(checkpoint_path),
            "device": str(device),
            "prediction": {
                "Steering": float(pred[0].item()),
                "acceleration": float(pred[1].item()),
            },
        }

    def _extract_frames(self, payload: dict[str, Any]) -> list[torch.Tensor]:
        if "frame_paths" in payload:
            raw_paths = payload["frame_paths"]
            if not isinstance(raw_paths, list):
                raise ValueError("frame_paths must be a list of strings")
            return [self._load_image_from_path(path) for path in raw_paths]

        if "frames_base64" in payload:
            raw_frames = payload["frames_base64"]
            if not isinstance(raw_frames, list):
                raise ValueError("frames_base64 must be a list of base64 strings")
            return [self._load_image_from_base64(item) for item in raw_frames]

        raise ValueError("request must include frame_paths or frames_base64")

    def _load_image_from_path(self, raw_path: Any) -> torch.Tensor:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("frame path entries must be non-empty strings")
        image_path = Path(normalize_windows_drive_path(raw_path))
        if not image_path.is_file():
            raise FileNotFoundError(f"frame image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        return self._transform(image)

    def _load_image_from_base64(self, raw_value: Any) -> torch.Tensor:
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError("frames_base64 entries must be non-empty strings")
        payload = raw_value.split(",", 1)[-1]
        image_bytes = base64.b64decode(payload)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self._transform(image)


class ModelServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler_cls: type[BaseHTTPRequestHandler], *, config_path: Path):
        super().__init__(server_address, handler_cls)
        self.runtime = ModelRuntime()
        self.config_path = config_path


class RequestHandler(BaseHTTPRequestHandler):
    server: ModelServer

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._write_json(HTTPStatus.OK, {"status": "ok"})
            return
        if self.path == "/model":
            self._write_json(HTTPStatus.OK, self.server.runtime.status())
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
        raw_device = payload.get("device", file_config.device)

        if not isinstance(raw_checkpoint, str) or not raw_checkpoint.strip():
            raise ValueError("checkpoint must be a non-empty string")
        if not isinstance(raw_device, str) or not raw_device.strip():
            raise ValueError("device must be a non-empty string")

        checkpoint_path = resolve_existing_path(raw_checkpoint, config_path)
        result = self.server.runtime.load_model(checkpoint_path, raw_device.strip().lower())
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
