from __future__ import annotations

import argparse
import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from dataset import FsdDataset
from models.planner import DrivingCNN


SUPPORTED_DEVICES = {"auto", "cpu", "cuda"}
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "train_config.toml"


@dataclass(frozen=True)
class InferenceConfig:
    checkpoint: str
    device: str
    data_root: str | None
    run_id: str | None
    sample_index: int
    output_json: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a trained planner checkpoint and run inference.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the TOML config. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to the .pt checkpoint you want to load.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=sorted(SUPPORTED_DEVICES),
        help="Device to use for inference.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Dataset root for sample inference.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run id to load from the dataset for sample inference.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Dataset sample index to run inference on.",
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Print structured JSON instead of plain text.",
    )
    return parser.parse_args()


def normalize_windows_drive_path(value: str) -> str:
    cleaned = value.strip().strip("\"'")
    if os.name == "nt" and len(cleaned) >= 2 and cleaned[1] == ":":
        if len(cleaned) == 2:
            cleaned += "\\"
        elif cleaned[2] not in ("\\", "/"):
            cleaned = f"{cleaned[:2]}\\{cleaned[2:]}"
    return cleaned


def load_config(path: Path) -> InferenceConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    inference_raw = raw.get("inference", {})
    dataset_raw = raw.get("dataset", {})

    checkpoint = str(inference_raw.get("checkpoint", "")).strip()
    if not checkpoint:
        raise ValueError(f"Missing inference.checkpoint in {path}")

    data_root_raw = inference_raw.get("data_root", dataset_raw.get("data_root"))
    data_root = None if data_root_raw is None else normalize_windows_drive_path(str(data_root_raw))
    run_id_raw = inference_raw.get("run_id")
    run_id = None if run_id_raw is None else str(run_id_raw).strip()

    return InferenceConfig(
        checkpoint=checkpoint,
        device=str(inference_raw.get("device", "auto")).strip().lower(),
        data_root=data_root,
        run_id=run_id,
        sample_index=int(inference_raw.get("sample_index", 0)),
        output_json=bool(inference_raw.get("output_json", False)),
    )


def resolve_config(args: argparse.Namespace) -> InferenceConfig:
    file_config = load_config(args.config)
    checkpoint = str(args.checkpoint) if args.checkpoint is not None else file_config.checkpoint
    device = args.device if args.device is not None else file_config.device
    data_root = normalize_windows_drive_path(args.data_root) if args.data_root is not None else file_config.data_root
    run_id = args.run_id if args.run_id is not None else file_config.run_id
    sample_index = args.sample_index if args.sample_index is not None else file_config.sample_index
    output_json = args.output_json or file_config.output_json

    if device not in SUPPORTED_DEVICES:
        raise ValueError(f"device must be one of: {', '.join(sorted(SUPPORTED_DEVICES))}")

    return InferenceConfig(
        checkpoint=checkpoint,
        device=device,
        data_root=data_root,
        run_id=run_id,
        sample_index=sample_index,
        output_json=output_json,
    )


def resolve_existing_path(raw_path: str, config_path: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate

    config_dir = config_path.resolve().parent
    project_root = config_dir.parent
    script_dir = Path(__file__).resolve().parent

    candidates = [
        Path.cwd() / candidate,
        config_dir / candidate,
        project_root / candidate,
        script_dir / candidate,
    ]

    for resolved in candidates:
        if resolved.is_file():
            return resolved

    tried = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        f"Checkpoint does not exist: {raw_path}\nTried:\n{tried}"
    )


def select_device(requested: str) -> torch.device:
    if requested not in SUPPORTED_DEVICES:
        raise ValueError(f"device must be one of: {', '.join(sorted(SUPPORTED_DEVICES))}")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint payload in {checkpoint_path}")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint is missing model_state_dict: {checkpoint_path}")
    return checkpoint


def build_model(checkpoint: dict[str, Any], device: torch.device) -> DrivingCNN:
    model = DrivingCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def run_sample_inference(
    model: DrivingCNN,
    device: torch.device,
    *,
    data_root: str,
    run_id: str,
    sample_index: int,
) -> dict[str, Any]:
    dataset = FsdDataset(run_id=run_id, data_root=data_root)
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample_index out of range: {sample_index}, dataset size={len(dataset)}")

    x, y = dataset[sample_index]
    sample_meta = dataset.samples[sample_index]
    x = x.unsqueeze(0).to(device, non_blocking=device.type == "cuda")

    with torch.no_grad():
        pred = model(x).squeeze(0).detach().cpu()

    return {
        "sample_index": sample_index,
        "frame_paths": [str(path) for path in sample_meta.get("frame_paths", [])],
        "prediction": {
            "Steering": float(pred[0].item()),
            "acceleration": float(pred[1].item()),
        },
        "target": {
            "Steering": float(y[0].item()),
            "acceleration": float(y[1].item()),
        },
    }


def build_output(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    device: torch.device,
    sample_result: dict[str, Any] | None,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "epoch": int(checkpoint.get("epoch", 0)),
        "train_metrics": checkpoint.get("train_metrics"),
        "val_metrics": checkpoint.get("val_metrics"),
    }
    if sample_result is not None:
        output["sample"] = sample_result
    return output


def print_plain(output: dict[str, Any]) -> None:
    print(f"Loaded checkpoint: {output['checkpoint']}")
    print(f"Device: {output['device']}")
    print(f"Epoch: {output['epoch']}")

    val_metrics = output.get("val_metrics") or {}
    if isinstance(val_metrics, dict) and val_metrics:
        print(
            "Validation metrics: "
            f"loss={float(val_metrics.get('loss', 0.0)):.6f} "
            f"overall_mae={float(val_metrics.get('overall_mae', 0.0)):.6f} "
            f"overall_rmse={float(val_metrics.get('overall_rmse', 0.0)):.6f}"
        )

    sample = output.get("sample")
    if isinstance(sample, dict):
        prediction = sample["prediction"]
        target = sample["target"]
        print()
        print(f"Sample index: {sample['sample_index']}")
        print(f"Prediction: Steering={prediction['Steering']:.6f} acceleration={prediction['acceleration']:.6f}")
        print(f"Target:     Steering={target['Steering']:.6f} acceleration={target['acceleration']:.6f}")
        frame_paths = sample.get("frame_paths") or []
        if frame_paths:
            print("Frames:")
            for frame_path in frame_paths:
                print(f"  {frame_path}")


def main() -> None:
    args = parse_args()
    config = resolve_config(args)
    device = select_device(config.device)
    checkpoint_path = resolve_existing_path(config.checkpoint, args.config)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = build_model(checkpoint, device)

    sample_result: dict[str, Any] | None = None
    if config.run_id or config.data_root:
        if not config.run_id or not config.data_root:
            raise ValueError("Both inference.run_id and inference.data_root are required for sample inference")
        sample_result = run_sample_inference(
            model,
            device,
            data_root=config.data_root,
            run_id=config.run_id,
            sample_index=config.sample_index,
        )

    output = build_output(checkpoint_path, checkpoint, device, sample_result)
    if config.output_json:
        print(json.dumps(output, indent=2))
    else:
        print_plain(output)


if __name__ == "__main__":
    main()
