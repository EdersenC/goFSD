from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from config import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    normalize_windows_drive_path,
    parse_dataset_window,
)
from dataset import FsdDataset
from heads import (
    head_layout_metadata,
    head_specs_metadata,
    resolve_checkpoint_head_specs,
    validate_checkpoint_head_layout,
)
from model_output import (
    single_control_prediction_from_output,
    single_prediction_from_output,
    single_tensor_mapping,
)
from models.planner import DrivingCNN
from state_inputs import (
    CURRENT_SPEED_KEY,
    ROUTE_FORWARD_DELTA_KEY,
    state_input_config_from_metadata,
    state_inputs_metadata,
)
from target_transforms import (
    legacy_delta_speed_target_transform,
    resolve_checkpoint_delta_speed_target_transform,
)


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
    image_width: int
    image_height: int
    window_size: int
    frame_stride: int
    sample_stride: int


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


def load_config(path: Path) -> InferenceConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    inference_raw = raw.get("inference", {})
    dataset_raw = raw.get("dataset", {})
    window_size, frame_stride, sample_stride = parse_dataset_window(raw)

    checkpoint = str(inference_raw.get("checkpoint", "")).strip()
    if not checkpoint:
        raise ValueError(f"Missing inference.checkpoint in {path}")

    data_root_raw = inference_raw.get("data_root", dataset_raw.get("data_root"))
    data_root = None if data_root_raw is None else normalize_windows_drive_path(str(data_root_raw))
    run_id_raw = inference_raw.get("run_id")
    run_id = None if run_id_raw is None else str(run_id_raw).strip()
    image_width = int(inference_raw.get("image_width", dataset_raw.get("image_width", DEFAULT_IMAGE_WIDTH)))
    image_height = int(inference_raw.get("image_height", dataset_raw.get("image_height", DEFAULT_IMAGE_HEIGHT)))

    return InferenceConfig(
        checkpoint=checkpoint,
        device=str(inference_raw.get("device", "auto")).strip().lower(),
        data_root=data_root,
        run_id=run_id,
        sample_index=int(inference_raw.get("sample_index", 0)),
        output_json=bool(inference_raw.get("output_json", False)),
        image_width=image_width,
        image_height=image_height,
        window_size=window_size,
        frame_stride=frame_stride,
        sample_stride=sample_stride,
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
        image_width=file_config.image_width,
        image_height=file_config.image_height,
        window_size=file_config.window_size,
        frame_stride=file_config.frame_stride,
        sample_stride=file_config.sample_stride,
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
    raise FileNotFoundError(f"Checkpoint does not exist: {raw_path}\nTried:\n{tried}")


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
    validate_checkpoint_head_layout(checkpoint)
    return checkpoint


def remap_legacy_state_dict_keys(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    remapped: dict[str, Any] = {}
    legacy_head_prefixes = (
        "heads.steer.",
        "heads.future_steer.",
        "heads.steering.",
    )
    for key, value in state_dict.items():
        new_key = key
        for legacy_prefix in legacy_head_prefixes:
            if key.startswith(legacy_prefix):
                new_key = f"heads.future_yaw_delta.{key[len(legacy_prefix):]}"
                break
        remapped[new_key] = value
    return remapped


def build_model(checkpoint: dict[str, Any], device: torch.device, frame_count: int) -> DrivingCNN:
    state_input_config = state_input_config_from_metadata(checkpoint.get("state_inputs"))
    head_specs = resolve_checkpoint_head_specs(checkpoint)
    model = DrivingCNN(frame_count=frame_count, head_specs=head_specs, state_input_config=state_input_config).to(device)
    model.load_state_dict(remap_legacy_state_dict_keys(checkpoint["model_state_dict"]))
    model.delta_speed_target_transform = resolve_checkpoint_delta_speed_target_transform(checkpoint)
    model.eval()
    return model


def resolve_checkpoint_frame_count(checkpoint: dict[str, Any], config: InferenceConfig) -> int:
    checkpoint_frame_count = checkpoint.get("frame_window_size")
    if checkpoint_frame_count is not None:
        expected = int(checkpoint_frame_count)
        if expected != config.window_size:
            raise ValueError(
                "Config/checkpoint frame-window mismatch: "
                f"config dataset.window_size={config.window_size}, "
                f"checkpoint frame_window_size={expected}"
            )
        return expected
    return config.window_size


def resolve_checkpoint_frame_stride(checkpoint: dict[str, Any], config: InferenceConfig) -> int:
    checkpoint_frame_stride = checkpoint.get("frame_stride", checkpoint.get("frame_window_stride"))
    if checkpoint_frame_stride is not None:
        expected = int(checkpoint_frame_stride)
        if expected != config.frame_stride:
            raise ValueError(
                "Config/checkpoint frame-stride mismatch: "
                f"config dataset.frame_stride={config.frame_stride}, "
                f"checkpoint frame_stride={expected}"
            )
        return expected
    return config.frame_stride


def checkpoint_sample_stride(checkpoint: dict[str, Any]) -> int:
    return int(checkpoint.get("sample_stride", 0) or 0)


def run_sample_inference(
    model: DrivingCNN,
    device: torch.device,
    *,
    data_root: str,
    run_id: str,
    sample_index: int,
    image_size: tuple[int, int],
    expected_window_size: int,
) -> dict[str, Any]:
    dataset = FsdDataset(
        run_id=run_id,
        data_root=data_root,
        image_size=image_size,
        expected_window_size=expected_window_size,
        state_input_config=model.state_input_config,
        head_specs=model.head_specs,
    )
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample_index out of range: {sample_index}, dataset size={len(dataset)}")

    x, state_inputs, targets = dataset[sample_index]
    sample_meta = dataset.samples[sample_index]
    x = x.unsqueeze(0).to(device, non_blocking=device.type == "cuda")
    current_speed = state_inputs.get(CURRENT_SPEED_KEY)
    route_forward_delta = state_inputs.get(ROUTE_FORWARD_DELTA_KEY)
    if current_speed is not None:
        current_speed = current_speed.unsqueeze(0).to(device, non_blocking=device.type == "cuda")
    if route_forward_delta is not None:
        route_forward_delta = route_forward_delta.unsqueeze(0).to(device, non_blocking=device.type == "cuda")

    with torch.no_grad():
        output = model(x, current_speed=current_speed, route_forward_delta=route_forward_delta)
    head_specs = model.head_specs
    delta_speed_transform = getattr(model, "delta_speed_target_transform", legacy_delta_speed_target_transform())

    return {
        "sample_index": sample_index,
        "frame_paths": [str(path) for path in sample_meta.get("frame_paths", [])],
        "state_inputs": single_tensor_mapping(state_inputs),
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
        "target": single_tensor_mapping(targets, delta_speed_transform=delta_speed_transform),
    }


def build_output(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    device: torch.device,
    sample_result: dict[str, Any] | None,
) -> dict[str, Any]:
    head_specs = resolve_checkpoint_head_specs(checkpoint)
    output: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "epoch": int(checkpoint.get("epoch", 0)),
        "frame_window_size": int(checkpoint.get("frame_window_size", 0) or 0),
        "frame_stride": int(checkpoint.get("frame_stride", checkpoint.get("frame_window_stride", 0)) or 0),
        "sample_stride": checkpoint_sample_stride(checkpoint),
        "input_channels": int(checkpoint.get("input_channels", 0) or 0),
        "delta_speed_target_transform": resolve_checkpoint_delta_speed_target_transform(checkpoint).metadata(),
        "head_specs": checkpoint.get("head_specs", head_specs_metadata(head_specs)),
        "head_layout": checkpoint.get("head_layout", head_layout_metadata(head_specs)),
        "state_inputs": checkpoint.get("state_inputs", state_inputs_metadata(state_input_config_from_metadata(None))),
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
    if int(output.get("frame_window_size", 0) or 0) > 0:
        print(
            "Frame window: "
            f"size={output['frame_window_size']} "
            f"frame_stride={output['frame_stride']} "
            f"sample_stride={output['sample_stride']} "
            f"input_channels={output['input_channels']}"
        )
    state_inputs = output.get("state_inputs") or {}
    current_speed = state_inputs.get("current_speed") if isinstance(state_inputs, dict) else None
    if isinstance(current_speed, dict) and current_speed.get("enabled"):
        print(
            "State inputs: "
            f"current_speed enabled cap={float(current_speed.get('cap', 0.0)):.3f} "
            f"fusion={current_speed.get('fusion', 'unknown')}"
        )
    route_forward_delta = state_inputs.get("route_forward_delta") if isinstance(state_inputs, dict) else None
    if isinstance(route_forward_delta, dict) and route_forward_delta.get("enabled"):
        print(
            "State inputs: "
            f"route_forward_delta enabled cap={float(route_forward_delta.get('cap', 0.0)):.3f} "
            f"fusion={route_forward_delta.get('fusion', 'unknown')}"
        )

    val_metrics = output.get("val_metrics") or {}
    if isinstance(val_metrics, dict) and val_metrics:
        print(
            "Validation metrics: "
            f"loss={float(val_metrics.get('loss', 0.0)):.6f} "
            f"control_mae={float(val_metrics.get('control_overall_mae', val_metrics.get('overall_mae', 0.0))):.6f} "
            f"control_rmse={float(val_metrics.get('control_overall_rmse', val_metrics.get('overall_rmse', 0.0))):.6f}"
        )

    sample = output.get("sample")
    if isinstance(sample, dict):
        control_outputs = sample["control_outputs"]
        all_outputs = sample["outputs"]
        target = sample["target"]
        aux_outputs = {key: value for key, value in all_outputs.items() if key not in control_outputs}
        print()
        print(f"Sample index: {sample['sample_index']}")
        print(f"Control outputs: {control_outputs}")
        if aux_outputs:
            print(f"Aux outputs:     {aux_outputs}")
        print(f"Targets:         {target}")
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
    frame_count = resolve_checkpoint_frame_count(checkpoint, config)
    _ = resolve_checkpoint_frame_stride(checkpoint, config)
    model = build_model(checkpoint, device, frame_count)

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
            image_size=(config.image_width, config.image_height),
            expected_window_size=config.window_size,
        )

    output = build_output(checkpoint_path, checkpoint, device, sample_result)
    if config.output_json:
        print(json.dumps(output, indent=2))
    else:
        print_plain(output)


if __name__ == "__main__":
    main()
