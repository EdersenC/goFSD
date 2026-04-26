from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from dataset import FsdDataset
from heads import HEAD_SPECS, head_layout_metadata
from inference import (
    DEFAULT_CONFIG_PATH,
    build_model,
    load_checkpoint,
    load_config as load_inference_config,
    remap_legacy_state_dict_keys,
    resolve_existing_path,
    select_device,
)
from models.planner import DrivingCNN
from state_inputs import StateInputConfig, training_state_input_config
from train import load_config as load_train_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print planner output shapes and attempt to batch a real dataset sample."
    )
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
        help="Optional checkpoint override. Defaults to inference.checkpoint from the TOML when present.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("auto", "cpu", "cuda"),
        help="Device for the shape check. Default: cpu.",
    )
    parser.add_argument(
        "--run-path",
        type=Path,
        default=None,
        help="Optional dataset run-path override. Defaults to the first configured training run path.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Deprecated run id override. Uses dataset.data_root/runs/<run-id> when provided.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index to inspect. Default: 0.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for the DataLoader smoke test. Default: 2.",
    )
    return parser.parse_args()


def _shape_map(mapping: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    return {key: list(value.shape) for key, value in mapping.items()}


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _stack_targets(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        key: torch.stack([sample[key] for sample in samples], dim=0)
        for key in samples[0]
    }


def _collect_valid_samples(
    dataset: FsdDataset,
    *,
    preferred_index: int,
    desired_count: int,
) -> tuple[list[int], list[torch.Tensor], list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]], list[str]]:
    ordered_indices = [preferred_index] + [index for index in range(len(dataset)) if index != preferred_index]
    selected_indices: list[int] = []
    selected_x: list[torch.Tensor] = []
    selected_state_inputs: list[dict[str, torch.Tensor]] = []
    selected_targets: list[dict[str, torch.Tensor]] = []
    skipped_errors: list[str] = []

    for index in ordered_indices:
        try:
            sample_x, sample_state_inputs, sample_targets = dataset[index]
        except Exception as exc:
            skipped_errors.append(f"sample {index}: {exc}")
            continue
        selected_indices.append(index)
        selected_x.append(sample_x)
        selected_state_inputs.append(sample_state_inputs)
        selected_targets.append(sample_targets)
        if len(selected_indices) >= desired_count:
            break

    if not selected_indices:
        raise ValueError(
            "Could not load any valid dataset samples for batching. "
            f"First errors: {skipped_errors[:5]}"
        )
    return selected_indices, selected_x, selected_state_inputs, selected_targets, skipped_errors


def _load_optional_checkpoint(model: DrivingCNN, checkpoint_path: Path | None, config_path: Path, device: torch.device) -> str | None:
    if checkpoint_path is None:
        return None
    resolved = resolve_existing_path(str(checkpoint_path), config_path)
    checkpoint = load_checkpoint(resolved, device)
    model.load_state_dict(remap_legacy_state_dict_keys(checkpoint["model_state_dict"]))
    return str(resolved)


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config)
    device = select_device(args.device)
    inference_config = load_inference_config(args.config)

    model = DrivingCNN(
        frame_count=config.dataset.window_size,
        state_input_config=training_state_input_config(),
        width_multiplier=config.model.width_multiplier,
    ).to(device)
    checkpoint_path = args.checkpoint
    if checkpoint_path is None and inference_config.checkpoint:
        checkpoint_path = Path(inference_config.checkpoint)
    checkpoint_error: str | None = None
    loaded_checkpoint: str | None = None
    try:
        loaded_checkpoint = _load_optional_checkpoint(model, checkpoint_path, args.config, device)
        if checkpoint_path is not None:
            checkpoint = load_checkpoint(resolve_existing_path(str(checkpoint_path), args.config), device)
            model = build_model(checkpoint, device, config.dataset.window_size)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        checkpoint_error = str(exc)
    model.eval()

    dummy = torch.zeros(
        (1, config.dataset.window_size * 3, config.dataset.image_height, config.dataset.image_width),
        dtype=torch.float32,
        device=device,
    )
    dummy_state_inputs = {
        key: torch.zeros((1,), dtype=torch.float32, device=device)
        for key in model.state_input_config.enabled_keys()
    }
    with torch.no_grad():
        dummy_outputs = model(dummy, state_inputs=dummy_state_inputs)
    model_without_speed = DrivingCNN(
        frame_count=config.dataset.window_size,
        state_input_config=StateInputConfig(),
        width_multiplier=config.model.width_multiplier,
    ).to(device)
    model_without_speed.eval()
    with torch.no_grad():
        dummy_outputs_without_speed = model_without_speed(dummy)

    payload: dict[str, Any] = {
        "device": str(device),
        "frame_window": {
            "size": config.dataset.window_size,
            "frame_stride": config.dataset.frame_stride,
            "sample_stride": config.dataset.sample_stride,
            "input_channels": config.dataset.window_size * 3,
        },
        "head_layout": head_layout_metadata(),
        "parameter_count": _parameter_count(model),
        "dummy_output_shapes": _shape_map(dummy_outputs),
        "dummy_output_shapes_without_current_speed": _shape_map(dummy_outputs_without_speed),
    }
    if loaded_checkpoint is not None:
        payload["checkpoint"] = loaded_checkpoint
    if checkpoint_error is not None:
        payload["checkpoint_error"] = checkpoint_error

    run_path = args.run_path
    if run_path is None and args.run_id is not None:
        run_path = Path(config.dataset.data_root) / "runs" / args.run_id
    if run_path is None:
        run_path = Path(config.dataset.train_run_paths[0])
    dataset = FsdDataset(
        run_paths=[run_path],
        image_size=(config.dataset.image_width, config.dataset.image_height),
        expected_window_size=config.dataset.window_size,
        state_input_config=config.state_inputs,
    )
    if len(dataset) <= 0:
        raise ValueError(f"Dataset is empty for run_path={run_path}")
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f"sample_index out of range: {args.sample_index}, dataset size={len(dataset)}")

    payload["dataset"] = {
        "run_path": str(run_path),
        "dataset_size": len(dataset),
        "head_names": [spec.name for spec in HEAD_SPECS],
    }
    try:
        selected_indices, selected_x, selected_state_inputs, selected_targets, skipped_errors = _collect_valid_samples(
            dataset,
            preferred_index=args.sample_index,
            desired_count=max(1, args.batch_size),
        )
    except ValueError as exc:
        payload["dataset"]["error"] = str(exc)
    else:
        sample_x = selected_x[0]
        sample_state_inputs = selected_state_inputs[0]
        sample_targets = selected_targets[0]
        batch_x = torch.stack(selected_x, dim=0)
        batch_state_inputs = _stack_targets(selected_state_inputs)
        batch_targets = _stack_targets(selected_targets)
        payload["dataset"].update({
            "sample_index": selected_indices[0],
            "sample_x_shape": list(sample_x.shape),
            "sample_state_input_shapes": _shape_map(sample_state_inputs),
            "sample_target_shapes": _shape_map(sample_targets),
            "batch_x_shape": list(batch_x.shape),
            "batch_state_input_shapes": _shape_map(batch_state_inputs),
            "batch_target_shapes": _shape_map(batch_targets),
            "batched_sample_indices": selected_indices,
            "skipped_invalid_samples": skipped_errors[:10],
        })

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
