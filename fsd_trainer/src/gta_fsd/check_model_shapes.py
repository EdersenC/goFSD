from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from dataset import FsdDataset
from inference import (
    DEFAULT_CONFIG_PATH,
    PLANNER_FORMAT,
    build_model,
    load_checkpoint,
    load_config as load_inference_config,
    resolve_existing_path,
    select_device,
)
from models.planner import DrivingCNN
from target_transforms import denormalize_target_tensor, target_transform_metadata
from train import load_config as load_train_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print temporal planner model, dataset, target, and prediction shapes."
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
        help="Optional temporal checkpoint override. Defaults to inference.checkpoint from TOML when present.",
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
        help="Dataset run id override. Uses dataset.data_root/runs/<run-id> when provided.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index to inspect first. Default: 0.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for the DataLoader smoke test. Default: 2.",
    )
    return parser.parse_args()


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _tensor_shape(value: torch.Tensor | None) -> list[int] | None:
    return None if value is None else list(value.shape)


def _first_sample_rows(
    values: torch.Tensor,
    names: tuple[str, ...],
    future_offsets: tuple[int, ...],
) -> list[dict[str, float | int]]:
    sample = values[0].detach().float().cpu()
    rows: list[dict[str, float | int]] = []
    for horizon_index, offset in enumerate(future_offsets):
        row: dict[str, float | int] = {"offset": int(offset)}
        for target_index, name in enumerate(names):
            row[name] = float(sample[horizon_index, target_index].item())
        rows.append(row)
    return rows


def _fresh_model(config: Any, device: torch.device) -> DrivingCNN:
    return DrivingCNN(
        frame_count=len(config.dataset.image_offsets),
        telemetry_feature_dim=len(config.dataset.telemetry_feature_names),
        telemetry_hidden_dim=config.model.telemetry_hidden_dim,
        telemetry_sequence_length=len(config.dataset.telemetry_offsets),
        horizon=len(config.dataset.future_offsets),
        control_dim=len(config.dataset.control_target_names),
        aux_dim=len(config.dataset.aux_target_names),
        state_input_dim=len(config.state_inputs.enabled_keys()),
        width_multiplier=config.model.width_multiplier,
        dropout=config.model.dropout,
        visual_temporal_enabled=config.model.visual_temporal.enabled,
        visual_temporal_type=config.model.visual_temporal.type,
        visual_temporal_hidden_dim=config.model.visual_temporal.hidden_dim,
        visual_temporal_num_layers=config.model.visual_temporal.num_layers,
        visual_temporal_bidirectional=config.model.visual_temporal.bidirectional,
        visual_temporal_dropout=config.model.visual_temporal.dropout,
        horizon_decoder_enabled=config.model.horizon_decoder.enabled,
        horizon_embed_dim=config.model.horizon_decoder.horizon_embed_dim,
        horizon_decoder_hidden_dim=config.model.horizon_decoder.hidden_dim,
        horizon_decoder_num_layers=config.model.horizon_decoder.num_layers,
        horizon_decoder_dropout=config.model.horizon_decoder.dropout,
    ).to(device)


def _load_checkpoint_model(
    *,
    checkpoint_path: Path | None,
    config_path: Path,
    device: torch.device,
    fallback_model: DrivingCNN,
    fallback_frame_count: int,
    strict_checkpoint: bool,
) -> tuple[DrivingCNN, str | None, str | None]:
    if checkpoint_path is None:
        return fallback_model, None, None

    try:
        resolved = resolve_existing_path(str(checkpoint_path), config_path)
        checkpoint = load_checkpoint(resolved, device)
        planner_format = str(checkpoint.get("planner_format", "")).strip()
        if planner_format != PLANNER_FORMAT:
            raise ValueError(
                "Legacy scalar-head planner has been removed. "
                "Use temporal planner outputs pred_controls/pred_aux."
            )
        return build_model(checkpoint, device, fallback_frame_count), str(resolved), None
    except Exception as exc:
        if strict_checkpoint:
            raise
        return fallback_model, None, str(exc)


def _resolve_run_path(args: argparse.Namespace, config: Any) -> Path:
    if args.run_path is not None:
        return args.run_path
    if args.run_id is not None:
        return Path(config.dataset.data_root) / "runs" / args.run_id
    return Path(config.dataset.train_run_paths[0])


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config)
    device = select_device(args.device)

    checkpoint_path = args.checkpoint
    strict_checkpoint = checkpoint_path is not None
    if checkpoint_path is None:
        try:
            inference_config = load_inference_config(args.config)
            checkpoint_path = Path(inference_config.checkpoint) if inference_config.checkpoint else None
        except Exception:
            checkpoint_path = None

    model, loaded_checkpoint, checkpoint_error = _load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        config_path=args.config,
        device=device,
        fallback_model=_fresh_model(config, device),
        fallback_frame_count=len(config.dataset.image_offsets),
        strict_checkpoint=strict_checkpoint,
    )
    model.eval()

    run_path = _resolve_run_path(args, config)
    dataset = FsdDataset(
        run_paths=[run_path],
        image_size=(config.dataset.image_width, config.dataset.image_height),
        expected_window_size=config.dataset.window_size,
        image_offsets=config.dataset.image_offsets,
        telemetry_offsets=config.dataset.telemetry_offsets,
        future_offsets=config.dataset.future_offsets,
        telemetry_feature_names=config.dataset.telemetry_feature_names,
        control_target_names=config.dataset.control_target_names,
        aux_target_names=config.dataset.aux_target_names,
        target_transforms=config.dataset.target_transforms,
        state_input_config=config.state_inputs,
    )
    if len(dataset) <= 0:
        raise ValueError(f"Dataset is empty for run_path={run_path}")
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f"sample_index out of range: {args.sample_index}, dataset size={len(dataset)}")

    batch_size = max(1, int(args.batch_size))
    end_index = min(len(dataset), args.sample_index + batch_size)
    subset_indices = list(range(args.sample_index, end_index))
    if len(subset_indices) < batch_size:
        subset_indices.extend(index for index in range(len(dataset)) if index not in subset_indices)
        subset_indices = subset_indices[:batch_size]

    loader = DataLoader(Subset(dataset, subset_indices), batch_size=len(subset_indices), shuffle=False, num_workers=0)
    images, telemetry, state_inputs, target_controls, target_aux = next(iter(loader))
    images = images.to(device)
    telemetry = telemetry.to(device)
    state_inputs = state_inputs.to(device)
    target_controls = target_controls.to(device)
    target_aux = target_aux.to(device)

    with torch.no_grad():
        output = model(images, telemetry, state_inputs)

    pred_controls = output["pred_controls"]
    pred_aux = output["pred_aux"]
    denorm_pred_controls = denormalize_target_tensor(
        pred_controls,
        config.dataset.control_target_names,
        config.dataset.target_transforms,
    )
    denorm_pred_aux = denormalize_target_tensor(
        pred_aux,
        config.dataset.aux_target_names,
        config.dataset.target_transforms,
    )
    denorm_target_controls = denormalize_target_tensor(
        target_controls,
        config.dataset.control_target_names,
        config.dataset.target_transforms,
    )
    denorm_target_aux = denormalize_target_tensor(
        target_aux,
        config.dataset.aux_target_names,
        config.dataset.target_transforms,
    )

    payload: dict[str, Any] = {
        "device": str(device),
        "checkpoint": loaded_checkpoint,
        "checkpoint_error": checkpoint_error,
        "parameter_count": _parameter_count(model),
        "dataset": {
            "run_path": str(run_path),
            "dataset_size": len(dataset),
            "sample_indices": subset_indices,
        },
        "metadata": {
            "planner_format": PLANNER_FORMAT,
            "image_offsets": list(config.dataset.image_offsets),
            "telemetry_offsets": list(config.dataset.telemetry_offsets),
            "future_offsets": list(config.dataset.future_offsets),
            "telemetry_feature_names": list(config.dataset.telemetry_feature_names),
            "control_target_names": list(config.dataset.control_target_names),
            "aux_target_names": list(config.dataset.aux_target_names),
            "target_transforms": target_transform_metadata(config.dataset.target_transforms),
            "image_order": "oldest_to_newest",
            "cnn_input_channels": int(model.conv1.in_channels),
            "visual_temporal": {
                "enabled": bool(getattr(model, "visual_temporal_enabled", True)),
                "type": str(getattr(model, "visual_temporal_type", "gru")),
                "hidden_dim": int(getattr(model, "visual_temporal_hidden_dim", 256)),
                "num_layers": int(getattr(model, "visual_temporal_num_layers", 1)),
                "bidirectional": bool(getattr(model, "visual_temporal_bidirectional", False)),
                "dropout": float(getattr(model, "visual_temporal_dropout", 0.0)),
            },
            "horizon_decoder": {
                "enabled": bool(getattr(model, "horizon_decoder_enabled", True)),
                "horizon_embed_dim": int(getattr(model, "horizon_embed_dim", 32)),
                "hidden_dim": int(getattr(model, "horizon_decoder_hidden_dim", 256)),
                "num_layers": int(getattr(model, "horizon_decoder_num_layers", 2)),
                "dropout": float(getattr(model, "horizon_decoder_dropout", 0.1)),
            },
        },
        "shapes": {
            "images": _tensor_shape(images),
            "telemetry": _tensor_shape(telemetry),
            "state_inputs": _tensor_shape(state_inputs),
            "target_controls": _tensor_shape(target_controls),
            "target_aux": _tensor_shape(target_aux),
            "pred_controls": _tensor_shape(pred_controls),
            "pred_aux": _tensor_shape(pred_aux),
        },
        "first_sample_normalized": {
            "target_controls": _first_sample_rows(
                target_controls,
                config.dataset.control_target_names,
                config.dataset.future_offsets,
            ),
            "target_aux": _first_sample_rows(
                target_aux,
                config.dataset.aux_target_names,
                config.dataset.future_offsets,
            ),
            "pred_controls": _first_sample_rows(
                pred_controls,
                config.dataset.control_target_names,
                config.dataset.future_offsets,
            ),
            "pred_aux": _first_sample_rows(
                pred_aux,
                config.dataset.aux_target_names,
                config.dataset.future_offsets,
            ),
        },
        "first_sample_denormalized": {
            "target_controls": _first_sample_rows(
                denorm_target_controls,
                config.dataset.control_target_names,
                config.dataset.future_offsets,
            ),
            "target_aux": _first_sample_rows(
                denorm_target_aux,
                config.dataset.aux_target_names,
                config.dataset.future_offsets,
            ),
            "pred_controls": _first_sample_rows(
                denorm_pred_controls,
                config.dataset.control_target_names,
                config.dataset.future_offsets,
            ),
            "pred_aux": _first_sample_rows(
                denorm_pred_aux,
                config.dataset.aux_target_names,
                config.dataset.future_offsets,
            ),
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
