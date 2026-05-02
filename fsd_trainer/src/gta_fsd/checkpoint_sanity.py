from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from dataset import FsdDataset
from inference import (
    DEFAULT_CONFIG_PATH,
    PLANNER_FORMAT,
    build_model,
    checkpoint_sample_stride,
    load_checkpoint,
    load_config as load_inference_config,
    resolve_checkpoint_frame_count,
    resolve_checkpoint_frame_stride,
    resolve_checkpoint_target_names,
    resolve_checkpoint_target_transform_registry,
    resolve_existing_path,
    select_device,
)
from state_inputs import state_input_config_from_metadata, state_inputs_metadata
from target_transforms import denormalize_target_tensor, target_transform_metadata
from train import TrainConfig, load_config as load_train_config


DEBUG_FRAMES_ROOT_NAME = "awesomeProject-inference-frames"
DEFAULT_SAMPLE_COUNT = 10
DEFAULT_DUMP_WINDOW_COUNT = 5
FLAT_CONTROL_RANGE_EPSILON = 1e-4
LEGACY_SCALAR_HEAD_ERROR = (
    "Legacy scalar-head planner has been removed. "
    "Use temporal planner outputs pred_controls/pred_aux."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a temporal checkpoint sanity pass against dataset samples."
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
        help="Optional temporal checkpoint override. Defaults to inference.checkpoint from the TOML.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=("auto", "cpu", "cuda"),
        help="Optional device override. Defaults to inference.device from the TOML.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Dataset run id override. Defaults to inference.run_id, then first val run, then first train run.",
    )
    parser.add_argument(
        "--sample-start",
        type=int,
        default=None,
        help="Dataset sample index to start from. Defaults to inference.sample_index.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help=f"Number of dataset samples to inspect. Default: {DEFAULT_SAMPLE_COUNT}.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Accepted for old command lines, but image-only debug frame checks are not valid for temporal planners.",
    )
    parser.add_argument(
        "--dump-window-count",
        type=int,
        default=DEFAULT_DUMP_WINDOW_COUNT,
        help="Accepted for old command lines; temporal sanity uses dataset telemetry instead.",
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Print structured JSON instead of plain text.",
    )
    return parser.parse_args()


def resolve_dataset_run_id(train_config: TrainConfig, inferred_run_id: str | None, override_run_id: str | None) -> str:
    if override_run_id is not None and override_run_id.strip():
        return override_run_id.strip()
    if inferred_run_id is not None and inferred_run_id.strip():
        return inferred_run_id.strip()
    if train_config.dataset.val_run_ids:
        return train_config.dataset.val_run_ids[0]
    if train_config.dataset.train_run_ids:
        return train_config.dataset.train_run_ids[0]
    raise ValueError("No dataset run ids are configured for checkpoint sanity checks")


def resolve_debug_frames_root() -> Path:
    return Path(tempfile.gettempdir()) / DEBUG_FRAMES_ROOT_NAME


def resolve_latest_debug_dir(root: Path) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"Debug frame root does not exist: {root}")

    candidates = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No debug frame dumps found under {root}")
    return candidates[0]


def resolve_debug_dir(override: Path | None) -> Path:
    if override is not None:
        debug_dir = override.expanduser().resolve()
        if not debug_dir.is_dir():
            raise FileNotFoundError(f"Debug frame directory does not exist: {debug_dir}")
        return debug_dir
    return resolve_latest_debug_dir(resolve_debug_frames_root())


def build_debug_windows(
    frame_paths: list[Path],
    *,
    window_size: int,
    frame_stride: int,
    limit: int,
) -> list[list[Path]]:
    if window_size < 1:
        raise ValueError("window_size must be > 0")
    if frame_stride < 1:
        raise ValueError("frame_stride must be > 0")
    if limit < 1:
        raise ValueError("limit must be > 0")

    required_frame_count = ((window_size - 1) * frame_stride) + 1
    if len(frame_paths) < required_frame_count:
        raise ValueError(
            "Not enough debug frames to build one inference window: "
            f"need at least {required_frame_count}, found {len(frame_paths)}"
        )

    max_start = len(frame_paths) - required_frame_count
    windows: list[list[Path]] = []
    for start_index in range(max_start + 1):
        window = [frame_paths[start_index + (offset * frame_stride)] for offset in range(window_size)]
        windows.append(window)
        if len(windows) >= limit:
            break
    return windows


def summarize_control_ranges(items: Iterable[dict[str, Any]]) -> dict[str, dict[str, float | bool]]:
    results = list(items)
    summary: dict[str, dict[str, float | bool]] = {}
    control_names: list[str] = []
    seen_control_names: set[str] = set()
    for item in results:
        names = item.get("control_target_names")
        if not isinstance(names, list):
            continue
        for raw_name in names:
            control_name = str(raw_name)
            if control_name in seen_control_names:
                continue
            seen_control_names.add(control_name)
            control_names.append(control_name)
    for control_index, control_name in enumerate(control_names):
        values: list[float] = []
        for item in results:
            pred_controls = item.get("pred_controls")
            if not isinstance(pred_controls, list):
                continue
            for batch in pred_controls:
                if not isinstance(batch, list):
                    continue
                for row in batch:
                    if not isinstance(row, list) or control_index >= len(row):
                        continue
                    values.append(float(row[control_index]))
        if not values:
            continue
        min_value = min(values)
        max_value = max(values)
        range_value = max_value - min_value
        summary[control_name] = {
            "min": min_value,
            "max": max_value,
            "mean": float(sum(values) / len(values)),
            "range": range_value,
            "flat": range_value <= FLAT_CONTROL_RANGE_EPSILON,
        }
    return summary


def _require_temporal_checkpoint(checkpoint: dict[str, Any]) -> None:
    if str(checkpoint.get("planner_format", "")).strip() != PLANNER_FORMAT:
        raise ValueError(LEGACY_SCALAR_HEAD_ERROR)


def _rows(
    values: torch.Tensor,
    names: tuple[str, ...],
    future_offsets: tuple[int, ...],
) -> list[dict[str, float | int]]:
    sample = values.detach().float().cpu()
    rows: list[dict[str, float | int]] = []
    for horizon_index, offset in enumerate(future_offsets):
        row: dict[str, float | int] = {"offset": int(offset)}
        for target_index, name in enumerate(names):
            row[name] = float(sample[horizon_index, target_index].item())
        rows.append(row)
    return rows


def _mean_or_zero(total: float, count: int) -> float:
    return 0.0 if count <= 0 else float(total / count)


def _init_target_sums(names: tuple[str, ...]) -> dict[str, dict[str, float | int]]:
    return {name: {"abs_sum": 0.0, "denorm_abs_sum": 0.0, "count": 0} for name in names}


def _update_target_sums(
    sums: dict[str, dict[str, float | int]],
    *,
    names: tuple[str, ...],
    normalized_abs: torch.Tensor,
    denorm_abs: torch.Tensor,
) -> None:
    for index, name in enumerate(names):
        sums[name]["abs_sum"] = float(sums[name]["abs_sum"]) + float(normalized_abs[..., index].sum().item())
        sums[name]["denorm_abs_sum"] = float(sums[name]["denorm_abs_sum"]) + float(denorm_abs[..., index].sum().item())
        sums[name]["count"] = int(sums[name]["count"]) + int(normalized_abs[..., index].numel())


def _finalize_target_sums(sums: dict[str, dict[str, float | int]]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "normalized_mae": _mean_or_zero(float(values["abs_sum"]), int(values["count"])),
            "denormalized_mae": _mean_or_zero(float(values["denorm_abs_sum"]), int(values["count"])),
        }
        for name, values in sums.items()
    }


def evaluate_dataset_samples(
    model: Any,
    device: torch.device,
    *,
    data_root: str,
    image_size: tuple[int, int],
    expected_window_size: int,
    run_id: str,
    sample_start: int,
    sample_count: int,
    image_offsets: tuple[int, ...],
    telemetry_offsets: tuple[int, ...],
    future_offsets: tuple[int, ...],
    telemetry_feature_names: tuple[str, ...],
    control_target_names: tuple[str, ...],
    aux_target_names: tuple[str, ...],
    target_transforms: dict[str, Any],
) -> dict[str, Any]:
    dataset = FsdDataset(
        run_id=run_id,
        data_root=data_root,
        image_size=image_size,
        expected_window_size=expected_window_size,
        image_offsets=image_offsets,
        telemetry_offsets=telemetry_offsets,
        future_offsets=future_offsets,
        telemetry_feature_names=telemetry_feature_names,
        control_target_names=control_target_names,
        aux_target_names=aux_target_names,
        target_transforms=target_transforms,
        state_input_config=getattr(model, "state_input_config", None),
    )
    if sample_start < 0:
        raise ValueError("sample_start must be >= 0")
    if sample_count < 1:
        raise ValueError("sample_count must be > 0")
    if sample_start >= len(dataset):
        raise IndexError(f"sample_start out of range: {sample_start}, dataset size={len(dataset)}")

    sample_end = min(len(dataset), sample_start + sample_count)
    control_sums = _init_target_sums(control_target_names)
    aux_sums = _init_target_sums(aux_target_names)
    results: list[dict[str, Any]] = []

    with torch.no_grad():
        for sample_index in range(sample_start, sample_end):
            images, telemetry, state_inputs, target_controls, target_aux = dataset[sample_index]
            sample_meta = dataset._load_sample(sample_index)
            images_batch = images.unsqueeze(0).to(device, non_blocking=device.type == "cuda")
            telemetry_batch = telemetry.unsqueeze(0).to(device, non_blocking=device.type == "cuda")
            state_inputs_batch = state_inputs.unsqueeze(0).to(device, non_blocking=device.type == "cuda")
            target_controls_batch = target_controls.unsqueeze(0).to(device, non_blocking=device.type == "cuda")
            target_aux_batch = target_aux.unsqueeze(0).to(device, non_blocking=device.type == "cuda")

            output = model(images_batch, telemetry_batch, state_inputs_batch)
            pred_controls = output["pred_controls"]
            pred_aux = output["pred_aux"]

            control_abs = (pred_controls - target_controls_batch).abs()
            aux_abs = (pred_aux - target_aux_batch).abs()
            denorm_pred_controls = denormalize_target_tensor(pred_controls, control_target_names, target_transforms)
            denorm_target_controls = denormalize_target_tensor(target_controls_batch, control_target_names, target_transforms)
            denorm_pred_aux = denormalize_target_tensor(pred_aux, aux_target_names, target_transforms)
            denorm_target_aux = denormalize_target_tensor(target_aux_batch, aux_target_names, target_transforms)
            denorm_control_abs = (denorm_pred_controls - denorm_target_controls).abs()
            denorm_aux_abs = (denorm_pred_aux - denorm_target_aux).abs()

            _update_target_sums(
                control_sums,
                names=control_target_names,
                normalized_abs=control_abs,
                denorm_abs=denorm_control_abs,
            )
            _update_target_sums(
                aux_sums,
                names=aux_target_names,
                normalized_abs=aux_abs,
                denorm_abs=denorm_aux_abs,
            )

            results.append({
                "sample_index": sample_index,
                "trip_key": sample_meta.get("trip_key"),
                "frame_paths": [str(path) for path in sample_meta.get("frame_paths", [])],
                "shapes": {
                    "images": list(images_batch.shape),
                    "telemetry": list(telemetry_batch.shape),
                    "state_inputs": list(state_inputs_batch.shape),
                    "pred_controls": list(pred_controls.shape),
                    "pred_aux": list(pred_aux.shape),
                    "target_controls": list(target_controls_batch.shape),
                    "target_aux": list(target_aux_batch.shape),
                },
                "pred_controls": _rows(denorm_pred_controls[0], control_target_names, future_offsets),
                "target_controls": _rows(denorm_target_controls[0], control_target_names, future_offsets),
                "pred_aux": _rows(denorm_pred_aux[0], aux_target_names, future_offsets),
                "target_aux": _rows(denorm_target_aux[0], aux_target_names, future_offsets),
            })

    return {
        "run_id": run_id,
        "dataset_size": len(dataset),
        "sample_start": sample_start,
        "sample_count": len(results),
        "control_target_mae": _finalize_target_sums(control_sums),
        "aux_target_mae": _finalize_target_sums(aux_sums),
        "results": results,
    }


def build_output(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    device: torch.device,
    dataset_results: dict[str, Any],
) -> dict[str, Any]:
    _require_temporal_checkpoint(checkpoint)
    future_offsets = tuple(int(value) for value in (checkpoint.get("future_offsets") or []))
    control_target_names, aux_target_names = resolve_checkpoint_target_names(
        checkpoint,
        future_steps=len(future_offsets),
    )
    target_transforms = resolve_checkpoint_target_transform_registry(checkpoint)
    return {
        "checkpoint": {
            "path": str(checkpoint_path),
            "device": str(device),
            "epoch": int(checkpoint.get("epoch", 0) or 0),
            "planner_format": PLANNER_FORMAT,
            "frame_window_size": int(checkpoint.get("frame_window_size", 0) or 0),
            "frame_stride": int(checkpoint.get("frame_stride", checkpoint.get("frame_window_stride", 0)) or 0),
            "sample_stride": checkpoint_sample_stride(checkpoint),
            "input_channels": int(checkpoint.get("input_channels", 0) or 0),
            "future_offsets": list(future_offsets),
            "telemetry_feature_names": list(checkpoint.get("telemetry_feature_names", [])),
            "control_target_names": list(control_target_names),
            "aux_target_names": list(aux_target_names),
            "target_transforms": target_transform_metadata(target_transforms),
            "state_inputs": checkpoint.get("state_inputs", state_inputs_metadata(state_input_config_from_metadata(None))),
        },
        "dataset_samples": dataset_results,
    }


def print_plain(output: dict[str, Any]) -> None:
    checkpoint = output["checkpoint"]
    dataset_samples = output["dataset_samples"]

    print(f"Checkpoint: {checkpoint['path']}")
    print(f"Device: {checkpoint['device']}")
    print(f"Planner format: {checkpoint['planner_format']}")
    print(
        "Model window: "
        f"size={checkpoint['frame_window_size']} "
        f"frame_stride={checkpoint['frame_stride']} "
        f"sample_stride={checkpoint['sample_stride']} "
        f"input_channels={checkpoint['input_channels']}"
    )
    print(f"Future offsets: {checkpoint['future_offsets']}")
    print(f"Control targets: {checkpoint['control_target_names']}")
    print(f"Aux targets: {checkpoint['aux_target_names']}")
    print()
    print(
        "Dataset samples: "
        f"run_id={dataset_samples['run_id']} "
        f"start={dataset_samples['sample_start']} "
        f"count={dataset_samples['sample_count']} "
        f"dataset_size={dataset_samples['dataset_size']}"
    )
    print("Control per-target MAE:")
    for target_name, values in dataset_samples.get("control_target_mae", {}).items():
        print(
            f"  {target_name}: "
            f"normalized={float(values['normalized_mae']):.6f} "
            f"denormalized={float(values['denormalized_mae']):.6f}"
        )
    print("Aux per-target MAE:")
    for target_name, values in dataset_samples.get("aux_target_mae", {}).items():
        print(
            f"  {target_name}: "
            f"normalized={float(values['normalized_mae']):.6f} "
            f"denormalized={float(values['denormalized_mae']):.6f}"
        )

    first_result = next(iter(dataset_samples.get("results", [])), None)
    if isinstance(first_result, dict):
        print()
        print(f"First sample: index={first_result['sample_index']} trip={first_result.get('trip_key')}")
        print(f"Shapes: {first_result['shapes']}")
        print(f"Pred controls: {first_result['pred_controls']}")
        print(f"Target controls: {first_result['target_controls']}")
        print(f"Pred aux: {first_result['pred_aux']}")
        print(f"Target aux: {first_result['target_aux']}")


def main() -> None:
    args = parse_args()
    train_config = load_train_config(args.config)
    inference_config = load_inference_config(args.config)

    checkpoint_raw = str(args.checkpoint) if args.checkpoint is not None else inference_config.checkpoint
    device_name = args.device if args.device is not None else inference_config.device
    sample_start = args.sample_start if args.sample_start is not None else inference_config.sample_index
    run_id = resolve_dataset_run_id(train_config, inference_config.run_id, args.run_id)
    data_root = inference_config.data_root or train_config.dataset.data_root

    checkpoint_path = resolve_existing_path(checkpoint_raw, args.config)
    device = select_device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)
    _require_temporal_checkpoint(checkpoint)
    frame_count = resolve_checkpoint_frame_count(checkpoint, inference_config)
    _ = resolve_checkpoint_frame_stride(checkpoint, inference_config)
    image_offsets = tuple(int(value) for value in (checkpoint.get("image_offsets") or train_config.dataset.image_offsets))
    telemetry_offsets = tuple(int(value) for value in (checkpoint.get("telemetry_offsets") or train_config.dataset.telemetry_offsets))
    future_offsets = tuple(int(value) for value in (checkpoint.get("future_offsets") or train_config.dataset.future_offsets))
    telemetry_feature_names = tuple(
        str(value)
        for value in (checkpoint.get("telemetry_feature_names") or train_config.dataset.telemetry_feature_names)
    )
    control_target_names, aux_target_names = resolve_checkpoint_target_names(
        checkpoint,
        future_steps=len(future_offsets),
    )
    target_transforms = resolve_checkpoint_target_transform_registry(checkpoint)
    model = build_model(checkpoint, device, frame_count)

    if args.debug_dir is not None:
        print(
            "debug-dir was provided, but temporal checkpoint sanity requires telemetry-backed dataset samples; "
            "image-only debug windows were skipped."
        )

    started = time.perf_counter()
    dataset_results = evaluate_dataset_samples(
        model,
        device,
        data_root=data_root,
        image_size=(train_config.dataset.image_width, train_config.dataset.image_height),
        expected_window_size=frame_count,
        run_id=run_id,
        sample_start=sample_start,
        sample_count=args.sample_count,
        image_offsets=image_offsets,
        telemetry_offsets=telemetry_offsets,
        future_offsets=future_offsets,
        telemetry_feature_names=telemetry_feature_names,
        control_target_names=control_target_names,
        aux_target_names=aux_target_names,
        target_transforms=target_transforms,
    )
    output = build_output(checkpoint_path, checkpoint, device, dataset_results)
    output["elapsed_s"] = time.perf_counter() - started

    if args.output_json:
        print(json.dumps(output, indent=2))
    else:
        print_plain(output)


if __name__ == "__main__":
    main()
