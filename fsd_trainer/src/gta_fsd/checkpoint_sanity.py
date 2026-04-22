from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Iterable

import torch

from dataset import FsdDataset
from inference import (
    DEFAULT_CONFIG_PATH,
    build_model,
    checkpoint_sample_stride,
    load_checkpoint,
    load_config as load_inference_config,
    resolve_checkpoint_frame_count,
    resolve_checkpoint_frame_stride,
    resolve_existing_path,
    select_device,
)
from heads import head_layout_metadata, head_specs_metadata, resolve_checkpoint_head_specs
from image_io import load_rgb_tensor_from_path
from model_output import single_control_prediction_from_output, single_tensor_mapping
from train import TrainConfig, load_config as load_train_config
from state_inputs import (
    CURRENT_SPEED_KEY,
    ROUTE_FORWARD_DELTA_KEY,
    StateInputConfig,
    state_input_config_from_metadata,
    state_inputs_metadata,
)
from target_transforms import (
    legacy_delta_speed_target_transform,
    resolve_checkpoint_delta_speed_target_transform,
)


DEBUG_FRAMES_ROOT_NAME = "awesomeProject-inference-frames"
DEFAULT_SAMPLE_COUNT = 10
DEFAULT_DUMP_WINDOW_COUNT = 5
FLAT_CONTROL_RANGE_EPSILON = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a strict control-only checkpoint sanity pass on dataset samples and dumped live frames."
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
        help="Optional checkpoint override. Defaults to inference.checkpoint from the TOML.",
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
        help="Optional debug frame dump directory. Defaults to the latest dump in the temp-frame root.",
    )
    parser.add_argument(
        "--dump-window-count",
        type=int,
        default=DEFAULT_DUMP_WINDOW_COUNT,
        help=f"Number of live frame windows to inspect. Default: {DEFAULT_DUMP_WINDOW_COUNT}.",
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


def load_frame_tensor(frame_path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    return load_rgb_tensor_from_path(frame_path, image_size)


def list_debug_frame_paths(debug_dir: Path) -> list[Path]:
    frame_paths = sorted(path for path in debug_dir.glob("frame-*.jpg") if path.is_file())
    if not frame_paths:
        raise FileNotFoundError(f"No debug frames found in {debug_dir}")
    return frame_paths


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


def predict_control_outputs(
    model: Any,
    device: torch.device,
    stacked_frames: torch.Tensor,
    *,
    state_inputs: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    x = stacked_frames.unsqueeze(0).to(device, non_blocking=device.type == "cuda")
    current_speed = None
    route_forward_delta = None
    if state_inputs is not None and CURRENT_SPEED_KEY in state_inputs:
        current_speed = state_inputs[CURRENT_SPEED_KEY].unsqueeze(0).to(device, non_blocking=device.type == "cuda")
    if state_inputs is not None and ROUTE_FORWARD_DELTA_KEY in state_inputs:
        route_forward_delta = state_inputs[ROUTE_FORWARD_DELTA_KEY].unsqueeze(0).to(
            device,
            non_blocking=device.type == "cuda",
        )
    with torch.no_grad():
        output = model(x, current_speed=current_speed, route_forward_delta=route_forward_delta)
    delta_speed_transform = getattr(model, "delta_speed_target_transform", legacy_delta_speed_target_transform())
    return single_control_prediction_from_output(
        output,
        head_specs=model.head_specs,
        delta_speed_transform=delta_speed_transform,
    )


def summarize_control_ranges(items: Iterable[dict[str, Any]]) -> dict[str, dict[str, float | bool]]:
    results = list(items)
    summary: dict[str, dict[str, float | bool]] = {}
    control_names = sorted({
        str(control_name)
        for item in results
        for control_name in (item.get("control_outputs") or {}).keys()
        if isinstance(item.get("control_outputs"), dict)
    })
    for control_name in control_names:
        values = [
            float(item["control_outputs"][control_name])
            for item in results
            if isinstance(item.get("control_outputs"), dict) and control_name in item["control_outputs"]
        ]
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
) -> dict[str, Any]:
    dataset = FsdDataset(
        run_id=run_id,
        data_root=data_root,
        image_size=image_size,
        expected_window_size=expected_window_size,
        state_input_config=model.state_input_config,
        head_specs=model.head_specs,
    )
    if sample_start < 0:
        raise ValueError("sample_start must be >= 0")
    if sample_count < 1:
        raise ValueError("sample_count must be > 0")
    if sample_start >= len(dataset):
        raise IndexError(f"sample_start out of range: {sample_start}, dataset size={len(dataset)}")

    sample_end = min(len(dataset), sample_start + sample_count)
    results: list[dict[str, Any]] = []
    delta_speed_transform = getattr(model, "delta_speed_target_transform", legacy_delta_speed_target_transform())
    for sample_index in range(sample_start, sample_end):
        stacked_frames, state_inputs, targets = dataset[sample_index]
        sample_meta = dataset.samples[sample_index]
        results.append({
            "sample_index": sample_index,
            "trip_key": sample_meta.get("trip_key"),
            "frame_paths": [str(path) for path in sample_meta.get("frame_paths", [])],
            "state_inputs": single_tensor_mapping(state_inputs),
            "control_outputs": predict_control_outputs(model, device, stacked_frames, state_inputs=state_inputs),
            "target": single_tensor_mapping(targets, delta_speed_transform=delta_speed_transform),
        })

    return {
        "run_id": run_id,
        "dataset_size": len(dataset),
        "sample_start": sample_start,
        "sample_count": len(results),
        "results": results,
        "control_ranges": summarize_control_ranges(results),
    }


def evaluate_debug_windows(
    model: Any,
    device: torch.device,
    *,
    debug_dir: Path,
    image_size: tuple[int, int],
    window_size: int,
    frame_stride: int,
    dump_window_count: int,
    state_input_config: StateInputConfig,
) -> dict[str, Any]:
    if dump_window_count < 1:
        raise ValueError("dump_window_count must be > 0")
    if state_input_config.current_speed_enabled or state_input_config.route_forward_delta_enabled:
        return {
            "debug_dir": str(debug_dir),
            "frame_count": 0,
            "window_size": window_size,
            "frame_stride": frame_stride,
            "window_count": 0,
            "results": [],
            "control_ranges": {},
            "skipped_reason": (
                "state-input-enabled checkpoints require live telemetry; "
                "debug frame dumps are image-only"
            ),
        }

    frame_paths = list_debug_frame_paths(debug_dir)
    windows = build_debug_windows(
        frame_paths,
        window_size=window_size,
        frame_stride=frame_stride,
        limit=dump_window_count,
    )

    results: list[dict[str, Any]] = []
    for window_index, window_paths in enumerate(windows):
        frame_tensors = [load_frame_tensor(path, image_size) for path in window_paths]
        stacked_frames = torch.cat(frame_tensors, dim=0)
        start_frame_index = frame_paths.index(window_paths[0])
        results.append({
            "window_index": window_index,
            "start_frame_index": start_frame_index,
            "frame_paths": [str(path) for path in window_paths],
            "control_outputs": predict_control_outputs(model, device, stacked_frames),
        })

    return {
        "debug_dir": str(debug_dir),
        "frame_count": len(frame_paths),
        "window_size": window_size,
        "frame_stride": frame_stride,
        "window_count": len(results),
        "results": results,
        "control_ranges": summarize_control_ranges(results),
    }


def build_output(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    device: torch.device,
    dataset_results: dict[str, Any],
    debug_results: dict[str, Any],
) -> dict[str, Any]:
    head_specs = resolve_checkpoint_head_specs(checkpoint)
    return {
        "checkpoint": {
            "path": str(checkpoint_path),
            "device": str(device),
            "epoch": int(checkpoint.get("epoch", 0) or 0),
            "frame_window_size": int(checkpoint.get("frame_window_size", 0) or 0),
            "frame_stride": int(checkpoint.get("frame_stride", checkpoint.get("frame_window_stride", 0)) or 0),
            "sample_stride": checkpoint_sample_stride(checkpoint),
            "input_channels": int(checkpoint.get("input_channels", 0) or 0),
            "delta_speed_target_transform": resolve_checkpoint_delta_speed_target_transform(checkpoint).metadata(),
            "head_layout": checkpoint.get("head_layout", head_layout_metadata(head_specs)),
            "head_specs": checkpoint.get("head_specs", head_specs_metadata(head_specs)),
            "state_inputs": checkpoint.get("state_inputs", state_inputs_metadata(state_input_config_from_metadata(None))),
        },
        "dataset_samples": dataset_results,
        "debug_windows": debug_results,
    }


def print_plain(output: dict[str, Any]) -> None:
    checkpoint = output["checkpoint"]
    dataset_samples = output["dataset_samples"]
    debug_windows = output["debug_windows"]

    print(f"Checkpoint: {checkpoint['path']}")
    print(
        "Model window: "
        f"size={checkpoint['frame_window_size']} "
        f"frame_stride={checkpoint['frame_stride']} "
        f"sample_stride={checkpoint['sample_stride']} "
        f"input_channels={checkpoint['input_channels']}"
    )
    print(f"Device: {checkpoint['device']}")
    print()
    print(
        "Dataset samples: "
        f"run_id={dataset_samples['run_id']} "
        f"start={dataset_samples['sample_start']} "
        f"count={dataset_samples['sample_count']} "
        f"dataset_size={dataset_samples['dataset_size']}"
    )
    for control_name, stats in dataset_samples.get("control_ranges", {}).items():
        print(
            f"dataset_{control_name}: "
            f"min={float(stats['min']):.6f} "
            f"max={float(stats['max']):.6f} "
            f"mean={float(stats['mean']):.6f} "
            f"range={float(stats['range']):.6f} "
            f"flat={bool(stats['flat'])}"
        )
    for item in dataset_samples.get("results", []):
        print(
            f"sample[{item['sample_index']}]: "
            f"controls={item['control_outputs']} "
            f"target={item['target']}"
        )
    print()
    print(
        "Debug windows: "
        f"dir={debug_windows['debug_dir']} "
        f"frames={debug_windows['frame_count']} "
        f"windows={debug_windows['window_count']}"
    )
    for control_name, stats in debug_windows.get("control_ranges", {}).items():
        print(
            f"debug_{control_name}: "
            f"min={float(stats['min']):.6f} "
            f"max={float(stats['max']):.6f} "
            f"mean={float(stats['mean']):.6f} "
            f"range={float(stats['range']):.6f} "
            f"flat={bool(stats['flat'])}"
        )
    for item in debug_windows.get("results", []):
        print(f"window[{item['window_index']}]: controls={item['control_outputs']}")


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
    frame_count = resolve_checkpoint_frame_count(checkpoint, inference_config)
    frame_stride = resolve_checkpoint_frame_stride(checkpoint, inference_config)
    model = build_model(checkpoint, device, frame_count)
    state_input_config = state_input_config_from_metadata(checkpoint.get("state_inputs"))

    dataset_results = evaluate_dataset_samples(
        model,
        device,
        data_root=data_root,
        image_size=(train_config.dataset.image_width, train_config.dataset.image_height),
        expected_window_size=frame_count,
        run_id=run_id,
        sample_start=sample_start,
        sample_count=args.sample_count,
    )
    debug_dir = resolve_debug_dir(args.debug_dir)
    debug_results = evaluate_debug_windows(
        model,
        device,
        debug_dir=debug_dir,
        image_size=(train_config.dataset.image_width, train_config.dataset.image_height),
        window_size=frame_count,
        frame_stride=frame_stride,
        dump_window_count=args.dump_window_count,
        state_input_config=state_input_config,
    )

    output = build_output(checkpoint_path, checkpoint, device, dataset_results, debug_results)
    if args.output_json:
        print(json.dumps(output, indent=2))
    else:
        print_plain(output)


if __name__ == "__main__":
    main()
