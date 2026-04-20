from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset as TorchDataset, Subset

from config import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    normalize_windows_drive_path,
    parse_delta_speed_target_transform,
    parse_dataset_window,
)
from dataset import DatasetStateInputs, DatasetTargets, FsdDataset
from heads import (
    HEAD_SPECS,
    HeadSpec,
    apply_loss_weight_overrides,
    compute_head_loss,
    head_layout_metadata,
    head_specs_metadata,
    inactive_loss_weight_override_names,
    normalize_head_tensor,
    supported_metric_names,
)
from models.planner import DrivingCNN
from state_inputs import CURRENT_SPEED_KEY, StateInputConfig, state_inputs_metadata, training_state_input_config
from target_transforms import (
    DeltaSpeedTargetTransform,
    default_delta_speed_target_transform,
    denormalize_delta_speed_tensor,
)


MetricPayload = dict[str, Any]


try:
    import psutil  # type: ignore[import-not-found]
except ImportError:
    psutil = None


@dataclass(frozen=True)
class DatasetConfig:
    data_root: str
    train_run_ids: tuple[str, ...]
    val_run_ids: tuple[str, ...]
    train_run_paths: tuple[str, ...]
    val_run_paths: tuple[str, ...]
    image_width: int
    image_height: int
    window_size: int
    frame_stride: int
    sample_stride: int
    delta_speed_transform: DeltaSpeedTargetTransform = field(default_factory=default_delta_speed_target_transform)


@dataclass(frozen=True)
class OutputConfig:
    base_dir: str


@dataclass(frozen=True)
class TrainingConfig:
    device: str
    epochs: int
    learning_rate: float
    early_stopping_metric: str
    early_stopping_patience: int
    early_stopping_min_delta: float
    head_loss_weights: dict[str, float]


@dataclass(frozen=True)
class LoaderConfig:
    train_batch_size: int
    train_num_workers: int
    train_pin_memory: bool
    train_prefetch_factor: int
    train_persistent_workers: bool
    val_batch_size: int
    val_num_workers: int
    val_pin_memory: bool
    val_prefetch_factor: int
    val_persistent_workers: bool
    log_every_n_batches: int
    val_split: float
    cpu_batch_size: int


@dataclass(frozen=True)
class TrainConfig:
    dataset: DatasetConfig
    output: OutputConfig
    training: TrainingConfig
    loader: LoaderConfig


@dataclass(frozen=True)
class TrainingContext:
    config: TrainConfig
    config_path: Path
    run_dir: Path
    run_metrics_path: Path
    data_root: Path
    train_dataset: FsdDataset
    val_dataset: FsdDataset
    val_subset: TorchDataset[tuple[Tensor, DatasetStateInputs, DatasetTargets]]
    selected_val_trip_count: int
    total_val_trip_count: int
    head_specs: tuple[HeadSpec, ...]
    model: Module
    device: torch.device
    optimizer: Optimizer
    scaler: torch.amp.GradScaler
    train_sample_shape: tuple[int, ...]
    train_state_input_shapes: dict[str, tuple[int, ...]]
    train_target_shapes: dict[str, tuple[int, ...]]
    ignored_loss_weight_overrides: tuple[str, ...]
    state_input_config: StateInputConfig
    delta_speed_transform: DeltaSpeedTargetTransform


@dataclass
class EarlyStoppingState:
    metric_name: str
    patience: int
    min_delta: float
    best_value: float = float("inf")
    best_epoch: int = 0
    bad_epoch_count: int = 0


@dataclass(frozen=True)
class EpochResult:
    epoch_index: int
    train_metrics: MetricPayload
    val_metrics: MetricPayload
    train_epoch_time: float
    val_epoch_time: float
    avg_batch_time: float
    avg_loader_wait_time: float
    avg_h2d_time: float
    avg_forward_backward_time: float
    avg_optimizer_time: float
    avg_iteration_time: float
    memory_snapshots: dict[str, dict[str, float | None]]


@dataclass(frozen=True)
class BatchTiming:
    loader_wait_s: float
    h2d_s: float
    forward_backward_s: float
    optimizer_s: float
    step_s: float

    @property
    def iteration_s(self) -> float:
        return self.loader_wait_s + self.step_s


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "train_config.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GTA FSD planner model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the training TOML config. Default: {DEFAULT_CONFIG_PATH}",
    )
    return parser.parse_args()


def load_config(path: Path) -> TrainConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    dataset_raw = raw["dataset"]
    output_raw = raw["output"]
    training_raw = raw["training"]
    loader_raw = raw["loader"]
    window_size, frame_stride, sample_stride = parse_dataset_window(raw)
    data_root = normalize_windows_drive_path(str(dataset_raw["data_root"]))
    train_run_ids = _resolve_training_run_ids(dataset_raw, "train_run_ids", fallback_key="run_id")
    val_run_ids = _resolve_training_run_ids(dataset_raw, "val_run_ids", fallback_key="val_id")

    legacy_batch_size = loader_raw.get("batch_size", loader_raw["cpu_batch_size"])
    legacy_num_workers = loader_raw.get("num_workers", 0)
    legacy_pin_memory = loader_raw.get("pin_memory", False)
    legacy_prefetch_factor = loader_raw.get("prefetch_factor", 1)
    legacy_persistent_workers = loader_raw.get("persistent_workers", False)

    return TrainConfig(
        dataset=DatasetConfig(
            data_root=data_root,
            train_run_ids=train_run_ids,
            val_run_ids=val_run_ids,
            train_run_paths=_resolve_run_paths_from_ids(data_root, train_run_ids),
            val_run_paths=_resolve_run_paths_from_ids(data_root, val_run_ids),
            image_width=int(dataset_raw.get("image_width", DEFAULT_IMAGE_WIDTH)),
            image_height=int(dataset_raw.get("image_height", DEFAULT_IMAGE_HEIGHT)),
            window_size=window_size,
            frame_stride=frame_stride,
            sample_stride=sample_stride,
            delta_speed_transform=parse_delta_speed_target_transform(raw),
        ),
        output=OutputConfig(base_dir=str(output_raw["base_dir"])),
        training=TrainingConfig(
            device=str(training_raw["device"]).strip().lower(),
            epochs=int(training_raw["epochs"]),
            learning_rate=float(training_raw["learning_rate"]),
            early_stopping_metric=str(training_raw.get("early_stopping_metric", "overall_mae")).strip(),
            early_stopping_patience=int(training_raw.get("early_stopping_patience", 3)),
            early_stopping_min_delta=float(training_raw.get("early_stopping_min_delta", 0.0)),
            head_loss_weights={
                str(name): float(weight)
                for name, weight in dict(training_raw.get("loss_weights", {})).items()
            },
        ),
        loader=LoaderConfig(
            train_batch_size=int(loader_raw.get("train_batch_size", legacy_batch_size)),
            train_num_workers=int(loader_raw.get("train_num_workers", legacy_num_workers)),
            train_pin_memory=bool(loader_raw.get("train_pin_memory", legacy_pin_memory)),
            train_prefetch_factor=int(loader_raw.get("train_prefetch_factor", legacy_prefetch_factor)),
            train_persistent_workers=bool(
                loader_raw.get("train_persistent_workers", legacy_persistent_workers)
            ),
            val_batch_size=int(loader_raw.get("val_batch_size", legacy_batch_size)),
            val_num_workers=int(loader_raw.get("val_num_workers", legacy_num_workers)),
            val_pin_memory=bool(loader_raw.get("val_pin_memory", legacy_pin_memory)),
            val_prefetch_factor=int(loader_raw.get("val_prefetch_factor", legacy_prefetch_factor)),
            val_persistent_workers=bool(
                loader_raw.get("val_persistent_workers", legacy_persistent_workers)
            ),
            log_every_n_batches=max(1, int(loader_raw.get("log_every_n_batches", 10))),
            val_split=float(loader_raw["val_split"]),
            cpu_batch_size=int(loader_raw["cpu_batch_size"]),
        ),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_training_run_ids(
    dataset_raw: dict[str, Any],
    key: str,
    *,
    fallback_key: str,
) -> tuple[str, ...]:
    raw_value = dataset_raw.get(key)
    if raw_value is not None:
        if not isinstance(raw_value, list):
            raise ValueError(f"dataset.{key} must be a TOML array of run ids")
        run_ids = tuple(
            str(item).strip()
            for item in raw_value
            if str(item).strip()
        )
        if not run_ids:
            raise ValueError(f"dataset.{key} must contain at least one run id")
        return run_ids

    fallback_run_id = _optional_str(dataset_raw.get(fallback_key))
    if fallback_run_id is None:
        raise ValueError(f"Missing dataset.{key} and deprecated dataset.{fallback_key}")
    return (fallback_run_id,)


def _resolve_run_paths_from_ids(data_root: str, run_ids: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(Path(data_root) / "runs" / run_id) for run_id in run_ids)


def prepare_output_paths(base_output_dir: Path) -> tuple[Path, Path]:
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base_output_dir / f"run-{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_dir / "run_metrics.json"


def save_epoch_artifacts(
    run_dir: Path,
    epoch_index: int,
    model: Module,
    optimizer: Optimizer,
    frame_window_size: int,
    frame_stride: int,
    sample_stride: int,
    train_metrics: MetricPayload,
    val_metrics: MetricPayload,
    train_epoch_time: float,
    val_epoch_time: float,
    avg_batch_time: float,
    avg_loader_wait_time: float,
    avg_h2d_time: float,
    avg_forward_backward_time: float,
    avg_optimizer_time: float,
    avg_iteration_time: float,
    head_specs: tuple[HeadSpec, ...],
    state_input_config: StateInputConfig,
    delta_speed_transform: DeltaSpeedTargetTransform,
) -> dict[str, Any]:
    checkpoint_path = run_dir / f"epoch-{epoch_index:03d}.pt"
    checkpoint_payload = {
        "epoch": epoch_index,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "frame_window_size": frame_window_size,
        "frame_stride": frame_stride,
        "sample_stride": sample_stride,
        "input_channels": frame_window_size * 3,
        "state_inputs": state_inputs_metadata(state_input_config),
        "delta_speed_target_transform": delta_speed_transform.metadata(),
        "head_specs": head_specs_metadata(head_specs),
        "head_layout": head_layout_metadata(head_specs),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_epoch_s": train_epoch_time,
        "val_epoch_s": val_epoch_time,
        "avg_batch_s": avg_batch_time,
        "avg_loader_wait_s": avg_loader_wait_time,
        "avg_h2d_s": avg_h2d_time,
        "avg_forward_backward_s": avg_forward_backward_time,
        "avg_optimizer_s": avg_optimizer_time,
        "avg_iteration_s": avg_iteration_time,
    }
    torch.save(checkpoint_payload, checkpoint_path)
    return {
        "epoch": epoch_index,
        "checkpoint": str(checkpoint_path),
        "frame_window_size": frame_window_size,
        "frame_stride": frame_stride,
        "sample_stride": sample_stride,
        "input_channels": frame_window_size * 3,
        "state_inputs": state_inputs_metadata(state_input_config),
        "delta_speed_target_transform": delta_speed_transform.metadata(),
        "head_specs": head_specs_metadata(head_specs),
        "head_layout": head_layout_metadata(head_specs),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_epoch_s": train_epoch_time,
        "val_epoch_s": val_epoch_time,
        "avg_batch_s": avg_batch_time,
        "avg_loader_wait_s": avg_loader_wait_time,
        "avg_h2d_s": avg_h2d_time,
        "avg_forward_backward_s": avg_forward_backward_time,
        "avg_optimizer_s": avg_optimizer_time,
        "avg_iteration_s": avg_iteration_time,
    }


def select_device(requested: str) -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("training.device must be one of: auto, cpu, cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("training.device is set to 'cuda' but CUDA is not available")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def probe_device(
    model: Module,
    device: torch.device,
    image_size: tuple[int, int],
    frame_count: int,
    state_input_config: StateInputConfig,
) -> bool:
    if device.type != "cuda":
        return True
    try:
        image_width, image_height = image_size
        probe_batch = torch.zeros((1, frame_count * 3, image_height, image_width), device=device)
        probe_speed = None
        if state_input_config.current_speed_enabled:
            probe_speed = torch.zeros((1,), device=device)
        with torch.no_grad():
            _ = model(probe_batch, current_speed=probe_speed)
        return True
    except RuntimeError as exc:
        message = str(exc).lower()
        if "no kernel image is available" in message or "not compatible with the current pytorch installation" in message:
            return False
        raise


def _build_phase_loader(
    dataset: TorchDataset[tuple[Tensor, DatasetStateInputs, DatasetTargets]],
    *,
    device: torch.device,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
    cpu_batch_size: int,
) -> DataLoader[tuple[Tensor, DatasetStateInputs, DatasetTargets]]:
    if device.type != "cuda":
        return DataLoader(
            dataset=dataset,
            batch_size=cpu_batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
        )

    resolved_num_workers = max(0, num_workers)
    resolved_prefetch_factor = prefetch_factor if resolved_num_workers > 0 else None
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=resolved_num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and resolved_num_workers > 0,
        prefetch_factor=resolved_prefetch_factor,
    )


def build_validation_subset(
    dataset: FsdDataset,
    val_split: float,
    *,
    seed: int = 42,
) -> tuple[FsdDataset | Subset[tuple[Tensor, DatasetTargets]], int, int]:
    total_trip_count = dataset.trip_count
    if total_trip_count <= 0:
        raise ValueError("Validation dataset has no trips")

    selected_trip_count = max(1, int(total_trip_count * val_split))
    if selected_trip_count >= total_trip_count:
        return dataset, total_trip_count, total_trip_count

    generator = torch.Generator().manual_seed(seed)
    shuffled_trip_indices = torch.randperm(total_trip_count, generator=generator).tolist()
    selected_trip_indices = sorted(shuffled_trip_indices[:selected_trip_count])

    subset_indices: list[int] = []
    trip_indices = dataset.trip_sample_indices()
    for trip_index in selected_trip_indices:
        subset_indices.extend(trip_indices[trip_index])

    return Subset(dataset, subset_indices), selected_trip_count, total_trip_count


def _loader_dataset_len(dataset: TorchDataset[tuple[Tensor, DatasetStateInputs, DatasetTargets]]) -> int:
    return len(dataset)


def _process_rss_bytes() -> int | None:
    if psutil is not None:
        return int(psutil.Process(os.getpid()).memory_info().rss)

    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
            process = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
            if ok:
                return int(counters.WorkingSetSize)
        except Exception:
            return None
        return None

    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except Exception:
            return None
    return None


def _memory_snapshot(device: torch.device) -> dict[str, float | None]:
    process_rss_bytes = _process_rss_bytes()
    snapshot: dict[str, float | None] = {
        "process_rss_mb": None if process_rss_bytes is None else process_rss_bytes / (1024.0 * 1024.0),
        "cuda_allocated_mb": None,
        "cuda_reserved_mb": None,
        "cuda_max_allocated_mb": None,
        "cuda_max_reserved_mb": None,
    }
    if device.type == "cuda":
        snapshot.update({
            "cuda_allocated_mb": torch.cuda.memory_allocated(device) / (1024.0 * 1024.0),
            "cuda_reserved_mb": torch.cuda.memory_reserved(device) / (1024.0 * 1024.0),
            "cuda_max_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0),
            "cuda_max_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0),
        })
    return snapshot


def _format_memory_snapshot(label: str, snapshot: dict[str, float | None]) -> str:
    parts = [label]
    for key in ("process_rss_mb", "cuda_allocated_mb", "cuda_reserved_mb", "cuda_max_allocated_mb", "cuda_max_reserved_mb"):
        value = snapshot.get(key)
        if value is not None:
            parts.append(f"{key}={value:.1f}")
    return " ".join(parts)


def _shutdown_loader_iterator(
    loader_iter: Iterator[tuple[Tensor, DatasetStateInputs, DatasetTargets]] | None,
) -> None:
    if loader_iter is None:
        return
    shutdown = getattr(loader_iter, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


def _release_phase_resources(
    device: torch.device,
    *,
    loader: DataLoader[tuple[Tensor, DatasetStateInputs, DatasetTargets]] | None = None,
    loader_iter: Iterator[tuple[Tensor, DatasetStateInputs, DatasetTargets]] | None = None,
) -> None:
    _shutdown_loader_iterator(loader_iter)
    del loader_iter
    del loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _empty_metric_totals(head_specs: tuple[HeadSpec, ...]) -> dict[str, Any]:
    return {
        "weighted_loss_sum": 0.0,
        "control_weighted_loss_sum": 0.0,
        "aux_weighted_loss_sum": 0.0,
        "loss_sum_by_head": {spec.name: 0.0 for spec in head_specs},
        "weighted_loss_sum_by_head": {spec.name: 0.0 for spec in head_specs},
        "abs_error_sum_by_head": {spec.name: 0.0 for spec in head_specs},
        "sq_error_sum_by_head": {spec.name: 0.0 for spec in head_specs},
        "element_count_by_head": {spec.name: 0 for spec in head_specs},
        "correct_count_by_head": {spec.name: 0.0 for spec in head_specs},
        "overall_abs_error_sum": 0.0,
        "overall_sq_error_sum": 0.0,
        "overall_element_count": 0,
        "control_abs_error_sum": 0.0,
        "control_sq_error_sum": 0.0,
        "control_element_count": 0,
        "aux_abs_error_sum": 0.0,
        "aux_sq_error_sum": 0.0,
        "aux_element_count": 0,
        "sample_count": 0,
        "batch_count": 0,
    }


def _mean_or_zero(total: float, count: int) -> float:
    return 0.0 if count <= 0 else float(total / count)


def _rmse_or_zero(total: float, count: int) -> float:
    return 0.0 if count <= 0 else float((total / count) ** 0.5)


def _empty_timing_totals() -> dict[str, float]:
    return {
        "loader_wait_s": 0.0,
        "h2d_s": 0.0,
        "forward_backward_s": 0.0,
        "optimizer_s": 0.0,
        "step_s": 0.0,
        "iteration_s": 0.0,
    }


def _update_timing_totals(totals: dict[str, float], timing: BatchTiming) -> None:
    totals["loader_wait_s"] += timing.loader_wait_s
    totals["h2d_s"] += timing.h2d_s
    totals["forward_backward_s"] += timing.forward_backward_s
    totals["optimizer_s"] += timing.optimizer_s
    totals["step_s"] += timing.step_s
    totals["iteration_s"] += timing.iteration_s


def _mean_timing(total: float, batch_count: int) -> float:
    return 0.0 if batch_count <= 0 else total / batch_count


def _move_targets_to_device(
    targets: DatasetTargets,
    device: torch.device,
    head_specs: tuple[HeadSpec, ...],
) -> DatasetTargets:
    return {
        spec.name: normalize_head_tensor(spec.name, targets[spec.name], spec).to(
            device, non_blocking=device.type == "cuda"
        )
        for spec in head_specs
        if spec.name in targets
    }


def _move_state_inputs_to_device(
    state_inputs: DatasetStateInputs,
    device: torch.device,
) -> DatasetStateInputs:
    return {
        name: value.to(device, non_blocking=device.type == "cuda")
        for name, value in state_inputs.items()
    }


def _metric_tensor_for_head(
    spec: HeadSpec,
    value: Tensor,
    *,
    delta_speed_transform: DeltaSpeedTargetTransform,
) -> Tensor:
    tensor = normalize_head_tensor(spec.name, value, spec)
    if spec.name == "delta_speed":
        return denormalize_delta_speed_tensor(tensor, delta_speed_transform)
    return tensor


def _compute_loss_and_update_totals(
    outputs: dict[str, Tensor],
    targets: DatasetTargets,
    totals: dict[str, Any],
    *,
    sample_count: int,
    head_specs: tuple[HeadSpec, ...],
    delta_speed_transform: DeltaSpeedTargetTransform,
) -> Tensor:
    loss_terms: list[Tensor] = []

    totals["sample_count"] += sample_count
    totals["batch_count"] += 1

    for spec in head_specs:
        if spec.name not in outputs:
            raise KeyError(f"model output is missing head '{spec.name}'")
        if spec.name not in targets:
            if spec.required_target:
                raise KeyError(f"batch targets are missing required head '{spec.name}'")
            continue

        prediction = normalize_head_tensor(spec.name, outputs[spec.name], spec)
        target = normalize_head_tensor(spec.name, targets[spec.name], spec).to(dtype=prediction.dtype)

        head_loss = compute_head_loss(spec, prediction, target)
        if head_loss is not None:
            head_loss_value = float(head_loss.item())
            weighted_loss_value = head_loss_value * spec.loss_weight
            totals["loss_sum_by_head"][spec.name] += head_loss_value
            totals["weighted_loss_sum_by_head"][spec.name] += weighted_loss_value
            totals["weighted_loss_sum"] += weighted_loss_value
            if spec.kind == "control":
                totals["control_weighted_loss_sum"] += weighted_loss_value
            else:
                totals["aux_weighted_loss_sum"] += weighted_loss_value
            if spec.loss_weight > 0.0:
                loss_terms.append(head_loss * spec.loss_weight)

        if spec.loss_type == "bce_with_logits":
            prediction_binary = (torch.sigmoid(prediction) >= 0.5).to(dtype=target.dtype)
            totals["correct_count_by_head"][spec.name] += float((prediction_binary == target).sum().item())
            totals["element_count_by_head"][spec.name] += int(target.numel())
            continue

        metric_prediction = _metric_tensor_for_head(
            spec,
            prediction,
            delta_speed_transform=delta_speed_transform,
        )
        metric_target = _metric_tensor_for_head(
            spec,
            target,
            delta_speed_transform=delta_speed_transform,
        ).to(dtype=metric_prediction.dtype)

        error = (metric_prediction - metric_target).float()
        abs_error_sum = float(error.abs().sum().item())
        sq_error_sum = float(error.pow(2).sum().item())
        element_count = int(error.numel())

        totals["abs_error_sum_by_head"][spec.name] += abs_error_sum
        totals["sq_error_sum_by_head"][spec.name] += sq_error_sum
        totals["element_count_by_head"][spec.name] += element_count

        totals["overall_abs_error_sum"] += abs_error_sum
        totals["overall_sq_error_sum"] += sq_error_sum
        totals["overall_element_count"] += element_count
        if spec.kind == "control":
            totals["control_abs_error_sum"] += abs_error_sum
            totals["control_sq_error_sum"] += sq_error_sum
            totals["control_element_count"] += element_count
        else:
            totals["aux_abs_error_sum"] += abs_error_sum
            totals["aux_sq_error_sum"] += sq_error_sum
            totals["aux_element_count"] += element_count

    if loss_terms:
        return torch.stack(loss_terms).sum()

    first_output = next(iter(outputs.values()))
    return torch.zeros((), dtype=first_output.dtype, device=first_output.device)


def finalize_metrics(totals: dict[str, Any], head_specs: tuple[HeadSpec, ...]) -> MetricPayload:
    batch_count = int(totals["batch_count"])
    control_specs = tuple(spec for spec in head_specs if spec.kind == "control")
    aux_specs = tuple(spec for spec in head_specs if spec.kind == "aux")
    metrics: MetricPayload = {
        "loss": _mean_or_zero(float(totals["weighted_loss_sum"]), batch_count),
        "overall_mae": _mean_or_zero(float(totals["overall_abs_error_sum"]), int(totals["overall_element_count"])),
        "overall_rmse": _rmse_or_zero(float(totals["overall_sq_error_sum"]), int(totals["overall_element_count"])),
    }
    if control_specs:
        metrics.update({
            "control_loss": _mean_or_zero(float(totals["control_weighted_loss_sum"]), batch_count),
            "control_overall_mae": _mean_or_zero(
                float(totals["control_abs_error_sum"]), int(totals["control_element_count"])
            ),
            "control_overall_rmse": _rmse_or_zero(
                float(totals["control_sq_error_sum"]), int(totals["control_element_count"])
            ),
        })
    if aux_specs:
        metrics.update({
            "aux_loss": _mean_or_zero(float(totals["aux_weighted_loss_sum"]), batch_count),
            "aux_overall_mae": _mean_or_zero(
                float(totals["aux_abs_error_sum"]), int(totals["aux_element_count"])
            ),
            "aux_overall_rmse": _rmse_or_zero(
                float(totals["aux_sq_error_sum"]), int(totals["aux_element_count"])
            ),
        })

    per_head: dict[str, dict[str, Any]] = {}
    control_head_metrics: dict[str, dict[str, Any]] = {}
    aux_head_metrics: dict[str, dict[str, Any]] = {}

    for spec in head_specs:
        head_metrics: dict[str, Any] = {
            "kind": spec.kind,
            "output_dim": spec.output_dim,
            "loss_type": spec.loss_type,
            "loss_weight": spec.loss_weight,
            "used_for_control": spec.used_for_control,
            "loss": _mean_or_zero(float(totals["loss_sum_by_head"][spec.name]), batch_count),
            "weighted_loss": _mean_or_zero(float(totals["weighted_loss_sum_by_head"][spec.name]), batch_count),
        }
        metrics[f"{spec.name}_loss"] = head_metrics["loss"]
        metrics[f"{spec.name}_weighted_loss"] = head_metrics["weighted_loss"]

        element_count = int(totals["element_count_by_head"][spec.name])
        if spec.loss_type == "bce_with_logits":
            accuracy = _mean_or_zero(float(totals["correct_count_by_head"][spec.name]), element_count)
            head_metrics["accuracy"] = accuracy
            metrics[f"{spec.name}_accuracy"] = accuracy
        else:
            mae = _mean_or_zero(float(totals["abs_error_sum_by_head"][spec.name]), element_count)
            rmse = _rmse_or_zero(float(totals["sq_error_sum_by_head"][spec.name]), element_count)
            head_metrics["mae"] = mae
            head_metrics["rmse"] = rmse
            metrics[f"{spec.name}_mae"] = mae
            metrics[f"{spec.name}_rmse"] = rmse
            if spec.name == "steer":
                metrics["steering_mae"] = mae
                metrics["steering_rmse"] = rmse

        per_head[spec.name] = head_metrics
        if spec.kind == "control":
            control_head_metrics[spec.name] = head_metrics
        else:
            aux_head_metrics[spec.name] = head_metrics

    metrics["per_head"] = per_head
    metrics["control_heads"] = control_head_metrics
    if aux_specs:
        metrics["aux_heads"] = aux_head_metrics
    metrics["sample_count"] = int(totals["sample_count"])
    metrics["batch_count"] = batch_count
    return metrics


def train_batch(
    x_batch: Tensor,
    state_inputs: DatasetStateInputs,
    targets: DatasetTargets,
    model: Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    head_specs: tuple[HeadSpec, ...],
    delta_speed_transform: DeltaSpeedTargetTransform,
) -> tuple[float, BatchTiming, dict[str, Any]]:
    step_start = time.perf_counter()
    transfer_start = time.perf_counter()
    x_batch = x_batch.to(device, non_blocking=device.type == "cuda", memory_format=torch.channels_last)
    state_inputs = _move_state_inputs_to_device(state_inputs, device)
    y_batch = _move_targets_to_device(targets, device, head_specs)
    h2d_time = time.perf_counter() - transfer_start

    optimizer.zero_grad(set_to_none=True)

    batch_totals = _empty_metric_totals(head_specs)
    forward_backward_start = time.perf_counter()
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        model_output = model(x_batch, current_speed=state_inputs.get(CURRENT_SPEED_KEY))
        loss = _compute_loss_and_update_totals(
            model_output,
            y_batch,
            batch_totals,
            sample_count=int(x_batch.shape[0]),
            head_specs=head_specs,
            delta_speed_transform=delta_speed_transform,
        )

    scaler.scale(loss).backward()
    forward_backward_time = time.perf_counter() - forward_backward_start

    optimizer_start = time.perf_counter()
    scaler.step(optimizer)
    scaler.update()
    optimizer_time = time.perf_counter() - optimizer_start

    timing = BatchTiming(
        loader_wait_s=0.0,
        h2d_s=h2d_time,
        forward_backward_s=forward_backward_time,
        optimizer_s=optimizer_time,
        step_s=time.perf_counter() - step_start,
    )
    return float(loss.item()), timing, batch_totals


def format_batch_timing(timing: BatchTiming) -> str:
    return (
        f"wait_s={timing.loader_wait_s:.3f} "
        f"batch_s={timing.step_s:.3f} "
        f"h2d_s={timing.h2d_s:.3f} "
        f"fwd_bwd_s={timing.forward_backward_s:.3f} "
        f"opt_s={timing.optimizer_s:.3f}"
    )


def train_epoch(
    loader: DataLoader[tuple[Tensor, DatasetStateInputs, DatasetTargets]],
    optimizer: Optimizer,
    model: Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    head_specs: tuple[HeadSpec, ...],
    delta_speed_transform: DeltaSpeedTargetTransform,
    log_every_n_batches: int,
) -> tuple[MetricPayload, float, dict[str, float]]:
    epoch_start = time.perf_counter()
    model.train()
    totals = _empty_metric_totals(head_specs)
    timing_totals = _empty_timing_totals()
    loader_iter: Iterator[tuple[Tensor, DatasetStateInputs, DatasetTargets]] | None = None
    total_batches = len(loader)

    try:
        loader_iter = iter(loader)
        for batch_index in range(1, total_batches + 1):
            wait_start = time.perf_counter()
            x_batch, state_inputs, targets = next(loader_iter)
            loader_wait_time = time.perf_counter() - wait_start
            if batch_index == 1:
                state_input_shapes = {name: tuple(value.shape) for name, value in state_inputs.items()}
                target_shapes = {name: tuple(value.shape) for name, value in targets.items()}
                print(
                    "first_train_batch "
                    f"x_shape={tuple(x_batch.shape)} "
                    f"state_input_shapes={state_input_shapes} "
                    f"target_shapes={target_shapes} "
                    f"dtype={x_batch.dtype}"
                )
            batch_loss, batch_timing, batch_totals = train_batch(
                x_batch,
                state_inputs,
                targets,
                model,
                optimizer,
                scaler,
                device,
                head_specs,
                delta_speed_transform,
            )
            batch_timing = BatchTiming(
                loader_wait_s=loader_wait_time,
                h2d_s=batch_timing.h2d_s,
                forward_backward_s=batch_timing.forward_backward_s,
                optimizer_s=batch_timing.optimizer_s,
                step_s=batch_timing.step_s,
            )
            _update_timing_totals(timing_totals, batch_timing)
            _merge_metric_totals(totals, batch_totals)
            should_log = (
                batch_index == 1
                or batch_index == total_batches
                or batch_index % log_every_n_batches == 0
            )
            if should_log:
                print(
                    f"batch={batch_index}/{total_batches} "
                    f"loss={batch_loss:.6f} "
                    f"{format_batch_timing(batch_timing)} "
                    f"weighted_heads={format_batch_weighted_losses(batch_totals, head_specs)}"
                )
    finally:
        _shutdown_loader_iterator(loader_iter)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    epoch_time = time.perf_counter() - epoch_start
    metrics = finalize_metrics(totals, head_specs)
    batch_count = int(totals["batch_count"])
    avg_timings = {
        key: _mean_timing(total, batch_count)
        for key, total in timing_totals.items()
    }
    return metrics, epoch_time, avg_timings


def _merge_metric_totals(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in (
        "weighted_loss_sum",
        "control_weighted_loss_sum",
        "aux_weighted_loss_sum",
        "overall_abs_error_sum",
        "overall_sq_error_sum",
        "overall_element_count",
        "control_abs_error_sum",
        "control_sq_error_sum",
        "control_element_count",
        "aux_abs_error_sum",
        "aux_sq_error_sum",
        "aux_element_count",
        "sample_count",
        "batch_count",
    ):
        target[key] += source[key]

    for key in (
        "loss_sum_by_head",
        "weighted_loss_sum_by_head",
        "abs_error_sum_by_head",
        "sq_error_sum_by_head",
        "element_count_by_head",
        "correct_count_by_head",
    ):
        for head_name, value in source[key].items():
            target[key][head_name] += value


def evaluate_epoch(
    loader: DataLoader[tuple[Tensor, DatasetStateInputs, DatasetTargets]],
    model: Module,
    device: torch.device,
    head_specs: tuple[HeadSpec, ...],
    delta_speed_transform: DeltaSpeedTargetTransform,
) -> tuple[MetricPayload, float]:
    eval_start = time.perf_counter()
    model.eval()
    totals = _empty_metric_totals(head_specs)

    loader_iter: Iterator[tuple[Tensor, DatasetStateInputs, DatasetTargets]] | None = None
    try:
        loader_iter = iter(loader)
        with torch.no_grad():
            for x_batch, state_inputs, targets in loader_iter:
                x_batch = x_batch.to(device, non_blocking=device.type == "cuda", memory_format=torch.channels_last)
                state_inputs = _move_state_inputs_to_device(state_inputs, device)
                y_batch = _move_targets_to_device(targets, device, head_specs)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    model_output = model(x_batch, current_speed=state_inputs.get(CURRENT_SPEED_KEY))
                    _compute_loss_and_update_totals(
                        model_output,
                        y_batch,
                        totals,
                        sample_count=int(x_batch.shape[0]),
                        head_specs=head_specs,
                        delta_speed_transform=delta_speed_transform,
                    )
    finally:
        _shutdown_loader_iterator(loader_iter)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    eval_time = time.perf_counter() - eval_start
    return finalize_metrics(totals, head_specs), eval_time


def build_run_summary(context: TrainingContext) -> dict[str, object]:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "completed_at": None,
        "elapsed_s": 0.0,
        "elapsed_hms": "00:00:00",
        "config_path": str(context.config_path),
        "dataset_root": str(context.data_root),
        "train_run_ids": list(context.config.dataset.train_run_ids),
        "val_run_ids": list(context.config.dataset.val_run_ids),
        "train_run_paths": list(context.config.dataset.train_run_paths),
        "val_run_paths": list(context.config.dataset.val_run_paths),
        "image_size": {
            "width": context.config.dataset.image_width,
            "height": context.config.dataset.image_height,
        },
        "frame_window": {
            "size": context.config.dataset.window_size,
            "frame_stride": context.config.dataset.frame_stride,
            "sample_stride": context.config.dataset.sample_stride,
            "input_channels": context.config.dataset.window_size * 3,
        },
        "state_inputs": state_inputs_metadata(context.state_input_config),
        "delta_speed_target_transform": context.delta_speed_transform.metadata(),
        "head_specs": head_specs_metadata(context.head_specs),
        "head_layout": head_layout_metadata(context.head_specs),
        "inactive_loss_weight_overrides": list(context.ignored_loss_weight_overrides),
        "device": str(context.device),
        "dataset_samples": len(context.train_dataset),
        "dataset_trips": context.train_dataset.trip_count,
        "validation_dataset_samples": len(context.val_dataset),
        "validation_dataset_trips": context.val_dataset.trip_count,
        "validation_selected_trips": context.selected_val_trip_count,
        "train_sample_shape": list(context.train_sample_shape),
        "train_state_input_shapes": {name: list(shape) for name, shape in context.train_state_input_shapes.items()},
        "train_target_shapes": {name: list(shape) for name, shape in context.train_target_shapes.items()},
        "train_loader_samples": len(context.train_dataset),
        "validation_loader_samples": _loader_dataset_len(context.val_subset),
        "loader": {
            "train_batch_size": context.config.loader.train_batch_size,
            "train_num_workers": context.config.loader.train_num_workers,
            "train_pin_memory": context.config.loader.train_pin_memory,
            "train_prefetch_factor": context.config.loader.train_prefetch_factor,
            "train_persistent_workers": context.config.loader.train_persistent_workers,
            "val_batch_size": context.config.loader.val_batch_size,
            "val_num_workers": context.config.loader.val_num_workers,
            "val_pin_memory": context.config.loader.val_pin_memory,
            "val_prefetch_factor": context.config.loader.val_prefetch_factor,
            "val_persistent_workers": context.config.loader.val_persistent_workers,
            "cpu_batch_size": context.config.loader.cpu_batch_size,
            "log_every_n_batches": context.config.loader.log_every_n_batches,
        },
        "epochs_total": context.config.training.epochs,
        "early_stopping": {
            "metric": context.config.training.early_stopping_metric,
            "patience": context.config.training.early_stopping_patience,
            "min_delta": context.config.training.early_stopping_min_delta,
        },
        "epochs": [],
    }


def prepare_model(
    device: torch.device,
    image_size: tuple[int, int],
    frame_count: int,
    head_specs: tuple[HeadSpec, ...],
    state_input_config: StateInputConfig,
) -> tuple[Module, torch.device]:
    model = DrivingCNN(
        frame_count=frame_count,
        head_specs=head_specs,
        state_input_config=state_input_config,
    ).to(device)
    if probe_device(model, device, image_size, frame_count, state_input_config):
        return model, device

    print(
        "CUDA is available but this PyTorch build cannot run on your GPU "
        "(likely missing sm_120 support). Falling back to CPU."
    )
    fallback_device = torch.device("cpu")
    return DrivingCNN(
        frame_count=frame_count,
        head_specs=head_specs,
        state_input_config=state_input_config,
    ).to(fallback_device), fallback_device


def build_training_context(config: TrainConfig, config_path: Path) -> TrainingContext:
    data_root = Path(config.dataset.data_root)
    requested_device = select_device(config.training.device)
    image_size = (config.dataset.image_width, config.dataset.image_height)
    train_dataset = FsdDataset(
        run_paths=config.dataset.train_run_paths,
        image_size=image_size,
        expected_window_size=config.dataset.window_size,
    )
    val_dataset = FsdDataset(
        run_paths=config.dataset.val_run_paths,
        image_size=image_size,
        expected_window_size=config.dataset.window_size,
    )
    train_sample_x, train_state_inputs, train_targets = train_dataset[0]
    ignored_loss_weight_overrides = inactive_loss_weight_override_names(config.training.head_loss_weights)
    head_specs = apply_loss_weight_overrides(config.training.head_loss_weights)
    state_input_config = training_state_input_config()
    model, device = prepare_model(
        requested_device,
        image_size,
        config.dataset.window_size,
        head_specs,
        state_input_config,
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    val_subset, selected_val_trip_count, total_val_trip_count = build_validation_subset(
        val_dataset,
        config.loader.val_split,
    )
    run_dir, run_metrics_path = prepare_output_paths(Path(config.output.base_dir))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    return TrainingContext(
        config=config,
        config_path=config_path,
        run_dir=run_dir,
        run_metrics_path=run_metrics_path,
        data_root=data_root,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        val_subset=val_subset,
        selected_val_trip_count=selected_val_trip_count,
        total_val_trip_count=total_val_trip_count,
        head_specs=head_specs,
        model=model,
        device=device,
        optimizer=optimizer,
        scaler=scaler,
        train_sample_shape=tuple(train_sample_x.shape),
        train_state_input_shapes={name: tuple(value.shape) for name, value in train_state_inputs.items()},
        train_target_shapes={name: tuple(value.shape) for name, value in train_targets.items()},
        ignored_loss_weight_overrides=ignored_loss_weight_overrides,
        state_input_config=state_input_config,
        delta_speed_transform=config.dataset.delta_speed_transform,
    )


def create_early_stopping(config: TrainingConfig, head_specs: tuple[HeadSpec, ...]) -> EarlyStoppingState:
    metric_name = config.early_stopping_metric
    if metric_name not in supported_metric_names(head_specs):
        raise ValueError(f"Unsupported early stopping metric: {metric_name}")

    return EarlyStoppingState(
        metric_name=metric_name,
        patience=max(0, config.early_stopping_patience),
        min_delta=max(0.0, config.early_stopping_min_delta),
    )


def run_epoch(context: TrainingContext, epoch_index: int) -> EpochResult:
    memory_snapshots: dict[str, dict[str, float | None]] = {}
    memory_snapshots["train_start"] = _memory_snapshot(context.device)
    print(_format_memory_snapshot("memory train_start", memory_snapshots["train_start"]))

    train_loader = _build_phase_loader(
        context.train_dataset,
        device=context.device,
        batch_size=context.config.loader.train_batch_size,
        shuffle=True,
        num_workers=context.config.loader.train_num_workers,
        pin_memory=context.config.loader.train_pin_memory,
        prefetch_factor=context.config.loader.train_prefetch_factor,
        persistent_workers=context.config.loader.train_persistent_workers,
        cpu_batch_size=context.config.loader.cpu_batch_size,
    )
    try:
        train_metrics, train_epoch_time, avg_timings = train_epoch(
            train_loader,
            context.optimizer,
            context.model,
            context.scaler,
            context.device,
            context.head_specs,
            context.delta_speed_transform,
            context.config.loader.log_every_n_batches,
        )
    finally:
        _release_phase_resources(context.device, loader=train_loader)

    memory_snapshots["train_end"] = _memory_snapshot(context.device)
    print(_format_memory_snapshot("memory train_end", memory_snapshots["train_end"]))
    memory_snapshots["val_start"] = _memory_snapshot(context.device)
    print(_format_memory_snapshot("memory val_start", memory_snapshots["val_start"]))

    val_loader = _build_phase_loader(
        context.val_subset,
        device=context.device,
        batch_size=context.config.loader.val_batch_size,
        shuffle=False,
        num_workers=context.config.loader.val_num_workers,
        pin_memory=context.config.loader.val_pin_memory,
        prefetch_factor=context.config.loader.val_prefetch_factor,
        persistent_workers=context.config.loader.val_persistent_workers,
        cpu_batch_size=context.config.loader.cpu_batch_size,
    )
    try:
        val_metrics, val_epoch_time = evaluate_epoch(
            val_loader,
            context.model,
            context.device,
            context.head_specs,
            context.delta_speed_transform,
        )
    finally:
        _release_phase_resources(context.device, loader=val_loader)

    memory_snapshots["val_end"] = _memory_snapshot(context.device)
    print(_format_memory_snapshot("memory val_end", memory_snapshots["val_end"]))
    return EpochResult(
        epoch_index=epoch_index,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        train_epoch_time=train_epoch_time,
        val_epoch_time=val_epoch_time,
        avg_batch_time=avg_timings["step_s"],
        avg_loader_wait_time=avg_timings["loader_wait_s"],
        avg_h2d_time=avg_timings["h2d_s"],
        avg_forward_backward_time=avg_timings["forward_backward_s"],
        avg_optimizer_time=avg_timings["optimizer_s"],
        avg_iteration_time=avg_timings["iteration_s"],
        memory_snapshots=memory_snapshots,
    )


def check_early_stopping(epoch_result: EpochResult, state: EarlyStoppingState) -> tuple[bool, bool, float]:
    metric_value = float(epoch_result.val_metrics[state.metric_name])
    improved = metric_value < (state.best_value - state.min_delta)
    if improved:
        state.best_value = metric_value
        state.best_epoch = epoch_result.epoch_index
        state.bad_epoch_count = 0
    else:
        state.bad_epoch_count += 1

    should_stop = state.patience > 0 and state.bad_epoch_count >= state.patience
    return improved, should_stop, metric_value


def record_epoch(
    context: TrainingContext,
    run_summary: dict[str, object],
    epoch_result: EpochResult,
    *,
    is_best: bool,
    monitored_value: float,
    early_stopping_state: EarlyStoppingState,
) -> dict[str, Any]:
    epoch_artifact = save_epoch_artifacts(
        run_dir=context.run_dir,
        epoch_index=epoch_result.epoch_index,
        model=context.model,
        optimizer=context.optimizer,
        frame_window_size=context.config.dataset.window_size,
        frame_stride=context.config.dataset.frame_stride,
        sample_stride=context.config.dataset.sample_stride,
        train_metrics=epoch_result.train_metrics,
        val_metrics=epoch_result.val_metrics,
        train_epoch_time=epoch_result.train_epoch_time,
        val_epoch_time=epoch_result.val_epoch_time,
        avg_batch_time=epoch_result.avg_batch_time,
        avg_loader_wait_time=epoch_result.avg_loader_wait_time,
        avg_h2d_time=epoch_result.avg_h2d_time,
        avg_forward_backward_time=epoch_result.avg_forward_backward_time,
        avg_optimizer_time=epoch_result.avg_optimizer_time,
        avg_iteration_time=epoch_result.avg_iteration_time,
        head_specs=context.head_specs,
        state_input_config=context.state_input_config,
        delta_speed_transform=context.delta_speed_transform,
    )
    epoch_artifact["is_best"] = is_best
    epoch_artifact["early_stopping_metric"] = early_stopping_state.metric_name
    epoch_artifact["early_stopping_value"] = monitored_value
    epoch_artifact["bad_epochs_in_a_row"] = early_stopping_state.bad_epoch_count
    epoch_artifact["memory_snapshots"] = epoch_result.memory_snapshots

    epochs_list = run_summary["epochs"]
    assert isinstance(epochs_list, list)
    epochs_list.append(epoch_artifact)

    run_summary["best_epoch"] = early_stopping_state.best_epoch
    run_summary["best_metric"] = early_stopping_state.best_value
    run_summary["stopped_early"] = False
    run_summary["epochs_completed"] = epoch_result.epoch_index
    context.run_metrics_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    return epoch_artifact


def _format_group_metrics(metrics: MetricPayload, specs: tuple[HeadSpec, ...]) -> str:
    parts: list[str] = []
    for spec in specs:
        head_metrics = metrics.get("per_head", {}).get(spec.name, {})
        if spec.loss_type == "bce_with_logits":
            parts.append(f"{spec.name}_acc={float(head_metrics.get('accuracy', 0.0)):.6f}")
        else:
            parts.append(f"{spec.name}_mae={float(head_metrics.get('mae', 0.0)):.6f}")
            parts.append(f"{spec.name}_rmse={float(head_metrics.get('rmse', 0.0)):.6f}")
    return " ".join(parts)


def format_batch_weighted_losses(batch_totals: dict[str, Any], head_specs: tuple[HeadSpec, ...]) -> str:
    parts: list[str] = []
    for spec in head_specs:
        parts.append(f"{spec.name}:{float(batch_totals['weighted_loss_sum_by_head'][spec.name]):.3f}")
    return " ".join(parts)


def _format_elapsed_hms(elapsed_s: float) -> str:
    total_seconds = max(0, int(round(elapsed_s)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def print_epoch_summary(
    epoch_result: EpochResult,
    *,
    head_specs: tuple[HeadSpec, ...],
    checkpoint: str,
    is_best: bool,
    monitored_value: float,
    early_stopping_state: EarlyStoppingState,
    elapsed_s: float,
) -> None:
    status_parts = [f"{early_stopping_state.metric_name}={monitored_value:.6f}"]
    if is_best:
        status_parts.append("new best")
    else:
        status_parts.append(f"no improvement ({early_stopping_state.bad_epoch_count} bad epochs)")

    control_specs = tuple(spec for spec in head_specs if spec.kind == "control")
    aux_specs = tuple(spec for spec in head_specs if spec.kind == "aux")
    control_summary = _format_group_metrics(epoch_result.val_metrics, control_specs)
    aux_summary = _format_group_metrics(epoch_result.val_metrics, aux_specs)
    summary_parts = [
        f"epoch={epoch_result.epoch_index}",
        f"train_loss={float(epoch_result.train_metrics['loss']):.6f}",
        f"val_loss={float(epoch_result.val_metrics['loss']):.6f}",
        f"train_control_mae={float(epoch_result.train_metrics['control_overall_mae']):.6f}",
        f"val_control_mae={float(epoch_result.val_metrics['control_overall_mae']):.6f}",
        f"train_control_rmse={float(epoch_result.train_metrics['control_overall_rmse']):.6f}",
        f"val_control_rmse={float(epoch_result.val_metrics['control_overall_rmse']):.6f}",
    ]
    if control_summary:
        summary_parts.append(control_summary)
    if aux_summary:
        summary_parts.append(aux_summary)
    summary_parts.extend([
        f"checkpoint={checkpoint}",
        f"train_epoch_s={epoch_result.train_epoch_time:.3f}",
        f"val_epoch_s={epoch_result.val_epoch_time:.3f}",
        f"avg_batch_s={epoch_result.avg_batch_time:.3f}",
        f"avg_wait_s={epoch_result.avg_loader_wait_time:.3f}",
        f"avg_h2d_s={epoch_result.avg_h2d_time:.3f}",
        f"avg_fwd_bwd_s={epoch_result.avg_forward_backward_time:.3f}",
        f"avg_opt_s={epoch_result.avg_optimizer_time:.3f}",
        f"avg_iteration_s={epoch_result.avg_iteration_time:.3f}",
        f"elapsed_s={elapsed_s:.3f}",
        f"elapsed_hms={_format_elapsed_hms(elapsed_s)}",
        f"[{', '.join(status_parts)}]",
    ])
    print(" ".join(summary_parts))


def execute_training(context: TrainingContext) -> dict[str, object]:
    run_summary = build_run_summary(context)
    early_stopping_state = create_early_stopping(context.config.training, context.head_specs)
    image_width = context.config.dataset.image_width
    image_height = context.config.dataset.image_height
    training_started_at = time.perf_counter()

    print(
        f"Using device: {context.device} with train_samples={len(context.train_dataset)} "
        f"train_trips={context.train_dataset.trip_count} "
        f"val_samples={len(context.val_dataset)} "
        f"val_trips={context.val_dataset.trip_count} "
        f"selected_val_trips={context.selected_val_trip_count}/{context.total_val_trip_count} "
        f"(train={len(context.train_dataset)}, val={_loader_dataset_len(context.val_subset)}) "
        f"run_dir={context.run_dir}"
    )
    print(
        "Input config: "
        f"image_size=({image_width}, {image_height}) "
        f"stacked_frames={context.config.dataset.window_size} "
        f"sample_x_shape={context.train_sample_shape} "
        f"sample_target_shapes={context.train_target_shapes}"
    )
    print(f"Planner heads: {[spec.name for spec in context.head_specs]}")
    print(
        "Head loss weights: "
        + " ".join(f"{spec.name}={spec.loss_weight:.4f}" for spec in context.head_specs)
    )
    if context.ignored_loss_weight_overrides:
        print(
            "Ignoring inactive auxiliary loss-weight overrides in control-only mode: "
            + ", ".join(context.ignored_loss_weight_overrides)
        )
    print(f"Starting training for {context.config.training.epochs} epochs...")

    for epoch_index in range(1, context.config.training.epochs + 1):
        epoch_result = run_epoch(context, epoch_index)
        is_best, should_stop, monitored_value = check_early_stopping(epoch_result, early_stopping_state)
        elapsed_s = time.perf_counter() - training_started_at
        epoch_artifact = record_epoch(
            context,
            run_summary,
            epoch_result,
            is_best=is_best,
            monitored_value=monitored_value,
            early_stopping_state=early_stopping_state,
        )
        run_summary["elapsed_s"] = elapsed_s
        run_summary["elapsed_hms"] = _format_elapsed_hms(elapsed_s)
        print_epoch_summary(
            epoch_result,
            head_specs=context.head_specs,
            checkpoint=str(epoch_artifact["checkpoint"]),
            is_best=is_best,
            monitored_value=monitored_value,
            early_stopping_state=early_stopping_state,
            elapsed_s=elapsed_s,
        )
        context.run_metrics_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

        if should_stop:
            run_summary["stopped_early"] = True
            run_summary["stop_reason"] = (
                f"no improvement in {early_stopping_state.metric_name} for "
                f"{early_stopping_state.bad_epoch_count} consecutive epochs"
            )
            run_summary["completed_at"] = datetime.now().isoformat(timespec="seconds")
            context.run_metrics_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
            print(
                f"Early stopping triggered at epoch {epoch_index}: "
                f"best_epoch={early_stopping_state.best_epoch} "
                f"best_{early_stopping_state.metric_name}={early_stopping_state.best_value:.6f} "
                f"elapsed_hms={run_summary['elapsed_hms']}"
            )
            break

    if run_summary["completed_at"] is None:
        elapsed_s = time.perf_counter() - training_started_at
        run_summary["elapsed_s"] = elapsed_s
        run_summary["elapsed_hms"] = _format_elapsed_hms(elapsed_s)
        run_summary["completed_at"] = datetime.now().isoformat(timespec="seconds")
        context.run_metrics_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    return run_summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    context = build_training_context(config, args.config)
    summary = execute_training(context)
    best_epoch = int(summary.get("best_epoch", 0))
    best_metric = float(summary.get("best_metric", 0.0))
    print(
        f"Training finished: run_dir={context.run_dir} "
        f"run_metrics={context.run_metrics_path} "
        f"epochs_completed={summary.get('epochs_completed', 0)}/{context.config.training.epochs} "
        f"stopped_early={summary.get('stopped_early', False)} "
        f"best_epoch={best_epoch} "
        f"best_{context.config.training.early_stopping_metric}={best_metric:.6f} "
        f"elapsed_s={float(summary.get('elapsed_s', 0.0)):.3f} "
        f"elapsed_hms={summary.get('elapsed_hms', '00:00:00')}"
    )


if __name__ == "__main__":
    main()
