from __future__ import annotations

import argparse
import gc
import json
import math
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
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset as TorchDataset, Subset

from config import (
    DEFAULT_AUX_LOSS_WEIGHT,
    DEFAULT_AUX_TARGET_NAMES,
    DEFAULT_CONTROL_TARGET_NAMES,
    DEFAULT_FUTURE_OFFSETS,
    DEFAULT_HORIZON_LOSS_WEIGHTS,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_OFFSETS,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_SAMPLE_STRIDE,
    DEFAULT_TELEMETRY_FEATURE_NAMES,
    DEFAULT_TELEMETRY_HIDDEN_DIM,
    DEFAULT_TELEMETRY_OFFSETS,
    normalize_windows_drive_path,
    parse_temporal_dataset_config,
)
from dataset import DatasetItem, FsdDataset
from models.planner import DrivingCNN


MetricPayload = dict[str, Any]
PLANNER_FORMAT = "temporal_telemetry_gru_v1"
DEFAULT_WIDTH_MULTIPLIER = 1.0


try:
    import psutil  # type: ignore[import-not-found]
except ImportError:
    psutil = None


@dataclass(frozen=True)
class YawLossWeightingConfig:
    enabled: bool = False
    base_weight: float = 1.0
    alpha: float = 2.0
    tau: float = 0.25
    max_scale: float = 3.0


@dataclass(frozen=True)
class TurnOversamplingConfig:
    enabled: bool = False
    straight_weight: float = 1.0
    light_turn_weight: float = 1.5
    medium_turn_weight: float = 2.5
    sharp_turn_weight: float = 4.0
    light_turn_threshold: float = 0.05
    medium_turn_threshold: float = 0.15
    sharp_turn_threshold: float = 0.30


@dataclass(frozen=True)
class DatasetConfig:
    data_root: str
    train_run_ids: tuple[str, ...]
    val_run_ids: tuple[str, ...]
    train_run_paths: tuple[str, ...]
    val_run_paths: tuple[str, ...]
    image_width: int = DEFAULT_IMAGE_WIDTH
    image_height: int = DEFAULT_IMAGE_HEIGHT
    image_offsets: tuple[int, ...] = DEFAULT_IMAGE_OFFSETS
    telemetry_offsets: tuple[int, ...] = DEFAULT_TELEMETRY_OFFSETS
    future_offsets: tuple[int, ...] = DEFAULT_FUTURE_OFFSETS
    telemetry_feature_names: tuple[str, ...] = DEFAULT_TELEMETRY_FEATURE_NAMES
    control_target_names: tuple[str, ...] = DEFAULT_CONTROL_TARGET_NAMES
    aux_target_names: tuple[str, ...] = DEFAULT_AUX_TARGET_NAMES
    window_size: int = len(DEFAULT_IMAGE_OFFSETS)
    frame_stride: int = 2
    sample_stride: int = DEFAULT_SAMPLE_STRIDE


@dataclass(frozen=True)
class OutputConfig:
    base_dir: str


@dataclass(frozen=True)
class ModelConfig:
    width_multiplier: float = DEFAULT_WIDTH_MULTIPLIER
    telemetry_hidden_dim: int = DEFAULT_TELEMETRY_HIDDEN_DIM


@dataclass(frozen=True)
class TrainingConfig:
    device: str
    epochs: int
    learning_rate: float
    early_stopping_metric: str = "val_loss"
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0
    aux_loss_weight: float = DEFAULT_AUX_LOSS_WEIGHT
    horizon_loss_weights: tuple[float, ...] = DEFAULT_HORIZON_LOSS_WEIGHTS
    head_loss_weights: dict[str, float] = field(default_factory=dict)
    yaw_consistency_weight: float = 0.0
    yaw_rate_scale_to_degrees: float = 57.29577951308232
    speed_consistency_weight: float = 0.0
    yaw_loss_weighting: YawLossWeightingConfig = field(default_factory=lambda: YawLossWeightingConfig())


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
    turn_oversampling: TurnOversamplingConfig = field(default_factory=lambda: TurnOversamplingConfig())


@dataclass(frozen=True)
class TrainConfig:
    dataset: DatasetConfig
    output: OutputConfig
    model: ModelConfig
    training: TrainingConfig
    loader: LoaderConfig
    state_inputs: Any | None = None


@dataclass(frozen=True)
class TrainingContext:
    config: TrainConfig
    config_path: Path
    run_dir: Path
    run_metrics_path: Path
    data_root: Path
    train_dataset: FsdDataset
    val_dataset: FsdDataset
    val_subset: TorchDataset[DatasetItem]
    selected_val_trip_count: int
    total_val_trip_count: int
    model: Module
    device: torch.device
    optimizer: Optimizer
    scaler: torch.amp.GradScaler
    train_image_shape: tuple[int, ...]
    train_telemetry_shape: tuple[int, ...]
    train_target_control_shape: tuple[int, ...]
    train_target_aux_shape: tuple[int, ...]


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
    parser = argparse.ArgumentParser(description="Train the GTA temporal telemetry GRU planner model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the training TOML config. Default: {DEFAULT_CONFIG_PATH}",
    )
    return parser.parse_args()


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
        run_ids = tuple(str(item).strip() for item in raw_value if str(item).strip())
        if not run_ids:
            raise ValueError(f"dataset.{key} must contain at least one run id")
        return run_ids

    fallback_run_id = _optional_str(dataset_raw.get(fallback_key))
    if fallback_run_id is None:
        raise ValueError(f"Missing dataset.{key} and deprecated dataset.{fallback_key}")
    return (fallback_run_id,)


def _resolve_run_paths_from_ids(data_root: str, run_ids: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(Path(data_root) / "runs" / run_id) for run_id in run_ids)


def _infer_frame_stride(image_offsets: tuple[int, ...]) -> int:
    if len(image_offsets) < 2:
        return 1
    strides = {image_offsets[index + 1] - image_offsets[index] for index in range(len(image_offsets) - 1)}
    if len(strides) == 1:
        return max(1, int(next(iter(strides))))
    return 1


def _parse_horizon_loss_weights(training_raw: dict[str, Any], horizon: int) -> tuple[float, ...]:
    raw_weights = training_raw.get("horizon_loss_weights", list(DEFAULT_HORIZON_LOSS_WEIGHTS))
    if not isinstance(raw_weights, list) or not raw_weights:
        raise ValueError("training.horizon_loss_weights must be a non-empty TOML array")
    weights = tuple(float(item) for item in raw_weights)
    if len(weights) != horizon:
        raise ValueError(
            "training.horizon_loss_weights must match dataset.future_offsets length: "
            f"weights={len(weights)} future_offsets={horizon}"
        )
    if any(weight <= 0.0 for weight in weights):
        raise ValueError("training.horizon_loss_weights must contain only positive values")
    return weights


def load_config(path: Path) -> TrainConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    dataset_raw = raw["dataset"]
    output_raw = raw["output"]
    model_raw = raw.get("model", {})
    training_raw = raw["training"]
    loader_raw = raw["loader"]

    data_root = normalize_windows_drive_path(str(dataset_raw["data_root"]))
    train_run_ids = _resolve_training_run_ids(dataset_raw, "train_run_ids", fallback_key="run_id")
    val_run_ids = _resolve_training_run_ids(dataset_raw, "val_run_ids", fallback_key="val_id")
    (
        image_offsets,
        telemetry_offsets,
        future_offsets,
        telemetry_feature_names,
        control_target_names,
        aux_target_names,
    ) = parse_temporal_dataset_config(raw)

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
            image_offsets=image_offsets,
            telemetry_offsets=telemetry_offsets,
            future_offsets=future_offsets,
            telemetry_feature_names=telemetry_feature_names,
            control_target_names=control_target_names,
            aux_target_names=aux_target_names,
            window_size=len(image_offsets),
            frame_stride=int(dataset_raw.get("frame_stride", _infer_frame_stride(image_offsets))),
            sample_stride=int(dataset_raw.get("sample_stride", max(future_offsets))),
        ),
        output=OutputConfig(base_dir=str(output_raw["base_dir"])),
        model=ModelConfig(
            width_multiplier=float(model_raw.get("width_multiplier", DEFAULT_WIDTH_MULTIPLIER)),
            telemetry_hidden_dim=int(model_raw.get("telemetry_hidden_dim", DEFAULT_TELEMETRY_HIDDEN_DIM)),
        ),
        training=TrainingConfig(
            device=str(training_raw.get("device", "auto")).strip().lower(),
            epochs=int(training_raw["epochs"]),
            learning_rate=float(training_raw["learning_rate"]),
            early_stopping_metric=str(training_raw.get("early_stopping_metric", "val_loss")).strip(),
            early_stopping_patience=int(training_raw.get("early_stopping_patience", 3)),
            early_stopping_min_delta=float(training_raw.get("early_stopping_min_delta", 0.0)),
            aux_loss_weight=float(training_raw.get("aux_loss_weight", DEFAULT_AUX_LOSS_WEIGHT)),
            horizon_loss_weights=_parse_horizon_loss_weights(training_raw, len(future_offsets)),
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
    config: TrainConfig,
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
) -> dict[str, Any]:
    checkpoint_path = run_dir / f"epoch-{epoch_index:03d}.pt"
    metadata = {
        "epoch": epoch_index,
        "planner_format": PLANNER_FORMAT,
        "planner_format_version": 1,
        "frame_window_size": config.dataset.window_size,
        "frame_stride": config.dataset.frame_stride,
        "sample_stride": config.dataset.sample_stride,
        "input_channels": config.dataset.window_size * 3,
        "image_size": {
            "width": config.dataset.image_width,
            "height": config.dataset.image_height,
        },
        "image_offsets": list(config.dataset.image_offsets),
        "telemetry_offsets": list(config.dataset.telemetry_offsets),
        "future_offsets": list(config.dataset.future_offsets),
        "telemetry_feature_names": list(config.dataset.telemetry_feature_names),
        "control_target_names": list(config.dataset.control_target_names),
        "aux_target_names": list(config.dataset.aux_target_names),
        "model": {
            "width_multiplier": config.model.width_multiplier,
            "telemetry_hidden_dim": config.model.telemetry_hidden_dim,
        },
        "training": {
            "aux_loss_weight": config.training.aux_loss_weight,
            "horizon_loss_weights": list(config.training.horizon_loss_weights),
            "early_stopping_metric": config.training.early_stopping_metric,
            "early_stopping_patience": config.training.early_stopping_patience,
            "early_stopping_min_delta": config.training.early_stopping_min_delta,
        },
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
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
    torch.save(metadata, checkpoint_path)
    return {
        "epoch": epoch_index,
        "checkpoint": str(checkpoint_path),
        **{key: value for key, value in metadata.items() if key not in {"model_state_dict", "optimizer_state_dict"}},
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
    *,
    image_size: tuple[int, int],
    frame_count: int,
    telemetry_length: int,
    telemetry_feature_dim: int,
) -> bool:
    if device.type != "cuda":
        return True
    try:
        image_width, image_height = image_size
        probe_images = torch.zeros((1, frame_count, 3, image_height, image_width), device=device)
        probe_telemetry = torch.zeros((1, telemetry_length, telemetry_feature_dim), device=device)
        with torch.no_grad():
            _ = model(probe_images, probe_telemetry)
        return True
    except RuntimeError as exc:
        message = str(exc).lower()
        if "no kernel image is available" in message or "not compatible with the current pytorch installation" in message:
            return False
        raise


def _build_phase_loader(
    dataset: TorchDataset[DatasetItem],
    *,
    device: torch.device,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
    cpu_batch_size: int,
) -> DataLoader[DatasetItem]:
    if device.type != "cuda":
        return DataLoader(
            dataset,
            batch_size=cpu_batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
        )

    resolved_num_workers = max(0, num_workers)
    resolved_pin_memory = pin_memory and resolved_num_workers > 0
    resolved_persistent_workers = persistent_workers and resolved_num_workers > 0
    resolved_prefetch_factor = prefetch_factor if resolved_num_workers > 0 else None

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=resolved_num_workers,
        pin_memory=resolved_pin_memory,
        persistent_workers=resolved_persistent_workers,
        prefetch_factor=resolved_prefetch_factor,
    )


def _format_loader_summary(
    phase_name: str,
    loader: DataLoader[DatasetItem],
) -> str:
    prefetch_factor = getattr(loader, "prefetch_factor", None)
    return (
        f"{phase_name}_loader batch_size={loader.batch_size} "
        f"num_workers={loader.num_workers} pin_memory={loader.pin_memory} "
        f"persistent_workers={loader.persistent_workers} "
        f"prefetch_factor={prefetch_factor} samples={_loader_dataset_len(loader)}"
    )


def build_validation_subset(
    dataset: FsdDataset,
    val_split: float,
    *,
    seed: int = 42,
) -> tuple[FsdDataset | Subset[DatasetItem], int, int]:
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


def _loader_dataset_len(dataset: TorchDataset[DatasetItem]) -> int:
    return len(dataset)


def _process_rss_bytes() -> int | None:
    if psutil is not None:
        return int(psutil.Process(os.getpid()).memory_info().rss)

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


def _shutdown_loader_iterator(loader_iter: Iterator[DatasetItem] | None) -> None:
    if loader_iter is None:
        return
    shutdown = getattr(loader_iter, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


def _release_phase_resources(
    device: torch.device,
    *,
    loader: DataLoader[DatasetItem] | None = None,
    loader_iter: Iterator[DatasetItem] | None = None,
) -> None:
    _shutdown_loader_iterator(loader_iter)
    del loader_iter
    del loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _mean_or_zero(total: float, count: int) -> float:
    return 0.0 if count <= 0 else float(total / count)


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


def _weighted_elementwise_mean(loss_tensor: Tensor, horizon_weights: tuple[float, ...] | None) -> Tensor:
    if loss_tensor.ndim < 2:
        raise ValueError(f"loss tensor must have at least rank 2 [B, H, ...], got {tuple(loss_tensor.shape)}")
    if horizon_weights is None:
        return loss_tensor.mean()

    weight_tensor = torch.as_tensor(horizon_weights, dtype=loss_tensor.dtype, device=loss_tensor.device)
    if weight_tensor.ndim != 1 or weight_tensor.numel() != loss_tensor.shape[1]:
        raise ValueError(
            "horizon_weights must be rank 1 and match the horizon dimension: "
            f"weights={tuple(weight_tensor.shape)} loss={tuple(loss_tensor.shape)}"
        )
    trailing_count = math.prod(loss_tensor.shape[2:]) if loss_tensor.ndim > 2 else 1
    if trailing_count <= 0:
        raise ValueError("loss tensor must have at least one element per horizon")

    view_shape = [1] * loss_tensor.ndim
    view_shape[1] = int(weight_tensor.numel())
    weighted_loss = loss_tensor * weight_tensor.view(*view_shape)
    denominator = loss_tensor.shape[0] * trailing_count * float(weight_tensor.sum().item())
    if denominator <= 0.0:
        raise ValueError("horizon loss weights must sum to > 0")
    return weighted_loss.sum() / denominator


def compute_planner_losses(
    pred_controls: Tensor,
    target_controls: Tensor,
    pred_aux: Tensor,
    target_aux: Tensor,
    *,
    aux_loss_weight: float,
    horizon_loss_weights: tuple[float, ...] | None,
) -> dict[str, Tensor]:
    control_loss = _weighted_elementwise_mean(
        F.smooth_l1_loss(pred_controls.float(), target_controls.float(), reduction="none"),
        horizon_loss_weights,
    )
    aux_loss = _weighted_elementwise_mean(
        F.smooth_l1_loss(pred_aux.float(), target_aux.float(), reduction="none"),
        horizon_loss_weights,
    )
    total_loss = control_loss + (aux_loss_weight * aux_loss)
    return {
        "loss": total_loss,
        "control_loss": control_loss,
        "aux_loss": aux_loss,
    }


def initialize_metric_totals(
    horizon: int,
    *,
    control_target_names: tuple[str, ...] = DEFAULT_CONTROL_TARGET_NAMES,
    aux_target_names: tuple[str, ...] = DEFAULT_AUX_TARGET_NAMES,
) -> dict[str, Any]:
    totals = {
        "loss_sum": 0.0,
        "control_loss_sum": 0.0,
        "aux_loss_sum": 0.0,
        "control_abs_sum": 0.0,
        "control_count": 0,
        "aux_abs_sum": 0.0,
        "aux_count": 0,
        "control_horizon_abs_sum": [0.0 for _ in range(horizon)],
        "control_horizon_count": [0 for _ in range(horizon)],
        "aux_horizon_abs_sum": [0.0 for _ in range(horizon)],
        "aux_horizon_count": [0 for _ in range(horizon)],
        "sample_count": 0,
        "batch_count": 0,
    }
    for name in control_target_names:
        totals[f"{name}_abs_sum"] = 0.0
        totals[f"{name}_count"] = 0
    for name in aux_target_names:
        totals[f"{name}_abs_sum"] = 0.0
        totals[f"{name}_count"] = 0
    return totals


def update_metric_totals(
    totals: dict[str, Any],
    *,
    pred_controls: Tensor,
    target_controls: Tensor,
    pred_aux: Tensor,
    target_aux: Tensor,
    losses: dict[str, Tensor],
    control_target_names: tuple[str, ...] = DEFAULT_CONTROL_TARGET_NAMES,
    aux_target_names: tuple[str, ...] = DEFAULT_AUX_TARGET_NAMES,
) -> None:
    control_abs = (pred_controls.float() - target_controls.float()).abs()
    aux_abs = (pred_aux.float() - target_aux.float()).abs()

    totals["loss_sum"] += float(losses["loss"].item())
    totals["control_loss_sum"] += float(losses["control_loss"].item())
    totals["aux_loss_sum"] += float(losses["aux_loss"].item())

    totals["control_abs_sum"] += float(control_abs.sum().item())
    totals["control_count"] += int(control_abs.numel())
    totals["aux_abs_sum"] += float(aux_abs.sum().item())
    totals["aux_count"] += int(aux_abs.numel())

    for index, name in enumerate(control_target_names):
        totals[f"{name}_abs_sum"] += float(control_abs[..., index].sum().item())
        totals[f"{name}_count"] += int(control_abs[..., index].numel())
    for index, name in enumerate(aux_target_names):
        totals[f"{name}_abs_sum"] += float(aux_abs[..., index].sum().item())
        totals[f"{name}_count"] += int(aux_abs[..., index].numel())

    for index in range(control_abs.shape[1]):
        totals["control_horizon_abs_sum"][index] += float(control_abs[:, index, :].sum().item())
        totals["control_horizon_count"][index] += int(control_abs[:, index, :].numel())
        totals["aux_horizon_abs_sum"][index] += float(aux_abs[:, index, :].sum().item())
        totals["aux_horizon_count"][index] += int(aux_abs[:, index, :].numel())

    totals["sample_count"] += int(pred_controls.shape[0])
    totals["batch_count"] += 1


def finalize_metrics(
    totals: dict[str, Any],
    future_offsets: tuple[int, ...],
    *,
    control_target_names: tuple[str, ...] = DEFAULT_CONTROL_TARGET_NAMES,
    aux_target_names: tuple[str, ...] = DEFAULT_AUX_TARGET_NAMES,
) -> MetricPayload:
    batch_count = int(totals["batch_count"])
    metrics: MetricPayload = {
        "loss": _mean_or_zero(float(totals["loss_sum"]), batch_count),
        "val_loss": _mean_or_zero(float(totals["loss_sum"]), batch_count),
        "control_loss": _mean_or_zero(float(totals["control_loss_sum"]), batch_count),
        "aux_loss": _mean_or_zero(float(totals["aux_loss_sum"]), batch_count),
        "control_mae_overall": _mean_or_zero(float(totals["control_abs_sum"]), int(totals["control_count"])),
        "control_overall_mae": _mean_or_zero(float(totals["control_abs_sum"]), int(totals["control_count"])),
        "aux_mae_overall": _mean_or_zero(float(totals["aux_abs_sum"]), int(totals["aux_count"])),
        "aux_overall_mae": _mean_or_zero(float(totals["aux_abs_sum"]), int(totals["aux_count"])),
        "sample_count": int(totals["sample_count"]),
        "batch_count": batch_count,
    }
    for name in control_target_names:
        metrics[f"{name}_mae"] = _mean_or_zero(float(totals[f"{name}_abs_sum"]), int(totals[f"{name}_count"]))
    for name in aux_target_names:
        metrics[f"{name}_mae"] = _mean_or_zero(float(totals[f"{name}_abs_sum"]), int(totals[f"{name}_count"]))
    for index, offset in enumerate(future_offsets):
        metrics[f"control_mae_t+{offset}"] = _mean_or_zero(
            float(totals["control_horizon_abs_sum"][index]),
            int(totals["control_horizon_count"][index]),
        )
        metrics[f"aux_mae_t+{offset}"] = _mean_or_zero(
            float(totals["aux_horizon_abs_sum"][index]),
            int(totals["aux_horizon_count"][index]),
        )
    return metrics


def format_first_batch_debug(
    images: Tensor,
    telemetry: Tensor,
    target_controls: Tensor,
    target_aux: Tensor,
) -> str:
    return (
        "first_train_batch "
        f"images_shape={tuple(images.shape)} "
        f"telemetry_shape={tuple(telemetry.shape)} "
        f"target_controls_shape={tuple(target_controls.shape)} "
        f"target_aux_shape={tuple(target_aux.shape)} "
        f"dtype={images.dtype}"
    )


def _round_nested(values: Any, digits: int = 4) -> Any:
    if isinstance(values, float):
        return round(values, digits)
    if isinstance(values, list):
        return [_round_nested(item, digits=digits) for item in values]
    return values


def _format_named_horizon_rows(
    values: Tensor,
    names: tuple[str, ...],
    future_offsets: tuple[int, ...],
) -> list[dict[str, Any]]:
    sample0 = values[0].detach().float().cpu().tolist()
    rows: list[dict[str, Any]] = []
    for horizon_index, offset in enumerate(future_offsets):
        row = {"horizon": f"t+{offset}"}
        for value_index, name in enumerate(names):
            row[name] = _round_nested(sample0[horizon_index][value_index])
        rows.append(row)
    return rows


def _format_named_horizon_section(
    title: str,
    values: Tensor,
    names: tuple[str, ...],
    future_offsets: tuple[int, ...],
) -> str:
    lines = [title]
    for row in _format_named_horizon_rows(values, names, future_offsets):
        horizon = str(row["horizon"])
        parts = [f"{name}={row[name]}" for name in names]
        lines.append(f"  {horizon}: " + " ".join(parts))
    return "\n".join(lines)


def format_first_batch_predictions(
    pred_controls: Tensor,
    target_controls: Tensor,
    pred_aux: Tensor,
    target_aux: Tensor,
    *,
    control_target_names: tuple[str, ...],
    aux_target_names: tuple[str, ...],
    future_offsets: tuple[int, ...],
    batch_index: int | None = None,
) -> str:
    header = "train_batch_predictions sample=0"
    if batch_index is not None:
        header += f" batch={batch_index}"
    return "\n".join((
        header,
        _format_named_horizon_section("pred_controls:", pred_controls, control_target_names, future_offsets),
        _format_named_horizon_section("target_controls:", target_controls, control_target_names, future_offsets),
        _format_named_horizon_section("pred_aux:", pred_aux, aux_target_names, future_offsets),
        _format_named_horizon_section("target_aux:", target_aux, aux_target_names, future_offsets),
    ))


def _merge_metric_totals(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in (
        "loss_sum",
        "control_loss_sum",
        "aux_loss_sum",
        "control_abs_sum",
        "control_count",
        "aux_abs_sum",
        "aux_count",
        "sample_count",
        "batch_count",
    ):
        target[key] += source[key]

    for key, value in source.items():
        if key.endswith("_abs_sum") or key.endswith("_count"):
            if key in target and key not in {
                "loss_sum",
                "control_loss_sum",
                "aux_loss_sum",
                "control_abs_sum",
                "control_count",
                "aux_abs_sum",
                "aux_count",
                "control_horizon_abs_sum",
                "control_horizon_count",
                "aux_horizon_abs_sum",
                "aux_horizon_count",
                "sample_count",
                "batch_count",
            }:
                target[key] += value

    for key in ("control_horizon_abs_sum", "control_horizon_count", "aux_horizon_abs_sum", "aux_horizon_count"):
        for index, value in enumerate(source[key]):
            target[key][index] += value


def train_batch(
    images: Tensor,
    telemetry: Tensor,
    target_controls: Tensor,
    target_aux: Tensor,
    model: Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    aux_loss_weight: float,
    horizon_loss_weights: tuple[float, ...],
    future_offsets: tuple[int, ...],
    control_target_names: tuple[str, ...],
    aux_target_names: tuple[str, ...],
    batch_index: int | None = None,
    capture_prediction_debug: bool = False,
) -> tuple[float, BatchTiming, dict[str, Any], str | None]:
    step_start = time.perf_counter()
    transfer_start = time.perf_counter()
    images = images.to(device, non_blocking=device.type == "cuda")
    telemetry = telemetry.to(device, non_blocking=device.type == "cuda")
    target_controls = target_controls.to(device, non_blocking=device.type == "cuda")
    target_aux = target_aux.to(device, non_blocking=device.type == "cuda")
    h2d_time = time.perf_counter() - transfer_start

    optimizer.zero_grad(set_to_none=True)
    batch_totals = initialize_metric_totals(
        target_controls.shape[1],
        control_target_names=control_target_names,
        aux_target_names=aux_target_names,
    )

    forward_backward_start = time.perf_counter()
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        output = model(images, telemetry)
        losses = compute_planner_losses(
            output["pred_controls"],
            target_controls,
            output["pred_aux"],
            target_aux,
            aux_loss_weight=aux_loss_weight,
            horizon_loss_weights=horizon_loss_weights,
        )
    scaler.scale(losses["loss"]).backward()
    forward_backward_time = time.perf_counter() - forward_backward_start

    optimizer_start = time.perf_counter()
    scaler.step(optimizer)
    scaler.update()
    optimizer_time = time.perf_counter() - optimizer_start

    update_metric_totals(
        batch_totals,
        pred_controls=output["pred_controls"].detach(),
        target_controls=target_controls.detach(),
        pred_aux=output["pred_aux"].detach(),
        target_aux=target_aux.detach(),
        losses=losses,
        control_target_names=control_target_names,
        aux_target_names=aux_target_names,
    )
    prediction_debug = None
    if capture_prediction_debug:
        prediction_debug = format_first_batch_predictions(
            output["pred_controls"],
            target_controls,
            output["pred_aux"],
            target_aux,
            control_target_names=control_target_names,
            aux_target_names=aux_target_names,
            future_offsets=future_offsets,
            batch_index=batch_index,
        )

    timing = BatchTiming(
        loader_wait_s=0.0,
        h2d_s=h2d_time,
        forward_backward_s=forward_backward_time,
        optimizer_s=optimizer_time,
        step_s=time.perf_counter() - step_start,
    )
    return float(losses["loss"].item()), timing, batch_totals, prediction_debug


def format_batch_timing(timing: BatchTiming) -> str:
    return (
        f"wait_s={timing.loader_wait_s:.3f} "
        f"batch_s={timing.step_s:.3f} "
        f"h2d_s={timing.h2d_s:.3f} "
        f"fwd_bwd_s={timing.forward_backward_s:.3f} "
        f"opt_s={timing.optimizer_s:.3f}"
    )


def train_epoch(
    loader: DataLoader[DatasetItem],
    optimizer: Optimizer,
    model: Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    future_offsets: tuple[int, ...],
    control_target_names: tuple[str, ...],
    aux_target_names: tuple[str, ...],
    aux_loss_weight: float,
    horizon_loss_weights: tuple[float, ...],
    log_every_n_batches: int,
) -> tuple[MetricPayload, float, dict[str, float]]:
    epoch_start = time.perf_counter()
    model.train()
    totals = initialize_metric_totals(
        len(future_offsets),
        control_target_names=control_target_names,
        aux_target_names=aux_target_names,
    )
    timing_totals = _empty_timing_totals()
    loader_iter: Iterator[DatasetItem] | None = None
    total_batches = len(loader)

    try:
        loader_iter = iter(loader)
        for batch_index in range(1, total_batches + 1):
            wait_start = time.perf_counter()
            images, telemetry, target_controls, target_aux = next(loader_iter)
            loader_wait_time = time.perf_counter() - wait_start
            if batch_index == 1:
                print(format_first_batch_debug(images, telemetry, target_controls, target_aux))
            batch_loss, batch_timing, batch_totals, prediction_debug = train_batch(
                images,
                telemetry,
                target_controls,
                target_aux,
                model,
                optimizer,
                scaler,
                device,
                aux_loss_weight=aux_loss_weight,
                horizon_loss_weights=horizon_loss_weights,
                future_offsets=future_offsets,
                control_target_names=control_target_names,
                aux_target_names=aux_target_names,
                batch_index=batch_index,
                capture_prediction_debug=batch_index == 1 or batch_index % 30 == 0 or batch_index == total_batches,
            )
            if prediction_debug is not None:
                print(prediction_debug)
            batch_timing = BatchTiming(
                loader_wait_s=loader_wait_time,
                h2d_s=batch_timing.h2d_s,
                forward_backward_s=batch_timing.forward_backward_s,
                optimizer_s=batch_timing.optimizer_s,
                step_s=batch_timing.step_s,
            )
            _update_timing_totals(timing_totals, batch_timing)
            _merge_metric_totals(totals, batch_totals)
            if batch_index == 1 or batch_index == total_batches or batch_index % log_every_n_batches == 0:
                batch_metrics = finalize_metrics(
                    batch_totals,
                    future_offsets,
                    control_target_names=control_target_names,
                    aux_target_names=aux_target_names,
                )
                print(
                    f"batch={batch_index}/{total_batches} "
                    f"loss={batch_loss:.6f} "
                    f"control_loss={float(batch_metrics['control_loss']):.6f} "
                    f"aux_loss={float(batch_metrics['aux_loss']):.6f} "
                    f"{format_batch_timing(batch_timing)}"
                )
    finally:
        _shutdown_loader_iterator(loader_iter)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    epoch_time = time.perf_counter() - epoch_start
    metrics = finalize_metrics(
        totals,
        future_offsets,
        control_target_names=control_target_names,
        aux_target_names=aux_target_names,
    )
    batch_count = int(totals["batch_count"])
    avg_timings = {key: _mean_timing(total, batch_count) for key, total in timing_totals.items()}
    return metrics, epoch_time, avg_timings


def evaluate_epoch(
    loader: DataLoader[DatasetItem],
    model: Module,
    device: torch.device,
    *,
    future_offsets: tuple[int, ...],
    control_target_names: tuple[str, ...],
    aux_target_names: tuple[str, ...],
    aux_loss_weight: float,
    horizon_loss_weights: tuple[float, ...],
) -> tuple[MetricPayload, float]:
    eval_start = time.perf_counter()
    model.eval()
    totals = initialize_metric_totals(
        len(future_offsets),
        control_target_names=control_target_names,
        aux_target_names=aux_target_names,
    )

    loader_iter: Iterator[DatasetItem] | None = None
    try:
        loader_iter = iter(loader)
        with torch.inference_mode():
            for images, telemetry, target_controls, target_aux in loader_iter:
                images = images.to(device, non_blocking=device.type == "cuda")
                telemetry = telemetry.to(device, non_blocking=device.type == "cuda")
                target_controls = target_controls.to(device, non_blocking=device.type == "cuda")
                target_aux = target_aux.to(device, non_blocking=device.type == "cuda")
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    output = model(images, telemetry)
                    losses = compute_planner_losses(
                        output["pred_controls"],
                        target_controls,
                        output["pred_aux"],
                        target_aux,
                        aux_loss_weight=aux_loss_weight,
                        horizon_loss_weights=horizon_loss_weights,
                    )
                update_metric_totals(
                    totals,
                    pred_controls=output["pred_controls"],
                    target_controls=target_controls,
                    pred_aux=output["pred_aux"],
                    target_aux=target_aux,
                    losses=losses,
                    control_target_names=control_target_names,
                    aux_target_names=aux_target_names,
                )
    finally:
        _shutdown_loader_iterator(loader_iter)

    if int(totals["batch_count"]) <= 0 or int(totals["sample_count"]) <= 0:
        raise ValueError(
            "Validation produced zero batches/samples. "
            "Check processed validation runs for empty dataset.jsonl files or missing temporal telemetry windows."
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    eval_time = time.perf_counter() - eval_start
    return finalize_metrics(
        totals,
        future_offsets,
        control_target_names=control_target_names,
        aux_target_names=aux_target_names,
    ), eval_time


def build_run_summary(context: TrainingContext) -> dict[str, object]:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "completed_at": None,
        "elapsed_s": 0.0,
        "elapsed_hms": "00:00:00",
        "config_path": str(context.config_path),
        "dataset_root": str(context.data_root),
        "planner_format": PLANNER_FORMAT,
        "train_run_ids": list(context.config.dataset.train_run_ids),
        "val_run_ids": list(context.config.dataset.val_run_ids),
        "train_run_paths": list(context.config.dataset.train_run_paths),
        "val_run_paths": list(context.config.dataset.val_run_paths),
        "image_size": {
            "width": context.config.dataset.image_width,
            "height": context.config.dataset.image_height,
        },
        "image_offsets": list(context.config.dataset.image_offsets),
        "telemetry_offsets": list(context.config.dataset.telemetry_offsets),
        "future_offsets": list(context.config.dataset.future_offsets),
        "telemetry_feature_names": list(context.config.dataset.telemetry_feature_names),
        "control_target_names": list(context.config.dataset.control_target_names),
        "aux_target_names": list(context.config.dataset.aux_target_names),
        "model": {
            "width_multiplier": context.config.model.width_multiplier,
            "telemetry_hidden_dim": context.config.model.telemetry_hidden_dim,
        },
        "training": {
            "epochs_total": context.config.training.epochs,
            "learning_rate": context.config.training.learning_rate,
            "aux_loss_weight": context.config.training.aux_loss_weight,
            "horizon_loss_weights": list(context.config.training.horizon_loss_weights),
            "early_stopping_metric": context.config.training.early_stopping_metric,
            "early_stopping_patience": context.config.training.early_stopping_patience,
            "early_stopping_min_delta": context.config.training.early_stopping_min_delta,
        },
        "device": str(context.device),
        "dataset_samples": len(context.train_dataset),
        "dataset_trips": context.train_dataset.trip_count,
        "validation_dataset_samples": len(context.val_dataset),
        "validation_dataset_trips": context.val_dataset.trip_count,
        "validation_selected_trips": context.selected_val_trip_count,
        "train_image_shape": list(context.train_image_shape),
        "train_telemetry_shape": list(context.train_telemetry_shape),
        "train_target_control_shape": list(context.train_target_control_shape),
        "train_target_aux_shape": list(context.train_target_aux_shape),
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
        "epochs": [],
    }


def prepare_model(
    device: torch.device,
    *,
    image_size: tuple[int, int],
    frame_count: int,
    telemetry_length: int,
    telemetry_feature_dim: int,
    telemetry_hidden_dim: int,
    horizon: int,
    control_dim: int,
    width_multiplier: float,
) -> tuple[Module, torch.device]:
    model = DrivingCNN(
        frame_count=frame_count,
        telemetry_feature_dim=telemetry_feature_dim,
        telemetry_hidden_dim=telemetry_hidden_dim,
        telemetry_sequence_length=telemetry_length,
        horizon=horizon,
        control_dim=control_dim,
        width_multiplier=width_multiplier,
    ).to(device)
    if probe_device(
        model,
        device,
        image_size=image_size,
        frame_count=frame_count,
        telemetry_length=telemetry_length,
        telemetry_feature_dim=telemetry_feature_dim,
    ):
        return model, device

    print(
        "CUDA is available but this PyTorch build cannot run on your GPU "
        "(likely missing sm_120 support). Falling back to CPU."
    )
    fallback_device = torch.device("cpu")
    return DrivingCNN(
        frame_count=frame_count,
        telemetry_feature_dim=telemetry_feature_dim,
        telemetry_hidden_dim=telemetry_hidden_dim,
        telemetry_sequence_length=telemetry_length,
        horizon=horizon,
        control_dim=control_dim,
        width_multiplier=width_multiplier,
    ).to(fallback_device), fallback_device


def build_training_context(config: TrainConfig, config_path: Path) -> TrainingContext:
    data_root = Path(config.dataset.data_root)
    requested_device = select_device(config.training.device)
    image_size = (config.dataset.image_width, config.dataset.image_height)

    train_dataset = FsdDataset(
        run_paths=config.dataset.train_run_paths,
        image_size=image_size,
        expected_window_size=config.dataset.window_size,
        image_offsets=config.dataset.image_offsets,
        telemetry_offsets=config.dataset.telemetry_offsets,
        future_offsets=config.dataset.future_offsets,
        telemetry_feature_names=config.dataset.telemetry_feature_names,
        control_target_names=config.dataset.control_target_names,
        aux_target_names=config.dataset.aux_target_names,
    )
    val_dataset = FsdDataset(
        run_paths=config.dataset.val_run_paths,
        image_size=image_size,
        expected_window_size=config.dataset.window_size,
        image_offsets=config.dataset.image_offsets,
        telemetry_offsets=config.dataset.telemetry_offsets,
        future_offsets=config.dataset.future_offsets,
        telemetry_feature_names=config.dataset.telemetry_feature_names,
        control_target_names=config.dataset.control_target_names,
        aux_target_names=config.dataset.aux_target_names,
    )
    if len(train_dataset) <= 0:
        raise ValueError(
            "Training dataset resolved to zero samples. "
            f"train_run_ids={list(config.dataset.train_run_ids)} "
            f"train_trips={train_dataset.trip_count} "
            f"{train_dataset.format_rejected_sample_summary()}"
        )
    if len(val_dataset) <= 0:
        raise ValueError(
            "Validation dataset resolved to zero samples before subset selection. "
            f"val_run_ids={list(config.dataset.val_run_ids)} "
            f"val_trips={val_dataset.trip_count}. "
            "Check processed validation runs for empty dataset.jsonl files. "
            f"{val_dataset.format_rejected_sample_summary()}"
        )

    train_images, train_telemetry, train_target_controls, train_target_aux = train_dataset[0]
    model, device = prepare_model(
        requested_device,
        image_size=image_size,
        frame_count=len(config.dataset.image_offsets),
        telemetry_length=len(config.dataset.telemetry_offsets),
        telemetry_feature_dim=len(config.dataset.telemetry_feature_names),
        telemetry_hidden_dim=config.model.telemetry_hidden_dim,
        horizon=len(config.dataset.future_offsets),
        control_dim=len(config.dataset.control_target_names),
        width_multiplier=config.model.width_multiplier,
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    val_subset, selected_val_trip_count, total_val_trip_count = build_validation_subset(
        val_dataset,
        config.loader.val_split,
    )
    if _loader_dataset_len(val_subset) <= 0:
        raise ValueError(
            "Validation subset resolved to zero usable samples. "
            f"val_run_ids={list(config.dataset.val_run_ids)} "
            f"validation_dataset_trips={val_dataset.trip_count} "
            f"validation_dataset_samples={len(val_dataset)} "
            f"selected_val_trips={selected_val_trip_count}/{total_val_trip_count}. "
            "Check processed validation runs for empty dataset.jsonl files or missing temporal telemetry windows."
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
        model=model,
        device=device,
        optimizer=optimizer,
        scaler=scaler,
        train_image_shape=tuple(train_images.shape),
        train_telemetry_shape=tuple(train_telemetry.shape),
        train_target_control_shape=tuple(train_target_controls.shape),
        train_target_aux_shape=tuple(train_target_aux.shape),
    )


def _supported_metric_names(
    future_offsets: tuple[int, ...],
    control_target_names: tuple[str, ...],
    aux_target_names: tuple[str, ...],
) -> set[str]:
    names = {
        "loss",
        "val_loss",
        "control_loss",
        "aux_loss",
        "control_mae_overall",
        "control_overall_mae",
        "aux_mae_overall",
        "aux_overall_mae",
    }
    for name in control_target_names:
        names.add(f"{name}_mae")
    for name in aux_target_names:
        names.add(f"{name}_mae")
    for offset in future_offsets:
        names.add(f"control_mae_t+{offset}")
        names.add(f"aux_mae_t+{offset}")
    return names


def create_early_stopping(
    config: TrainingConfig,
    future_offsets: tuple[int, ...],
    control_target_names: tuple[str, ...],
    aux_target_names: tuple[str, ...],
) -> EarlyStoppingState:
    metric_name = config.early_stopping_metric
    if metric_name not in _supported_metric_names(future_offsets, control_target_names, aux_target_names):
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
    print(_format_loader_summary("train", train_loader))
    try:
        train_metrics, train_epoch_time, avg_timings = train_epoch(
            train_loader,
            context.optimizer,
            context.model,
            context.scaler,
            context.device,
            future_offsets=context.config.dataset.future_offsets,
            control_target_names=context.config.dataset.control_target_names,
            aux_target_names=context.config.dataset.aux_target_names,
            aux_loss_weight=context.config.training.aux_loss_weight,
            horizon_loss_weights=context.config.training.horizon_loss_weights,
            log_every_n_batches=context.config.loader.log_every_n_batches,
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
    print(_format_loader_summary("val", val_loader))
    try:
        val_metrics, val_epoch_time = evaluate_epoch(
            val_loader,
            context.model,
            context.device,
            future_offsets=context.config.dataset.future_offsets,
            control_target_names=context.config.dataset.control_target_names,
            aux_target_names=context.config.dataset.aux_target_names,
            aux_loss_weight=context.config.training.aux_loss_weight,
            horizon_loss_weights=context.config.training.horizon_loss_weights,
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
        config=context.config,
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


def _format_elapsed_hms(elapsed_s: float) -> str:
    total_seconds = max(0, int(round(elapsed_s)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def print_epoch_summary(
    epoch_result: EpochResult,
    *,
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

    summary_parts = [
        f"epoch={epoch_result.epoch_index}",
        f"train_loss={float(epoch_result.train_metrics['loss']):.6f}",
        f"val_loss={float(epoch_result.val_metrics['val_loss']):.6f}",
        f"train_control_mae={float(epoch_result.train_metrics['control_mae_overall']):.6f}",
        f"val_control_mae={float(epoch_result.val_metrics['control_mae_overall']):.6f}",
        f"train_aux_mae={float(epoch_result.train_metrics['aux_mae_overall']):.6f}",
        f"val_aux_mae={float(epoch_result.val_metrics['aux_mae_overall']):.6f}",
        f"steering_mae={float(epoch_result.val_metrics['steering_mae']):.6f}",
        f"acceleration_mae={float(epoch_result.val_metrics['acceleration_mae']):.6f}",
        f"future_speed_mae={float(epoch_result.val_metrics['future_speed_mae']):.6f}",
        f"future_yaw_delta_mae={float(epoch_result.val_metrics['future_yaw_delta_mae']):.6f}",
        f"future_yaw_rate_mae={float(epoch_result.val_metrics['future_yaw_rate_mae']):.6f}",
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
    ]
    print(" ".join(summary_parts))


def execute_training(context: TrainingContext) -> dict[str, object]:
    run_summary = build_run_summary(context)
    early_stopping_state = create_early_stopping(
        context.config.training,
        context.config.dataset.future_offsets,
        context.config.dataset.control_target_names,
        context.config.dataset.aux_target_names,
    )
    training_started_at = time.perf_counter()

    print(
        f"Using device: {context.device} with train_samples={len(context.train_dataset)} "
        f"train_trips={context.train_dataset.trip_count} "
        f"val_samples={len(context.val_dataset)} "
        f"val_trips={context.val_dataset.trip_count} "
        f"selected_val_trips={context.selected_val_trip_count}/{context.total_val_trip_count} "
        f"run_dir={context.run_dir}"
    )
    print(
        "Sequence config: "
        f"image_size=({context.config.dataset.image_width}, {context.config.dataset.image_height}) "
        f"image_offsets={list(context.config.dataset.image_offsets)} "
        f"telemetry_offsets={list(context.config.dataset.telemetry_offsets)} "
        f"future_offsets={list(context.config.dataset.future_offsets)} "
        f"telemetry_features={list(context.config.dataset.telemetry_feature_names)}"
    )
    print(
        "Model config: "
        f"width_multiplier={context.config.model.width_multiplier:.3f} "
        f"telemetry_hidden_dim={context.config.model.telemetry_hidden_dim} "
        f"sample_images_shape={context.train_image_shape} "
        f"sample_telemetry_shape={context.train_telemetry_shape} "
        f"sample_target_controls_shape={context.train_target_control_shape} "
        f"sample_target_aux_shape={context.train_target_aux_shape}"
    )
    print(
        "Loss config: "
        f"aux_loss_weight={context.config.training.aux_loss_weight:.3f} "
        f"horizon_loss_weights={list(context.config.training.horizon_loss_weights)}"
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
    execute_training(context)


if __name__ == "__main__":
    main()
