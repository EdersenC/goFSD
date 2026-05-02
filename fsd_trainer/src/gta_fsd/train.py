from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import tempfile
import time
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LambdaLR
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
    DEFAULT_LOSS_FUNCTION,
    DEFAULT_SAMPLE_STRIDE,
    DEFAULT_SMOOTH_L1_BETA,
    DEFAULT_TELEMETRY_FEATURE_NAMES,
    DEFAULT_TELEMETRY_HIDDEN_DIM,
    DEFAULT_TELEMETRY_OFFSETS,
    normalize_windows_drive_path,
    parse_temporal_dataset_config,
)
from dataset import DatasetItem, FsdDataset
from models.planner import DrivingCNN
from target_transforms import (
    DEFAULT_FUTURE_SPEED_DELTA_CLIP,
    DEFAULT_FUTURE_SPEED_DELTA_NORMALIZE,
    build_target_transform_registry,
    denormalize_target_tensor,
    TARGET_TRANSFORM_TYPE_SIGNED_CAP,
    TargetTransform,
    target_transform_metadata,
)
from state_inputs import (
    ROUTE_DIRECTION_KEEP_STRAIGHT_KEY,
    ROUTE_DIRECTION_REROUTE_WRONG_WAY_KEY,
    ROUTE_DIRECTION_TURN_LEFT_KEY,
    ROUTE_DIRECTION_TURN_RIGHT_KEY,
    ROUTE_DIRECTION_UNKNOWN_KEY,
    StateInputConfig,
    default_inference_state_input_config,
    state_input_definition,
    state_input_config_from_metadata,
    state_inputs_metadata,
)


MetricPayload = dict[str, Any]
PLANNER_FORMAT = "temporal_telemetry_gru_v1"
DEFAULT_WIDTH_MULTIPLIER = 1.0
DEFAULT_EARLY_STOPPING_METRIC = "drive_score"
DRIVE_SCORE_CONTROL_MAE_WEIGHT = 0.25
DRIVE_SCORE_GENERALIZATION_GAP_WEIGHT = 0.10
SUPPORTED_LOSS_FUNCTIONS = {DEFAULT_LOSS_FUNCTION}
SUPPORTED_OPTIMIZERS = {"adam", "adamw"}
SUPPORTED_SCHEDULERS = {"none", "cosine"}


try:
    import psutil  # type: ignore[import-not-found]
except ImportError:
    psutil = None


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
    target_transforms: dict[str, TargetTransform] = field(default_factory=dict)
    window_size: int = len(DEFAULT_IMAGE_OFFSETS)
    frame_stride: int = 2
    sample_stride: int = DEFAULT_SAMPLE_STRIDE


@dataclass(frozen=True)
class OutputConfig:
    base_dir: str


@dataclass(frozen=True)
class VisualTemporalConfig:
    enabled: bool = True
    type: str = "gru"
    hidden_dim: int = 256
    num_layers: int = 1
    bidirectional: bool = False
    dropout: float = 0.0


@dataclass(frozen=True)
class HorizonDecoderConfig:
    enabled: bool = True
    horizon_embed_dim: int = 32
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1


@dataclass(frozen=True)
class ModelConfig:
    width_multiplier: float = DEFAULT_WIDTH_MULTIPLIER
    telemetry_hidden_dim: int = DEFAULT_TELEMETRY_HIDDEN_DIM
    dropout: float = 0.1
    visual_temporal: VisualTemporalConfig = field(default_factory=VisualTemporalConfig)
    horizon_decoder: HorizonDecoderConfig = field(default_factory=HorizonDecoderConfig)


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adam"
    lr: float | None = None
    weight_decay: float = 0.0
    grad_clip_norm: float | None = None


@dataclass(frozen=True)
class SchedulerConfig:
    name: str = "none"
    warmup_fraction: float = 0.0
    min_lr_ratio: float = 0.0
    step_frequency: str = "none"


@dataclass(frozen=True)
class EmaConfig:
    enabled: bool = False
    decay: float = 0.999


@dataclass(frozen=True)
class TrainingConfig:
    device: str
    epochs: int
    learning_rate: float
    early_stopping_metric: str = DEFAULT_EARLY_STOPPING_METRIC
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0
    loss_function: str = DEFAULT_LOSS_FUNCTION
    smooth_l1_beta: float = DEFAULT_SMOOTH_L1_BETA
    aux_loss_weight: float = DEFAULT_AUX_LOSS_WEIGHT
    horizon_loss_weights: tuple[float, ...] = DEFAULT_HORIZON_LOSS_WEIGHTS
    target_loss_weights: dict[str, float] = field(default_factory=dict)
    consistency_settings: dict[str, float | bool | str] = field(default_factory=dict)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    ema: EmaConfig = field(default_factory=EmaConfig)


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
    state_inputs: StateInputConfig = field(default_factory=default_inference_state_input_config)


@dataclass(frozen=True)
class DatasetPhaseStats:
    sample_count: int
    trip_count: int
    selected_trip_count: int | None = None
    total_trip_count: int | None = None
    loader_sample_count: int | None = None


@dataclass(frozen=True)
class TrainingContext:
    config: TrainConfig
    config_path: Path
    run_dir: Path
    run_metrics_path: Path
    data_root: Path
    output_base_dir: Path
    train_stats: DatasetPhaseStats
    val_stats: DatasetPhaseStats
    model: Module
    device: torch.device
    optimizer: Optimizer
    scheduler: LRScheduler | None
    ema: ModelEma | None
    scaler: torch.amp.GradScaler
    train_steps_per_epoch: int
    total_train_steps: int
    train_image_shape: tuple[int, ...]
    train_telemetry_shape: tuple[int, ...]
    train_state_input_shape: tuple[int, ...]
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
    avg_grad_norm: float | None
    avg_learning_rate: float
    memory_snapshots: dict[str, dict[str, float | None]]


@dataclass(frozen=True)
class BatchTiming:
    loader_wait_s: float
    h2d_s: float
    forward_backward_s: float
    optimizer_s: float
    step_s: float
    grad_norm: float | None
    learning_rate: float

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


class ModelEma:
    def __init__(self, model: Module, decay: float) -> None:
        self.decay = float(decay)
        self._shadow_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        self._backup_state: dict[str, Tensor] | None = None

    @torch.no_grad()
    def update(self, model: Module) -> None:
        model_state = model.state_dict()
        for key, value in model_state.items():
            shadow = self._shadow_state[key]
            source = value.detach()
            if torch.is_floating_point(shadow):
                shadow.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                shadow.copy_(source)

    @torch.no_grad()
    def apply_to(self, model: Module) -> None:
        self._backup_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        model.load_state_dict(self._shadow_state, strict=True)

    @torch.no_grad()
    def restore(self, model: Module) -> None:
        if self._backup_state is None:
            return
        model.load_state_dict(self._backup_state, strict=True)
        self._backup_state = None

    def state_dict(self) -> dict[str, Tensor]:
        return {
            key: value.detach().clone()
            for key, value in self._shadow_state.items()
        }


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


def _parse_loss_function(training_raw: dict[str, Any]) -> str:
    loss_function = str(training_raw.get("loss_function", DEFAULT_LOSS_FUNCTION)).strip().lower()
    if loss_function not in SUPPORTED_LOSS_FUNCTIONS:
        raise ValueError(f"training.loss_function must be one of: {', '.join(sorted(SUPPORTED_LOSS_FUNCTIONS))}")
    return loss_function


def _parse_smooth_l1_beta(training_raw: dict[str, Any]) -> float:
    beta = float(training_raw.get("smooth_l1_beta", DEFAULT_SMOOTH_L1_BETA))
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("training.smooth_l1_beta must be a positive finite number")
    return beta


def _parse_target_loss_weights(
    training_raw: dict[str, Any],
    control_target_names: tuple[str, ...],
    aux_target_names: tuple[str, ...],
) -> dict[str, float]:
    raw_weights = training_raw.get("target_loss_weights")
    weight_table_name = "training.target_loss_weights"
    if raw_weights is None:
        raw_weights = training_raw.get("loss_weights", {})
        weight_table_name = "training.loss_weights"
        if raw_weights is None:
            raw_weights = {}
    elif training_raw.get("loss_weights") is not None:
        raw_weights = {**(training_raw.get("loss_weights") or {}), **raw_weights}

    if raw_weights is None:
        return {}
    if not isinstance(raw_weights, dict):
        raise ValueError(f"{weight_table_name} must be a TOML table")

    supported_names = set(control_target_names) | set(aux_target_names)
    weights: dict[str, float] = {}
    unknown: list[str] = []
    for raw_name, raw_weight in raw_weights.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError(f"{weight_table_name} contains an empty target name")

        if name not in supported_names:
            unknown.append(name)
            continue
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"{weight_table_name}.{name} must be a finite number >= 0")
        weights[name] = weight

    if unknown:
        hint = ""
        if "yaw_rate" in unknown and "future_yaw_rate" in supported_names:
            hint = "; use future_yaw_rate for the auxiliary future yaw-rate target"
        raise ValueError(
            "unknown target loss weight name(s): "
            f"{', '.join(sorted(unknown))}. Supported targets: {', '.join(sorted(supported_names))}{hint}"
        )
    return weights


def _parse_consistency_settings(training_raw: dict[str, Any]) -> dict[str, float | bool | str]:
    raw_settings = training_raw.get("consistency")
    if raw_settings is None:
        return {}
    if not isinstance(raw_settings, dict):
        raise ValueError("training.consistency must be a TOML table")

    parsed: dict[str, float | bool | str] = {}
    for key, value in raw_settings.items():
        name = str(key).strip()
        if not name:
            raise ValueError("training.consistency contains an empty key")
        if isinstance(value, bool):
            parsed[name] = bool(value)
        elif isinstance(value, (int, float)):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"training.consistency.{name} must be finite")
            parsed[name] = numeric
        elif isinstance(value, str):
            parsed[name] = str(value)
        else:
            raise ValueError(
                f"training.consistency.{name} must be bool, finite number, or string"
            )
    return parsed


def _parse_visual_temporal_config(model_raw: dict[str, Any]) -> VisualTemporalConfig:
    raw = model_raw.get("visual_temporal")
    if raw is None:
        return VisualTemporalConfig()
    if not isinstance(raw, dict):
        raise ValueError("model.visual_temporal must be a TOML table")

    encoder_type = str(raw.get("type", "gru")).strip().lower()
    if encoder_type != "gru":
        raise ValueError("model.visual_temporal.type must be 'gru'")

    hidden_dim = int(raw.get("hidden_dim", 256))
    if hidden_dim < 1:
        raise ValueError("model.visual_temporal.hidden_dim must be > 0")
    num_layers = int(raw.get("num_layers", 1))
    if num_layers < 1:
        raise ValueError("model.visual_temporal.num_layers must be > 0")
    dropout = float(raw.get("dropout", 0.0))
    if not math.isfinite(dropout) or dropout < 0.0 or dropout > 1.0:
        raise ValueError("model.visual_temporal.dropout must be a finite number in [0, 1]")

    return VisualTemporalConfig(
        enabled=bool(raw.get("enabled", True)),
        type=encoder_type,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        bidirectional=bool(raw.get("bidirectional", False)),
        dropout=dropout,
    )


def _parse_horizon_decoder_config(model_raw: dict[str, Any]) -> HorizonDecoderConfig:
    raw = model_raw.get("horizon_decoder")
    if raw is None:
        return HorizonDecoderConfig()
    if not isinstance(raw, dict):
        raise ValueError("model.horizon_decoder must be a TOML table")

    enabled = bool(raw.get("enabled", True))
    if not enabled:
        raise ValueError("model.horizon_decoder.enabled must be true for the temporal planner")
    horizon_embed_dim = int(raw.get("horizon_embed_dim", 32))
    if horizon_embed_dim < 1:
        raise ValueError("model.horizon_decoder.horizon_embed_dim must be > 0")
    hidden_dim = int(raw.get("hidden_dim", 256))
    if hidden_dim < 1:
        raise ValueError("model.horizon_decoder.hidden_dim must be > 0")
    num_layers = int(raw.get("num_layers", 2))
    if num_layers < 1:
        raise ValueError("model.horizon_decoder.num_layers must be > 0")
    dropout = float(raw.get("dropout", 0.1))
    if not math.isfinite(dropout) or dropout < 0.0 or dropout > 1.0:
        raise ValueError("model.horizon_decoder.dropout must be a finite number in [0, 1]")

    return HorizonDecoderConfig(
        enabled=enabled,
        horizon_embed_dim=horizon_embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )


def _parse_turn_oversampling_config(loader_raw: dict[str, Any]) -> TurnOversamplingConfig:
    raw = loader_raw.get("turn_oversampling")
    if raw is None:
        return TurnOversamplingConfig()
    if not isinstance(raw, dict):
        raise ValueError("loader.turn_oversampling must be a TOML table")

    config = TurnOversamplingConfig(
        enabled=bool(raw.get("enabled", False)),
        straight_weight=float(raw.get("straight_weight", 1.0)),
        light_turn_weight=float(raw.get("light_turn_weight", 1.5)),
        medium_turn_weight=float(raw.get("medium_turn_weight", 2.5)),
        sharp_turn_weight=float(raw.get("sharp_turn_weight", 4.0)),
        light_turn_threshold=float(raw.get("light_turn_threshold", 0.05)),
        medium_turn_threshold=float(raw.get("medium_turn_threshold", 0.15)),
        sharp_turn_threshold=float(raw.get("sharp_turn_threshold", 0.30)),
    )
    if config.light_turn_threshold < 0.0:
        raise ValueError("loader.turn_oversampling.light_turn_threshold must be >= 0")
    if config.medium_turn_threshold < config.light_turn_threshold:
        raise ValueError("loader.turn_oversampling.medium_turn_threshold must be >= light_turn_threshold")
    if config.sharp_turn_threshold < config.medium_turn_threshold:
        raise ValueError("loader.turn_oversampling.sharp_turn_threshold must be >= medium_turn_threshold")
    for field_name in (
        "straight_weight",
        "light_turn_weight",
        "medium_turn_weight",
        "sharp_turn_weight",
    ):
        weight = float(getattr(config, field_name))
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"loader.turn_oversampling.{field_name} must be a positive finite number")
    return config


def _parse_optimizer_config(training_raw: dict[str, Any], *, default_lr: float) -> OptimizerConfig:
    optimizer_raw = training_raw.get("optimizer")
    if optimizer_raw is None:
        return OptimizerConfig(name="adam", lr=default_lr, weight_decay=0.0, grad_clip_norm=None)
    if not isinstance(optimizer_raw, dict):
        raise ValueError("training.optimizer must be a TOML table")

    name = str(optimizer_raw.get("name", "adamw")).strip().lower()
    if name not in SUPPORTED_OPTIMIZERS:
        raise ValueError(f"training.optimizer.name must be one of: {', '.join(sorted(SUPPORTED_OPTIMIZERS))}")

    lr = float(optimizer_raw.get("lr", default_lr))
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError("training.optimizer.lr must be a positive finite number")

    weight_decay = float(optimizer_raw.get("weight_decay", 0.0001))
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("training.optimizer.weight_decay must be a finite number >= 0")

    grad_clip_raw = optimizer_raw.get("grad_clip_norm", 1.0)
    if grad_clip_raw is None:
        grad_clip_norm = None
    else:
        grad_clip_norm = float(grad_clip_raw)
        if not math.isfinite(grad_clip_norm) or grad_clip_norm <= 0.0:
            raise ValueError("training.optimizer.grad_clip_norm must be a positive finite number or null")

    return OptimizerConfig(
        name=name,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
    )


def _parse_scheduler_config(training_raw: dict[str, Any]) -> SchedulerConfig:
    scheduler_raw = training_raw.get("scheduler")
    if scheduler_raw is None:
        return SchedulerConfig()
    if not isinstance(scheduler_raw, dict):
        raise ValueError("training.scheduler must be a TOML table")

    name = str(scheduler_raw.get("name", "cosine")).strip().lower()
    if name in {"off", "disabled"}:
        name = "none"
    if name not in SUPPORTED_SCHEDULERS:
        raise ValueError(f"training.scheduler.name must be one of: {', '.join(sorted(SUPPORTED_SCHEDULERS))}")
    if name == "none":
        return SchedulerConfig(name="none", warmup_fraction=0.0, min_lr_ratio=0.0, step_frequency="none")

    warmup_fraction = float(scheduler_raw.get("warmup_fraction", 0.05))
    if not math.isfinite(warmup_fraction) or warmup_fraction < 0.0 or warmup_fraction > 1.0:
        raise ValueError("training.scheduler.warmup_fraction must be a finite number in [0, 1]")

    min_lr_ratio = float(scheduler_raw.get("min_lr_ratio", 0.05))
    if not math.isfinite(min_lr_ratio) or min_lr_ratio < 0.0 or min_lr_ratio > 1.0:
        raise ValueError("training.scheduler.min_lr_ratio must be a finite number in [0, 1]")

    return SchedulerConfig(
        name="cosine",
        warmup_fraction=warmup_fraction,
        min_lr_ratio=min_lr_ratio,
        step_frequency="per_step",
    )


def _parse_ema_config(training_raw: dict[str, Any]) -> EmaConfig:
    ema_raw = training_raw.get("ema")
    if ema_raw is None:
        return EmaConfig(enabled=False, decay=0.999)
    if not isinstance(ema_raw, dict):
        raise ValueError("training.ema must be a TOML table")

    enabled = bool(ema_raw.get("enabled", True))
    decay = float(ema_raw.get("decay", 0.999))
    if not math.isfinite(decay) or decay < 0.0 or decay >= 1.0:
        raise ValueError("training.ema.decay must be a finite number in [0, 1)")
    return EmaConfig(enabled=enabled, decay=decay)


def resolved_target_loss_weights(
    target_names: tuple[str, ...],
    target_loss_weights: dict[str, float] | None,
) -> dict[str, float]:
    overrides = target_loss_weights or {}
    return {name: float(overrides.get(name, 1.0)) for name in target_names}


def _parse_target_transforms(
    dataset_raw: dict[str, Any],
    control_target_names: tuple[str, ...],
    aux_target_names: tuple[str, ...],
) -> dict[str, TargetTransform]:
    target_names = tuple(control_target_names) + tuple(aux_target_names)
    raw_target_transforms = dataset_raw.get("target_transforms")
    if raw_target_transforms is not None and not isinstance(raw_target_transforms, dict):
        raise ValueError("dataset.target_transforms must be a TOML table")
    transforms = build_target_transform_registry(
        target_names,
        raw_target_transforms or {},
    )

    if "future_speed_delta" in target_names and "future_speed_delta" not in transforms:
        future_speed_delta = TargetTransform(
            target_name="future_speed_delta",
            transform_type=TARGET_TRANSFORM_TYPE_SIGNED_CAP,
            range_min=-float(dataset_raw.get("future_speed_delta_clip", DEFAULT_FUTURE_SPEED_DELTA_CLIP)),
            range_max=float(dataset_raw.get("future_speed_delta_clip", DEFAULT_FUTURE_SPEED_DELTA_CLIP)),
            normalize=bool(dataset_raw.get("future_speed_delta_normalize", DEFAULT_FUTURE_SPEED_DELTA_NORMALIZE)),
        )
        transforms["future_speed_delta"] = future_speed_delta

    return transforms


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
    learning_rate = float(training_raw["learning_rate"])
    optimizer_config = _parse_optimizer_config(training_raw, default_lr=learning_rate)
    scheduler_config = _parse_scheduler_config(training_raw)
    ema_config = _parse_ema_config(training_raw)
    consistency_settings = _parse_consistency_settings(training_raw)
    turn_oversampling_config = _parse_turn_oversampling_config(loader_raw)

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
                target_transforms=_parse_target_transforms(
                    dataset_raw,
                    control_target_names,
                    aux_target_names,
                ),
                window_size=len(image_offsets),
                frame_stride=int(dataset_raw.get("frame_stride", _infer_frame_stride(image_offsets))),
                sample_stride=int(dataset_raw.get("sample_stride", max(future_offsets))),
            ),
        output=OutputConfig(base_dir=normalize_windows_drive_path(str(output_raw["base_dir"]))),
        model=ModelConfig(
            width_multiplier=float(model_raw.get("width_multiplier", DEFAULT_WIDTH_MULTIPLIER)),
            telemetry_hidden_dim=int(model_raw.get("telemetry_hidden_dim", DEFAULT_TELEMETRY_HIDDEN_DIM)),
            dropout=float(model_raw.get("dropout", 0.1)),
            visual_temporal=_parse_visual_temporal_config(model_raw),
            horizon_decoder=_parse_horizon_decoder_config(model_raw),
        ),
        training=TrainingConfig(
            device=str(training_raw.get("device", "auto")).strip().lower(),
            epochs=int(training_raw["epochs"]),
            learning_rate=learning_rate,
            early_stopping_metric=str(
                training_raw.get("early_stopping_metric", DEFAULT_EARLY_STOPPING_METRIC)
            ).strip(),
            early_stopping_patience=int(training_raw.get("early_stopping_patience", 3)),
            early_stopping_min_delta=float(training_raw.get("early_stopping_min_delta", 0.0)),
            loss_function=_parse_loss_function(training_raw),
            smooth_l1_beta=_parse_smooth_l1_beta(training_raw),
            aux_loss_weight=float(training_raw.get("aux_loss_weight", DEFAULT_AUX_LOSS_WEIGHT)),
            horizon_loss_weights=_parse_horizon_loss_weights(training_raw, len(future_offsets)),
            target_loss_weights=_parse_target_loss_weights(
                training_raw,
                control_target_names,
                aux_target_names,
            ),
            consistency_settings=consistency_settings,
            optimizer=optimizer_config,
            scheduler=scheduler_config,
            ema=ema_config,
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
            turn_oversampling=turn_oversampling_config,
        ),
        state_inputs=state_input_config_from_metadata(raw.get("state_inputs")),
    )


def prepare_output_paths(base_output_dir: Path) -> tuple[Path, Path]:
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base_output_dir / f"run-{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_dir / "run_metrics.json"


def _optimizer_learning_rate(optimizer: Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def _build_checkpoint_metadata(
    *,
    config: TrainConfig,
    epoch_index: int,
    model_state_dict: Mapping[str, Tensor],
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    ema: ModelEma | None,
    checkpoint_variant: str,
    eval_model_variant: str,
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
    avg_grad_norm: float | None,
    avg_learning_rate: float,
    train_steps_per_epoch: int,
    total_train_steps: int,
) -> dict[str, Any]:
    resolved_loss_weights = {
        **resolved_target_loss_weights(config.dataset.control_target_names, config.training.target_loss_weights),
        **resolved_target_loss_weights(config.dataset.aux_target_names, config.training.target_loss_weights),
    }
    optimizer_lr = config.training.learning_rate if config.training.optimizer.lr is None else config.training.optimizer.lr
    metadata = {
        "epoch": epoch_index,
        "planner_format": PLANNER_FORMAT,
        "planner_format_version": 1,
        "checkpoint_variant": checkpoint_variant,
        "frame_window_size": config.dataset.window_size,
        "frame_stride": config.dataset.frame_stride,
        "sample_stride": config.dataset.sample_stride,
        "input_channels": 3,
        "frame_input_channels": 3,
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
        "target_transforms": target_transform_metadata(config.dataset.target_transforms),
        "state_inputs": state_inputs_metadata(config.state_inputs),
        "model": {
            "width_multiplier": config.model.width_multiplier,
            "telemetry_hidden_dim": config.model.telemetry_hidden_dim,
            "dropout": config.model.dropout,
            "visual_temporal": {
                "enabled": config.model.visual_temporal.enabled,
                "type": config.model.visual_temporal.type,
                "hidden_dim": config.model.visual_temporal.hidden_dim,
                "num_layers": config.model.visual_temporal.num_layers,
                "bidirectional": config.model.visual_temporal.bidirectional,
                "dropout": config.model.visual_temporal.dropout,
                "image_order": "oldest_to_newest",
            },
            "horizon_decoder": {
                "enabled": config.model.horizon_decoder.enabled,
                "horizon_embed_dim": config.model.horizon_decoder.horizon_embed_dim,
                "hidden_dim": config.model.horizon_decoder.hidden_dim,
                "num_layers": config.model.horizon_decoder.num_layers,
                "dropout": config.model.horizon_decoder.dropout,
            },
        },
        "training": {
            "loss_function": config.training.loss_function,
            "smooth_l1_beta": config.training.smooth_l1_beta,
            "aux_loss_weight": config.training.aux_loss_weight,
            "horizon_loss_weights": list(config.training.horizon_loss_weights),
            "target_loss_weights": resolved_loss_weights,
            "consistency": dict(config.training.consistency_settings),
            "early_stopping_metric": config.training.early_stopping_metric,
            "early_stopping_patience": config.training.early_stopping_patience,
            "early_stopping_min_delta": config.training.early_stopping_min_delta,
            "optimizer": {
                "name": config.training.optimizer.name,
                "lr": float(optimizer_lr),
                "current_lr": _optimizer_learning_rate(optimizer),
                "weight_decay": config.training.optimizer.weight_decay,
                "grad_clip_norm": config.training.optimizer.grad_clip_norm,
            },
            "scheduler": {
                "name": config.training.scheduler.name,
                "warmup_fraction": config.training.scheduler.warmup_fraction,
                "min_lr_ratio": config.training.scheduler.min_lr_ratio,
                "step_frequency": config.training.scheduler.step_frequency,
                "train_steps_per_epoch": train_steps_per_epoch,
                "total_train_steps": total_train_steps,
            },
            "ema": {
                "enabled": config.training.ema.enabled,
                "decay": config.training.ema.decay,
                "eval_model": eval_model_variant,
            },
        },
        "loader": {
            "turn_oversampling": {
                "enabled": config.loader.turn_oversampling.enabled,
                "straight_weight": config.loader.turn_oversampling.straight_weight,
                "light_turn_weight": config.loader.turn_oversampling.light_turn_weight,
                "medium_turn_weight": config.loader.turn_oversampling.medium_turn_weight,
                "sharp_turn_weight": config.loader.turn_oversampling.sharp_turn_weight,
                "light_turn_threshold": config.loader.turn_oversampling.light_turn_threshold,
                "medium_turn_threshold": config.loader.turn_oversampling.medium_turn_threshold,
                "sharp_turn_threshold": config.loader.turn_oversampling.sharp_turn_threshold,
            }
        },
        "model_state_dict": dict(model_state_dict),
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
        "avg_grad_norm": avg_grad_norm,
        "avg_learning_rate": avg_learning_rate,
    }
    if scheduler is not None:
        metadata["scheduler_state_dict"] = scheduler.state_dict()
    if ema is not None:
        metadata["ema_state_dict"] = ema.state_dict()
    return metadata


def save_epoch_artifacts(
    run_dir: Path,
    epoch_index: int,
    model: Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    ema: ModelEma | None,
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
    avg_grad_norm: float | None,
    avg_learning_rate: float,
    train_steps_per_epoch: int,
    total_train_steps: int,
) -> dict[str, Any]:
    checkpoint_path = run_dir / f"epoch-{epoch_index:03d}.pt"
    eval_model_variant = "ema" if str(val_metrics.get("eval_model", "")).strip().lower() == "ema" else "model"
    metadata = _build_checkpoint_metadata(
        config=config,
        epoch_index=epoch_index,
        model_state_dict=model.state_dict(),
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        checkpoint_variant="model",
        eval_model_variant=eval_model_variant,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        train_epoch_time=train_epoch_time,
        val_epoch_time=val_epoch_time,
        avg_batch_time=avg_batch_time,
        avg_loader_wait_time=avg_loader_wait_time,
        avg_h2d_time=avg_h2d_time,
        avg_forward_backward_time=avg_forward_backward_time,
        avg_optimizer_time=avg_optimizer_time,
        avg_iteration_time=avg_iteration_time,
        avg_grad_norm=avg_grad_norm,
        avg_learning_rate=avg_learning_rate,
        train_steps_per_epoch=train_steps_per_epoch,
        total_train_steps=total_train_steps,
    )
    torch.save(metadata, checkpoint_path)
    artifact = {
        "epoch": epoch_index,
        "checkpoint": str(checkpoint_path),
        **{
            key: value
            for key, value in metadata.items()
            if key not in {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "ema_state_dict"}
        },
    }
    if ema is not None:
        ema_checkpoint_path = run_dir / f"epoch-{epoch_index:03d}-ema.pt"
        ema_metadata = _build_checkpoint_metadata(
            config=config,
            epoch_index=epoch_index,
            model_state_dict=ema.state_dict(),
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            checkpoint_variant="ema",
            eval_model_variant=eval_model_variant,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            train_epoch_time=train_epoch_time,
            val_epoch_time=val_epoch_time,
            avg_batch_time=avg_batch_time,
            avg_loader_wait_time=avg_loader_wait_time,
            avg_h2d_time=avg_h2d_time,
            avg_forward_backward_time=avg_forward_backward_time,
            avg_optimizer_time=avg_optimizer_time,
            avg_iteration_time=avg_iteration_time,
            avg_grad_norm=avg_grad_norm,
            avg_learning_rate=avg_learning_rate,
            train_steps_per_epoch=train_steps_per_epoch,
            total_train_steps=total_train_steps,
        )
        torch.save(ema_metadata, ema_checkpoint_path)
        artifact["ema_checkpoint"] = str(ema_checkpoint_path)
    return artifact


def _effective_train_batch_size(config: TrainConfig, device: torch.device) -> int:
    batch_size = config.loader.train_batch_size if device.type == "cuda" else config.loader.cpu_batch_size
    if batch_size <= 0:
        raise ValueError("train batch size must be >= 1")
    return int(batch_size)


def _estimate_train_steps_per_epoch(sample_count: int, *, batch_size: int) -> int:
    if sample_count <= 0:
        raise ValueError("sample_count must be >= 1")
    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1")
    return max(1, math.ceil(sample_count / batch_size))


def build_optimizer(model: Module, config: OptimizerConfig, *, fallback_lr: float) -> Optimizer:
    lr = fallback_lr if config.lr is None else config.lr
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError("optimizer lr must be a positive finite number")
    if config.name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=float(config.weight_decay),
        )
    if config.name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=float(config.weight_decay),
        )
    raise ValueError(f"unsupported optimizer: {config.name}")


def build_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig,
    *,
    total_train_steps: int,
) -> LRScheduler | None:
    if config.name == "none":
        return None
    if config.name != "cosine":
        raise ValueError(f"unsupported scheduler: {config.name}")

    total_steps = max(1, int(total_train_steps))
    warmup_steps = int(total_steps * config.warmup_fraction)
    if config.warmup_fraction > 0.0 and warmup_steps <= 0:
        warmup_steps = 1

    def lr_lambda(step_index: int) -> float:
        if warmup_steps > 0 and step_index < warmup_steps:
            return float(step_index + 1) / float(warmup_steps)

        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step_index - warmup_steps) / float(max(1, total_steps - warmup_steps))
        clamped_progress = min(1.0, max(0.0, progress))
        cosine_term = 0.5 * (1.0 + math.cos(math.pi * clamped_progress))
        return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine_term

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


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
    state_input_dim: int,
) -> bool:
    if device.type != "cuda":
        return True
    try:
        image_width, image_height = image_size
        probe_images = torch.zeros((1, frame_count, 3, image_height, image_width), device=device)
        probe_telemetry = torch.zeros((1, telemetry_length, telemetry_feature_dim), device=device)
        probe_state_inputs = None if state_input_dim <= 0 else torch.zeros((1, state_input_dim), device=device)
        with torch.no_grad():
            _ = model(probe_images, probe_telemetry, probe_state_inputs)
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


def _build_configured_dataset(config: TrainConfig, run_paths: tuple[str, ...]) -> FsdDataset:
    return FsdDataset(
        run_paths=run_paths,
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
    dataset: FsdDataset | None = None,
) -> None:
    _shutdown_loader_iterator(loader_iter)
    if dataset is not None:
        dataset.close()
    del loader_iter
    del loader
    del dataset
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
        "grad_norm_sum": 0.0,
        "grad_norm_count": 0.0,
        "learning_rate_sum": 0.0,
    }


def _update_timing_totals(totals: dict[str, float], timing: BatchTiming) -> None:
    totals["loader_wait_s"] += timing.loader_wait_s
    totals["h2d_s"] += timing.h2d_s
    totals["forward_backward_s"] += timing.forward_backward_s
    totals["optimizer_s"] += timing.optimizer_s
    totals["step_s"] += timing.step_s
    totals["iteration_s"] += timing.iteration_s
    totals["learning_rate_sum"] += timing.learning_rate
    if timing.grad_norm is not None:
        totals["grad_norm_sum"] += timing.grad_norm
        totals["grad_norm_count"] += 1.0


def _mean_timing(total: float, batch_count: int) -> float:
    return 0.0 if batch_count <= 0 else total / batch_count


def _elementwise_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    loss_function: str,
    smooth_l1_beta: float,
) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target shapes must match: pred={tuple(prediction.shape)} target={tuple(target.shape)}")
    if loss_function != DEFAULT_LOSS_FUNCTION:
        raise ValueError(f"unsupported loss_function: {loss_function}")
    if not math.isfinite(float(smooth_l1_beta)) or smooth_l1_beta <= 0.0:
        raise ValueError("smooth_l1_beta must be a positive finite number")
    return F.smooth_l1_loss(
        prediction.float(),
        target.float(),
        reduction="none",
        beta=float(smooth_l1_beta),
    )


def _target_weight_tensor(
    *,
    target_names: tuple[str, ...],
    target_loss_weights: dict[str, float] | None,
    dtype: torch.dtype,
    device: torch.device,
    expected_target_count: int,
) -> Tensor:
    if len(target_names) != expected_target_count:
        raise ValueError(
            "target_names must match the final loss dimension: "
            f"names={len(target_names)} targets={expected_target_count}"
        )
    weights: list[float] = []
    overrides = target_loss_weights or {}
    for name in target_names:
        weight = float(overrides.get(name, 1.0))
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"loss weight for target '{name}' must be a finite number >= 0")
        weights.append(weight)
    return torch.as_tensor(weights, dtype=dtype, device=device)


def _weighted_elementwise_mean(
    loss_tensor: Tensor,
    horizon_weights: tuple[float, ...] | None,
    target_weights: Tensor | None = None,
) -> Tensor:
    if loss_tensor.ndim < 2:
        raise ValueError(f"loss tensor must have at least rank 2 [B, H, ...], got {tuple(loss_tensor.shape)}")
    if horizon_weights is None and target_weights is None:
        return loss_tensor.mean()

    horizon_tensor: Tensor | None = None
    if horizon_weights is not None:
        horizon_tensor = torch.as_tensor(horizon_weights, dtype=loss_tensor.dtype, device=loss_tensor.device)
        if horizon_tensor.ndim != 1 or horizon_tensor.numel() != loss_tensor.shape[1]:
            raise ValueError(
                "horizon_weights must be rank 1 and match the horizon dimension: "
                f"weights={tuple(horizon_tensor.shape)} loss={tuple(loss_tensor.shape)}"
            )
        if float(horizon_tensor.sum().item()) <= 0.0:
            raise ValueError("horizon loss weights must sum to > 0")

    trailing_count = math.prod(loss_tensor.shape[2:]) if loss_tensor.ndim > 2 else 1
    if trailing_count <= 0:
        raise ValueError("loss tensor must have at least one element per horizon")

    weighted_loss = loss_tensor
    denominator = float(loss_tensor.shape[0] * trailing_count * loss_tensor.shape[1])
    if horizon_tensor is not None:
        view_shape = [1] * loss_tensor.ndim
        view_shape[1] = int(horizon_tensor.numel())
        weighted_loss = weighted_loss * horizon_tensor.view(*view_shape)
        denominator = float(loss_tensor.shape[0] * trailing_count) * float(horizon_tensor.sum().item())

    if target_weights is not None:
        if loss_tensor.ndim < 3:
            raise ValueError("target_weights require a loss tensor with a target dimension")
        if target_weights.ndim != 1 or target_weights.numel() != loss_tensor.shape[-1]:
            raise ValueError(
                "target_weights must be rank 1 and match the final loss dimension: "
                f"weights={tuple(target_weights.shape)} loss={tuple(loss_tensor.shape)}"
            )
        target_weight_sum = float(target_weights.sum().item())
        if target_weight_sum <= 0.0:
            return loss_tensor.sum() * 0.0
        view_shape = [1] * loss_tensor.ndim
        view_shape[-1] = int(target_weights.numel())
        weighted_loss = weighted_loss * target_weights.view(*view_shape)
        target_count = int(loss_tensor.shape[-1])
        denominator = denominator * target_weight_sum / float(target_count)

    if denominator <= 0.0:
        raise ValueError("loss weights must sum to > 0")
    return weighted_loss.sum() / denominator


def _named_target_losses(
    loss_tensor: Tensor,
    target_names: tuple[str, ...],
    horizon_loss_weights: tuple[float, ...] | None,
) -> dict[str, Tensor]:
    if loss_tensor.ndim < 3 or loss_tensor.shape[-1] != len(target_names):
        raise ValueError(
            "target loss tensor must have final dimension matching target_names: "
            f"loss={tuple(loss_tensor.shape)} target_names={len(target_names)}"
        )
    return {
        f"{name}_loss": _weighted_elementwise_mean(loss_tensor[..., index], horizon_loss_weights)
        for index, name in enumerate(target_names)
    }


def compute_planner_losses(
    pred_controls: Tensor,
    target_controls: Tensor,
    pred_aux: Tensor,
    target_aux: Tensor,
    *,
    aux_loss_weight: float,
    horizon_loss_weights: tuple[float, ...] | None,
    target_loss_weights: dict[str, float] | None = None,
    control_target_names: tuple[str, ...] = DEFAULT_CONTROL_TARGET_NAMES,
    aux_target_names: tuple[str, ...] = DEFAULT_AUX_TARGET_NAMES,
    loss_function: str = DEFAULT_LOSS_FUNCTION,
    smooth_l1_beta: float = DEFAULT_SMOOTH_L1_BETA,
) -> dict[str, Tensor]:
    control_element_losses = _elementwise_loss(
        pred_controls,
        target_controls,
        loss_function=loss_function,
        smooth_l1_beta=smooth_l1_beta,
    )
    aux_element_losses = _elementwise_loss(
        pred_aux,
        target_aux,
        loss_function=loss_function,
        smooth_l1_beta=smooth_l1_beta,
    )
    control_target_weights = _target_weight_tensor(
        target_names=control_target_names,
        target_loss_weights=target_loss_weights,
        dtype=control_element_losses.dtype,
        device=control_element_losses.device,
        expected_target_count=control_element_losses.shape[-1],
    )
    aux_target_weights = _target_weight_tensor(
        target_names=aux_target_names,
        target_loss_weights=target_loss_weights,
        dtype=aux_element_losses.dtype,
        device=aux_element_losses.device,
        expected_target_count=aux_element_losses.shape[-1],
    )
    control_loss = _weighted_elementwise_mean(
        control_element_losses,
        horizon_loss_weights,
        control_target_weights,
    )
    aux_loss = _weighted_elementwise_mean(
        aux_element_losses,
        horizon_loss_weights,
        aux_target_weights,
    )
    total_loss = control_loss + (aux_loss_weight * aux_loss)
    losses = {
        "loss": total_loss,
        "control_loss": control_loss,
        "aux_loss": aux_loss,
    }
    losses.update(_named_target_losses(control_element_losses, control_target_names, horizon_loss_weights))
    losses.update(_named_target_losses(aux_element_losses, aux_target_names, horizon_loss_weights))
    return losses


def initialize_metric_totals(
    horizon: int,
    target_transforms: Mapping[str, Any] | None = None,
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
        totals[f"{name}_loss_sum"] = 0.0
        totals[f"{name}_denorm_abs_sum"] = 0.0
    for name in aux_target_names:
        totals[f"{name}_abs_sum"] = 0.0
        totals[f"{name}_count"] = 0
        totals[f"{name}_loss_sum"] = 0.0
        totals[f"{name}_denorm_abs_sum"] = 0.0
    return totals


def update_metric_totals(
    totals: dict[str, Any],
    *,
    pred_controls: Tensor,
    target_controls: Tensor,
    pred_aux: Tensor,
    target_aux: Tensor,
    target_transforms: Mapping[str, Any] | None = None,
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
        totals[f"{name}_loss_sum"] += float(losses[f"{name}_loss"].item())
        if target_transforms:
            denorm_error = (denormalize_target_tensor(
                pred_controls[:, :, index].unsqueeze(-1),
                (name,),
                target_transforms,
            ) - denormalize_target_tensor(
                target_controls[:, :, index].unsqueeze(-1),
                (name,),
                target_transforms,
            )).abs()
            totals[f"{name}_denorm_abs_sum"] += float(denorm_error.sum().item())
    for index, name in enumerate(aux_target_names):
        totals[f"{name}_abs_sum"] += float(aux_abs[..., index].sum().item())
        totals[f"{name}_count"] += int(aux_abs[..., index].numel())
        totals[f"{name}_loss_sum"] += float(losses[f"{name}_loss"].item())
        if target_transforms:
            denorm_error = (denormalize_target_tensor(
                pred_aux[:, :, index].unsqueeze(-1),
                (name,),
                target_transforms,
            ) - denormalize_target_tensor(
                target_aux[:, :, index].unsqueeze(-1),
                (name,),
                target_transforms,
            )).abs()
            totals[f"{name}_denorm_abs_sum"] += float(denorm_error.sum().item())

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
    target_transforms: Mapping[str, Any] | None = None,
    *,
    control_target_names: tuple[str, ...] = DEFAULT_CONTROL_TARGET_NAMES,
    aux_target_names: tuple[str, ...] = DEFAULT_AUX_TARGET_NAMES,
) -> MetricPayload:
    def _group_mean(metric_names: tuple[str, ...], suffix: str) -> float | None:
        values = [metrics[f"{name}{suffix}"] for name in metric_names if f"{name}{suffix}" in metrics]
        if not values:
            return None
        return float(sum(float(value) for value in values) / len(values))

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
        metrics[f"{name}_mae_denorm"] = _mean_or_zero(
            float(totals[f"{name}_denorm_abs_sum"]),
            int(totals[f"{name}_count"]),
        ) if target_transforms is not None else _mean_or_zero(
            float(totals[f"{name}_abs_sum"]),
            int(totals[f"{name}_count"]),
        )
        metrics[f"{name}_loss"] = _mean_or_zero(float(totals[f"{name}_loss_sum"]), batch_count)
    for name in aux_target_names:
        metrics[f"{name}_mae"] = _mean_or_zero(float(totals[f"{name}_abs_sum"]), int(totals[f"{name}_count"]))
        metrics[f"{name}_mae_denorm"] = _mean_or_zero(
            float(totals[f"{name}_denorm_abs_sum"]),
            int(totals[f"{name}_count"]),
        ) if target_transforms is not None else _mean_or_zero(
            float(totals[f"{name}_abs_sum"]),
            int(totals[f"{name}_count"]),
        )
        metrics[f"{name}_loss"] = _mean_or_zero(float(totals[f"{name}_loss_sum"]), batch_count)
    for index, offset in enumerate(future_offsets):
        metrics[f"control_mae_t+{offset}"] = _mean_or_zero(
            float(totals["control_horizon_abs_sum"][index]),
            int(totals["control_horizon_count"][index]),
        )
        metrics[f"aux_mae_t+{offset}"] = _mean_or_zero(
            float(totals["aux_horizon_abs_sum"][index]),
            int(totals["aux_horizon_count"][index]),
        )

    if target_transforms is None:
        metrics["control_mae_overall_denorm"] = float(metrics["control_mae_overall"])
        metrics["control_overall_mae_denorm"] = float(metrics["control_overall_mae"])
        metrics["aux_mae_overall_denorm"] = float(metrics["aux_mae_overall"])
        metrics["aux_overall_mae_denorm"] = float(metrics["aux_overall_mae"])
    else:
        control_denorm_abs_sum = sum(float(totals[f"{name}_denorm_abs_sum"]) for name in control_target_names)
        control_denorm_count = sum(int(totals[f"{name}_count"]) for name in control_target_names)
        aux_denorm_abs_sum = sum(float(totals[f"{name}_denorm_abs_sum"]) for name in aux_target_names)
        aux_denorm_count = sum(int(totals[f"{name}_count"]) for name in aux_target_names)
        metrics["control_mae_overall_denorm"] = _mean_or_zero(control_denorm_abs_sum, control_denorm_count)
        metrics["control_overall_mae_denorm"] = float(metrics["control_mae_overall_denorm"])
        metrics["aux_mae_overall_denorm"] = _mean_or_zero(aux_denorm_abs_sum, aux_denorm_count)
        metrics["aux_overall_mae_denorm"] = float(metrics["aux_mae_overall_denorm"])

    longitudinal_targets = tuple(
        name for name in aux_target_names if name in {"future_speed", "future_speed_delta"}
    )
    if longitudinal_targets:
        longitudinal_norm = _group_mean(longitudinal_targets, "_mae")
        if longitudinal_norm is not None:
            metrics["longitudinal_aux_mae"] = longitudinal_norm
        longitudinal_denorm = _group_mean(longitudinal_targets, "_mae_denorm")
        if longitudinal_denorm is not None:
            metrics["longitudinal_aux_mae_denorm"] = longitudinal_denorm

    lateral_targets = tuple(
        name for name in aux_target_names if name in {"future_yaw_delta", "future_yaw_rate"}
    )
    if lateral_targets:
        lateral_norm = _group_mean(lateral_targets, "_mae")
        if lateral_norm is not None:
            metrics["lateral_aux_mae"] = lateral_norm
        lateral_denorm = _group_mean(lateral_targets, "_mae_denorm")
        if lateral_denorm is not None:
            metrics["lateral_aux_mae_denorm"] = lateral_denorm
    return metrics


def compute_drive_score(
    train_metrics: MetricPayload,
    val_metrics: MetricPayload,
    *,
    control_mae_weight: float = DRIVE_SCORE_CONTROL_MAE_WEIGHT,
    generalization_gap_weight: float = DRIVE_SCORE_GENERALIZATION_GAP_WEIGHT,
) -> float:
    val_control_loss = float(val_metrics["control_loss"])
    val_control_mae = float(val_metrics["control_mae_overall"])
    train_control_mae = float(train_metrics["control_mae_overall"])
    generalization_gap = max(0.0, val_control_mae - train_control_mae)
    return val_control_loss + (control_mae_weight * val_control_mae) + (generalization_gap_weight * generalization_gap)


def attach_drive_score(train_metrics: MetricPayload, val_metrics: MetricPayload) -> float:
    score = compute_drive_score(train_metrics, val_metrics)
    val_metrics["drive_score"] = score
    val_metrics["control_mae_generalization_gap"] = max(
        0.0,
        float(val_metrics["control_mae_overall"]) - float(train_metrics["control_mae_overall"]),
    )
    return score


def format_first_batch_debug(
    images: Tensor,
    telemetry: Tensor,
    state_inputs: Tensor,
    target_controls: Tensor,
    target_aux: Tensor,
) -> str:
    return (
        "first_train_batch "
        f"images_shape={tuple(images.shape)} "
        f"telemetry_shape={tuple(telemetry.shape)} "
        f"state_inputs_shape={tuple(state_inputs.shape)} "
        f"target_controls_shape={tuple(target_controls.shape)} "
        f"target_aux_shape={tuple(target_aux.shape)} "
        f"dtype={images.dtype}"
    )


def _format_state_input_values(
    state_inputs: Tensor,
    state_input_names: tuple[str, ...],
) -> list[str]:
    if state_inputs.ndim < 2 or not state_input_names:
        return []
    sample0 = state_inputs[0].detach().float().cpu().tolist()
    pairs: list[str] = []
    for index, key in enumerate(state_input_names):
        if index >= len(sample0):
            break
        definition = state_input_definition(key)
        pairs.append(f"{definition.camel_key}={round(float(sample0[index]), 4)}")
    return pairs


def _navigation_choice_from_state_inputs(
    state_inputs: Tensor,
    state_input_names: tuple[str, ...],
) -> str:
    if state_inputs.ndim < 2 or not state_input_names:
        return "unavailable"
    sample0 = state_inputs[0].detach().float().cpu().tolist()
    value_by_key = {
        key: float(sample0[index])
        for index, key in enumerate(state_input_names)
        if index < len(sample0)
    }
    nav_candidates = (
        (ROUTE_DIRECTION_UNKNOWN_KEY, "unknown"),
        (ROUTE_DIRECTION_KEEP_STRAIGHT_KEY, "keep_straight"),
        (ROUTE_DIRECTION_TURN_LEFT_KEY, "turn_left"),
        (ROUTE_DIRECTION_TURN_RIGHT_KEY, "turn_right"),
        (ROUTE_DIRECTION_REROUTE_WRONG_WAY_KEY, "reroute_wrong_way"),
    )
    available = [
        (label, value_by_key[key])
        for key, label in nav_candidates
        if key in value_by_key
    ]
    if not available:
        return "unavailable"
    best_label, best_value = max(available, key=lambda item: item[1])
    if best_value <= 0.0:
        return "inactive"
    active = [label for label, value in available if value >= 0.5]
    if len(active) > 1:
        return "|".join(active)
    return best_label


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
    state_inputs: Tensor,
    pred_controls: Tensor,
    target_controls: Tensor,
    pred_aux: Tensor,
    target_aux: Tensor,
    *,
    state_input_names: tuple[str, ...],
    control_target_names: tuple[str, ...],
    aux_target_names: tuple[str, ...],
    future_offsets: tuple[int, ...],
    batch_index: int | None = None,
) -> str:
    header = "train_batch_predictions sample=0"
    if batch_index is not None:
        header += f" batch={batch_index}"
    state_input_pairs = _format_state_input_values(state_inputs, state_input_names)
    navigation_choice = _navigation_choice_from_state_inputs(state_inputs, state_input_names)
    return "\n".join((
        header,
        f"navigation_choice: {navigation_choice}",
        "state_inputs_sample0: " + (" ".join(state_input_pairs) if state_input_pairs else "<none>"),
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
        elif key.endswith("_loss_sum"):
            if key in target and key not in {"loss_sum", "control_loss_sum", "aux_loss_sum"}:
                target[key] += value

    for key in ("control_horizon_abs_sum", "control_horizon_count", "aux_horizon_abs_sum", "aux_horizon_count"):
        for index, value in enumerate(source[key]):
            target[key][index] += value


def train_batch(
    images: Tensor,
    telemetry: Tensor,
    state_inputs: Tensor,
    target_controls: Tensor,
    target_aux: Tensor,
    model: Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    ema: ModelEma | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    grad_clip_norm: float | None,
    aux_loss_weight: float,
    horizon_loss_weights: tuple[float, ...],
    target_loss_weights: dict[str, float],
    loss_function: str,
    smooth_l1_beta: float,
    future_offsets: tuple[int, ...],
    target_transforms: Mapping[str, Any] | None,
    state_input_names: tuple[str, ...],
    control_target_names: tuple[str, ...],
    aux_target_names: tuple[str, ...],
    batch_index: int | None = None,
    capture_prediction_debug: bool = False,
) -> tuple[float, BatchTiming, dict[str, Any], str | None]:
    step_start = time.perf_counter()
    transfer_start = time.perf_counter()
    images = images.to(device, non_blocking=device.type == "cuda")
    telemetry = telemetry.to(device, non_blocking=device.type == "cuda")
    state_inputs = state_inputs.to(device, non_blocking=device.type == "cuda")
    target_controls = target_controls.to(device, non_blocking=device.type == "cuda")
    target_aux = target_aux.to(device, non_blocking=device.type == "cuda")
    h2d_time = time.perf_counter() - transfer_start

    optimizer.zero_grad(set_to_none=True)
    batch_totals = initialize_metric_totals(
        target_controls.shape[1],
        target_transforms=target_transforms,
        control_target_names=control_target_names,
        aux_target_names=aux_target_names,
    )

    forward_backward_start = time.perf_counter()
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        output = model(images, telemetry, state_inputs)
        losses = compute_planner_losses(
            output["pred_controls"],
            target_controls,
            output["pred_aux"],
            target_aux,
            aux_loss_weight=aux_loss_weight,
            horizon_loss_weights=horizon_loss_weights,
            target_loss_weights=target_loss_weights,
            control_target_names=control_target_names,
            aux_target_names=aux_target_names,
            loss_function=loss_function,
            smooth_l1_beta=smooth_l1_beta,
        )
    scaler.scale(losses["loss"]).backward()
    forward_backward_time = time.perf_counter() - forward_backward_start

    grad_norm: float | None = None
    if grad_clip_norm is not None:
        scaler.unscale_(optimizer)
        grad_norm = float(clip_grad_norm_(model.parameters(), grad_clip_norm))

    optimizer_start = time.perf_counter()
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    step_skipped = scale_after < scale_before
    # Warmup+cosine scheduling is intentionally per optimizer update (once per batch step).
    if scheduler is not None and not step_skipped:
        scheduler.step()
    if ema is not None and not step_skipped:
        ema.update(model)
    optimizer_time = time.perf_counter() - optimizer_start

    update_metric_totals(
        batch_totals,
        pred_controls=output["pred_controls"].detach(),
        target_controls=target_controls.detach(),
        pred_aux=output["pred_aux"].detach(),
        target_aux=target_aux.detach(),
        losses=losses,
        target_transforms=target_transforms,
        control_target_names=control_target_names,
        aux_target_names=aux_target_names,
    )
    prediction_debug = None
    if capture_prediction_debug:
        prediction_debug = format_first_batch_predictions(
            state_inputs,
            output["pred_controls"],
            target_controls,
            output["pred_aux"],
            target_aux,
            state_input_names=state_input_names,
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
        grad_norm=grad_norm,
        learning_rate=_optimizer_learning_rate(optimizer),
    )
    return float(losses["loss"].item()), timing, batch_totals, prediction_debug


def format_batch_timing(timing: BatchTiming) -> str:
    parts = [
        f"wait_s={timing.loader_wait_s:.3f} "
        f"batch_s={timing.step_s:.3f} "
        f"h2d_s={timing.h2d_s:.3f} "
        f"fwd_bwd_s={timing.forward_backward_s:.3f} "
        f"opt_s={timing.optimizer_s:.3f} "
        f"lr={timing.learning_rate:.7f}"
    ]
    if timing.grad_norm is not None:
        parts.append(f"grad_norm={timing.grad_norm:.4f}")
    return " ".join(parts)


def train_epoch(
    loader: DataLoader[DatasetItem],
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    ema: ModelEma | None,
    model: Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    grad_clip_norm: float | None,
    future_offsets: tuple[int, ...],
    state_input_names: tuple[str, ...],
    control_target_names: tuple[str, ...],
    aux_target_names: tuple[str, ...],
    aux_loss_weight: float,
    horizon_loss_weights: tuple[float, ...],
    target_loss_weights: dict[str, float],
    loss_function: str,
    smooth_l1_beta: float,
    log_every_n_batches: int,
    target_transforms: Mapping[str, Any] | None,
) -> tuple[MetricPayload, float, dict[str, float | None]]:
    epoch_start = time.perf_counter()
    model.train()
    totals = initialize_metric_totals(
        len(future_offsets),
        target_transforms=target_transforms,
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
            images, telemetry, state_inputs, target_controls, target_aux = next(loader_iter)
            loader_wait_time = time.perf_counter() - wait_start
            if batch_index == 1:
                print(format_first_batch_debug(images, telemetry, state_inputs, target_controls, target_aux))
            batch_loss, batch_timing, batch_totals, prediction_debug = train_batch(
                images,
                telemetry,
                state_inputs,
                target_controls,
                target_aux,
                model,
                optimizer,
                scheduler,
                ema,
                scaler,
                device,
                grad_clip_norm=grad_clip_norm,
                aux_loss_weight=aux_loss_weight,
                horizon_loss_weights=horizon_loss_weights,
                target_loss_weights=target_loss_weights,
                loss_function=loss_function,
                smooth_l1_beta=smooth_l1_beta,
                target_transforms=target_transforms,
                future_offsets=future_offsets,
                state_input_names=state_input_names,
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
                grad_norm=batch_timing.grad_norm,
                learning_rate=batch_timing.learning_rate,
            )
            _update_timing_totals(timing_totals, batch_timing)
            _merge_metric_totals(totals, batch_totals)
            if batch_index == 1 or batch_index == total_batches or batch_index % log_every_n_batches == 0:
                batch_metrics = finalize_metrics(
                    batch_totals,
                    future_offsets,
                    target_transforms=target_transforms,
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
        target_transforms=target_transforms,
        control_target_names=control_target_names,
        aux_target_names=aux_target_names,
    )
    batch_count = int(totals["batch_count"])
    avg_timings = {
        "loader_wait_s": _mean_timing(timing_totals["loader_wait_s"], batch_count),
        "h2d_s": _mean_timing(timing_totals["h2d_s"], batch_count),
        "forward_backward_s": _mean_timing(timing_totals["forward_backward_s"], batch_count),
        "optimizer_s": _mean_timing(timing_totals["optimizer_s"], batch_count),
        "step_s": _mean_timing(timing_totals["step_s"], batch_count),
        "iteration_s": _mean_timing(timing_totals["iteration_s"], batch_count),
        "grad_norm": (
            None
            if timing_totals["grad_norm_count"] <= 0.0
            else timing_totals["grad_norm_sum"] / timing_totals["grad_norm_count"]
        ),
        "learning_rate": _mean_timing(timing_totals["learning_rate_sum"], batch_count),
    }
    return metrics, epoch_time, avg_timings


def evaluate_epoch(
    loader: DataLoader[DatasetItem],
    model: Module,
    device: torch.device,
    *,
    future_offsets: tuple[int, ...],
    control_target_names: tuple[str, ...],
    aux_target_names: tuple[str, ...],
    target_transforms: Mapping[str, Any] | None,
    aux_loss_weight: float,
    horizon_loss_weights: tuple[float, ...],
    target_loss_weights: dict[str, float],
    loss_function: str,
    smooth_l1_beta: float,
) -> tuple[MetricPayload, float]:
    eval_start = time.perf_counter()
    model.eval()
    totals = initialize_metric_totals(
        len(future_offsets),
        target_transforms=target_transforms,
        control_target_names=control_target_names,
        aux_target_names=aux_target_names,
    )

    loader_iter: Iterator[DatasetItem] | None = None
    try:
        loader_iter = iter(loader)
        with torch.inference_mode():
            for images, telemetry, state_inputs, target_controls, target_aux in loader_iter:
                images = images.to(device, non_blocking=device.type == "cuda")
                telemetry = telemetry.to(device, non_blocking=device.type == "cuda")
                state_inputs = state_inputs.to(device, non_blocking=device.type == "cuda")
                target_controls = target_controls.to(device, non_blocking=device.type == "cuda")
                target_aux = target_aux.to(device, non_blocking=device.type == "cuda")
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    output = model(images, telemetry, state_inputs)
                    losses = compute_planner_losses(
                        output["pred_controls"],
                        target_controls,
                        output["pred_aux"],
                        target_aux,
                        aux_loss_weight=aux_loss_weight,
                        horizon_loss_weights=horizon_loss_weights,
                        target_loss_weights=target_loss_weights,
                        control_target_names=control_target_names,
                        aux_target_names=aux_target_names,
                        loss_function=loss_function,
                        smooth_l1_beta=smooth_l1_beta,
                    )
                update_metric_totals(
                    totals,
                    pred_controls=output["pred_controls"],
                    target_controls=target_controls,
                    pred_aux=output["pred_aux"],
                    target_aux=target_aux,
                    losses=losses,
                    target_transforms=target_transforms,
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
        target_transforms=target_transforms,
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
        "output_base_dir": str(context.output_base_dir),
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
        "target_transforms": target_transform_metadata(context.config.dataset.target_transforms),
        "model": {
            "width_multiplier": context.config.model.width_multiplier,
            "telemetry_hidden_dim": context.config.model.telemetry_hidden_dim,
            "dropout": context.config.model.dropout,
            "visual_temporal": {
                "enabled": context.config.model.visual_temporal.enabled,
                "type": context.config.model.visual_temporal.type,
                "hidden_dim": context.config.model.visual_temporal.hidden_dim,
                "num_layers": context.config.model.visual_temporal.num_layers,
                "bidirectional": context.config.model.visual_temporal.bidirectional,
                "dropout": context.config.model.visual_temporal.dropout,
                "image_order": "oldest_to_newest",
            },
            "horizon_decoder": {
                "enabled": context.config.model.horizon_decoder.enabled,
                "horizon_embed_dim": context.config.model.horizon_decoder.horizon_embed_dim,
                "hidden_dim": context.config.model.horizon_decoder.hidden_dim,
                "num_layers": context.config.model.horizon_decoder.num_layers,
                "dropout": context.config.model.horizon_decoder.dropout,
            },
        },
        "training": {
            "epochs_total": context.config.training.epochs,
            "learning_rate": context.config.training.learning_rate,
            "loss_function": context.config.training.loss_function,
            "smooth_l1_beta": context.config.training.smooth_l1_beta,
            "aux_loss_weight": context.config.training.aux_loss_weight,
            "horizon_loss_weights": list(context.config.training.horizon_loss_weights),
            "target_loss_weights": {
                **resolved_target_loss_weights(
                    context.config.dataset.control_target_names,
                    context.config.training.target_loss_weights,
                ),
                **resolved_target_loss_weights(
                    context.config.dataset.aux_target_names,
                    context.config.training.target_loss_weights,
                ),
            },
            "consistency": dict(context.config.training.consistency_settings),
            "early_stopping_metric": context.config.training.early_stopping_metric,
            "early_stopping_patience": context.config.training.early_stopping_patience,
            "early_stopping_min_delta": context.config.training.early_stopping_min_delta,
            "optimizer": {
                "name": context.config.training.optimizer.name,
                "lr": context.config.training.learning_rate
                if context.config.training.optimizer.lr is None
                else context.config.training.optimizer.lr,
                "weight_decay": context.config.training.optimizer.weight_decay,
                "grad_clip_norm": context.config.training.optimizer.grad_clip_norm,
            },
            "scheduler": {
                "name": context.config.training.scheduler.name,
                "warmup_fraction": context.config.training.scheduler.warmup_fraction,
                "min_lr_ratio": context.config.training.scheduler.min_lr_ratio,
                "step_frequency": context.config.training.scheduler.step_frequency,
                "train_steps_per_epoch": context.train_steps_per_epoch,
                "total_train_steps": context.total_train_steps,
            },
            "ema": {
                "enabled": context.config.training.ema.enabled,
                "decay": context.config.training.ema.decay,
            },
        },
        "device": str(context.device),
        "dataset_samples": context.train_stats.sample_count,
        "dataset_trips": context.train_stats.trip_count,
        "validation_dataset_samples": context.val_stats.sample_count,
        "validation_dataset_trips": context.val_stats.trip_count,
        "validation_selected_trips": context.val_stats.selected_trip_count,
        "train_image_shape": list(context.train_image_shape),
        "train_telemetry_shape": list(context.train_telemetry_shape),
        "train_state_input_shape": list(context.train_state_input_shape),
        "train_target_control_shape": list(context.train_target_control_shape),
        "train_target_aux_shape": list(context.train_target_aux_shape),
        "train_loader_samples": context.train_stats.loader_sample_count,
        "validation_loader_samples": context.val_stats.loader_sample_count,
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
            "turn_oversampling": {
                "enabled": context.config.loader.turn_oversampling.enabled,
                "straight_weight": context.config.loader.turn_oversampling.straight_weight,
                "light_turn_weight": context.config.loader.turn_oversampling.light_turn_weight,
                "medium_turn_weight": context.config.loader.turn_oversampling.medium_turn_weight,
                "sharp_turn_weight": context.config.loader.turn_oversampling.sharp_turn_weight,
                "light_turn_threshold": context.config.loader.turn_oversampling.light_turn_threshold,
                "medium_turn_threshold": context.config.loader.turn_oversampling.medium_turn_threshold,
                "sharp_turn_threshold": context.config.loader.turn_oversampling.sharp_turn_threshold,
            },
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
    aux_dim: int,
    state_input_dim: int,
    width_multiplier: float,
    dropout: float,
    visual_temporal: VisualTemporalConfig,
    horizon_decoder: HorizonDecoderConfig,
    state_input_config: StateInputConfig,
) -> tuple[Module, torch.device]:
    model = DrivingCNN(
        frame_count=frame_count,
        telemetry_feature_dim=telemetry_feature_dim,
        telemetry_hidden_dim=telemetry_hidden_dim,
        telemetry_sequence_length=telemetry_length,
        horizon=horizon,
        control_dim=control_dim,
        aux_dim=aux_dim,
        state_input_dim=state_input_dim,
        width_multiplier=width_multiplier,
        dropout=dropout,
        visual_temporal_enabled=visual_temporal.enabled,
        visual_temporal_type=visual_temporal.type,
        visual_temporal_hidden_dim=visual_temporal.hidden_dim,
        visual_temporal_num_layers=visual_temporal.num_layers,
        visual_temporal_bidirectional=visual_temporal.bidirectional,
        visual_temporal_dropout=visual_temporal.dropout,
        horizon_decoder_enabled=horizon_decoder.enabled,
        horizon_embed_dim=horizon_decoder.horizon_embed_dim,
        horizon_decoder_hidden_dim=horizon_decoder.hidden_dim,
        horizon_decoder_num_layers=horizon_decoder.num_layers,
        horizon_decoder_dropout=horizon_decoder.dropout,
    ).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.state_input_config = state_input_config
    if probe_device(
        model,
        device,
        image_size=image_size,
        frame_count=frame_count,
        telemetry_length=telemetry_length,
        telemetry_feature_dim=telemetry_feature_dim,
        state_input_dim=state_input_dim,
    ):
        return model, device

    print(
        "CUDA is available but this PyTorch build cannot run on your GPU "
        "(likely missing sm_120 support). Falling back to CPU."
    )
    fallback_device = torch.device("cpu")
    fallback_model = DrivingCNN(
        frame_count=frame_count,
        telemetry_feature_dim=telemetry_feature_dim,
        telemetry_hidden_dim=telemetry_hidden_dim,
        telemetry_sequence_length=telemetry_length,
        horizon=horizon,
        control_dim=control_dim,
        aux_dim=aux_dim,
        state_input_dim=state_input_dim,
        width_multiplier=width_multiplier,
        dropout=dropout,
        visual_temporal_enabled=visual_temporal.enabled,
        visual_temporal_type=visual_temporal.type,
        visual_temporal_hidden_dim=visual_temporal.hidden_dim,
        visual_temporal_num_layers=visual_temporal.num_layers,
        visual_temporal_bidirectional=visual_temporal.bidirectional,
        visual_temporal_dropout=visual_temporal.dropout,
        horizon_decoder_enabled=horizon_decoder.enabled,
        horizon_embed_dim=horizon_decoder.horizon_embed_dim,
        horizon_decoder_hidden_dim=horizon_decoder.hidden_dim,
        horizon_decoder_num_layers=horizon_decoder.num_layers,
        horizon_decoder_dropout=horizon_decoder.dropout,
    ).to(fallback_device)
    fallback_model.state_input_config = state_input_config
    return fallback_model, fallback_device


def build_training_context(config: TrainConfig, config_path: Path) -> TrainingContext:
    data_root = Path(config.dataset.data_root)
    requested_device = select_device(config.training.device)
    train_dataset = _build_configured_dataset(config, config.dataset.train_run_paths)
    try:
        if len(train_dataset) <= 0:
            raise ValueError(
                "Training dataset resolved to zero samples. "
                f"train_run_ids={list(config.dataset.train_run_ids)} "
                f"train_trips={train_dataset.trip_count} "
                f"{train_dataset.format_rejected_sample_summary()}"
            )

        train_images, train_telemetry, train_state_inputs, train_target_controls, train_target_aux = train_dataset[0]
        train_stats = DatasetPhaseStats(
            sample_count=len(train_dataset),
            trip_count=train_dataset.trip_count,
            loader_sample_count=len(train_dataset),
        )
        train_image_shape = tuple(train_images.shape)
        train_telemetry_shape = tuple(train_telemetry.shape)
        train_state_input_shape = tuple(train_state_inputs.shape)
        train_target_control_shape = tuple(train_target_controls.shape)
        train_target_aux_shape = tuple(train_target_aux.shape)
        state_input_dim = int(train_state_inputs.shape[-1]) if train_state_inputs.ndim > 0 else 0
    finally:
        _release_phase_resources(requested_device, dataset=train_dataset)
        del train_dataset

    model, device = prepare_model(
        requested_device,
        image_size=(config.dataset.image_width, config.dataset.image_height),
        frame_count=len(config.dataset.image_offsets),
        telemetry_length=len(config.dataset.telemetry_offsets),
        telemetry_feature_dim=len(config.dataset.telemetry_feature_names),
        telemetry_hidden_dim=config.model.telemetry_hidden_dim,
        horizon=len(config.dataset.future_offsets),
        control_dim=len(config.dataset.control_target_names),
        aux_dim=len(config.dataset.aux_target_names),
        state_input_dim=state_input_dim,
        width_multiplier=config.model.width_multiplier,
        dropout=config.model.dropout,
        visual_temporal=config.model.visual_temporal,
        horizon_decoder=config.model.horizon_decoder,
        state_input_config=config.state_inputs,
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    val_dataset = _build_configured_dataset(config, config.dataset.val_run_paths)
    try:
        if len(val_dataset) <= 0:
            raise ValueError(
                "Validation dataset resolved to zero samples before subset selection. "
                f"val_run_ids={list(config.dataset.val_run_ids)} "
                f"val_trips={val_dataset.trip_count}. "
                "Check processed validation runs for empty dataset.jsonl files. "
                f"{val_dataset.format_rejected_sample_summary()}"
            )

        val_subset, selected_val_trip_count, total_val_trip_count = build_validation_subset(
            val_dataset,
            config.loader.val_split,
        )
        val_loader_sample_count = _loader_dataset_len(val_subset)
        if val_loader_sample_count <= 0:
            raise ValueError(
                "Validation subset resolved to zero usable samples. "
                f"val_run_ids={list(config.dataset.val_run_ids)} "
                f"validation_dataset_trips={val_dataset.trip_count} "
                f"validation_dataset_samples={len(val_dataset)} "
                f"selected_val_trips={selected_val_trip_count}/{total_val_trip_count}. "
                "Check processed validation runs for empty dataset.jsonl files or missing temporal telemetry windows."
            )
        val_stats = DatasetPhaseStats(
            sample_count=len(val_dataset),
            trip_count=val_dataset.trip_count,
            selected_trip_count=selected_val_trip_count,
            total_trip_count=total_val_trip_count,
            loader_sample_count=val_loader_sample_count,
        )
    finally:
        _release_phase_resources(device, dataset=val_dataset)
        del val_dataset

    output_base_dir = Path(config.output.base_dir)
    run_dir, run_metrics_path = prepare_output_paths(output_base_dir)
    train_batch_size = _effective_train_batch_size(config, device)
    train_steps_per_epoch = _estimate_train_steps_per_epoch(
        train_stats.sample_count,
        batch_size=train_batch_size,
    )
    total_train_steps = max(1, train_steps_per_epoch * max(1, config.training.epochs))
    optimizer = build_optimizer(
        model,
        config.training.optimizer,
        fallback_lr=config.training.learning_rate,
    )
    scheduler = build_scheduler(
        optimizer,
        config.training.scheduler,
        total_train_steps=total_train_steps,
    )
    ema = ModelEma(model, config.training.ema.decay) if config.training.ema.enabled else None
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    return TrainingContext(
        config=config,
        config_path=config_path,
        run_dir=run_dir,
        run_metrics_path=run_metrics_path,
        data_root=data_root,
        output_base_dir=output_base_dir,
        train_stats=train_stats,
        val_stats=val_stats,
        model=model,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        scaler=scaler,
        train_steps_per_epoch=train_steps_per_epoch,
        total_train_steps=total_train_steps,
        train_image_shape=train_image_shape,
        train_telemetry_shape=train_telemetry_shape,
        train_state_input_shape=train_state_input_shape,
        train_target_control_shape=train_target_control_shape,
        train_target_aux_shape=train_target_aux_shape,
    )


def _supported_metric_names(
    future_offsets: tuple[int, ...],
    control_target_names: tuple[str, ...],
    aux_target_names: tuple[str, ...],
) -> set[str]:
    names = {
        "drive_score",
        "loss",
        "val_loss",
        "control_loss",
        "aux_loss",
        "control_mae_generalization_gap",
        "control_mae_overall",
        "control_overall_mae",
        "aux_mae_overall",
        "aux_overall_mae",
    }
    for name in control_target_names:
        names.add(f"{name}_mae")
        names.add(f"{name}_loss")
    for name in aux_target_names:
        names.add(f"{name}_mae")
        names.add(f"{name}_loss")
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

    train_dataset: FsdDataset | None = None
    train_loader: DataLoader[DatasetItem] | None = None
    try:
        train_dataset = _build_configured_dataset(context.config, context.config.dataset.train_run_paths)
        if len(train_dataset) <= 0:
            raise ValueError(
                "Training dataset resolved to zero samples during epoch load. "
                f"train_run_ids={list(context.config.dataset.train_run_ids)} "
                f"train_trips={train_dataset.trip_count} "
                f"{train_dataset.format_rejected_sample_summary()}"
            )
        train_loader = _build_phase_loader(
            train_dataset,
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
        train_metrics, train_epoch_time, avg_timings = train_epoch(
            train_loader,
            context.optimizer,
            context.scheduler,
            context.ema,
            context.model,
            context.scaler,
            context.device,
            grad_clip_norm=context.config.training.optimizer.grad_clip_norm,
            future_offsets=context.config.dataset.future_offsets,
            state_input_names=context.config.state_inputs.enabled_keys(),
            control_target_names=context.config.dataset.control_target_names,
            aux_target_names=context.config.dataset.aux_target_names,
            aux_loss_weight=context.config.training.aux_loss_weight,
            horizon_loss_weights=context.config.training.horizon_loss_weights,
            target_loss_weights=context.config.training.target_loss_weights,
            loss_function=context.config.training.loss_function,
            smooth_l1_beta=context.config.training.smooth_l1_beta,
            log_every_n_batches=context.config.loader.log_every_n_batches,
            target_transforms=context.config.dataset.target_transforms,
        )
    finally:
        _release_phase_resources(context.device, loader=train_loader, dataset=train_dataset)
        del train_loader
        del train_dataset

    memory_snapshots["train_end"] = _memory_snapshot(context.device)
    print(_format_memory_snapshot("memory train_end", memory_snapshots["train_end"]))
    memory_snapshots["val_start"] = _memory_snapshot(context.device)
    print(_format_memory_snapshot("memory val_start", memory_snapshots["val_start"]))

    val_dataset: FsdDataset | None = None
    val_subset: TorchDataset[DatasetItem] | None = None
    val_loader: DataLoader[DatasetItem] | None = None
    try:
        val_dataset = _build_configured_dataset(context.config, context.config.dataset.val_run_paths)
        if len(val_dataset) <= 0:
            raise ValueError(
                "Validation dataset resolved to zero samples during epoch load. "
                f"val_run_ids={list(context.config.dataset.val_run_ids)} "
                f"val_trips={val_dataset.trip_count}. "
                "Check processed validation runs for empty dataset.jsonl files. "
                f"{val_dataset.format_rejected_sample_summary()}"
            )
        val_subset, _, _ = build_validation_subset(val_dataset, context.config.loader.val_split)
        if _loader_dataset_len(val_subset) <= 0:
            raise ValueError(
                "Validation subset resolved to zero usable samples during epoch load. "
                f"val_run_ids={list(context.config.dataset.val_run_ids)} "
                "Check processed validation runs for empty dataset.jsonl files or missing temporal telemetry windows."
            )
        val_loader = _build_phase_loader(
            val_subset,
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
        use_ema_for_eval = context.ema is not None and context.config.training.ema.enabled
        if use_ema_for_eval:
            # Validation runs on the EMA weights when enabled.
            context.ema.apply_to(context.model)
        try:
            val_metrics, val_epoch_time = evaluate_epoch(
                val_loader,
                context.model,
                context.device,
                future_offsets=context.config.dataset.future_offsets,
                control_target_names=context.config.dataset.control_target_names,
                aux_target_names=context.config.dataset.aux_target_names,
                target_transforms=context.config.dataset.target_transforms,
                aux_loss_weight=context.config.training.aux_loss_weight,
                horizon_loss_weights=context.config.training.horizon_loss_weights,
                target_loss_weights=context.config.training.target_loss_weights,
                loss_function=context.config.training.loss_function,
                smooth_l1_beta=context.config.training.smooth_l1_beta,
            )
        finally:
            if use_ema_for_eval:
                context.ema.restore(context.model)
    finally:
        _release_phase_resources(context.device, loader=val_loader, dataset=val_dataset)
        del val_loader
        del val_subset
        del val_dataset

    memory_snapshots["val_end"] = _memory_snapshot(context.device)
    print(_format_memory_snapshot("memory val_end", memory_snapshots["val_end"]))
    val_metrics["eval_model"] = "ema" if (context.ema is not None and context.config.training.ema.enabled) else "model"
    attach_drive_score(train_metrics, val_metrics)
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
        avg_grad_norm=avg_timings["grad_norm"],
        avg_learning_rate=float(avg_timings["learning_rate"]),
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
        scheduler=context.scheduler,
        ema=context.ema,
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
        avg_grad_norm=epoch_result.avg_grad_norm,
        avg_learning_rate=epoch_result.avg_learning_rate,
        train_steps_per_epoch=context.train_steps_per_epoch,
        total_train_steps=context.total_train_steps,
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
    run_summary["best_metric_name"] = early_stopping_state.metric_name
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
        f"drive_score={float(epoch_result.val_metrics['drive_score']):.6f}",
        f"train_control_mae={float(epoch_result.train_metrics['control_mae_overall']):.6f}",
        f"val_control_mae={float(epoch_result.val_metrics['control_mae_overall']):.6f}",
        f"train_aux_mae={float(epoch_result.train_metrics['aux_mae_overall']):.6f}",
        f"val_aux_mae={float(epoch_result.val_metrics['aux_mae_overall']):.6f}",
        f"longitudinal_aux_mae={float(epoch_result.val_metrics.get('longitudinal_aux_mae', float('nan'))):.6f}",
        f"lateral_aux_mae={float(epoch_result.val_metrics.get('lateral_aux_mae', float('nan'))):.6f}",
        f"steering_mae={float(epoch_result.val_metrics.get('steering_mae', float('nan'))):.6f}",
        f"acceleration_mae={float(epoch_result.val_metrics.get('acceleration_mae', float('nan'))):.6f}",
        f"future_speed_mae={float(epoch_result.val_metrics.get('future_speed_mae', float('nan'))):.6f}",
        f"future_speed_delta_mae={float(epoch_result.val_metrics.get('future_speed_delta_mae', 0.0)):.6f}",
        f"future_speed_delta_loss={float(epoch_result.val_metrics.get('future_speed_delta_loss', 0.0)):.6f}",
        f"future_yaw_delta_mae={float(epoch_result.val_metrics.get('future_yaw_delta_mae', float('nan'))):.6f}",
        f"future_yaw_rate_mae={float(epoch_result.val_metrics.get('future_yaw_rate_mae', float('nan'))):.6f}",
        f"checkpoint={checkpoint}",
        f"train_epoch_s={epoch_result.train_epoch_time:.3f}",
        f"val_epoch_s={epoch_result.val_epoch_time:.3f}",
        f"avg_batch_s={epoch_result.avg_batch_time:.3f}",
        f"avg_wait_s={epoch_result.avg_loader_wait_time:.3f}",
        f"avg_h2d_s={epoch_result.avg_h2d_time:.3f}",
        f"avg_fwd_bwd_s={epoch_result.avg_forward_backward_time:.3f}",
        f"avg_opt_s={epoch_result.avg_optimizer_time:.3f}",
        f"avg_iteration_s={epoch_result.avg_iteration_time:.3f}",
        f"avg_lr={epoch_result.avg_learning_rate:.7f}",
        f"elapsed_s={elapsed_s:.3f}",
        f"elapsed_hms={_format_elapsed_hms(elapsed_s)}",
        f"[{', '.join(status_parts)}]",
    ]
    if epoch_result.avg_grad_norm is not None:
        summary_parts.append(f"avg_grad_norm={epoch_result.avg_grad_norm:.4f}")
    print(" ".join(summary_parts))


def _torch_hub_dir() -> str:
    try:
        return str(torch.hub.get_dir())
    except Exception:
        return "unavailable"


def format_runtime_paths(context: TrainingContext) -> str:
    return (
        "Runtime paths: "
        f"cwd={Path.cwd()} "
        f"config={context.config_path} "
        f"data_root={context.data_root} "
        f"output_base_dir={context.output_base_dir} "
        f"run_dir={context.run_dir} "
        f"temp_dir={tempfile.gettempdir()} "
        f"torch_hub_dir={_torch_hub_dir()}"
    )


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
        f"Using device: {context.device} with train_samples={context.train_stats.sample_count} "
        f"train_trips={context.train_stats.trip_count} "
        f"val_samples={context.val_stats.sample_count} "
        f"val_trips={context.val_stats.trip_count} "
        f"selected_val_trips={context.val_stats.selected_trip_count}/{context.val_stats.total_trip_count} "
        f"run_dir={context.run_dir}"
    )
    print(format_runtime_paths(context))
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
        f"dropout={context.config.model.dropout:.3f} "
        f"visual_temporal={context.config.model.visual_temporal.type}:"
        f"{context.config.model.visual_temporal.hidden_dim} "
        f"horizon_decoder_embed_dim={context.config.model.horizon_decoder.horizon_embed_dim} "
        f"sample_images_shape={context.train_image_shape} "
        f"sample_telemetry_shape={context.train_telemetry_shape} "
        f"sample_state_inputs_shape={context.train_state_input_shape} "
        f"sample_target_controls_shape={context.train_target_control_shape} "
        f"sample_target_aux_shape={context.train_target_aux_shape}"
    )
    resolved_loss_weights = {
        **resolved_target_loss_weights(
            context.config.dataset.control_target_names,
            context.config.training.target_loss_weights,
        ),
        **resolved_target_loss_weights(
            context.config.dataset.aux_target_names,
            context.config.training.target_loss_weights,
        ),
    }
    print(
        "Loss config: "
        f"loss_function={context.config.training.loss_function} "
        f"smooth_l1_beta={context.config.training.smooth_l1_beta:.6f} "
        f"aux_loss_weight={context.config.training.aux_loss_weight:.3f} "
        f"horizon_loss_weights={list(context.config.training.horizon_loss_weights)} "
        f"target_loss_weights={resolved_loss_weights}"
    )
    print(
        "Optimization config: "
        f"optimizer={context.config.training.optimizer.name} "
        f"lr={_optimizer_learning_rate(context.optimizer):.7f} "
        f"weight_decay={context.config.training.optimizer.weight_decay:.6f} "
        f"grad_clip_norm={context.config.training.optimizer.grad_clip_norm} "
        f"scheduler={context.config.training.scheduler.name} "
        f"scheduler_step_frequency={context.config.training.scheduler.step_frequency} "
        f"warmup_fraction={context.config.training.scheduler.warmup_fraction:.4f} "
        f"min_lr_ratio={context.config.training.scheduler.min_lr_ratio:.4f} "
        f"ema_enabled={context.config.training.ema.enabled} "
        f"ema_decay={context.config.training.ema.decay:.6f} "
        f"train_steps_per_epoch={context.train_steps_per_epoch} "
        f"total_train_steps={context.total_train_steps}"
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
