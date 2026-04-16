from __future__ import annotations

import argparse
import json
import os
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, random_split

from dataset import FsdDataset
from models.planner import DrivingCNN


@dataclass(frozen=True)
class DatasetConfig:
    data_root: str
    run_id: str
    val_id: str


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


@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int
    num_workers: int
    pin_memory: bool
    prefetch_factor: int
    persistent_workers: bool
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
    train_loader: DataLoader[tuple[Tensor, Tensor]]
    val_loader: DataLoader[tuple[Tensor, Tensor]]
    model: Module
    device: torch.device
    criterion: Module
    optimizer: Optimizer
    scaler: torch.amp.GradScaler


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
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    train_epoch_time: float
    val_epoch_time: float
    avg_batch_time: float


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

    return TrainConfig(
        dataset=DatasetConfig(
            data_root=normalize_windows_drive_path(str(dataset_raw["data_root"])),
            run_id=str(dataset_raw["run_id"]),
            val_id=str(dataset_raw["val_id"]),
        ),
        output=OutputConfig(
            base_dir=str(output_raw["base_dir"]),
        ),
        training=TrainingConfig(
            device=str(training_raw["device"]).strip().lower(),
            epochs=int(training_raw["epochs"]),
            learning_rate=float(training_raw["learning_rate"]),
            early_stopping_metric=str(training_raw.get("early_stopping_metric", "overall_mae")).strip(),
            early_stopping_patience=int(training_raw.get("early_stopping_patience", 3)),
            early_stopping_min_delta=float(training_raw.get("early_stopping_min_delta", 0.0)),
        ),
        loader=LoaderConfig(
            batch_size=int(loader_raw["batch_size"]),
            num_workers=int(loader_raw["num_workers"]),
            pin_memory=bool(loader_raw["pin_memory"]),
            prefetch_factor=int(loader_raw["prefetch_factor"]),
            persistent_workers=bool(loader_raw["persistent_workers"]),
            val_split=float(loader_raw["val_split"]),
            cpu_batch_size=int(loader_raw["cpu_batch_size"]),
        ),
    )


def normalize_windows_drive_path(value: str) -> str:
    cleaned = value.strip().strip("\"'")
    if os.name == "nt" and len(cleaned) >= 2 and cleaned[1] == ":":
        if len(cleaned) == 2:
            cleaned += "\\"
        elif cleaned[2] not in ("\\", "/"):
            cleaned = f"{cleaned[:2]}\\{cleaned[2:]}"
    return cleaned


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
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    train_epoch_time: float,
    val_epoch_time: float,
    avg_batch_time: float,
) -> dict[str, float | int | str | bool]:
    checkpoint_path = run_dir / f"epoch-{epoch_index:03d}.pt"
    torch.save(
        {
            "epoch": epoch_index,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "train_epoch_s": train_epoch_time,
            "val_epoch_s": val_epoch_time,
            "avg_batch_s": avg_batch_time,
        },
        checkpoint_path,
    )
    return {
        "epoch": epoch_index,
        "checkpoint": str(checkpoint_path),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_epoch_s": train_epoch_time,
        "val_epoch_s": val_epoch_time,
        "avg_batch_s": avg_batch_time,
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


def probe_device(model: Module, device: torch.device) -> bool:
    if device.type != "cuda":
        return True
    try:
        probe_batch = torch.zeros((1, 9, 224, 224), device=device)
        with torch.no_grad():
            _ = model(probe_batch)
        return True
    except RuntimeError as exc:
        message = str(exc).lower()
        if "no kernel image is available" in message or "not compatible with the current pytorch installation" in message:
            return False
        raise


def build_loaders(
    dataset: FsdDataset,
    val_dataset: FsdDataset,
    device: torch.device,
    config: LoaderConfig,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    total_len = len(val_dataset)
    if total_len <= 0:
        raise ValueError("Validation dataset is empty")

    val_len = max(1, int(total_len * config.val_split))
    if val_len >= total_len:
        val_subset = val_dataset
    else:
        split_generator = torch.Generator().manual_seed(42)
        val_subset, _ = random_split(
            val_dataset,
            [val_len, total_len - val_len],
            generator=split_generator,
        )

    if device.type == "cuda":
        prefetch_factor = config.prefetch_factor if config.num_workers > 0 else None
        train_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers and config.num_workers > 0,
            prefetch_factor=prefetch_factor,
        )
        val_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
            dataset=val_subset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers and config.num_workers > 0,
            prefetch_factor=prefetch_factor,
        )
        return train_loader, val_loader

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=config.cpu_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        dataset=val_subset,
        batch_size=config.cpu_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, val_loader


def finalize_metrics(
    total_loss: float,
    total_batches: int,
    total_abs_error: Tensor,
    total_sq_error: Tensor,
    total_samples: int,
) -> dict[str, float]:
    if total_batches == 0 or total_samples == 0:
        return {
            "loss": 0.0,
            "steering_mae": 0.0,
            "accel_mae": 0.0,
            "steering_rmse": 0.0,
            "accel_rmse": 0.0,
            "overall_mae": 0.0,
            "overall_rmse": 0.0,
        }

    steering_mae = float(total_abs_error[0].item() / total_samples)
    accel_mae = float(total_abs_error[1].item() / total_samples)
    steering_rmse = float((total_sq_error[0].item() / total_samples) ** 0.5)
    accel_rmse = float((total_sq_error[1].item() / total_samples) ** 0.5)

    both_targets = total_samples * 2
    overall_mae = float(total_abs_error.sum().item() / both_targets)
    overall_rmse = float((total_sq_error.sum().item() / both_targets) ** 0.5)

    return {
        "loss": total_loss / total_batches,
        "steering_mae": steering_mae,
        "accel_mae": accel_mae,
        "steering_rmse": steering_rmse,
        "accel_rmse": accel_rmse,
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
    }


def train_batch(
    x_batch: Tensor,
    y_batch: Tensor,
    model: Module,
    criterion: Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[float, float, Tensor, Tensor, int]:
    batch_start = time.perf_counter()
    x_batch = x_batch.to(device, non_blocking=device.type == "cuda", memory_format=torch.channels_last)
    y_batch = y_batch.to(device, non_blocking=device.type == "cuda")

    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        preds = model(x_batch)
        loss = criterion(preds, y_batch)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    error = (preds.detach() - y_batch.detach()).float()
    abs_error = error.abs().sum(dim=0).cpu()
    sq_error = error.pow(2).sum(dim=0).cpu()

    return float(loss.item()), time.perf_counter() - batch_start, abs_error, sq_error, int(y_batch.shape[0])


def train_epoch(
    loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: Module,
    optimizer: Optimizer,
    model: Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[dict[str, float], float, float]:
    epoch_start = time.perf_counter()
    model.train()
    total_loss = 0.0
    total_batch_time = 0.0
    total_samples = 0
    total_batches = 0
    total_abs_error = torch.zeros(2, dtype=torch.float64)
    total_sq_error = torch.zeros(2, dtype=torch.float64)

    for batch_index, (x_batch, y_batch) in enumerate(loader, start=1):
        batch_loss, batch_time, batch_abs_error, batch_sq_error, batch_samples = train_batch(
            x_batch, y_batch, model, criterion, optimizer, scaler, device
        )
        total_loss += batch_loss
        total_batch_time += batch_time
        total_abs_error += batch_abs_error
        total_sq_error += batch_sq_error
        total_samples += batch_samples
        total_batches += 1

        print(f"batch={batch_index}/{len(loader)} loss={batch_loss:.6f} batch_s={batch_time:.3f}")

    epoch_time = time.perf_counter() - epoch_start
    metrics = finalize_metrics(total_loss, total_batches, total_abs_error, total_sq_error, total_samples)
    avg_batch_time = total_batch_time / total_batches if total_batches else 0.0
    return metrics, epoch_time, avg_batch_time


def evaluate_epoch(
    loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: Module,
    model: Module,
    device: torch.device,
) -> tuple[dict[str, float], float]:
    eval_start = time.perf_counter()
    model.eval()

    total_loss = 0.0
    total_samples = 0
    total_batches = 0
    total_abs_error = torch.zeros(2, dtype=torch.float64)
    total_sq_error = torch.zeros(2, dtype=torch.float64)

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=device.type == "cuda")
            y_batch = y_batch.to(device, non_blocking=device.type == "cuda")

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                preds = model(x_batch)
                loss = criterion(preds, y_batch)

            error = (preds - y_batch).float()
            total_loss += float(loss.item())
            total_abs_error += error.abs().sum(dim=0).cpu()
            total_sq_error += error.pow(2).sum(dim=0).cpu()
            total_samples += int(y_batch.shape[0])
            total_batches += 1

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    eval_time = time.perf_counter() - eval_start
    metrics = finalize_metrics(total_loss, total_batches, total_abs_error, total_sq_error, total_samples)
    return metrics, eval_time


def build_run_summary(context: TrainingContext) -> dict[str, object]:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(context.config_path),
        "dataset_root": str(context.data_root),
        "run_id": context.config.dataset.run_id,
        "val_id": context.config.dataset.val_id,
        "device": str(context.device),
        "dataset_samples": len(context.train_dataset),
        "validation_dataset_samples": len(context.val_dataset),
        "train_loader_samples": len(context.train_loader.dataset),
        "validation_loader_samples": len(context.val_loader.dataset),
        "epochs_total": context.config.training.epochs,
        "early_stopping": {
            "metric": context.config.training.early_stopping_metric,
            "patience": context.config.training.early_stopping_patience,
            "min_delta": context.config.training.early_stopping_min_delta,
        },
        "epochs": [],
    }


def prepare_model(device: torch.device) -> tuple[Module, torch.device]:
    model = DrivingCNN().to(device)
    if probe_device(model, device):
        return model, device

    print(
        "CUDA is available but this PyTorch build cannot run on your GPU "
        "(likely missing sm_120 support). Falling back to CPU."
    )
    fallback_device = torch.device("cpu")
    return DrivingCNN().to(fallback_device), fallback_device


def build_training_context(config: TrainConfig, config_path: Path) -> TrainingContext:
    data_root = Path(config.dataset.data_root)
    requested_device = select_device(config.training.device)
    train_dataset = FsdDataset(run_id=config.dataset.run_id, data_root=data_root)
    val_dataset = FsdDataset(run_id=config.dataset.val_id, data_root=data_root)
    model, device = prepare_model(requested_device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader = build_loaders(train_dataset, val_dataset, device, config.loader)
    run_dir, run_metrics_path = prepare_output_paths(Path(config.output.base_dir))
    criterion = torch.nn.MSELoss()
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
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
    )


def create_early_stopping(config: TrainingConfig) -> EarlyStoppingState:
    metric_name = config.early_stopping_metric
    if metric_name not in {
        "loss",
        "steering_mae",
        "accel_mae",
        "steering_rmse",
        "accel_rmse",
        "overall_mae",
        "overall_rmse",
    }:
        raise ValueError(f"Unsupported early stopping metric: {metric_name}")

    return EarlyStoppingState(
        metric_name=metric_name,
        patience=max(0, config.early_stopping_patience),
        min_delta=max(0.0, config.early_stopping_min_delta),
    )


def run_epoch(context: TrainingContext, epoch_index: int) -> EpochResult:
    train_metrics, train_epoch_time, avg_batch_time = train_epoch(
        context.train_loader,
        context.criterion,
        context.optimizer,
        context.model,
        context.scaler,
        context.device,
    )
    val_metrics, val_epoch_time = evaluate_epoch(
        context.val_loader,
        context.criterion,
        context.model,
        context.device,
    )
    return EpochResult(
        epoch_index=epoch_index,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        train_epoch_time=train_epoch_time,
        val_epoch_time=val_epoch_time,
        avg_batch_time=avg_batch_time,
    )


def check_early_stopping(epoch_result: EpochResult, state: EarlyStoppingState) -> tuple[bool, bool, float]:
    metric_value = epoch_result.val_metrics[state.metric_name]
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
) -> dict[str, float | int | str | bool]:
    epoch_artifact = save_epoch_artifacts(
        run_dir=context.run_dir,
        epoch_index=epoch_result.epoch_index,
        model=context.model,
        optimizer=context.optimizer,
        train_metrics=epoch_result.train_metrics,
        val_metrics=epoch_result.val_metrics,
        train_epoch_time=epoch_result.train_epoch_time,
        val_epoch_time=epoch_result.val_epoch_time,
        avg_batch_time=epoch_result.avg_batch_time,
    )
    epoch_artifact["is_best"] = is_best
    epoch_artifact["early_stopping_metric"] = early_stopping_state.metric_name
    epoch_artifact["early_stopping_value"] = monitored_value
    epoch_artifact["bad_epochs_in_a_row"] = early_stopping_state.bad_epoch_count

    epochs_list = run_summary["epochs"]
    assert isinstance(epochs_list, list)
    epochs_list.append(epoch_artifact)

    run_summary["best_epoch"] = early_stopping_state.best_epoch
    run_summary["best_metric"] = early_stopping_state.best_value
    run_summary["stopped_early"] = False
    run_summary["epochs_completed"] = epoch_result.epoch_index
    context.run_metrics_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    return epoch_artifact


def print_epoch_summary(
    epoch_result: EpochResult,
    *,
    checkpoint: str,
    is_best: bool,
    monitored_value: float,
    early_stopping_state: EarlyStoppingState,
) -> None:
    status_parts = [f"{early_stopping_state.metric_name}={monitored_value:.6f}"]
    if is_best:
        status_parts.append("new best")
    else:
        status_parts.append(f"no improvement ({early_stopping_state.bad_epoch_count} bad epochs)")

    print(
        f"epoch={epoch_result.epoch_index} "
        f"train_loss={epoch_result.train_metrics['loss']:.6f} "
        f"val_loss={epoch_result.val_metrics['loss']:.6f} "
        f"train_mae={epoch_result.train_metrics['overall_mae']:.6f} "
        f"val_mae={epoch_result.val_metrics['overall_mae']:.6f} "
        f"train_rmse={epoch_result.train_metrics['overall_rmse']:.6f} "
        f"val_rmse={epoch_result.val_metrics['overall_rmse']:.6f} "
        f"steer_mae={epoch_result.val_metrics['steering_mae']:.6f} "
        f"accel_mae={epoch_result.val_metrics['accel_mae']:.6f} "
        f"steer_rmse={epoch_result.val_metrics['steering_rmse']:.6f} "
        f"accel_rmse={epoch_result.val_metrics['accel_rmse']:.6f} "
        f"checkpoint={checkpoint} "
        f"train_epoch_s={epoch_result.train_epoch_time:.3f} "
        f"val_epoch_s={epoch_result.val_epoch_time:.3f} "
        f"avg_batch_s={epoch_result.avg_batch_time:.3f} "
        f"[{', '.join(status_parts)}]"
    )


def execute_training(context: TrainingContext) -> dict[str, object]:
    run_summary = build_run_summary(context)
    early_stopping_state = create_early_stopping(context.config.training)

    print(
        f"Using device: {context.device} with train_samples={len(context.train_dataset)} "
        f"val_samples={len(context.val_dataset)} "
        f"(train={len(context.train_loader.dataset)}, val={len(context.val_loader.dataset)}) "
        f"run_dir={context.run_dir}"
    )
    print(f"Starting training for {context.config.training.epochs} epochs...")

    for epoch_index in range(1, context.config.training.epochs + 1):
        epoch_result = run_epoch(context, epoch_index)
        is_best, should_stop, monitored_value = check_early_stopping(epoch_result, early_stopping_state)
        epoch_artifact = record_epoch(
            context,
            run_summary,
            epoch_result,
            is_best=is_best,
            monitored_value=monitored_value,
            early_stopping_state=early_stopping_state,
        )
        print_epoch_summary(
            epoch_result,
            checkpoint=str(epoch_artifact["checkpoint"]),
            is_best=is_best,
            monitored_value=monitored_value,
            early_stopping_state=early_stopping_state,
        )

        if should_stop:
            run_summary["stopped_early"] = True
            run_summary["stop_reason"] = (
                f"no improvement in {early_stopping_state.metric_name} for "
                f"{early_stopping_state.bad_epoch_count} consecutive epochs"
            )
            context.run_metrics_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
            print(
                f"Early stopping triggered at epoch {epoch_index}: "
                f"best_epoch={early_stopping_state.best_epoch} "
                f"best_{early_stopping_state.metric_name}={early_stopping_state.best_value:.6f}"
            )
            break

    return run_summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    context = build_training_context(config, args.config)
    summary = execute_training(context)
    print(f'{summary}')


if __name__ == "__main__":
    main()
