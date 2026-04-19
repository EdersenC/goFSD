from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from heads import HEAD_SPECS


@dataclass(frozen=True)
class RankedRun:
    run_metrics_path: Path
    run_name: str
    best_epoch: "RankedEpoch"


@dataclass(frozen=True)
class RankedEpoch:
    rank_score: float
    epoch: int
    checkpoint: str
    val_loss: float
    val_control_rmse: float
    val_control_mae: float
    train_control_rmse: float
    train_control_mae: float
    rmse_gap: float
    mae_gap: float
    control_head_metrics: dict[str, float]
    why: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank training epochs from best to worst using validation metrics."
    )
    parser.add_argument(
        "--run-metrics",
        type=Path,
        default=None,
        help="Path to a single run_metrics.json file. Defaults to all runs under fsd_trainer.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Show only the top N epochs. Default 0 prints all epochs.",
    )
    return parser.parse_args()


def find_all_run_metrics() -> list[Path]:
    project_root = Path(__file__).resolve().parents[2]
    candidates = sorted(project_root.rglob("run_metrics.json"))
    if not candidates:
        raise FileNotFoundError(f"No run_metrics.json files found under {project_root}")
    return candidates


def require_float(metrics: dict[str, Any], key: str, *, epoch: int, section: str) -> float:
    value = metrics.get(key)
    if value is None:
        raise KeyError(f"Missing {section}.{key} for epoch {epoch}")
    return float(value)


def _require_metric_with_fallback(
    metrics: dict[str, Any],
    primary: str,
    *,
    epoch: int,
    section: str,
    fallback: str | None = None,
) -> float:
    if primary in metrics:
        return float(metrics[primary])
    if fallback is not None and fallback in metrics:
        return float(metrics[fallback])
    raise KeyError(f"Missing {section}.{primary} for epoch {epoch}")


def build_reason(
    *,
    val_loss: float,
    val_control_rmse: float,
    val_control_mae: float,
    control_head_metrics: dict[str, float],
    rmse_gap: float,
    mae_gap: float,
) -> str:
    reason_parts = [
        f"val control RMSE {val_control_rmse:.6f}",
        f"val control MAE {val_control_mae:.6f}",
        f"val loss {val_loss:.6f}",
    ]
    for key, value in control_head_metrics.items():
        reason_parts.append(f"{key} {value:.6f}")
    reason_parts.append(f"RMSE gap {rmse_gap:+.6f}")
    reason_parts.append(f"MAE gap {mae_gap:+.6f}")
    return ", ".join(reason_parts)


def rank_epochs(run_metrics_path: Path) -> list[RankedEpoch]:
    payload = json.loads(run_metrics_path.read_text(encoding="utf-8"))
    raw_epochs = payload.get("epochs", [])
    if not isinstance(raw_epochs, list) or not raw_epochs:
        raise ValueError(f"No epochs found in {run_metrics_path}")

    ranked: list[RankedEpoch] = []
    for raw_epoch in raw_epochs:
        epoch = int(raw_epoch["epoch"])
        checkpoint = str(raw_epoch.get("checkpoint", ""))
        train_metrics = raw_epoch.get("train_metrics", {})
        val_metrics = raw_epoch.get("val_metrics", {})
        if not isinstance(train_metrics, dict) or not isinstance(val_metrics, dict):
            raise ValueError(f"Invalid metrics section for epoch {epoch}")

        val_loss = require_float(val_metrics, "loss", epoch=epoch, section="val_metrics")
        val_control_rmse = _require_metric_with_fallback(
            val_metrics,
            "control_overall_rmse",
            epoch=epoch,
            section="val_metrics",
            fallback="overall_rmse",
        )
        val_control_mae = _require_metric_with_fallback(
            val_metrics,
            "control_overall_mae",
            epoch=epoch,
            section="val_metrics",
            fallback="overall_mae",
        )
        train_control_rmse = _require_metric_with_fallback(
            train_metrics,
            "control_overall_rmse",
            epoch=epoch,
            section="train_metrics",
            fallback="overall_rmse",
        )
        train_control_mae = _require_metric_with_fallback(
            train_metrics,
            "control_overall_mae",
            epoch=epoch,
            section="train_metrics",
            fallback="overall_mae",
        )

        rmse_gap = val_control_rmse - train_control_rmse
        mae_gap = val_control_mae - train_control_mae

        control_head_metrics: dict[str, float] = {}
        for spec in HEAD_SPECS:
            if spec.kind != "control":
                continue
            metric_name = f"{spec.name}_rmse"
            fallback = "steering_rmse" if spec.name == "steer" else None
            control_head_metrics[metric_name] = _require_metric_with_fallback(
                val_metrics,
                metric_name,
                epoch=epoch,
                section="val_metrics",
                fallback=fallback,
            )

        rank_score = (
            val_control_rmse * 1000.0
            + val_control_mae * 100.0
            + val_loss * 10.0
            + abs(rmse_gap) * 20.0
            + abs(mae_gap) * 10.0
        )

        ranked.append(
            RankedEpoch(
                rank_score=rank_score,
                epoch=epoch,
                checkpoint=checkpoint,
                val_loss=val_loss,
                val_control_rmse=val_control_rmse,
                val_control_mae=val_control_mae,
                train_control_rmse=train_control_rmse,
                train_control_mae=train_control_mae,
                rmse_gap=rmse_gap,
                mae_gap=mae_gap,
                control_head_metrics=control_head_metrics,
                why=build_reason(
                    val_loss=val_loss,
                    val_control_rmse=val_control_rmse,
                    val_control_mae=val_control_mae,
                    control_head_metrics=control_head_metrics,
                    rmse_gap=rmse_gap,
                    mae_gap=mae_gap,
                ),
            )
        )

    ranked.sort(key=lambda item: (item.rank_score, item.val_control_rmse, item.val_control_mae, item.epoch))
    return ranked


def rank_runs(run_metrics_paths: list[Path]) -> list[RankedRun]:
    ranked_runs: list[RankedRun] = []
    for run_metrics_path in run_metrics_paths:
        ranked_epochs = rank_epochs(run_metrics_path)
        ranked_runs.append(
            RankedRun(
                run_metrics_path=run_metrics_path,
                run_name=run_metrics_path.parent.name,
                best_epoch=ranked_epochs[0],
            )
        )

    ranked_runs.sort(
        key=lambda item: (
            item.best_epoch.rank_score,
            item.best_epoch.val_control_rmse,
            item.best_epoch.val_control_mae,
            item.run_name,
        )
    )
    return ranked_runs


def print_run_leaderboard(ranked_runs: list[RankedRun], top: int) -> None:
    visible = ranked_runs if top <= 0 else ranked_runs[:top]
    recommended = ranked_runs[0]
    recommended_epoch = recommended.best_epoch
    print(
        "Recommended checkpoint: "
        f"{recommended.run_name} / epoch-{recommended_epoch.epoch:03d} / "
        f"{recommended_epoch.checkpoint}"
    )
    print()
    print()
    print("Run leaderboard")
    print(f"Ranked {len(ranked_runs)} runs using the best epoch from each run.")
    print()
    print()

    for index, item in enumerate(visible, start=1):
        best = item.best_epoch
        print(f"{index}. {item.run_name} :: best_epoch={best.epoch} :: score={best.rank_score:.6f}")
        print(f"   metrics={item.run_metrics_path}")
        print(f"   checkpoint={best.checkpoint}")
        print(f"   why: {best.why}")
        print()
    print()


def print_rankings(run_metrics_path: Path, ranked: list[RankedEpoch], top: int) -> None:
    visible = ranked if top <= 0 else ranked[:top]
    print(f"Run metrics: {run_metrics_path}")
    print(f"Ranked {len(ranked)} epochs by validation quality and generalization gap.")
    print()
    print()

    best = ranked[0]
    print("Best epoch")
    print(
        f"  epoch {best.epoch} :: checkpoint={best.checkpoint} :: "
        f"score={best.rank_score:.6f}"
    )
    print(f"  why: {best.why}")
    print()
    print()

    print("All epochs")
    for index, item in enumerate(visible, start=1):
        print(
            f"{index}. epoch {item.epoch} :: score={item.rank_score:.6f} :: "
            f"checkpoint={item.checkpoint}"
        )
        print(f"   why: {item.why}")
        print()
    print()


def main() -> None:
    args = parse_args()
    run_metrics_paths = [args.run_metrics] if args.run_metrics else find_all_run_metrics()
    if len(run_metrics_paths) > 1:
        ranked_runs = rank_runs(run_metrics_paths)
        print_run_leaderboard(ranked_runs, args.top)
        print()
        print()
    for index, run_metrics_path in enumerate(run_metrics_paths):
        ranked = rank_epochs(run_metrics_path)
        print_rankings(run_metrics_path, ranked, args.top)
        if index != len(run_metrics_paths) - 1:
            print()
            print()


if __name__ == "__main__":
    main()
