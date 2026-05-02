from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DRIVE_SCORE_CONTROL_MAE_WEIGHT = 0.25
DRIVE_SCORE_GENERALIZATION_GAP_WEIGHT = 0.10
GENERIC_LOSS_KEYS = {"loss", "val_loss", "control_loss", "aux_loss"}


@dataclass(frozen=True)
class RankedRun:
    run_metrics_path: Path
    run_name: str
    best_epoch: "RankedEpoch"


@dataclass(frozen=True)
class RankedEpoch:
    rank_score: float
    drive_score: float
    epoch: int
    checkpoint: str
    val_loss: float | None
    val_control_loss: float | None
    val_control_mae: float | None
    train_control_mae: float | None
    mae_gap: float | None
    control_overall_mae: float | None
    aux_overall_mae: float | None
    control_overall_mae_denorm: float | None
    aux_overall_mae_denorm: float | None
    longitudinal_aux_mae: float | None
    lateral_aux_mae: float | None
    longitudinal_aux_mae_denorm: float | None
    lateral_aux_mae_denorm: float | None
    control_target_names: tuple[str, ...]
    aux_target_names: tuple[str, ...]
    target_metrics: dict[str, dict[str, float | None]]
    target_loss_metrics: dict[str, float]
    drive_score_components: dict[str, float]
    missing_metrics: tuple[str, ...]
    why: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank training epochs from best to worst using temporal validation metrics."
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


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _metric(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in metrics:
            value = _as_float(metrics.get(key))
            if value is not None:
                return value
    return None


def _fmt(value: float | None) -> str:
    return "missing" if value is None else f"{value:.6f}"


def _normalize_names(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    out: list[str] = []
    for item in value:
        name = str(item).strip()
        if name:
            out.append(name)
    return tuple(out)


def _infer_targets_from_metrics(metrics: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    inferred_losses: list[str] = []
    for key in metrics:
        if not key.endswith("_loss"):
            continue
        if key in GENERIC_LOSS_KEYS:
            continue
        target = key[:-5].strip()
        if not target:
            continue
        inferred_losses.append(target)
    unique_targets = tuple(dict.fromkeys(inferred_losses))
    control_targets = tuple(name for name in unique_targets if not name.startswith("future_"))
    aux_targets = tuple(name for name in unique_targets if name.startswith("future_"))
    return control_targets, aux_targets


def _resolve_target_names(raw_epoch: dict[str, Any], payload: dict[str, Any], val_metrics: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    control_target_names = _normalize_names(raw_epoch.get("control_target_names"))
    if not control_target_names:
        control_target_names = _normalize_names(payload.get("control_target_names"))

    aux_target_names = _normalize_names(raw_epoch.get("aux_target_names"))
    if not aux_target_names:
        aux_target_names = _normalize_names(payload.get("aux_target_names"))

    inferred_control, inferred_aux = _infer_targets_from_metrics(val_metrics)
    if not control_target_names:
        control_target_names = inferred_control
    if not aux_target_names:
        aux_target_names = inferred_aux

    control_target_names = tuple(dict.fromkeys([*control_target_names, *inferred_control]))
    aux_target_names = tuple(dict.fromkeys([*aux_target_names, *inferred_aux]))
    return control_target_names, aux_target_names


def _group_target_mae(
    *,
    val_metrics: dict[str, Any],
    target_names: tuple[str, ...],
    preferred_names: tuple[str, ...],
) -> tuple[float | None, float | None]:
    selected = tuple(name for name in preferred_names if name in target_names)
    if not selected:
        return None, None
    norm_values = [value for value in (_metric(val_metrics, f"{name}_mae") for name in selected) if value is not None]
    denorm_values = [
        value for value in (_metric(val_metrics, f"{name}_mae_denorm") for name in selected) if value is not None
    ]
    norm = None if not norm_values else sum(norm_values) / len(norm_values)
    denorm = None if not denorm_values else sum(denorm_values) / len(denorm_values)
    return norm, denorm


def _drive_score_components(
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
) -> tuple[float, dict[str, float], list[str], float | None, float | None, float | None, float | None]:
    missing: list[str] = []
    val_control_loss = _metric(val_metrics, "control_loss", "val_loss", "loss")
    if val_control_loss is None:
        missing.append("val_metrics.control_loss")
    val_control_mae = _metric(val_metrics, "control_mae_overall", "control_overall_mae")
    if val_control_mae is None:
        missing.append("val_metrics.control_mae_overall")
    train_control_mae = _metric(train_metrics, "control_mae_overall", "control_overall_mae")
    if train_control_mae is None:
        missing.append("train_metrics.control_mae_overall")

    effective_control_loss = 0.0 if val_control_loss is None else val_control_loss
    effective_val_control_mae = 0.0 if val_control_mae is None else val_control_mae
    effective_train_control_mae = 0.0 if train_control_mae is None else train_control_mae
    mae_gap = max(0.0, effective_val_control_mae - effective_train_control_mae)

    control_loss_component = effective_control_loss
    control_mae_component = DRIVE_SCORE_CONTROL_MAE_WEIGHT * effective_val_control_mae
    gap_component = DRIVE_SCORE_GENERALIZATION_GAP_WEIGHT * mae_gap
    derived_drive_score = control_loss_component + control_mae_component + gap_component

    explicit_drive_score = _metric(val_metrics, "drive_score")
    drive_score = derived_drive_score if explicit_drive_score is None else explicit_drive_score
    if explicit_drive_score is None:
        missing.append("val_metrics.drive_score")

    components = {
        "control_loss_component": control_loss_component,
        "control_mae_component": control_mae_component,
        "gap_component": gap_component,
        "derived_drive_score": derived_drive_score,
    }
    return drive_score, components, missing, val_control_loss, val_control_mae, train_control_mae, mae_gap


def _build_reason(
    *,
    drive_score: float,
    drive_score_components: dict[str, float],
    val_loss: float | None,
    control_overall_mae: float | None,
    aux_overall_mae: float | None,
    longitudinal_aux_mae: float | None,
    lateral_aux_mae: float | None,
    target_loss_metrics: dict[str, float],
    missing_metrics: tuple[str, ...],
) -> str:
    reason_parts = [
        f"drive_score={drive_score:.6f}",
        (
            "drive_components="
            f"{drive_score_components['control_loss_component']:.6f}"
            f"+{drive_score_components['control_mae_component']:.6f}"
            f"+{drive_score_components['gap_component']:.6f}"
            f" (derived={drive_score_components['derived_drive_score']:.6f})"
        ),
        f"val_loss={_fmt(val_loss)}",
        f"control_overall_mae={_fmt(control_overall_mae)}",
        f"aux_overall_mae={_fmt(aux_overall_mae)}",
        f"longitudinal_aux_mae={_fmt(longitudinal_aux_mae)}",
        f"lateral_aux_mae={_fmt(lateral_aux_mae)}",
    ]
    for metric_name, metric_value in target_loss_metrics.items():
        reason_parts.append(f"{metric_name}={metric_value:.6f}")
    if missing_metrics:
        reason_parts.append("missing=" + ", ".join(missing_metrics))
    return ", ".join(reason_parts)


def _sort_value(value: float | None) -> float:
    return float("inf") if value is None else value


def rank_epochs(run_metrics_path: Path) -> list[RankedEpoch]:
    payload = json.loads(run_metrics_path.read_text(encoding="utf-8"))
    raw_epochs = payload.get("epochs", [])
    if not isinstance(raw_epochs, list) or not raw_epochs:
        raise ValueError(f"No epochs found in {run_metrics_path}")

    ranked: list[RankedEpoch] = []
    for raw_epoch in raw_epochs:
        if not isinstance(raw_epoch, dict):
            continue
        epoch = int(raw_epoch.get("epoch", 0) or 0)
        checkpoint = str(raw_epoch.get("checkpoint", ""))
        train_metrics = raw_epoch.get("train_metrics", {})
        val_metrics = raw_epoch.get("val_metrics", {})
        if not isinstance(train_metrics, dict):
            train_metrics = {}
        if not isinstance(val_metrics, dict):
            val_metrics = {}

        control_target_names, aux_target_names = _resolve_target_names(raw_epoch, payload, val_metrics)
        drive_score, drive_components, missing_drive, val_control_loss, val_control_mae, train_control_mae, mae_gap = (
            _drive_score_components(train_metrics, val_metrics)
        )
        val_loss = _metric(val_metrics, "val_loss", "loss")
        if val_loss is None:
            missing_drive.append("val_metrics.val_loss")

        control_overall_mae = _metric(val_metrics, "control_overall_mae", "control_mae_overall")
        aux_overall_mae = _metric(val_metrics, "aux_overall_mae", "aux_mae_overall")
        control_overall_mae_denorm = _metric(val_metrics, "control_overall_mae_denorm", "control_mae_overall_denorm")
        aux_overall_mae_denorm = _metric(val_metrics, "aux_overall_mae_denorm", "aux_mae_overall_denorm")

        longitudinal_aux_mae = _metric(val_metrics, "longitudinal_aux_mae")
        longitudinal_aux_mae_denorm = _metric(val_metrics, "longitudinal_aux_mae_denorm")
        if longitudinal_aux_mae is None or longitudinal_aux_mae_denorm is None:
            computed_norm, computed_denorm = _group_target_mae(
                val_metrics=val_metrics,
                target_names=aux_target_names,
                preferred_names=("future_speed", "future_speed_delta"),
            )
            if longitudinal_aux_mae is None:
                longitudinal_aux_mae = computed_norm
            if longitudinal_aux_mae_denorm is None:
                longitudinal_aux_mae_denorm = computed_denorm

        lateral_aux_mae = _metric(val_metrics, "lateral_aux_mae")
        lateral_aux_mae_denorm = _metric(val_metrics, "lateral_aux_mae_denorm")
        if lateral_aux_mae is None or lateral_aux_mae_denorm is None:
            computed_norm, computed_denorm = _group_target_mae(
                val_metrics=val_metrics,
                target_names=aux_target_names,
                preferred_names=("future_yaw_delta", "future_yaw_rate"),
            )
            if lateral_aux_mae is None:
                lateral_aux_mae = computed_norm
            if lateral_aux_mae_denorm is None:
                lateral_aux_mae_denorm = computed_denorm

        ordered_targets = tuple(dict.fromkeys([*control_target_names, *aux_target_names]))
        target_metrics: dict[str, dict[str, float | None]] = {}
        target_loss_metrics: dict[str, float] = {}
        missing_metrics = list(missing_drive)
        for target_name in ordered_targets:
            loss_value = _metric(val_metrics, f"{target_name}_loss")
            mae_value = _metric(val_metrics, f"{target_name}_mae")
            mae_denorm_value = _metric(val_metrics, f"{target_name}_mae_denorm")
            if loss_value is None and mae_value is None and mae_denorm_value is None:
                missing_metrics.append(f"val_metrics.{target_name}_loss_or_mae")
                continue
            target_metrics[target_name] = {
                "loss": loss_value,
                "mae": mae_value,
                "mae_denorm": mae_denorm_value,
            }
            if loss_value is not None:
                target_loss_metrics[f"{target_name}_loss"] = loss_value

        missing_metrics_tuple = tuple(dict.fromkeys(missing_metrics))
        rank_score = drive_score
        ranked.append(
            RankedEpoch(
                rank_score=rank_score,
                drive_score=drive_score,
                epoch=epoch,
                checkpoint=checkpoint,
                val_loss=val_loss,
                val_control_loss=val_control_loss,
                val_control_mae=val_control_mae,
                train_control_mae=train_control_mae,
                mae_gap=mae_gap,
                control_overall_mae=control_overall_mae,
                aux_overall_mae=aux_overall_mae,
                control_overall_mae_denorm=control_overall_mae_denorm,
                aux_overall_mae_denorm=aux_overall_mae_denorm,
                longitudinal_aux_mae=longitudinal_aux_mae,
                lateral_aux_mae=lateral_aux_mae,
                longitudinal_aux_mae_denorm=longitudinal_aux_mae_denorm,
                lateral_aux_mae_denorm=lateral_aux_mae_denorm,
                control_target_names=control_target_names,
                aux_target_names=aux_target_names,
                target_metrics=target_metrics,
                target_loss_metrics=target_loss_metrics,
                drive_score_components=drive_components,
                missing_metrics=missing_metrics_tuple,
                why=_build_reason(
                    drive_score=drive_score,
                    drive_score_components=drive_components,
                    val_loss=val_loss,
                    control_overall_mae=control_overall_mae,
                    aux_overall_mae=aux_overall_mae,
                    longitudinal_aux_mae=longitudinal_aux_mae,
                    lateral_aux_mae=lateral_aux_mae,
                    target_loss_metrics=target_loss_metrics,
                    missing_metrics=missing_metrics_tuple,
                ),
            )
        )

    ranked.sort(
        key=lambda item: (
            _sort_value(item.rank_score),
            _sort_value(item.control_overall_mae),
            _sort_value(item.aux_overall_mae),
            _sort_value(item.val_loss),
            item.epoch,
        )
    )
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
            _sort_value(item.best_epoch.rank_score),
            _sort_value(item.best_epoch.control_overall_mae),
            _sort_value(item.best_epoch.aux_overall_mae),
            _sort_value(item.best_epoch.val_loss),
            item.run_name,
        )
    )
    return ranked_runs


def _format_target_metric_row(name: str, payload: dict[str, float | None]) -> str:
    return (
        f"{name}: "
        f"loss={_fmt(payload.get('loss'))} "
        f"mae={_fmt(payload.get('mae'))} "
        f"mae_denorm={_fmt(payload.get('mae_denorm'))}"
    )


def _print_epoch_details(item: RankedEpoch, *, indent: str = "   ") -> None:
    print(
        f"{indent}summary "
        f"control_overall_mae={_fmt(item.control_overall_mae)} "
        f"aux_overall_mae={_fmt(item.aux_overall_mae)} "
        f"longitudinal_aux_mae={_fmt(item.longitudinal_aux_mae)} "
        f"lateral_aux_mae={_fmt(item.lateral_aux_mae)}"
    )
    print(
        f"{indent}summary_denorm "
        f"control_overall_mae_denorm={_fmt(item.control_overall_mae_denorm)} "
        f"aux_overall_mae_denorm={_fmt(item.aux_overall_mae_denorm)} "
        f"longitudinal_aux_mae_denorm={_fmt(item.longitudinal_aux_mae_denorm)} "
        f"lateral_aux_mae_denorm={_fmt(item.lateral_aux_mae_denorm)}"
    )
    print(
        f"{indent}drive_components "
        f"control_loss_component={item.drive_score_components['control_loss_component']:.6f} "
        f"control_mae_component={item.drive_score_components['control_mae_component']:.6f} "
        f"gap_component={item.drive_score_components['gap_component']:.6f} "
        f"derived={item.drive_score_components['derived_drive_score']:.6f} "
        f"ranked_drive_score={item.drive_score:.6f}"
    )
    if item.control_target_names:
        print(f"{indent}control_targets={list(item.control_target_names)}")
    if item.aux_target_names:
        print(f"{indent}aux_targets={list(item.aux_target_names)}")
    if item.target_metrics:
        print(f"{indent}target_metrics")
        for name in [*item.control_target_names, *item.aux_target_names]:
            payload = item.target_metrics.get(name)
            if payload is None:
                continue
            print(f"{indent}  {_format_target_metric_row(name, payload)}")
        extra_names = sorted(set(item.target_metrics) - set(item.control_target_names) - set(item.aux_target_names))
        for name in extra_names:
            print(f"{indent}  {_format_target_metric_row(name, item.target_metrics[name])}")
    if item.missing_metrics:
        print(f"{indent}missing={list(item.missing_metrics)}")


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
        _print_epoch_details(best, indent="   ")
        print()
    print()


def print_rankings(run_metrics_path: Path, ranked: list[RankedEpoch], top: int) -> None:
    visible = ranked if top <= 0 else ranked[:top]
    print(f"Run metrics: {run_metrics_path}")
    print("Ranked epochs by drive score and temporal control/auxiliary quality.")
    print()
    print()

    best = ranked[0]
    print("Best epoch")
    print(
        f"  epoch {best.epoch} :: checkpoint={best.checkpoint} :: "
        f"score={best.rank_score:.6f}"
    )
    print(f"  why: {best.why}")
    _print_epoch_details(best, indent="  ")
    print()
    print()

    print("All epochs")
    for index, item in enumerate(visible, start=1):
        print(
            f"{index}. epoch {item.epoch} :: score={item.rank_score:.6f} :: "
            f"checkpoint={item.checkpoint}"
        )
        print(f"   why: {item.why}")
        _print_epoch_details(item, indent="   ")
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
