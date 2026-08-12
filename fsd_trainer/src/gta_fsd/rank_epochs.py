from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import STOP_SIGN_AUX_TARGET_NAMES
from control_contract import STOP_SIGN_CONTROL_TARGET_NAMES


STOP_SIGN_SCORE_MOTION_PLAN_MAE_WEIGHT = 0.25
STOP_SIGN_SCORE_GENERALIZATION_GAP_WEIGHT = 0.10


@dataclass(frozen=True)
class RankedRun:
    run_metrics_path: Path
    run_name: str
    best_epoch: "RankedEpoch"


@dataclass(frozen=True)
class RankedEpoch:
    rank_score: float
    stop_sign_score: float
    epoch: int
    checkpoint: str
    val_loss: float | None
    val_motion_plan_loss: float | None
    val_motion_plan_mae: float | None
    train_motion_plan_mae: float | None
    mae_gap: float | None
    motion_plan_mae_denorm: float | None
    diagnostic_mae: float | None
    diagnostic_mae_denorm: float | None
    control_target_names: tuple[str, ...]
    aux_target_names: tuple[str, ...]
    target_metrics: dict[str, dict[str, float | None]]
    target_loss_metrics: dict[str, float]
    score_components: dict[str, float]
    missing_metrics: tuple[str, ...]
    why: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank stop-sign temporal-planner epochs from best to worst."
    )
    parser.add_argument(
        "--run-metrics",
        type=Path,
        default=None,
        help="Path to one run_metrics.json; defaults to all trainer runs.",
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
    if value is None or isinstance(value, bool):
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if math.isfinite(resolved) else None


def _metric(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _as_float(metrics.get(key))
        if value is not None:
            return value
    return None


def _fmt(value: float | None) -> str:
    return "missing" if value is None else f"{value:.6f}"


def _normalize_names(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    names = tuple(str(item).strip() for item in value if str(item).strip())
    if len(names) != len(value) or len(set(names)) != len(names):
        raise ValueError(f"{field_name} must contain unique non-empty names")
    return names


def _resolve_target_names(
    raw_epoch: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    raw_control_names = raw_epoch.get(
        "control_target_names",
        payload.get("control_target_names"),
    )
    raw_aux_names = raw_epoch.get(
        "aux_target_names",
        payload.get("aux_target_names"),
    )
    control_names = _normalize_names(
        raw_control_names,
        field_name="control_target_names",
    )
    aux_names = _normalize_names(raw_aux_names, field_name="aux_target_names")
    if control_names != STOP_SIGN_CONTROL_TARGET_NAMES:
        raise ValueError(
            "run metrics do not use the stop-sign motion-plan targets: "
            f"{list(control_names)}"
        )
    if aux_names != STOP_SIGN_AUX_TARGET_NAMES:
        raise ValueError(
            "run metrics do not use the stop-sign diagnostic targets: "
            f"{list(aux_names)}"
        )
    return control_names, aux_names


def _stop_sign_score_components(
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
) -> tuple[
    float,
    dict[str, float],
    list[str],
    float | None,
    float | None,
    float | None,
    float | None,
]:
    missing: list[str] = []
    val_motion_plan_loss = _metric(val_metrics, "control_loss")
    val_motion_plan_mae = _metric(val_metrics, "control_mae_overall")
    train_motion_plan_mae = _metric(train_metrics, "control_mae_overall")
    if val_motion_plan_loss is None:
        missing.append("val_metrics.control_loss")
    if val_motion_plan_mae is None:
        missing.append("val_metrics.control_mae_overall")
    if train_motion_plan_mae is None:
        missing.append("train_metrics.control_mae_overall")

    loss = 0.0 if val_motion_plan_loss is None else val_motion_plan_loss
    val_mae = 0.0 if val_motion_plan_mae is None else val_motion_plan_mae
    train_mae = 0.0 if train_motion_plan_mae is None else train_motion_plan_mae
    mae_gap = max(0.0, val_mae - train_mae)
    components = {
        "motion_plan_loss": loss,
        "motion_plan_mae": STOP_SIGN_SCORE_MOTION_PLAN_MAE_WEIGHT * val_mae,
        "generalization_gap": STOP_SIGN_SCORE_GENERALIZATION_GAP_WEIGHT * mae_gap,
    }
    derived_score = sum(components.values())
    explicit_score = _metric(val_metrics, "stop_sign_score")
    if explicit_score is None:
        missing.append("val_metrics.stop_sign_score")
    score = derived_score if explicit_score is None else explicit_score
    components["derived_stop_sign_score"] = derived_score
    return (
        score,
        components,
        missing,
        val_motion_plan_loss,
        val_motion_plan_mae,
        train_motion_plan_mae,
        mae_gap,
    )


def _target_metrics(
    metrics: dict[str, Any],
    names: tuple[str, ...],
) -> tuple[dict[str, dict[str, float | None]], dict[str, float], list[str]]:
    targets: dict[str, dict[str, float | None]] = {}
    target_losses: dict[str, float] = {}
    missing: list[str] = []
    for name in names:
        loss = _metric(metrics, f"{name}_loss")
        mae = _metric(metrics, f"{name}_mae")
        mae_denorm = _metric(metrics, f"{name}_mae_denorm")
        if loss is None and mae is None and mae_denorm is None:
            missing.append(f"val_metrics.{name}_loss_or_mae")
            continue
        targets[name] = {"loss": loss, "mae": mae, "mae_denorm": mae_denorm}
        if loss is not None:
            target_losses[f"{name}_loss"] = loss
    return targets, target_losses, missing


def _build_reason(item: RankedEpoch) -> str:
    components = item.score_components
    parts = [
        f"stop_sign_score={item.stop_sign_score:.6f}",
        (
            "score_components="
            f"{components['motion_plan_loss']:.6f}"
            f"+{components['motion_plan_mae']:.6f}"
            f"+{components['generalization_gap']:.6f}"
            f" (derived={components['derived_stop_sign_score']:.6f})"
        ),
        f"val_loss={_fmt(item.val_loss)}",
        f"motion_plan_mae={_fmt(item.val_motion_plan_mae)}",
        f"diagnostic_mae={_fmt(item.diagnostic_mae)}",
    ]
    parts.extend(
        f"{name}={value:.6f}"
        for name, value in item.target_loss_metrics.items()
    )
    if item.missing_metrics:
        parts.append("missing=" + ", ".join(item.missing_metrics))
    return ", ".join(parts)


def _sort_value(value: float | None) -> float:
    return float("inf") if value is None else value


def rank_epochs(run_metrics_path: Path) -> list[RankedEpoch]:
    payload = json.loads(run_metrics_path.read_text(encoding="utf-8"))
    raw_epochs = payload.get("epochs")
    if not isinstance(raw_epochs, list) or not raw_epochs:
        raise ValueError(f"No epochs found in {run_metrics_path}")

    ranked: list[RankedEpoch] = []
    for raw_epoch in raw_epochs:
        if not isinstance(raw_epoch, dict):
            raise ValueError("run metrics epochs must be objects")
        train_metrics = raw_epoch.get("train_metrics")
        val_metrics = raw_epoch.get("val_metrics")
        if not isinstance(train_metrics, dict) or not isinstance(val_metrics, dict):
            raise ValueError("each epoch must include train_metrics and val_metrics objects")
        control_names, aux_names = _resolve_target_names(raw_epoch, payload)
        (
            score,
            components,
            missing,
            val_plan_loss,
            val_plan_mae,
            train_plan_mae,
            mae_gap,
        ) = _stop_sign_score_components(train_metrics, val_metrics)
        val_loss = _metric(val_metrics, "val_loss")
        if val_loss is None:
            missing.append("val_metrics.val_loss")
        all_names = control_names + aux_names
        targets, losses, missing_targets = _target_metrics(val_metrics, all_names)
        item = RankedEpoch(
            rank_score=score,
            stop_sign_score=score,
            epoch=int(raw_epoch.get("epoch", 0) or 0),
            checkpoint=str(raw_epoch.get("checkpoint", "")),
            val_loss=val_loss,
            val_motion_plan_loss=val_plan_loss,
            val_motion_plan_mae=val_plan_mae,
            train_motion_plan_mae=train_plan_mae,
            mae_gap=mae_gap,
            motion_plan_mae_denorm=_metric(val_metrics, "control_mae_overall_denorm"),
            diagnostic_mae=_metric(val_metrics, "aux_mae_overall"),
            diagnostic_mae_denorm=_metric(val_metrics, "aux_mae_overall_denorm"),
            control_target_names=control_names,
            aux_target_names=aux_names,
            target_metrics=targets,
            target_loss_metrics=losses,
            score_components=components,
            missing_metrics=tuple(dict.fromkeys([*missing, *missing_targets])),
            why="",
        )
        ranked.append(RankedEpoch(**{**item.__dict__, "why": _build_reason(item)}))

    ranked.sort(
        key=lambda item: (
            item.rank_score,
            _sort_value(item.val_motion_plan_mae),
            _sort_value(item.diagnostic_mae),
            _sort_value(item.val_loss),
            item.epoch,
        )
    )
    return ranked


def rank_runs(run_metrics_paths: list[Path]) -> list[RankedRun]:
    ranked_runs = [
        RankedRun(
            run_metrics_path=path,
            run_name=path.parent.name,
            best_epoch=rank_epochs(path)[0],
        )
        for path in run_metrics_paths
    ]
    ranked_runs.sort(key=lambda item: (item.best_epoch.rank_score, item.run_name))
    return ranked_runs


def _format_target_metric_row(
    name: str,
    payload: dict[str, float | None],
) -> str:
    return (
        f"{name}: loss={_fmt(payload.get('loss'))} "
        f"mae={_fmt(payload.get('mae'))} "
        f"mae_denorm={_fmt(payload.get('mae_denorm'))}"
    )


def _print_epoch_details(item: RankedEpoch, *, indent: str = "   ") -> None:
    print(
        f"{indent}summary motion_plan_mae={_fmt(item.val_motion_plan_mae)} "
        f"diagnostic_mae={_fmt(item.diagnostic_mae)} "
        f"motion_plan_mae_denorm={_fmt(item.motion_plan_mae_denorm)} "
        f"diagnostic_mae_denorm={_fmt(item.diagnostic_mae_denorm)}"
    )
    print(
        f"{indent}score_components "
        f"motion_plan_loss={item.score_components['motion_plan_loss']:.6f} "
        f"motion_plan_mae={item.score_components['motion_plan_mae']:.6f} "
        f"gap={item.score_components['generalization_gap']:.6f} "
        f"derived={item.score_components['derived_stop_sign_score']:.6f} "
        f"ranked_stop_sign_score={item.stop_sign_score:.6f}"
    )
    print(f"{indent}motion_plan_targets={list(item.control_target_names)}")
    print(f"{indent}diagnostic_targets={list(item.aux_target_names)}")
    for name in (*item.control_target_names, *item.aux_target_names):
        target = item.target_metrics.get(name)
        if target is not None:
            print(f"{indent}  {_format_target_metric_row(name, target)}")
    if item.missing_metrics:
        print(f"{indent}missing={list(item.missing_metrics)}")


def print_run_leaderboard(ranked_runs: list[RankedRun], top: int) -> None:
    visible = ranked_runs if top <= 0 else ranked_runs[:top]
    recommended = ranked_runs[0]
    best = recommended.best_epoch
    print(
        "Recommended checkpoint: "
        f"{recommended.run_name} / epoch-{best.epoch:03d} / {best.checkpoint}"
    )
    print("\nRun leaderboard")
    print(f"Ranked {len(ranked_runs)} runs using the best epoch from each run.\n")
    for index, item in enumerate(visible, start=1):
        best = item.best_epoch
        print(
            f"{index}. {item.run_name} :: best_epoch={best.epoch} "
            f":: score={best.rank_score:.6f}"
        )
        print(f"   metrics={item.run_metrics_path}")
        print(f"   checkpoint={best.checkpoint}")
        print(f"   why: {best.why}")
        _print_epoch_details(best)
        print()


def print_rankings(
    run_metrics_path: Path,
    ranked: list[RankedEpoch],
    top: int,
) -> None:
    visible = ranked if top <= 0 else ranked[:top]
    print(f"Run metrics: {run_metrics_path}")
    print("Ranked epochs by stop-sign motion-plan and diagnostic quality.\n")
    best = ranked[0]
    print(
        f"Best epoch\n  epoch {best.epoch} :: checkpoint={best.checkpoint} "
        f":: score={best.rank_score:.6f}"
    )
    print(f"  why: {best.why}")
    _print_epoch_details(best, indent="  ")
    print("\nAll epochs")
    for index, item in enumerate(visible, start=1):
        print(
            f"{index}. epoch {item.epoch} :: score={item.rank_score:.6f} "
            f":: checkpoint={item.checkpoint}"
        )
        print(f"   why: {item.why}")


def main() -> None:
    args = parse_args()
    if args.top < 0:
        raise ValueError("--top must be >= 0")
    if args.run_metrics is not None:
        print_rankings(args.run_metrics, rank_epochs(args.run_metrics), args.top)
        return
    print_run_leaderboard(rank_runs(find_all_run_metrics()), args.top)


if __name__ == "__main__":
    main()
