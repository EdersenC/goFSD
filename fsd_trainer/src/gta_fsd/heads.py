"""Central planner head definitions.

Developer note: to add a new head:
1. Add a `HeadSpec` entry to `CONTROL_HEAD_SPECS` or `AUX_HEAD_SPECS`.
2. Point `target_builder` at the raw label field or slice you want to train against.
3. Set `loss_type` and `loss_weight`.
4. Set `used_for_control=True` only if the head should be turned into applied controls.

Control heads and trainable auxiliary heads share the same runtime registry. Older checkpoints may
still advertise a smaller supported layout, so inference resolves heads from checkpoint metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Callable, Iterable, Literal, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from target_transforms import DELTA_SPEED_TARGET_LABEL_KEY


HeadKind = Literal["control", "aux"]
LossType = Literal["smooth_l1", "mse", "bce_with_logits"] | None
TargetBuilder = Callable[[dict[str, Any]], Tensor]


def _require_label_key(label: dict[str, Any], raw_key: str) -> Any:
    if raw_key not in label:
        raise KeyError(f"missing label key '{raw_key}'")
    return label[raw_key]


def _build_scalar_target(label: dict[str, Any], *, raw_key: str) -> Tensor:
    value = _require_label_key(label, raw_key)
    return _coerce_scalar(value, raw_key)


def _build_xy_slice_target(label: dict[str, Any], *, raw_key: str) -> Tensor:
    value = _require_label_key(label, raw_key)
    return _coerce_vector_slice(value, raw_key, 2)


def _build_binary_target(label: dict[str, Any], *, raw_key: str) -> Tensor:
    value = _require_label_key(label, raw_key)
    return _coerce_binary(value, raw_key)


@dataclass(frozen=True)
class HeadSpec:
    name: str
    kind: HeadKind
    output_dim: int
    loss_type: LossType
    loss_weight: float
    target_source: str
    target_builder: TargetBuilder
    used_for_control: bool
    required_target: bool = True

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "output_dim": self.output_dim,
            "loss_type": self.loss_type,
            "loss_weight": self.loss_weight,
            "target_source": self.target_source,
            "used_for_control": self.used_for_control,
            "required_target": self.required_target,
        }

    def layout_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "output_dim": self.output_dim,
            "used_for_control": self.used_for_control,
        }


def scalar_target(raw_key: str) -> TargetBuilder:
    return partial(_build_scalar_target, raw_key=raw_key)


def xy_slice_target(raw_key: str) -> TargetBuilder:
    return partial(_build_xy_slice_target, raw_key=raw_key)


def binary_target(raw_key: str) -> TargetBuilder:
    return partial(_build_binary_target, raw_key=raw_key)


CONTROL_HEAD_SPECS: tuple[HeadSpec, ...] = (
    HeadSpec(
        name="steer",
        kind="control",
        output_dim=1,
        loss_type="smooth_l1",
        loss_weight=1.0,
        target_source="label.Steering",
        target_builder=scalar_target("Steering"),
        used_for_control=True,
    ),
    HeadSpec(
        name="delta_speed",
        kind="control",
        output_dim=1,
        loss_type="smooth_l1",
        loss_weight=1.0,
        target_source=f"label.{DELTA_SPEED_TARGET_LABEL_KEY}",
        target_builder=scalar_target(DELTA_SPEED_TARGET_LABEL_KEY),
        used_for_control=True,
    ),
)

AUX_HEAD_SPECS: tuple[HeadSpec, ...] = (
    HeadSpec(
        name="future_speed",
        kind="aux",
        output_dim=1,
        loss_type="smooth_l1",
        loss_weight=0.35,
        target_source="label.future_speed",
        target_builder=scalar_target("future_speed"),
        used_for_control=False,
    ),
    HeadSpec(
        name="route_xy",
        kind="aux",
        output_dim=2,
        loss_type="smooth_l1",
        loss_weight=0.35,
        target_source="label.gps[:2]",
        target_builder=xy_slice_target("gps"),
        used_for_control=False,
    ),
    HeadSpec(
        name="speed",
        kind="aux",
        output_dim=1,
        loss_type="smooth_l1",
        loss_weight=0.0,
        target_source="label.currentSpeed",
        target_builder=scalar_target("currentSpeed"),
        used_for_control=False,
    ),
    HeadSpec(
        name="is_stopped",
        kind="aux",
        output_dim=1,
        loss_type="bce_with_logits",
        loss_weight=0.0,
        target_source="label.isStopped (bool or 0/1)",
        target_builder=binary_target("isStopped"),
        used_for_control=False,
    ),
)

ALL_HEAD_SPECS: tuple[HeadSpec, ...] = CONTROL_HEAD_SPECS + AUX_HEAD_SPECS
ALL_HEAD_SPECS_BY_NAME: dict[str, HeadSpec] = {spec.name: spec for spec in ALL_HEAD_SPECS}
ACTIVE_HEAD_SPECS: tuple[HeadSpec, ...] = (
    ALL_HEAD_SPECS_BY_NAME["steer"],
    ALL_HEAD_SPECS_BY_NAME["delta_speed"],
    ALL_HEAD_SPECS_BY_NAME["future_speed"],
)
HEAD_SPECS: tuple[HeadSpec, ...] = ACTIVE_HEAD_SPECS
HEAD_SPECS_BY_NAME: dict[str, HeadSpec] = {spec.name: spec for spec in HEAD_SPECS}

if len(ALL_HEAD_SPECS_BY_NAME) != len(ALL_HEAD_SPECS):
    raise ValueError("planner head names must be unique")
for _spec in ALL_HEAD_SPECS:
    if _spec.output_dim < 1:
        raise ValueError(f"planner head '{_spec.name}' must have output_dim > 0")


def head_names(head_specs: Iterable[HeadSpec] = HEAD_SPECS) -> tuple[str, ...]:
    return tuple(spec.name for spec in head_specs)


def control_head_specs(head_specs: Iterable[HeadSpec] = HEAD_SPECS) -> tuple[HeadSpec, ...]:
    return tuple(spec for spec in head_specs if spec.used_for_control)


def aux_head_specs(head_specs: Iterable[HeadSpec] = HEAD_SPECS) -> tuple[HeadSpec, ...]:
    return tuple(spec for spec in head_specs if not spec.used_for_control)


def trainable_head_specs(head_specs: Iterable[HeadSpec] = HEAD_SPECS) -> tuple[HeadSpec, ...]:
    return tuple(spec for spec in head_specs if spec.loss_type is not None and spec.loss_weight > 0.0)


def head_specs_metadata(head_specs: Iterable[HeadSpec] = HEAD_SPECS) -> list[dict[str, Any]]:
    return [spec.metadata() for spec in head_specs]


def head_layout_metadata(head_specs: Iterable[HeadSpec] = HEAD_SPECS) -> list[dict[str, Any]]:
    return [spec.layout_metadata() for spec in head_specs]


def resolve_head_specs_from_metadata(raw_specs: Any) -> tuple[HeadSpec, ...]:
    if not isinstance(raw_specs, list):
        raise ValueError("head metadata must be a list")

    resolved: list[HeadSpec] = []
    seen: set[str] = set()
    for item in raw_specs:
        if not isinstance(item, dict):
            raise ValueError("head metadata entries must be dicts")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError("head metadata entry is missing a non-empty name")
        if name in seen:
            raise ValueError(f"duplicate head metadata entry: {name}")
        spec = ALL_HEAD_SPECS_BY_NAME.get(name)
        if spec is None:
            raise ValueError(f"unknown head in metadata: {name}")
        if "kind" in item and item.get("kind") != spec.kind:
            raise ValueError(f"head metadata kind mismatch for '{name}'")
        if "output_dim" in item and int(item.get("output_dim")) != spec.output_dim:
            raise ValueError(f"head metadata output_dim mismatch for '{name}'")
        if "used_for_control" in item and bool(item.get("used_for_control")) != spec.used_for_control:
            raise ValueError(f"head metadata used_for_control mismatch for '{name}'")
        seen.add(name)
        resolved.append(spec)
    return tuple(resolved)


def resolve_checkpoint_head_specs(checkpoint: Mapping[str, Any]) -> tuple[HeadSpec, ...]:
    raw_specs = checkpoint.get("head_specs")
    if isinstance(raw_specs, list):
        return resolve_head_specs_from_metadata(raw_specs)

    raw_layout = checkpoint.get("head_layout")
    if isinstance(raw_layout, list):
        return resolve_head_specs_from_metadata(raw_layout)

    raise ValueError(
        "checkpoint is missing head_layout/head_specs metadata and predates the head-spec refactor; "
        "retrain or export a new checkpoint"
    )


def apply_loss_weight_overrides(loss_weight_overrides: Mapping[str, float] | None) -> tuple[HeadSpec, ...]:
    if not loss_weight_overrides:
        return HEAD_SPECS

    normalized = {str(name): float(weight) for name, weight in loss_weight_overrides.items()}
    unknown = sorted(set(normalized) - set(ALL_HEAD_SPECS_BY_NAME))
    if unknown:
        raise ValueError(f"unknown head loss weight override(s): {', '.join(unknown)}")
    for name, weight in normalized.items():
        if weight < 0.0:
            raise ValueError(f"loss weight for head '{name}' must be >= 0")

    resolved: list[HeadSpec] = []
    for spec in HEAD_SPECS:
        if spec.name in normalized:
            weight = normalized[spec.name]
            resolved.append(replace(spec, loss_weight=weight))
        else:
            resolved.append(spec)
    return tuple(resolved)


def inactive_loss_weight_override_names(loss_weight_overrides: Mapping[str, float] | None) -> tuple[str, ...]:
    if not loss_weight_overrides:
        return ()
    active = set(HEAD_SPECS_BY_NAME)
    all_names = set(ALL_HEAD_SPECS_BY_NAME)
    return tuple(sorted(name for name in loss_weight_overrides if name in all_names and name not in active))


def build_targets_from_label(label: dict[str, Any], head_specs: Iterable[HeadSpec] = HEAD_SPECS) -> dict[str, Tensor]:
    targets: dict[str, Tensor] = {}
    for spec in tuple(head_specs):
        try:
            targets[spec.name] = spec.target_builder(label)
        except (KeyError, TypeError, ValueError) as exc:
            if spec.required_target:
                raise ValueError(
                    f"failed to build target for head '{spec.name}' from {spec.target_source}: {exc}"
                ) from exc
    return targets


def get_control_outputs(outputs: Mapping[str, Tensor], head_specs: Iterable[HeadSpec] = HEAD_SPECS) -> dict[str, Tensor]:
    return {spec.name: outputs[spec.name] for spec in control_head_specs(head_specs)}


def normalize_head_tensor(name: str, value: Tensor, spec: HeadSpec) -> Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"head '{name}' must be a torch.Tensor, got {type(value).__name__}")

    tensor = value.float()
    if spec.output_dim == 1:
        if tensor.ndim > 1 and tensor.shape[-1] == 1:
            tensor = tensor.squeeze(-1)
        if tensor.ndim > 1:
            raise ValueError(f"head '{name}' expected a scalar tensor, got shape {tuple(tensor.shape)}")
        return tensor

    if tensor.ndim == 0:
        raise ValueError(f"head '{name}' expected a vector tensor, got shape {tuple(tensor.shape)}")
    if tensor.shape[-1] != spec.output_dim:
        raise ValueError(
            f"head '{name}' expected last dimension {spec.output_dim}, got shape {tuple(tensor.shape)}"
        )
    return tensor


def compute_head_loss(spec: HeadSpec, prediction: Tensor, target: Tensor) -> Tensor | None:
    if spec.loss_type is None:
        return None

    pred = normalize_head_tensor(spec.name, prediction, spec)
    truth = normalize_head_tensor(spec.name, target, spec).to(dtype=pred.dtype)

    if spec.loss_type == "smooth_l1":
        return F.smooth_l1_loss(pred, truth)
    if spec.loss_type == "mse":
        return F.mse_loss(pred, truth)
    if spec.loss_type == "bce_with_logits":
        return F.binary_cross_entropy_with_logits(pred, truth)
    raise ValueError(f"unsupported loss_type for head '{spec.name}': {spec.loss_type}")


def supported_metric_names(head_specs: Iterable[HeadSpec] = HEAD_SPECS) -> set[str]:
    head_specs_tuple = tuple(head_specs)
    control_specs = control_head_specs(head_specs_tuple)
    aux_specs = aux_head_specs(head_specs_tuple)
    names = {
        "loss",
        "overall_mae",
        "overall_rmse",
    }
    if control_specs:
        names.update({
            "control_loss",
            "control_overall_mae",
            "control_overall_rmse",
        })
    if aux_specs:
        names.update({
            "aux_loss",
            "aux_overall_mae",
            "aux_overall_rmse",
        })
    head_spec_names = {spec.name for spec in head_specs_tuple}
    for spec in head_specs_tuple:
        names.add(f"{spec.name}_loss")
        names.add(f"{spec.name}_weighted_loss")
        if spec.loss_type == "bce_with_logits":
            names.add(f"{spec.name}_accuracy")
        else:
            names.add(f"{spec.name}_mae")
            names.add(f"{spec.name}_rmse")
    if "steer" in head_spec_names:
        names.add("steering_mae")
        names.add("steering_rmse")
    return names


def validate_checkpoint_head_layout(checkpoint: Mapping[str, Any]) -> None:
    raw_layout = checkpoint.get("head_layout")
    if not isinstance(raw_layout, list):
        raise ValueError(
            "checkpoint is missing head_layout metadata and predates the head-spec refactor; "
            "retrain or export a new checkpoint"
        )
    resolve_head_specs_from_metadata(raw_layout)


def _coerce_scalar(value: Any, raw_key: str) -> Tensor:
    if isinstance(value, bool):
        return torch.tensor(float(value), dtype=torch.float32)
    if isinstance(value, (int, float)):
        return torch.tensor(float(value), dtype=torch.float32)
    raise TypeError(f"label key '{raw_key}' must be numeric, got {type(value).__name__}")


def _coerce_binary(value: Any, raw_key: str) -> Tensor:
    if isinstance(value, bool):
        return torch.tensor(1.0 if value else 0.0, dtype=torch.float32)
    if isinstance(value, (int, float)):
        return torch.tensor(1.0 if float(value) != 0.0 else 0.0, dtype=torch.float32)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return torch.tensor(1.0, dtype=torch.float32)
        if normalized in {"0", "false", "no", "n", ""}:
            return torch.tensor(0.0, dtype=torch.float32)
    raise TypeError(f"label key '{raw_key}' must be bool-like or 0/1, got {type(value).__name__}")


def _coerce_vector_slice(value: Any, raw_key: str, size: int) -> Tensor:
    if not isinstance(value, list) or len(value) < size:
        raise TypeError(f"label key '{raw_key}' must be a numeric list with at least {size} entries")
    out: list[float] = []
    for index, item in enumerate(value[:size]):
        if not isinstance(item, (int, float)):
            raise TypeError(f"label key '{raw_key}[{index}]' must be numeric, got {type(item).__name__}")
        out.append(float(item))
    return torch.tensor(out, dtype=torch.float32)
