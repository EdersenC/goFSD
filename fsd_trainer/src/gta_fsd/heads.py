from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

from torch import Tensor


LEGACY_SCALAR_HEAD_ERROR = (
    "Legacy scalar-head planner has been removed. "
    "Use temporal planner outputs pred_controls/pred_aux."
)

HeadKind = Literal["control", "aux"]
LossType = Literal["smooth_l1", "mse", "bce_with_logits"] | None
@dataclass(frozen=True)
class HeadSpec:
    name: str
    kind: HeadKind = "aux"
    output_dim: int = 1
    loss_type: LossType = None
    loss_weight: float = 0.0
    target_source: str = ""
    target_builder: Any = None
    used_for_control: bool = False
    required_target: bool = False

    def metadata(self) -> dict[str, Any]:
        raise RuntimeError(LEGACY_SCALAR_HEAD_ERROR)

    def layout_metadata(self) -> dict[str, Any]:
        raise RuntimeError(LEGACY_SCALAR_HEAD_ERROR)


HEAD_SPECS: tuple[HeadSpec, ...] = ()
HEAD_SPECS_BY_NAME: dict[str, HeadSpec] = {}
ALL_HEAD_SPECS: tuple[HeadSpec, ...] = ()
ALL_HEAD_SPECS_BY_NAME: dict[str, HeadSpec] = {}
ACTIVE_HEAD_SPECS: tuple[HeadSpec, ...] = ()


def _removed() -> None:
    raise RuntimeError(LEGACY_SCALAR_HEAD_ERROR)


def head_names(head_specs: Iterable[HeadSpec] = HEAD_SPECS) -> tuple[str, ...]:
    _removed()


def canonical_head_name(name: str) -> str:
    _removed()


def control_head_specs(head_specs: Iterable[HeadSpec] = HEAD_SPECS) -> tuple[HeadSpec, ...]:
    _removed()


def aux_head_specs(head_specs: Iterable[HeadSpec] = HEAD_SPECS) -> tuple[HeadSpec, ...]:
    _removed()


def trainable_head_specs(head_specs: Iterable[HeadSpec] = HEAD_SPECS) -> tuple[HeadSpec, ...]:
    _removed()


def head_specs_metadata(head_specs: Iterable[HeadSpec] = HEAD_SPECS) -> list[dict[str, Any]]:
    _removed()


def head_layout_metadata(head_specs: Iterable[HeadSpec] = HEAD_SPECS) -> list[dict[str, Any]]:
    _removed()


def resolve_head_specs_from_metadata(raw_specs: Any) -> tuple[HeadSpec, ...]:
    _removed()


def resolve_checkpoint_head_specs(checkpoint: Mapping[str, Any]) -> tuple[HeadSpec, ...]:
    _removed()


def apply_loss_weight_overrides(loss_weight_overrides: Mapping[str, float] | None) -> tuple[HeadSpec, ...]:
    _removed()


def inactive_loss_weight_override_names(loss_weight_overrides: Mapping[str, float] | None) -> tuple[str, ...]:
    _removed()


def build_targets_from_label(label: dict[str, Any], head_specs: Iterable[HeadSpec] = HEAD_SPECS) -> dict[str, Tensor]:
    _removed()


def get_control_outputs(outputs: Mapping[str, Tensor], head_specs: Iterable[HeadSpec] = HEAD_SPECS) -> dict[str, Tensor]:
    _removed()


def normalize_head_tensor(name: str, value: Tensor, spec: HeadSpec | None = None) -> Tensor:
    _removed()


def compute_head_loss(spec: HeadSpec, prediction: Tensor, target: Tensor) -> Tensor | None:
    _removed()


def supported_metric_names(head_specs: Iterable[HeadSpec] = HEAD_SPECS) -> set[str]:
    _removed()


def validate_checkpoint_head_layout(checkpoint: Mapping[str, Any]) -> None:
    _removed()
