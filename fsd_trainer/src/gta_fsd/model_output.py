from __future__ import annotations

from typing import Any, Mapping

from torch import Tensor

from heads import HeadSpec, LEGACY_SCALAR_HEAD_ERROR


def _removed() -> None:
    raise RuntimeError(LEGACY_SCALAR_HEAD_ERROR)


def control_tensor_from_output(
    output: Any,
    *,
    head_specs: tuple[HeadSpec, ...] | None = None,
) -> Tensor:
    _removed()


def single_prediction_from_output(
    output: Any,
    *,
    head_specs: tuple[HeadSpec, ...] | None = None,
) -> dict[str, float | list[float]]:
    _removed()


def single_control_prediction_from_output(
    output: Any,
    *,
    head_specs: tuple[HeadSpec, ...] | None = None,
) -> dict[str, float | list[float]]:
    _removed()


def require_output_mapping(output: Any) -> dict[str, Tensor]:
    _removed()


def single_tensor_mapping(
    mapping: Mapping[str, Any],
    *,
    keys: tuple[str, ...] | None = None,
    skip_keys: set[str] | None = None,
) -> dict[str, float | list[float]]:
    _removed()
