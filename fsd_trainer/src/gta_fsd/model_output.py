from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from heads import HEAD_SPECS_BY_NAME, HeadSpec, control_head_specs, get_control_outputs, normalize_head_tensor
from target_transforms import (
    DeltaSpeedTargetTransform,
    denormalize_delta_speed_tensor,
    legacy_delta_speed_target_transform,
)


def control_tensor_from_output(
    output: Any,
    *,
    head_specs: tuple[HeadSpec, ...] | None = None,
    delta_speed_transform: DeltaSpeedTargetTransform | None = None,
) -> Tensor:
    mapping = require_output_mapping(output)
    resolved_head_specs = tuple(HEAD_SPECS_BY_NAME.values()) if head_specs is None else head_specs
    resolved_delta_speed_transform = (
        legacy_delta_speed_target_transform() if delta_speed_transform is None else delta_speed_transform
    )
    control_tensors = [
        _display_tensor(
            spec.name,
            normalize_head_tensor(spec.name, _require_tensor(mapping, spec.name), spec),
            delta_speed_transform=resolved_delta_speed_transform,
        )
        for spec in control_head_specs(resolved_head_specs)
    ]
    if not control_tensors:
        raise ValueError("no control heads are configured")
    return torch.stack(control_tensors, dim=-1)


def single_prediction_from_output(
    output: Any,
    *,
    head_specs: tuple[HeadSpec, ...] | None = None,
    delta_speed_transform: DeltaSpeedTargetTransform | None = None,
) -> dict[str, float | list[float]]:
    mapping = require_output_mapping(output)
    resolved_head_specs = tuple(HEAD_SPECS_BY_NAME.values()) if head_specs is None else head_specs
    keys = tuple(spec.name for spec in resolved_head_specs if spec.name in mapping)
    return single_tensor_mapping(mapping, keys=keys, delta_speed_transform=delta_speed_transform)


def single_control_prediction_from_output(
    output: Any,
    *,
    head_specs: tuple[HeadSpec, ...] | None = None,
    delta_speed_transform: DeltaSpeedTargetTransform | None = None,
) -> dict[str, float | list[float]]:
    resolved_head_specs = tuple(HEAD_SPECS_BY_NAME.values()) if head_specs is None else head_specs
    return single_tensor_mapping(
        get_control_outputs(require_output_mapping(output), head_specs=resolved_head_specs),
        delta_speed_transform=delta_speed_transform,
    )


def require_output_mapping(output: Any) -> dict[str, Tensor]:
    if not isinstance(output, dict):
        raise TypeError(f"planner output must be a dict, got {type(output).__name__}")

    normalized: dict[str, Tensor] = {}
    for key, value in output.items():
        tensor = _require_tensor(output, key)
        spec = HEAD_SPECS_BY_NAME.get(key)
        normalized[key] = normalize_head_tensor(key, tensor, spec) if spec is not None else tensor.float()
    return normalized


def _require_tensor(output: Mapping[str, Any], key: str) -> Tensor:
    if key not in output:
        raise KeyError(f"planner output is missing '{key}'")
    value = output[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"planner output '{key}' must be a torch.Tensor, got {type(value).__name__}")
    return value


def _display_tensor(
    key: str,
    tensor: Tensor,
    *,
    delta_speed_transform: DeltaSpeedTargetTransform,
) -> Tensor:
    if key == "delta_speed":
        return denormalize_delta_speed_tensor(tensor, delta_speed_transform)
    return tensor


def _single_sample_value(
    key: str,
    value: Any,
    *,
    delta_speed_transform: DeltaSpeedTargetTransform,
) -> float | list[float]:
    tensor = value
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"planner output '{key}' must be a torch.Tensor, got {type(value).__name__}")

    spec = HEAD_SPECS_BY_NAME.get(key)
    if spec is not None:
        tensor = normalize_head_tensor(key, tensor, spec)
    tensor = _display_tensor(key, tensor, delta_speed_transform=delta_speed_transform)

    single = tensor.squeeze(0).detach().cpu()
    if single.ndim == 0:
        return float(single.item())
    if single.ndim == 1:
        return [float(item) for item in single.tolist()]
    raise ValueError(
        f"planner output '{key}' must reduce to a scalar or vector for single-sample inference, "
        f"got shape {tuple(single.shape)}"
    )


def single_tensor_mapping(
    mapping: Mapping[str, Any],
    *,
    keys: tuple[str, ...] | None = None,
    skip_keys: set[str] | None = None,
    delta_speed_transform: DeltaSpeedTargetTransform | None = None,
) -> dict[str, float | list[float]]:
    output: dict[str, float | list[float]] = {}
    selected_keys = tuple(mapping.keys()) if keys is None else keys
    skipped = skip_keys or set()
    resolved_delta_speed_transform = (
        legacy_delta_speed_target_transform() if delta_speed_transform is None else delta_speed_transform
    )
    for key in selected_keys:
        if key in skipped:
            continue
        if key not in mapping:
            raise KeyError(f"mapping is missing '{key}'")
        output[key] = _single_sample_value(
            key,
            mapping[key],
            delta_speed_transform=resolved_delta_speed_transform,
        )
    return output
