from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from config import (
    DEFAULT_AUX_TARGET_NAMES,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    STOP_SIGN_AUX_TARGET_NAMES,
    normalize_windows_drive_path,
    parse_dataset_window,
    resolve_optional_data_root,
)
from control_contract import (
    PLANNER_FORMAT,
    PLANNER_FORMAT_VERSION,
    control_contract_metadata,
    require_stop_sign_control_target_names,
    resolve_checkpoint_control_horizon_dt_ms,
    validate_checkpoint_control_contract,
)
from dataset import FsdDataset
from models.planner import StopSignTemporalPlanner
from state_inputs import (
    DEFAULT_WIDTH_MULTIPLIER,
    state_input_definition,
    state_input_config_from_metadata,
    state_inputs_metadata,
)
from stop_sign_contract import release_policy_metadata
from target_transforms import (
    TargetTransform,
    denormalize_target_tensor,
    target_transform_metadata,
    resolve_checkpoint_target_transforms,
    validate_stop_sign_control_target_transforms,
)


SUPPORTED_DEVICES = {"auto", "cpu", "cuda"}
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "train_config.toml"
LEGACY_SCALAR_HEAD_ERROR = (
    "Legacy scalar-head planner has been removed. "
    "Use temporal planner outputs pred_controls/pred_aux."
)


@dataclass(frozen=True)
class InferenceConfig:
    checkpoint: str
    device: str
    data_root: str | None
    run_id: str | None
    sample_index: int
    output_json: bool
    metadata_only: bool
    image_width: int
    image_height: int
    window_size: int
    frame_stride: int
    sample_stride: int


@dataclass(frozen=True)
class CheckpointTimeline:
    image_offsets: tuple[int, ...]
    telemetry_offsets: tuple[int, ...]
    future_offsets: tuple[int, ...]
    telemetry_sample_interval_ms: int


def resolve_checkpoint_image_size(checkpoint: Mapping[str, Any]) -> tuple[int, int]:
    raw_image_size = checkpoint.get("image_size")
    if not isinstance(raw_image_size, Mapping):
        raise ValueError("stop-sign checkpoint must include image_size")

    dimensions: list[int] = []
    for name in ("width", "height"):
        value = raw_image_size.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"checkpoint image_size.{name} must be a positive integer")
        dimensions.append(value)
    return dimensions[0], dimensions[1]


def require_checkpoint_image_size(
    checkpoint: Mapping[str, Any],
    expected_image_size: tuple[int, int],
) -> tuple[int, int]:
    checkpoint_image_size = resolve_checkpoint_image_size(checkpoint)
    if checkpoint_image_size != expected_image_size:
        raise ValueError(
            "Config/checkpoint image-size mismatch: "
            f"config={expected_image_size[0]}x{expected_image_size[1]}, "
            f"checkpoint={checkpoint_image_size[0]}x{checkpoint_image_size[1]}"
        )
    return checkpoint_image_size


def _checkpoint_offsets(
    checkpoint: Mapping[str, Any],
    key: str,
    *,
    allow_negative: bool,
    allow_zero: bool,
) -> tuple[int, ...]:
    raw_offsets = checkpoint.get(key)
    if not isinstance(raw_offsets, list) or not raw_offsets:
        raise ValueError(f"stop-sign checkpoint must include non-empty {key}")

    offsets: list[int] = []
    for index, value in enumerate(raw_offsets):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key}[{index}] must be an integer")
        offsets.append(value)

    result = tuple(offsets)
    if tuple(sorted(result)) != result or len(set(result)) != len(result):
        raise ValueError(f"{key} must be strictly increasing")
    if not allow_negative and any(value < 0 for value in result):
        raise ValueError(f"{key} must not contain negative offsets")
    if not allow_zero and any(value == 0 for value in result):
        raise ValueError(f"{key} must not contain zero")
    if allow_negative and any(value > 0 for value in result):
        raise ValueError(f"{key} must not contain positive offsets")
    return result


def resolve_checkpoint_timeline(checkpoint: Mapping[str, Any]) -> CheckpointTimeline:
    image_offsets = _checkpoint_offsets(
        checkpoint,
        "image_offsets",
        allow_negative=True,
        allow_zero=True,
    )
    telemetry_offsets = _checkpoint_offsets(
        checkpoint,
        "telemetry_offsets",
        allow_negative=True,
        allow_zero=True,
    )
    future_offsets = _checkpoint_offsets(
        checkpoint,
        "future_offsets",
        allow_negative=False,
        allow_zero=False,
    )
    if image_offsets[-1] != 0:
        raise ValueError("image_offsets must end at 0")
    if telemetry_offsets[-1] != 0:
        raise ValueError("telemetry_offsets must end at 0")

    raw_interval = checkpoint.get("telemetry_sample_interval_ms")
    if isinstance(raw_interval, bool) or not isinstance(raw_interval, int) or raw_interval <= 0:
        raise ValueError("telemetry_sample_interval_ms must be a positive integer")

    checkpoint_frame_count = checkpoint.get("frame_window_size")
    if checkpoint_frame_count is not None:
        if (
            isinstance(checkpoint_frame_count, bool)
            or not isinstance(checkpoint_frame_count, int)
            or checkpoint_frame_count <= 0
        ):
            raise ValueError("frame_window_size must be a positive integer")
        if checkpoint_frame_count != len(image_offsets):
            raise ValueError(
                "checkpoint frame_window_size must match image_offsets: "
                f"frame_window_size={checkpoint_frame_count} image_offsets={len(image_offsets)}"
            )

    return CheckpointTimeline(
        image_offsets=image_offsets,
        telemetry_offsets=telemetry_offsets,
        future_offsets=future_offsets,
        telemetry_sample_interval_ms=raw_interval,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a trained planner checkpoint and run inference.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the TOML config. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to the .pt checkpoint you want to load.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=sorted(SUPPORTED_DEVICES),
        help="Device to use for inference.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Dataset root for sample inference.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run id to load from the dataset for sample inference.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Dataset sample index to run inference on.",
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Print structured JSON instead of plain text.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Load checkpoint metadata without running dataset sample inference.",
    )
    return parser.parse_args()


def load_config(path: Path, *, require_checkpoint: bool = True) -> InferenceConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    inference_raw = raw.get("inference", {})
    dataset_raw = raw.get("dataset", {})
    window_size, frame_stride, sample_stride = parse_dataset_window(raw)

    checkpoint = str(inference_raw.get("checkpoint", "")).strip()
    if require_checkpoint and not checkpoint:
        raise ValueError(f"Missing inference.checkpoint in {path}")

    data_root_raw = inference_raw.get("data_root", dataset_raw.get("data_root"))
    data_root = resolve_optional_data_root(data_root_raw)
    run_id_raw = inference_raw.get("run_id")
    run_id_value = "" if run_id_raw is None else str(run_id_raw).strip()
    run_id = run_id_value or None
    image_width = int(inference_raw.get("image_width", dataset_raw.get("image_width", DEFAULT_IMAGE_WIDTH)))
    image_height = int(inference_raw.get("image_height", dataset_raw.get("image_height", DEFAULT_IMAGE_HEIGHT)))

    return InferenceConfig(
        checkpoint=checkpoint,
        device=str(inference_raw.get("device", "auto")).strip().lower(),
        data_root=data_root,
        run_id=run_id,
        sample_index=int(inference_raw.get("sample_index", 0)),
        output_json=bool(inference_raw.get("output_json", False)),
        metadata_only=bool(inference_raw.get("metadata_only", False)),
        image_width=image_width,
        image_height=image_height,
        window_size=window_size,
        frame_stride=frame_stride,
        sample_stride=sample_stride,
    )


def resolve_config(args: argparse.Namespace) -> InferenceConfig:
    file_config = load_config(
        args.config,
        require_checkpoint=args.checkpoint is None,
    )
    checkpoint = str(args.checkpoint) if args.checkpoint is not None else file_config.checkpoint
    device = args.device if args.device is not None else file_config.device
    data_root = normalize_windows_drive_path(args.data_root) if args.data_root is not None else file_config.data_root
    run_id = args.run_id if args.run_id is not None else file_config.run_id
    sample_index = args.sample_index if args.sample_index is not None else file_config.sample_index
    output_json = args.output_json or file_config.output_json
    metadata_only = bool(args.metadata_only or file_config.metadata_only)

    if device not in SUPPORTED_DEVICES:
        raise ValueError(f"device must be one of: {', '.join(sorted(SUPPORTED_DEVICES))}")

    return InferenceConfig(
        checkpoint=checkpoint,
        device=device,
        data_root=data_root,
        run_id=run_id,
        sample_index=sample_index,
        output_json=output_json,
        metadata_only=metadata_only,
        image_width=file_config.image_width,
        image_height=file_config.image_height,
        window_size=file_config.window_size,
        frame_stride=file_config.frame_stride,
        sample_stride=file_config.sample_stride,
    )


def resolve_existing_path(raw_path: str, config_path: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate

    config_dir = config_path.resolve().parent
    project_root = config_dir.parent
    script_dir = Path(__file__).resolve().parent

    candidates = [
        Path.cwd() / candidate,
        config_dir / candidate,
        project_root / candidate,
        script_dir / candidate,
    ]

    for resolved in candidates:
        if resolved.is_file():
            return resolved

    tried = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(f"Checkpoint does not exist: {raw_path}\nTried:\n{tried}")


def select_device(requested: str) -> torch.device:
    if requested not in SUPPORTED_DEVICES:
        raise ValueError(f"device must be one of: {', '.join(sorted(SUPPORTED_DEVICES))}")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint payload in {checkpoint_path}")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint is missing model_state_dict: {checkpoint_path}")
    try:
        validate_checkpoint_control_contract(checkpoint)
        resolve_checkpoint_image_size(checkpoint)
        resolve_checkpoint_timeline(checkpoint)
        validate_stop_sign_control_target_transforms(
            resolve_checkpoint_target_transform_registry(checkpoint)
        )
        if checkpoint.get("telemetry_feature_names") != ["current_speed"]:
            raise ValueError(
                "stop-sign checkpoint telemetry_feature_names must be exactly ['current_speed']"
            )
        state_input_config_from_metadata(checkpoint.get("state_inputs"))
    except ValueError as exc:
        raise ValueError(f"{exc} checkpoint={checkpoint_path}") from exc
    return checkpoint


def resolve_checkpoint_width_multiplier(checkpoint: dict[str, Any]) -> float:
    model_metadata = checkpoint.get("model", {})
    if isinstance(model_metadata, dict) and model_metadata.get("width_multiplier") is not None:
        return float(model_metadata["width_multiplier"])
    if checkpoint.get("width_multiplier") is not None:
        return float(checkpoint["width_multiplier"])

    state_dict = checkpoint.get("model_state_dict")
    if isinstance(state_dict, Mapping):
        stem_weight = state_dict.get("stem.conv.weight")
        if isinstance(stem_weight, torch.Tensor) and stem_weight.ndim >= 1 and int(stem_weight.shape[0]) > 0:
            return float(stem_weight.shape[0]) / 64.0
    return DEFAULT_WIDTH_MULTIPLIER


def resolve_checkpoint_dropout(checkpoint: dict[str, Any], *, fallback: float = 0.1) -> float:
    model_metadata = checkpoint.get("model", {})
    if isinstance(model_metadata, dict) and model_metadata.get("dropout") is not None:
        return float(model_metadata["dropout"])
    return float(fallback)


def resolve_checkpoint_visual_temporal(checkpoint: dict[str, Any]) -> dict[str, Any]:
    model_metadata = checkpoint.get("model", {})
    raw = model_metadata.get("visual_temporal", {}) if isinstance(model_metadata, dict) else {}
    if not isinstance(raw, dict):
        raw = {}
    return {
        "enabled": bool(raw.get("enabled", True)),
        "type": str(raw.get("type", "gru")).strip().lower(),
        "hidden_dim": int(raw.get("hidden_dim", 256)),
        "num_layers": int(raw.get("num_layers", 1)),
        "bidirectional": bool(raw.get("bidirectional", False)),
        "dropout": float(raw.get("dropout", 0.0)),
        "image_order": str(raw.get("image_order", "oldest_to_newest")),
    }


def resolve_checkpoint_horizon_decoder(checkpoint: dict[str, Any]) -> dict[str, Any]:
    model_metadata = checkpoint.get("model", {})
    raw = model_metadata.get("horizon_decoder", {}) if isinstance(model_metadata, dict) else {}
    if not isinstance(raw, dict):
        raw = {}
    return {
        "enabled": bool(raw.get("enabled", True)),
        "horizon_embed_dim": int(raw.get("horizon_embed_dim", 32)),
        "hidden_dim": int(raw.get("hidden_dim", 256)),
        "num_layers": int(raw.get("num_layers", 2)),
        "dropout": float(raw.get("dropout", 0.1)),
    }


def resolve_checkpoint_aux_target_names(checkpoint: dict[str, Any], *, future_steps: int) -> list[str]:
    del future_steps
    names = checkpoint.get("aux_target_names")
    if not isinstance(names, list):
        raise ValueError("stop-sign checkpoint must include aux_target_names")
    normalized = tuple(str(name).strip() for name in names if str(name).strip())
    if normalized != STOP_SIGN_AUX_TARGET_NAMES:
        raise ValueError(
            "checkpoint aux_target_names must exactly match the stop-sign diagnostic contract: "
            f"expected={list(STOP_SIGN_AUX_TARGET_NAMES)} actual={list(normalized)}"
        )
    return list(normalized)


def resolve_checkpoint_target_names(checkpoint: dict[str, Any], *, future_steps: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    future_offsets = checkpoint.get("future_offsets") or [1, 2, 3, 4, 5, 6]
    control_target_names = resolve_checkpoint_control_target_names(
        checkpoint,
        future_steps=len(future_offsets) if future_steps <= 0 else future_steps,
    )
    aux_target_names = resolve_checkpoint_aux_target_names(
        checkpoint,
        future_steps=len(future_offsets) if future_steps <= 0 else future_steps,
    )
    return tuple(control_target_names), tuple(aux_target_names)


def resolve_checkpoint_target_transform_registry(checkpoint: dict[str, Any]) -> dict[str, TargetTransform]:
    future_steps = len(checkpoint.get("future_offsets") or [1, 2, 3, 4, 5, 6])
    control_target_names, aux_target_names = resolve_checkpoint_target_names(checkpoint, future_steps=future_steps)
    target_names = tuple(control_target_names) + tuple(aux_target_names)
    transforms = resolve_checkpoint_target_transforms(checkpoint, target_names)
    validate_stop_sign_control_target_transforms(transforms)
    return transforms


def build_model(checkpoint: dict[str, Any], device: torch.device, frame_count: int) -> StopSignTemporalPlanner:
    validate_checkpoint_control_contract(checkpoint)
    timeline = resolve_checkpoint_timeline(checkpoint)
    if frame_count != len(timeline.image_offsets):
        raise ValueError(
            "model frame_count must match checkpoint image_offsets: "
            f"frame_count={frame_count} image_offsets={len(timeline.image_offsets)}"
        )
    planner_format = str(checkpoint.get("planner_format", "")).strip()
    width_multiplier = resolve_checkpoint_width_multiplier(checkpoint)
    if planner_format == PLANNER_FORMAT:
        telemetry_feature_names = checkpoint.get("telemetry_feature_names")
        if telemetry_feature_names != ["current_speed"]:
            raise ValueError(
                "stop-sign checkpoint telemetry_feature_names must be exactly ['current_speed']"
            )
        control_target_names = resolve_checkpoint_control_target_names(
            checkpoint,
            future_steps=len(timeline.future_offsets),
        )
        aux_target_names = resolve_checkpoint_aux_target_names(
            checkpoint,
            future_steps=len(timeline.future_offsets),
        )
        state_input_config = state_input_config_from_metadata(checkpoint.get("state_inputs"))
        model_metadata = checkpoint.get("model", {})
        telemetry_hidden_dim = 128
        if isinstance(model_metadata, dict) and model_metadata.get("telemetry_hidden_dim") is not None:
            telemetry_hidden_dim = int(model_metadata["telemetry_hidden_dim"])
        dropout = resolve_checkpoint_dropout(checkpoint)
        visual_temporal = resolve_checkpoint_visual_temporal(checkpoint)
        horizon_decoder = resolve_checkpoint_horizon_decoder(checkpoint)
        model = StopSignTemporalPlanner(
            frame_count=frame_count,
            telemetry_feature_dim=len(telemetry_feature_names),
            telemetry_hidden_dim=telemetry_hidden_dim,
            telemetry_sequence_length=len(timeline.telemetry_offsets),
            horizon=len(timeline.future_offsets),
            control_dim=len(control_target_names),
            control_target_names=tuple(control_target_names),
            aux_dim=len(aux_target_names),
            state_input_dim=len(state_input_config.enabled_keys()),
            width_multiplier=width_multiplier,
            dropout=dropout,
            visual_temporal_enabled=bool(visual_temporal["enabled"]),
            visual_temporal_type=str(visual_temporal["type"]),
            visual_temporal_hidden_dim=int(visual_temporal["hidden_dim"]),
            visual_temporal_num_layers=int(visual_temporal["num_layers"]),
            visual_temporal_bidirectional=bool(visual_temporal["bidirectional"]),
            visual_temporal_dropout=float(visual_temporal["dropout"]),
            horizon_decoder_enabled=bool(horizon_decoder["enabled"]),
            horizon_embed_dim=int(horizon_decoder["horizon_embed_dim"]),
            horizon_decoder_hidden_dim=int(horizon_decoder["hidden_dim"]),
            horizon_decoder_num_layers=int(horizon_decoder["num_layers"]),
            horizon_decoder_dropout=float(horizon_decoder["dropout"]),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.state_input_config = state_input_config
        model.eval()
        return model

    raise ValueError(LEGACY_SCALAR_HEAD_ERROR)


def resolve_checkpoint_control_target_names(checkpoint: dict[str, Any], *, future_steps: int) -> list[str]:
    names = checkpoint.get("control_target_names")
    if not isinstance(names, list):
        raise ValueError("stop-sign checkpoint must include control_target_names")
    return list(require_stop_sign_control_target_names(
        names,
        source="checkpoint control_target_names",
    ))


def resolve_checkpoint_frame_count(checkpoint: dict[str, Any], config: InferenceConfig) -> int:
    timeline_frame_count = len(resolve_checkpoint_timeline(checkpoint).image_offsets)
    checkpoint_frame_count = checkpoint.get("frame_window_size")
    expected = timeline_frame_count if checkpoint_frame_count is None else int(checkpoint_frame_count)
    if expected != config.window_size:
        raise ValueError(
            "Config/checkpoint frame-window mismatch: "
            f"config dataset.window_size={config.window_size}, "
            f"checkpoint frame_window_size={expected}"
        )
    return expected


def resolve_checkpoint_frame_stride(checkpoint: dict[str, Any], config: InferenceConfig) -> int:
    checkpoint_frame_stride = checkpoint.get("frame_stride", checkpoint.get("frame_window_stride"))
    if checkpoint_frame_stride is not None:
        expected = int(checkpoint_frame_stride)
        if expected != config.frame_stride:
            raise ValueError(
                "Config/checkpoint frame-stride mismatch: "
                f"config dataset.frame_stride={config.frame_stride}, "
                f"checkpoint frame_stride={expected}"
            )
        return expected
    return config.frame_stride


def checkpoint_sample_stride(checkpoint: dict[str, Any]) -> int:
    return int(checkpoint.get("sample_stride", 0) or 0)


def run_sample_inference(
    model: StopSignTemporalPlanner,
    device: torch.device,
    *,
    data_root: str,
    run_id: str,
    sample_index: int,
    image_size: tuple[int, int],
    expected_window_size: int,
    timeline: CheckpointTimeline,
    control_target_names: tuple[str, ...],
    aux_target_names: tuple[str, ...],
    target_transforms: dict[str, TargetTransform] | None,
) -> dict[str, Any]:
    dataset = FsdDataset(
        run_id=run_id,
        data_root=data_root,
        image_size=image_size,
        expected_window_size=expected_window_size,
        image_offsets=timeline.image_offsets,
        telemetry_offsets=timeline.telemetry_offsets,
        future_offsets=timeline.future_offsets,
        telemetry_sample_interval_ms=timeline.telemetry_sample_interval_ms,
        control_target_names=control_target_names,
        aux_target_names=aux_target_names,
        target_transforms=target_transforms,
        state_input_config=getattr(model, "state_input_config", None),
    )
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample_index out of range: {sample_index}, dataset size={len(dataset)}")

    images, telemetry, state_inputs, target_controls, target_aux = dataset[sample_index]
    sample_meta = dataset._load_sample(sample_index)
    images = images.unsqueeze(0).to(device, non_blocking=device.type == "cuda")
    telemetry = telemetry.unsqueeze(0).to(device, non_blocking=device.type == "cuda")
    state_inputs = state_inputs.unsqueeze(0).to(device, non_blocking=device.type == "cuda")

    with torch.no_grad():
        output = model(images, telemetry, state_inputs)

    pred_controls = denormalize_target_tensor(
        output["pred_controls"],
        control_target_names,
        target_transforms or {},
    )
    pred_aux = denormalize_target_tensor(
        output["pred_aux"],
        aux_target_names,
        target_transforms or {},
    )
    denorm_target_controls = denormalize_target_tensor(
        target_controls,
        control_target_names,
        target_transforms or {},
    )
    denorm_target_aux = denormalize_target_tensor(
        target_aux,
        aux_target_names,
        target_transforms or {},
    )

    return {
        "sample_index": sample_index,
        "frame_paths": [str(path) for path in sample_meta.get("frame_paths", [])],
        "state_inputs": {
            definition.camel_key: float(state_inputs[0, index].item())
            for index, key in enumerate(dataset.state_input_config.enabled_keys())
            for definition in [state_input_definition(key)]
        },
        "pred_controls": pred_controls.detach().cpu().tolist(),
        "motion_plan": {
            "future_speed_mps": pred_controls[..., 0].detach().cpu().tolist(),
            "stop_intent": pred_controls[..., 1].detach().cpu().tolist(),
            "release_policy": release_policy_metadata(),
        },
        "pred_aux": pred_aux.detach().cpu().tolist(),
        "target_controls": denorm_target_controls.detach().cpu().tolist(),
        "target_aux": denorm_target_aux.detach().cpu().tolist(),
        "pred_controls_normalized": output["pred_controls"].detach().cpu().tolist(),
        "pred_aux_normalized": output["pred_aux"].detach().cpu().tolist(),
        "target_controls_normalized": target_controls.detach().cpu().tolist(),
        "target_aux_normalized": target_aux.detach().cpu().tolist(),
        "control_target_names": list(control_target_names),
        "aux_target_names": list(aux_target_names),
    }


def build_output(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    device: torch.device,
    sample_result: dict[str, Any] | None,
) -> dict[str, Any]:
    timeline = resolve_checkpoint_timeline(checkpoint)
    checkpoint_image_size = resolve_checkpoint_image_size(checkpoint)
    control_target_names, aux_target_names = resolve_checkpoint_target_names(
        checkpoint,
        future_steps=len(timeline.future_offsets),
    )
    target_transforms = resolve_checkpoint_target_transform_registry(checkpoint)
    output: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "epoch": int(checkpoint.get("epoch", 0)),
        "planner_format": PLANNER_FORMAT,
        "planner_format_version": PLANNER_FORMAT_VERSION,
        "control_contract": control_contract_metadata(),
        "release_policy": release_policy_metadata(),
        "frame_window_size": int(checkpoint.get("frame_window_size", 0) or 0),
        "frame_stride": int(checkpoint.get("frame_stride", checkpoint.get("frame_window_stride", 0)) or 0),
        "sample_stride": checkpoint_sample_stride(checkpoint),
        "input_channels": int(checkpoint.get("input_channels", 0) or 0),
        "image_size": {
            "width": checkpoint_image_size[0],
            "height": checkpoint_image_size[1],
        },
        "model": {
            "width_multiplier": resolve_checkpoint_width_multiplier(checkpoint),
            "dropout": resolve_checkpoint_dropout(checkpoint),
            "visual_temporal": resolve_checkpoint_visual_temporal(checkpoint),
            "horizon_decoder": resolve_checkpoint_horizon_decoder(checkpoint),
        },
        "target_transforms": target_transform_metadata(target_transforms),
        "state_inputs": checkpoint.get("state_inputs", state_inputs_metadata(state_input_config_from_metadata(None))),
        "image_offsets": list(timeline.image_offsets),
        "telemetry_offsets": list(timeline.telemetry_offsets),
        "future_offsets": list(timeline.future_offsets),
        "telemetry_sample_interval_ms": timeline.telemetry_sample_interval_ms,
        "control_horizon_dt_ms": list(resolve_checkpoint_control_horizon_dt_ms(checkpoint)),
        "telemetry_feature_names": list(checkpoint.get("telemetry_feature_names", [])),
        "control_target_names": list(control_target_names),
        "aux_target_names": list(aux_target_names),
        "train_metrics": checkpoint.get("train_metrics"),
        "val_metrics": checkpoint.get("val_metrics"),
    }
    if sample_result is not None:
        output["sample"] = sample_result
    return output


def print_plain(output: dict[str, Any]) -> None:
    print(f"Loaded checkpoint: {output['checkpoint']}")
    print(f"Device: {output['device']}")
    print(f"Epoch: {output['epoch']}")
    if int(output.get("frame_window_size", 0) or 0) > 0:
        print(
            "Frame window: "
            f"size={output['frame_window_size']} "
            f"frame_stride={output['frame_stride']} "
            f"sample_stride={output['sample_stride']} "
            f"input_channels={output['input_channels']}"
        )
    model_metadata = output.get("model") or {}
    if isinstance(model_metadata, dict):
        print(
            "Model: "
            f"width_multiplier={float(model_metadata.get('width_multiplier', DEFAULT_WIDTH_MULTIPLIER)):.3f} "
            f"dropout={float(model_metadata.get('dropout', 0.0)):.3f}"
        )
        visual_temporal = model_metadata.get("visual_temporal")
        if isinstance(visual_temporal, dict):
            print(
                "Visual temporal: "
                f"enabled={bool(visual_temporal.get('enabled', True))} "
                f"type={visual_temporal.get('type', 'gru')} "
                f"hidden_dim={int(visual_temporal.get('hidden_dim', 256))} "
                f"image_order={visual_temporal.get('image_order', 'oldest_to_newest')}"
            )
        horizon_decoder = model_metadata.get("horizon_decoder")
        if isinstance(horizon_decoder, dict):
            print(
                "Horizon decoder: "
                f"enabled={bool(horizon_decoder.get('enabled', True))} "
                f"horizon_embed_dim={int(horizon_decoder.get('horizon_embed_dim', 32))} "
                f"hidden_dim={int(horizon_decoder.get('hidden_dim', 256))} "
                f"num_layers={int(horizon_decoder.get('num_layers', 2))}"
            )
    state_inputs = output.get("state_inputs") or {}
    if isinstance(state_inputs, dict):
        for name, item in state_inputs.items():
            if not isinstance(item, dict) or not item.get("enabled"):
                continue
            description = f"State input: {name}"
            if "cap" in item:
                description += f" cap={float(item.get('cap', 0.0)):.3f}"
            print(description)

    val_metrics = output.get("val_metrics") or {}
    if isinstance(val_metrics, dict) and val_metrics:
        print(
            "Validation metrics: "
            f"loss={float(val_metrics.get('loss', 0.0)):.6f} "
            f"control_mae={float(val_metrics.get('control_overall_mae', val_metrics.get('overall_mae', 0.0))):.6f} "
            f"control_rmse={float(val_metrics.get('control_overall_rmse', val_metrics.get('overall_rmse', 0.0))):.6f}"
        )

    sample = output.get("sample")
    if isinstance(sample, dict):
        print()
        print(f"Sample index: {sample['sample_index']}")
        print(f"Control target names: {sample.get('control_target_names', [])}")
        print(f"Aux target names: {sample.get('aux_target_names', [])}")
        print(f"Pred controls: {sample.get('pred_controls')}")
        print(f"Pred aux:      {sample.get('pred_aux')}")
        print(f"Target controls: {sample.get('target_controls')}")
        print(f"Target aux:      {sample.get('target_aux')}")
        frame_paths = sample.get("frame_paths") or []
        if frame_paths:
            print("Frames:")
            for frame_path in frame_paths:
                print(f"  {frame_path}")
        return


def main() -> None:
    args = parse_args()
    config = resolve_config(args)
    device = select_device(config.device)
    checkpoint_path = resolve_existing_path(config.checkpoint, args.config)
    checkpoint = load_checkpoint(checkpoint_path, device)
    timeline = resolve_checkpoint_timeline(checkpoint)
    image_size = require_checkpoint_image_size(
        checkpoint,
        (config.image_width, config.image_height),
    )
    frame_count = resolve_checkpoint_frame_count(checkpoint, config)
    _ = resolve_checkpoint_frame_stride(checkpoint, config)
    model = build_model(checkpoint, device, frame_count)
    control_target_names, aux_target_names = resolve_checkpoint_target_names(
        checkpoint,
        future_steps=len(timeline.future_offsets),
    )
    target_transforms = resolve_checkpoint_target_transform_registry(checkpoint)

    sample_result: dict[str, Any] | None = None
    if not config.metadata_only and config.run_id:
        if not config.data_root:
            raise ValueError("inference.data_root is required when inference.run_id is configured")
        sample_result = run_sample_inference(
            model,
            device,
            data_root=config.data_root,
            run_id=config.run_id,
            sample_index=config.sample_index,
            image_size=image_size,
            expected_window_size=frame_count,
            timeline=timeline,
            control_target_names=control_target_names,
            aux_target_names=aux_target_names,
            target_transforms=target_transforms,
        )

    output = build_output(checkpoint_path, checkpoint, device, sample_result)
    if config.output_json:
        print(json.dumps(output, indent=2))
    else:
        print_plain(output)


if __name__ == "__main__":
    main()
