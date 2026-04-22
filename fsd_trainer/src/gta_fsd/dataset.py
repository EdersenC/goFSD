"""Helpers for loading processed trip datasets from one or more run directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor
from torch.utils.data import Dataset

from heads import (
    CURRENT_SPEED_TARGET_KEY,
    FUTURE_HORIZON_SECONDS_TARGET_KEY,
    HEAD_SPECS,
    HEAD_SPECS_BY_NAME,
    HeadSpec,
    build_targets_from_label,
)
from image_io import load_rgb_tensor_from_path
from state_inputs import StateInputConfig, build_state_inputs_from_label, training_state_input_config


DatasetTargets = dict[str, Tensor]
DatasetStateInputs = dict[str, Tensor]


class Trip:
    def __init__(self, trip_dir: str | Path, *, run_path: str | Path):
        self.trip_dir = Path(trip_dir)
        self.run_path = Path(run_path)
        self.sample_indices: list[int] = []

    @property
    def trip_key(self) -> str:
        relative_trip_dir = self.trip_dir.relative_to(self.run_path)
        return (Path(self.run_path.name) / relative_trip_dir).as_posix()

    def load_data(self) -> dict[str, Any]:
        metadata_path = self.trip_dir / "metadata.json"
        video_path = self.trip_dir / "video.mkv"
        log_path = self.trip_dir / "video.log"
        dataset_path = self.trip_dir / "dataset.jsonl"
        processing_path = self.trip_dir / "processing.json"

        if not self.trip_dir.is_dir():
            raise FileNotFoundError(f"Trip directory does not exist: {self.trip_dir}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Trip metadata is missing: {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        return {
            "trip_dir": self.trip_dir,
            "trip_name": self.trip_dir.name,
            "trip_key": self.trip_key,
            "run_path": self.run_path,
            "run_name": self.run_path.name,
            "metadata": metadata,
            "metadata_path": metadata_path,
            "video_path": video_path,
            "log_path": log_path,
            "dataset_path": dataset_path,
            "processing_path": processing_path,
            "samples": self.load_samples(),
            "sample_indices": list(self.sample_indices),
            "run_id": metadata.get("runId"),
            "scene_id": metadata.get("sceneId"),
            "scene_variant": metadata.get("sceneVariant"),
            "trip_index": metadata.get("tripIndex"),
        }

    def load_samples(self) -> list[dict[str, Any]]:
        dataset_path = self.trip_dir / "dataset.jsonl"
        if not dataset_path.is_file():
            return []

        samples: list[dict[str, Any]] = []
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                sample["trip_dir"] = self.trip_dir
                sample["run_path"] = self.run_path
                sample["run_name"] = self.run_path.name
                sample["trip_key"] = self.trip_key
                sample["frame_paths"] = [self.trip_dir / Path(path) for path in sample.get("frame_paths", [])]
                samples.append(sample)
        return samples


class FsdDataset(Dataset[tuple[Tensor, DatasetStateInputs, DatasetTargets]]):
    def __init__(
        self,
        run_paths: Iterable[str | Path] | None = None,
        *,
        run_id: str | None = None,
        data_root: str | Path | None = None,
        image_size: tuple[int, int] = (480, 480),
        expected_window_size: int = 3,
        state_input_config: StateInputConfig | None = None,
        head_specs: tuple[HeadSpec, ...] = HEAD_SPECS,
    ):
        if expected_window_size < 1:
            raise ValueError("expected_window_size must be > 0")

        self.expected_window_size = expected_window_size
        self.image_size = image_size
        self.state_input_config = state_input_config or training_state_input_config()
        self.head_specs = head_specs
        self.data_root = None if data_root is None else Path(data_root)
        self.run_paths: list[Path] = self._resolve_run_paths(run_paths, run_id=run_id, data_root=data_root)
        self.trips: list[Trip] = self._load_trips()
        self.samples: list[dict[str, Any]] = self._load_samples()
        self._scalar_target_cache: dict[str, tuple[float, ...]] = {}

    def _resolve_run_paths(
        self,
        run_paths: Iterable[str | Path] | None,
        *,
        run_id: str | None,
        data_root: str | Path | None,
    ) -> list[Path]:
        resolved: list[Path] = []
        if run_paths is not None:
            for raw_path in run_paths:
                path = Path(raw_path)
                if not str(path).strip():
                    raise ValueError("run_paths entries must be non-empty")
                resolved.append(path)

        if resolved:
            return resolved

        if run_id is None or data_root is None:
            raise ValueError("Provide run_paths or both run_id and data_root")
        return [Path(data_root) / "runs" / run_id]

    def _resolve_scene_dirs(self, run_path: Path) -> list[Path]:
        if not run_path.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {run_path}")

        scene_dirs = sorted(path for path in run_path.iterdir() if path.is_dir())
        if not scene_dirs:
            raise FileNotFoundError(f"No scene directory found in run: {run_path}")
        return scene_dirs

    def _load_trips(self) -> list[Trip]:
        trips: list[Trip] = []
        for run_path in self.run_paths:
            for scene_dir in self._resolve_scene_dirs(run_path):
                trip_dirs = sorted(
                    path
                    for path in scene_dir.iterdir()
                    if path.is_dir() and path.name.startswith("trip-")
                )
                for trip_dir in trip_dirs:
                    trips.append(Trip(trip_dir, run_path=run_path))
        return trips

    def _load_samples(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for trip in self.trips:
            start_index = len(samples)
            trip_samples = trip.load_samples()
            samples.extend(trip_samples)
            trip.sample_indices = list(range(start_index, len(samples)))
        return samples

    @property
    def trip_count(self) -> int:
        return len(self.trips)

    def trip_sample_indices(self) -> list[list[int]]:
        return [list(trip.sample_indices) for trip in self.trips]

    def load_trip_data(self, trip_index: int) -> dict[str, Any]:
        if trip_index < 0 or trip_index >= len(self.trips):
            raise IndexError(f"Trip index out of range: {trip_index}")
        return self.trips[trip_index].load_data()

    def __len__(self) -> int:
        return len(self.samples)

    def scalar_target_values(self, head_name: str) -> tuple[float, ...]:
        cached = self._scalar_target_cache.get(head_name)
        if cached is not None:
            return cached

        spec = next((item for item in self.head_specs if item.name == head_name), None)
        if spec is None:
            spec = HEAD_SPECS_BY_NAME.get(head_name)
        if spec is None:
            raise KeyError(f"unknown head target: {head_name}")

        values: list[float] = []
        for sample in self.samples:
            target = spec.target_builder(sample["label"]).detach()
            if target.numel() != 1:
                raise ValueError(
                    f"head '{head_name}' target_builder must produce a scalar per sample, got shape {tuple(target.shape)}"
                )
            values.append(float(target.reshape(()).item()))

        resolved = tuple(values)
        self._scalar_target_cache[head_name] = resolved
        return resolved

    def _load_frame_tensor(self, frame_path: Path) -> Tensor:
        return load_rgb_tensor_from_path(frame_path, self.image_size)

    def __getitem__(self, index: int) -> tuple[Tensor, DatasetStateInputs, DatasetTargets]:
        sample = self.samples[index]
        frame_paths = sample.get("frame_paths", [])
        if not frame_paths:
            raise ValueError(f"No frame paths found for sample at index {index}")
        if len(frame_paths) != self.expected_window_size:
            raise ValueError(
                "Expected "
                f"{self.expected_window_size} frame paths for sample at index {index}, "
                f"found {len(frame_paths)}"
            )

        frame_tensors: list[Tensor] = []
        for frame_path in frame_paths:
            if not frame_path.is_file():
                raise FileNotFoundError(f"Frame image not found: {frame_path}")
            frame_tensors.append(self._load_frame_tensor(frame_path))

        label = sample["label"]
        x = torch.cat(frame_tensors, dim=0)
        state_inputs = build_state_inputs_from_label(label, self.state_input_config)
        targets = build_targets(label, head_specs=self.head_specs)
        return x, state_inputs, targets


def build_targets(
    label: dict[str, Any],
    *,
    steering: float | None = None,
    delta_speed: float | None = None,
    head_specs: tuple[HeadSpec, ...] = HEAD_SPECS,
) -> DatasetTargets:
    del steering, delta_speed
    targets = build_targets_from_label(label, head_specs=head_specs)
    if FUTURE_HORIZON_SECONDS_TARGET_KEY in label:
        targets[FUTURE_HORIZON_SECONDS_TARGET_KEY] = torch.tensor(
            float(label[FUTURE_HORIZON_SECONDS_TARGET_KEY]),
            dtype=torch.float32,
        )
    if "currentSpeed" in label:
        targets[CURRENT_SPEED_TARGET_KEY] = torch.tensor(
            float(label["currentSpeed"]),
            dtype=torch.float32,
        )
    return targets
