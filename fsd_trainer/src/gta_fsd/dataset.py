"""Helpers for loading processed trip datasets from a single run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class Trip:
    def __init__(self, trip_dir: str | Path):
        self.trip_dir = Path(trip_dir)

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
            "metadata": metadata,
            "metadata_path": metadata_path,
            "video_path": video_path,
            "log_path": log_path,
            "dataset_path": dataset_path,
            "processing_path": processing_path,
            "samples": self.load_samples(),
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
                sample["frame_paths"] = [self.trip_dir / Path(path) for path in sample.get("frame_paths", [])]
                samples.append(sample)
        return samples


class FsdDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, run_id: str, data_root: str | Path, image_size: tuple[int, int] = (224, 224)):
        self.data_root: Path = Path(data_root)
        self.run_id: str = run_id
        self.run_dir: Path = self.data_root / "runs" / run_id
        self.scene_dir: Path = self._resolve_scene_dir()
        self.trips: list[Trip] = self._load_trips()
        self.samples: list[dict[str, Any]] = self._load_samples()
        self.transform: Callable[[Image.Image], Tensor] = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

    def _resolve_scene_dir(self) -> Path:
        if not self.run_dir.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {self.run_dir}")

        scene_dirs = sorted(path for path in self.run_dir.iterdir() if path.is_dir())
        if not scene_dirs:
            raise FileNotFoundError(f"No scene directory found in run: {self.run_dir}")
        if len(scene_dirs) > 1:
            raise ValueError(
                f"Expected one scene directory in {self.run_dir}, found {len(scene_dirs)}"
            )

        return scene_dirs[0]

    def _load_trips(self) -> list[Trip]:
        trip_dirs = sorted(
            path
            for path in self.scene_dir.iterdir()
            if path.is_dir() and path.name.startswith("trip-")
        )
        return [Trip(trip_dir) for trip_dir in trip_dirs]

    def _load_samples(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for trip in self.trips:
            samples.extend(trip.load_samples())
        return samples

    def load_trip_data(self, trip_index: int) -> dict[str, Any]:
        if trip_index < 0 or trip_index >= len(self.trips):
            raise IndexError(f"Trip index out of range: {trip_index}")
        return self.trips[trip_index].load_data()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample = self.samples[index]
        frame_paths = sample.get("frame_paths", [])
        if not frame_paths:
            raise ValueError(f"No frame paths found for sample at index {index}")
        if len(frame_paths) != 3:
            raise ValueError(f"Expected 3 frame paths for sample at index {index}, found {len(frame_paths)}")

        frame_tensors: list[Tensor] = []
        for frame_path in frame_paths:
            if not frame_path.is_file():
                raise FileNotFoundError(f"Frame image not found: {frame_path}")
            image = Image.open(frame_path).convert("RGB")
            image_tensor = self.transform(image)
            frame_tensors.append(image_tensor)

        label = sample["label"]
        steering = float(label["Steering"])
        acceleration = float(label["acceleration"])
        x = torch.cat(frame_tensors, dim=0)
        y = torch.tensor([steering, acceleration], dtype=torch.float32)
        return x, y


