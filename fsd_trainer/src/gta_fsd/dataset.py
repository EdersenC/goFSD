"""Helpers for loading processed trip datasets from a single run."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


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


class FsdDataset:
    def __init__(self, run_id: str, data_root: str | Path):
        self.data_root = Path(data_root)
        self.run_id = run_id
        self.run_dir = self.data_root / "runs" / run_id
        self.scene_dir = self._resolve_scene_dir()
        self.trips = self._load_trips()
        self.samples = self._load_samples()

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

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def main():
    default_data_root = r"S:\fsd_fivem_data" if os.name == "nt" else "/mnt/s/fsd_fivem_data"
    configured_root = os.environ.get("FSD_DATA_ROOT", default_data_root).strip().strip("\"'")
    if os.name == "nt" and len(configured_root) >= 2 and configured_root[1] == ":":
        if len(configured_root) == 2:
            configured_root += "\\"
        elif configured_root[2] not in ("\\", "/"):
            configured_root = f"{configured_root[:2]}\\{configured_root[2:]}"
    data_root = Path(configured_root)

    dataset = FsdDataset(run_id="2026-04-12_09-13-44PM_7zspe7", data_root=data_root)
    print(f"Found {len(dataset.trips)} trips in the run.")
    print(f"Found {len(dataset)} processed samples in the dataset.")

    for i, trip in enumerate(dataset.trips):
        trip_data = dataset.load_trip_data(i)
        print(f"Trip {i} dataset path: {trip_data['dataset_path']}")
        print(f"Trip {i} sample count: {len(trip_data['samples'])}")


if __name__ == "__main__":
    main()
