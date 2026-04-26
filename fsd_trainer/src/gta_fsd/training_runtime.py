from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path
from typing import Any

from state_inputs import (
    ACTIVE_HEAD_NAMES,
    DEFAULT_WIDTH_MULTIPLIER,
    STATE_INPUT_DEFINITIONS,
    STATE_INPUT_DEFINITIONS_BY_CAMEL,
)


ALLOWED_LOSS_WEIGHT_KEYS = [
    "future_yaw_delta",
    "future_speed",
    "move_intent",
    "delta_speed",
    "yaw_rate",
]

ALLOWED_CONSISTENCY_KEYS = [
    "yaw_delta_vs_yaw_rate_weight",
    "yaw_rate_scale_to_degrees",
    "future_speed_vs_delta_speed_weight",
]

ALLOWED_STATE_INPUT_HEADS = list(ACTIVE_HEAD_NAMES)


class TrainingJobError(RuntimeError):
    pass


class TrainingJobNotFoundError(TrainingJobError):
    pass


class TrainingJobNotPendingError(TrainingJobError):
    pass


class TrainingJobNotActiveError(TrainingJobError):
    pass


class TrainingJobNotTerminalError(TrainingJobError):
    pass


class TrainingJobNotRequeueableError(TrainingJobError):
    pass


class TrainingJobRequestError(TrainingJobError):
    pass


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time() % 1) * 1_000_000_000):09d}Z"


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _clone_job(job: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(job)


def _sort_jobs_by_created_desc(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(jobs, key=lambda item: str(item.get("createdAt", "")), reverse=True)


def _is_terminal_status(status: str) -> bool:
    return status in {"completed", "failed", "canceled", "stopped"}


def _tail_text(text: str, tail_lines: int) -> str:
    if tail_lines <= 0:
        return text
    lines = text.splitlines()
    if len(lines) <= tail_lines:
        return text
    return "\n".join(lines[-tail_lines:]) + "\n"


def _slugify_job_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", name.strip().lower())
    cleaned = cleaned.strip("-")
    return cleaned or "train"


def _normalize_runtime_path(raw_value: str) -> Path:
    cleaned = str(raw_value).strip().strip("\"'")
    if os.name == "nt" and len(cleaned) >= 2 and cleaned[1] == ":":
        if len(cleaned) == 2:
            cleaned += "\\"
        elif len(cleaned) > 2 and cleaned[2] not in ("\\", "/"):
            cleaned = f"{cleaned[:2]}\\{cleaned[2:]}"
    if os.name != "nt" and len(cleaned) >= 3 and cleaned[1] == ":" and cleaned[2] in ("\\", "/"):
        drive = cleaned[0].lower()
        rest = cleaned[2:].replace("\\", "/")
        return Path(f"/mnt/{drive}{rest}")
    return Path(cleaned)


def _normalize_turn_oversampling_thresholds(raw_map: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(raw_map)
    light = normalized.get("light_turn_threshold")
    medium = normalized.get("medium_turn_threshold")
    sharp = normalized.get("sharp_turn_threshold")
    if isinstance(light, (int, float)) and isinstance(medium, (int, float)) and float(medium) < float(light):
        normalized["medium_turn_threshold"] = float(light)
        medium = float(light)
    if isinstance(medium, (int, float)) and isinstance(sharp, (int, float)) and float(sharp) < float(medium):
        normalized["sharp_turn_threshold"] = float(medium)
    return normalized


def _resolve_jobs_dir(config_path: Path) -> Path:
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    backend_training = raw.get("backend", {}).get("training", {})
    jobs_dir_raw = str(backend_training.get("jobs_dir", "")).strip()
    if jobs_dir_raw:
        jobs_dir = _normalize_runtime_path(jobs_dir_raw)
        if jobs_dir.is_absolute():
            return jobs_dir
        return (config_path.parent / jobs_dir).resolve()

    dataset_raw = raw.get("dataset", {})
    data_root_raw = str(dataset_raw.get("data_root", "")).strip()
    if data_root_raw:
        return _normalize_runtime_path(data_root_raw) / "training_jobs"
    return config_path.resolve().parent.parent / "training_jobs"


def _load_page_defaults(config_path: Path, jobs_dir: Path) -> dict[str, Any]:
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    dataset_raw = raw.get("dataset", {})
    loader_raw = raw.get("loader", {})
    training_raw = raw.get("training", {})
    model_raw = raw.get("model", {})
    loss_weights = training_raw.get("loss_weights", {})
    consistency = training_raw.get("consistency", {})
    turn_oversampling = loader_raw.get("turn_oversampling", {})
    yaw_loss_weighting = training_raw.get("yaw_loss_weighting", {})
    state_inputs = raw.get("state_inputs", {})
    state_inputs_payload: dict[str, Any] = {}
    for definition in STATE_INPUT_DEFINITIONS:
        item = state_inputs.get(definition.key, {})
        payload = {
            "enabled": bool(item.get("enabled", definition.default_enabled)),
            "heads": [str(value) for value in item.get("heads", definition.default_heads) if str(value).strip()],
        }
        if definition.default_cap is not None:
            payload["cap"] = float(item.get("cap", definition.default_cap) or definition.default_cap)
        state_inputs_payload[definition.camel_key] = payload
    return {
        "configPath": str(config_path),
        "pythonBin": sys.executable,
        "trainScript": str((Path(__file__).resolve().parent / "train.py").resolve()),
        "jobsDir": str(jobs_dir),
        "epochs": int(training_raw.get("epochs", 0) or 0),
        "learningRate": float(training_raw.get("learning_rate", 0.0) or 0.0),
        "widthMultiplier": float(model_raw.get("width_multiplier", DEFAULT_WIDTH_MULTIPLIER) or DEFAULT_WIDTH_MULTIPLIER),
        "trainRunIds": [str(value) for value in dataset_raw.get("train_run_ids", []) if str(value).strip()],
        "valRunIds": [str(value) for value in dataset_raw.get("val_run_ids", []) if str(value).strip()],
        "lossWeights": {key: float(loss_weights.get(key, 0.0) or 0.0) for key in ALLOWED_LOSS_WEIGHT_KEYS},
        "consistency": {key: float(consistency.get(key, 0.0) or 0.0) for key in ALLOWED_CONSISTENCY_KEYS},
        "turnOversampling": {
            "enabled": bool(turn_oversampling.get("enabled", False)),
            "straight_weight": float(turn_oversampling.get("straight_weight", 1.0) or 1.0),
            "light_turn_weight": float(turn_oversampling.get("light_turn_weight", 1.5) or 1.5),
            "medium_turn_weight": float(turn_oversampling.get("medium_turn_weight", 2.5) or 2.5),
            "sharp_turn_weight": float(turn_oversampling.get("sharp_turn_weight", 4.0) or 4.0),
            "light_turn_threshold": float(turn_oversampling.get("light_turn_threshold", 0.05) or 0.05),
            "medium_turn_threshold": float(turn_oversampling.get("medium_turn_threshold", 0.15) or 0.15),
            "sharp_turn_threshold": float(turn_oversampling.get("sharp_turn_threshold", 0.30) or 0.30),
        },
        "yawLossWeighting": {
            "enabled": bool(yaw_loss_weighting.get("enabled", False)),
            "base_weight": float(yaw_loss_weighting.get("base_weight", 1.0) or 1.0),
            "alpha": float(yaw_loss_weighting.get("alpha", 2.0) or 2.0),
            "tau": float(yaw_loss_weighting.get("tau", 0.25) or 0.25),
            "max_scale": float(yaw_loss_weighting.get("max_scale", 3.0) or 3.0),
        },
        "stateInputs": state_inputs_payload,
        "allowedLossWeightKeys": list(ALLOWED_LOSS_WEIGHT_KEYS),
        "allowedConsistencyKeys": list(ALLOWED_CONSISTENCY_KEYS),
        "allowedStateInputHeads": list(ALLOWED_STATE_INPUT_HEADS),
    }


def _parse_job_specs(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        specs = payload
    elif isinstance(payload, dict) and isinstance(payload.get("jobs"), list):
        specs = payload["jobs"]
    elif isinstance(payload, dict):
        specs = [payload]
    else:
        raise TrainingJobRequestError("request body must be a job object, job array, or {\"jobs\": [...]}")

    if not specs:
        raise TrainingJobRequestError("at least one training job is required")

    normalized_specs: list[dict[str, Any]] = []
    for spec in specs:
        if not isinstance(spec, dict):
            raise TrainingJobRequestError("each training job must be a JSON object")
        name = str(spec.get("name", "") or "").strip()
        notes = str(spec.get("notes", "") or "").strip()
        learning_rate = spec.get("learningRate")
        if learning_rate is not None:
            if not _is_finite_number(learning_rate) or float(learning_rate) <= 0:
                raise TrainingJobRequestError("learningRate must be a positive finite number")
            learning_rate = float(learning_rate)
        epochs = spec.get("epochs")
        if epochs is not None:
            if not _is_finite_number(epochs) or int(epochs) < 1:
                raise TrainingJobRequestError("epochs must be an integer >= 1")
            epochs = int(epochs)
        width_multiplier = spec.get("widthMultiplier")
        if width_multiplier is not None:
            if not _is_finite_number(width_multiplier) or float(width_multiplier) <= 0:
                raise TrainingJobRequestError("widthMultiplier must be a positive finite number")
            width_multiplier = float(width_multiplier)

        def normalize_float_map(raw_map: Any, field_name: str) -> dict[str, float]:
            if raw_map is None:
                return {}
            if not isinstance(raw_map, dict):
                raise TrainingJobRequestError(f"{field_name} must be an object map")
            out: dict[str, float] = {}
            for key, value in raw_map.items():
                key_name = str(key).strip()
                if not key_name or not _is_finite_number(value):
                    raise TrainingJobRequestError(f"{field_name} must only contain finite numeric values")
                out[key_name] = float(value)
            return out

        def normalize_string_list(raw_list: Any, field_name: str) -> list[str] | None:
            if raw_list is None:
                return None
            if not isinstance(raw_list, list):
                raise TrainingJobRequestError(f"{field_name} must be an array of run ids")
            out: list[str] = []
            for value in raw_list:
                item = str(value).strip()
                if item:
                    out.append(item)
            return out

        def normalize_bool(raw_value: Any, field_name: str) -> bool:
            if isinstance(raw_value, bool):
                return raw_value
            raise TrainingJobRequestError(f"{field_name} must be true or false")

        def normalize_optional_positive_float(raw_value: Any, field_name: str) -> float:
            if not _is_finite_number(raw_value) or float(raw_value) <= 0:
                raise TrainingJobRequestError(f"{field_name} must be a positive finite number")
            return float(raw_value)

        def normalize_toggle_map(raw_map: Any, field_name: str) -> dict[str, Any]:
            if raw_map is None:
                return {}
            if not isinstance(raw_map, dict):
                raise TrainingJobRequestError(f"{field_name} must be an object map")
            normalized: dict[str, Any] = {}
            for key, value in raw_map.items():
                key_name = str(key).strip()
                if not key_name:
                    raise TrainingJobRequestError(f"{field_name} contains an invalid key")
                if key_name == "enabled":
                    normalized[key_name] = normalize_bool(value, f"{field_name}.enabled")
                else:
                    normalized[key_name] = normalize_optional_positive_float(value, f"{field_name}.{key_name}")
            return normalized

        def normalize_turn_oversampling(raw_map: Any) -> dict[str, Any]:
            normalized = normalize_toggle_map(raw_map, "turnOversampling")
            return _normalize_turn_oversampling_thresholds(normalized)

        def normalize_state_inputs(raw_map: Any) -> dict[str, Any]:
            if raw_map is None:
                return {}
            if not isinstance(raw_map, dict):
                raise TrainingJobRequestError("stateInputs must be an object map")
            normalized: dict[str, Any] = {}
            for source_key, value in raw_map.items():
                source_name = str(source_key).strip()
                definition = STATE_INPUT_DEFINITIONS_BY_CAMEL.get(source_name)
                if definition is None:
                    raise TrainingJobRequestError(
                        f"stateInputs only supports {', '.join(item.camel_key for item in STATE_INPUT_DEFINITIONS)}"
                    )
                if not isinstance(value, dict):
                    raise TrainingJobRequestError(f"stateInputs.{source_name} must be an object")
                source_config: dict[str, Any] = {}
                if "enabled" in value:
                    source_config["enabled"] = normalize_bool(value["enabled"], f"stateInputs.{source_name}.enabled")
                if "cap" in value:
                    if definition.default_cap is None:
                        raise TrainingJobRequestError(f"stateInputs.{source_name}.cap is not supported for boolean inputs")
                    source_config["cap"] = normalize_optional_positive_float(value["cap"], f"stateInputs.{source_name}.cap")
                if "heads" in value:
                    if not isinstance(value["heads"], list):
                        raise TrainingJobRequestError(f"stateInputs.{source_name}.heads must be an array")
                    heads: list[str] = []
                    for raw_head in value["heads"]:
                        head = str(raw_head).strip()
                        if not head:
                            continue
                        if head not in ALLOWED_STATE_INPUT_HEADS:
                            raise TrainingJobRequestError(
                                f"stateInputs.{source_name}.heads must only use {', '.join(ALLOWED_STATE_INPUT_HEADS)}"
                            )
                        if head not in heads:
                            heads.append(head)
                    source_config["heads"] = heads
                normalized[source_name] = source_config
            return normalized

        normalized_specs.append({
            "name": name,
            "notes": notes,
            "epochs": epochs,
            "learningRate": learning_rate,
            "widthMultiplier": width_multiplier,
            "trainRunIds": normalize_string_list(spec.get("trainRunIds"), "trainRunIds"),
            "valRunIds": normalize_string_list(spec.get("valRunIds"), "valRunIds"),
            "lossWeights": normalize_float_map(spec.get("lossWeights"), "lossWeights"),
            "consistency": normalize_float_map(spec.get("consistency"), "consistency"),
            "turnOversampling": normalize_turn_oversampling(spec.get("turnOversampling")),
            "yawLossWeighting": normalize_toggle_map(spec.get("yawLossWeighting"), "yawLossWeighting"),
            "stateInputs": normalize_state_inputs(spec.get("stateInputs")),
        })
    return normalized_specs


def _ensure_table(parent_path: tuple[str, ...], lines: list[str], emitted: set[tuple[str, ...]]) -> None:
    if not parent_path or parent_path in emitted:
        return
    if len(parent_path) > 1:
        _ensure_table(parent_path[:-1], lines, emitted)
    lines.append("")
    lines.append(f"[{'.'.join(parent_path)}]")
    emitted.add(parent_path)


def _format_toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return f"{value:.1f}"
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_format_toml_scalar(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {type(value)!r}")


def _dump_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []
    emitted_tables: set[tuple[str, ...]] = set()

    def write_table(table: dict[str, Any], path: tuple[str, ...]) -> None:
        scalar_items: list[tuple[str, Any]] = []
        child_tables: list[tuple[str, dict[str, Any]]] = []
        for key, value in table.items():
            if isinstance(value, dict):
                child_tables.append((key, value))
            else:
                scalar_items.append((key, value))

        if path:
            _ensure_table(path, lines, emitted_tables)
        for key, value in scalar_items:
            lines.append(f"{key} = {_format_toml_scalar(value)}")
        for child_key, child_table in child_tables:
            write_table(child_table, path + (child_key,))

    write_table(data, ())
    return "\n".join(line for line in lines if line is not None).strip() + "\n"


class TrainingManager:
    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path.resolve()
        self._project_root = self._config_path.parent.parent.resolve()
        self._train_script = (Path(__file__).resolve().parent / "train.py").resolve()
        self._jobs_dir = _resolve_jobs_dir(self._config_path).resolve()
        self._page_config = _load_page_defaults(self._config_path, self._jobs_dir)
        self._lock = threading.Lock()
        self._signal = threading.Event()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._queue: list[str] = []
        self._active_job_id: str | None = None
        self._active_process: subprocess.Popen[str] | None = None
        self._active_stop_requested = False
        self._recent_limit = 100

        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing_jobs()
        self._thread = threading.Thread(target=self._run_loop, name="training-queue", daemon=True)
        self._thread.start()

    def page_config(self) -> dict[str, Any]:
        payload = copy.deepcopy(self._page_config)
        payload["historyLimit"] = self._recent_limit
        return payload

    def state(self) -> dict[str, Any]:
        with self._lock:
            queued_jobs = [_clone_job(self._jobs[job_id]) for job_id in self._queue if job_id in self._jobs]
            active_job = _clone_job(self._jobs[self._active_job_id]) if self._active_job_id and self._active_job_id in self._jobs else None
            recent_jobs = _sort_jobs_by_created_desc([
                _clone_job(job)
                for job in self._jobs.values()
                if _is_terminal_status(str(job.get("status", "")))
            ])[:self._recent_limit]
        return {
            "activeJobId": self._active_job_id,
            "queuedCount": len(queued_jobs),
            "running": self._active_job_id is not None,
            "queuedJobs": queued_jobs,
            "activeJob": active_job,
            "recentJobs": recent_jobs,
            "jobsDirectory": str(self._jobs_dir),
        }

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return _sort_jobs_by_created_desc([_clone_job(job) for job in self._jobs.values()])

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(str(job_id).strip())
            if job is None:
                raise TrainingJobNotFoundError("training job not found")
            return _clone_job(job)

    def read_log(self, job_id: str, tail_lines: int = 0) -> str:
        job = self.get_job(job_id)
        log_path = Path(str(job.get("logPath", "")).strip())
        if not log_path:
            return ""
        if not log_path.is_file():
            return ""
        text = log_path.read_text(encoding="utf-8", errors="replace")
        return _tail_text(text, tail_lines)

    def enqueue(self, payload: Any) -> list[dict[str, Any]]:
        specs = _parse_job_specs(payload)
        created_at = _iso_now()
        created: list[dict[str, Any]] = []
        with self._lock:
            for index, spec in enumerate(specs, start=1):
                job = self._create_queued_job_locked(spec, created_at=created_at, index=index)
                self._jobs[str(job["id"])] = job
                self._queue.append(str(job["id"]))
                self._persist_job(job)
                created.append(_clone_job(job))
        self._signal.set()
        return created

    def requeue(self, job_id: str) -> dict[str, Any]:
        source_job = self.get_job(job_id)
        source_status = str(source_job.get("status", ""))
        if source_status not in {"failed", "stopped"}:
            raise TrainingJobNotRequeueableError("only failed or stopped training jobs can be requeued")
        created = self.enqueue(self._requeue_spec_from_job(source_job))
        return created[0]

    def _next_named_job_id_locked(self, raw_name: str) -> str:
        base = _slugify_job_name(raw_name)
        candidate = base
        suffix = 2
        while candidate in self._jobs or (self._jobs_dir / candidate).exists():
            candidate = f"{base}-{suffix}"
            suffix += 1
        return candidate

    def _create_queued_job_locked(self, spec: dict[str, Any], *, created_at: str, index: int) -> dict[str, Any]:
        if spec["name"]:
            job_id = self._next_named_job_id_locked(spec["name"])
        else:
            job_id = f"train-{time.time_ns()}-{index:03d}"
        job_dir = self._jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return {
            "id": job_id,
            "name": spec["name"],
            "notes": spec["notes"],
            "status": "queued",
            "epochs": spec["epochs"],
            "learningRate": spec["learningRate"],
            "widthMultiplier": spec["widthMultiplier"],
            "trainRunIds": list(spec["trainRunIds"]) if spec["trainRunIds"] is not None else None,
            "valRunIds": list(spec["valRunIds"]) if spec["valRunIds"] is not None else None,
            "lossWeights": dict(spec["lossWeights"]),
            "consistency": dict(spec["consistency"]),
            "turnOversampling": dict(spec["turnOversampling"]),
            "yawLossWeighting": dict(spec["yawLossWeighting"]),
            "stateInputs": copy.deepcopy(spec["stateInputs"]),
            "createdAt": created_at,
            "lastUpdatedAt": created_at,
            "configPath": str(job_dir / "derived_train_config.toml"),
            "logPath": str(job_dir / "train.log"),
            "jobDir": str(job_dir),
            "runDir": "",
            "runMetricsPath": "",
            "exitCode": None,
            "error": "",
            "command": [],
            "cancelRequested": False,
            "stopRequested": False,
        }

    def _requeue_spec_from_job(self, job: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": str(job.get("name", "") or ""),
            "notes": str(job.get("notes", "") or ""),
            "epochs": int(job["epochs"]) if job.get("epochs") is not None else None,
            "learningRate": float(job["learningRate"]) if job.get("learningRate") is not None else None,
            "widthMultiplier": float(job["widthMultiplier"]) if job.get("widthMultiplier") is not None else None,
            "trainRunIds": list(job["trainRunIds"]) if job.get("trainRunIds") is not None else None,
            "valRunIds": list(job["valRunIds"]) if job.get("valRunIds") is not None else None,
            "lossWeights": dict(job.get("lossWeights") or {}),
            "consistency": dict(job.get("consistency") or {}),
            "turnOversampling": dict(job.get("turnOversampling") or {}),
            "yawLossWeighting": dict(job.get("yawLossWeighting") or {}),
            "stateInputs": copy.deepcopy(job.get("stateInputs") or {}),
        }

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(str(job_id).strip())
            if job is None:
                raise TrainingJobNotFoundError("training job not found")
            if job.get("status") != "queued":
                raise TrainingJobNotPendingError("training job is not pending")
            job["status"] = "canceled"
            job["cancelRequested"] = True
            job["finishedAt"] = _iso_now()
            job["lastUpdatedAt"] = job["finishedAt"]
            self._queue = [queued_id for queued_id in self._queue if queued_id != job["id"]]
            self._persist_job(job)
            return _clone_job(job)

    def stop(self, job_id: str) -> dict[str, Any]:
        process: subprocess.Popen[str] | None = None
        with self._lock:
            job = self._jobs.get(str(job_id).strip())
            if job is None:
                raise TrainingJobNotFoundError("training job not found")
            if job.get("id") != self._active_job_id or self._active_process is None:
                raise TrainingJobNotActiveError("training job is not active")
            job["stopRequested"] = True
            job["lastUpdatedAt"] = _iso_now()
            self._active_stop_requested = True
            process = self._active_process
            self._persist_job(job)

        if process is not None and process.poll() is None:
            process.terminate()
            deadline = time.time() + 3.0
            while time.time() < deadline:
                if process.poll() is not None:
                    break
                time.sleep(0.05)
            if process.poll() is None:
                process.kill()
        return self.get_job(job_id)

    def delete_job(self, job_id: str) -> dict[str, Any]:
        job_dir: Path | None = None
        with self._lock:
            job = self._jobs.get(str(job_id).strip())
            if job is None:
                raise TrainingJobNotFoundError("training job not found")
            if not _is_terminal_status(str(job.get("status", ""))):
                raise TrainingJobNotTerminalError("training job must be terminal before it can be deleted")
            deleted = _clone_job(job)
            job_dir = Path(str(job.get("jobDir", "")).strip()) if job.get("jobDir") else None
            self._jobs.pop(job["id"], None)
            self._queue = [queued_id for queued_id in self._queue if queued_id != job["id"]]
        if job_dir:
            shutil.rmtree(job_dir, ignore_errors=True)
        return deleted

    def clear_history(self) -> dict[str, Any]:
        deleted_jobs: list[dict[str, Any]] = []
        job_dirs: list[Path] = []
        with self._lock:
            for job_id, job in list(self._jobs.items()):
                if not _is_terminal_status(str(job.get("status", ""))):
                    continue
                deleted_jobs.append(_clone_job(job))
                if job.get("jobDir"):
                    job_dirs.append(Path(str(job["jobDir"])))
                self._jobs.pop(job_id, None)
        for job_dir in job_dirs:
            shutil.rmtree(job_dir, ignore_errors=True)
        return {
            "status": "cleared",
            "deletedCount": len(deleted_jobs),
            "jobs": deleted_jobs,
        }

    def _run_loop(self) -> None:
        while True:
            self._signal.wait()
            self._signal.clear()
            while True:
                with self._lock:
                    if self._active_job_id is not None:
                        break
                    next_job_id = None
                    while self._queue:
                        candidate = self._queue.pop(0)
                        job = self._jobs.get(candidate)
                        if job and job.get("status") == "queued":
                            next_job_id = candidate
                            self._active_job_id = candidate
                            self._active_stop_requested = False
                            break
                    if next_job_id is None:
                        break
                self._run_job(next_job_id)

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            started_at = _iso_now()
            job["status"] = "starting"
            job["startedAt"] = started_at
            job["lastUpdatedAt"] = started_at
            job["stopRequested"] = False
            job["cancelRequested"] = False
            self._persist_job(job)

        try:
            self._write_derived_config(job)
            run_dir = ""
            run_metrics = ""
            log_lock = threading.Lock()
            command = [sys.executable, str(self._train_script), "--config", str(job["configPath"])]
            process = subprocess.Popen(
                command,
                cwd=str(self._project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            with self._lock:
                self._active_process = process
                job["status"] = "running"
                job["command"] = list(command)
                job["lastUpdatedAt"] = _iso_now()
                self._persist_job(job)

            log_path = Path(str(job["logPath"]))
            with log_path.open("w", encoding="utf-8") as log_file:
                def consume(stream: Any, prefix: str) -> None:
                    nonlocal run_dir, run_metrics
                    assert stream is not None
                    for line in stream:
                        clean = line.rstrip("\n")
                        with log_lock:
                            log_file.write(f"[{prefix}] {clean}\n")
                            log_file.flush()
                        if "run_dir=" in clean:
                            parsed = clean.split("run_dir=", 1)[1].strip().split()[0].strip("\"'")
                            if parsed:
                                run_dir = parsed
                        if "run_metrics=" in clean:
                            parsed = clean.split("run_metrics=", 1)[1].strip().split()[0].strip("\"'")
                            if parsed:
                                run_metrics = parsed

                stdout_thread = threading.Thread(target=consume, args=(process.stdout, "stdout"), daemon=True)
                stderr_thread = threading.Thread(target=consume, args=(process.stderr, "stderr"), daemon=True)
                stdout_thread.start()
                stderr_thread.start()
                exit_code = process.wait()
                stdout_thread.join()
                stderr_thread.join()

            with self._lock:
                job = self._jobs[job_id]
                job["runDir"] = run_dir
                job["runMetricsPath"] = run_metrics
                job["exitCode"] = int(exit_code)
                job["finishedAt"] = _iso_now()
                job["lastUpdatedAt"] = job["finishedAt"]
                if exit_code == 0 and not self._active_stop_requested:
                    job["status"] = "completed"
                    job["error"] = ""
                elif self._active_stop_requested:
                    job["status"] = "stopped"
                    job["error"] = "training process stopped"
                else:
                    job["status"] = "failed"
                    job["error"] = f"training process failed: exit status {exit_code}"
                self._persist_job(job)
                pruned_dirs = self._prune_terminal_history_locked()
        except Exception as exc:
            with self._lock:
                job = self._jobs[job_id]
                job["finishedAt"] = _iso_now()
                job["lastUpdatedAt"] = job["finishedAt"]
                job["status"] = "stopped" if self._active_stop_requested else "failed"
                job["error"] = str(exc)
                self._persist_job(job)
                pruned_dirs = self._prune_terminal_history_locked()
        finally:
            for job_dir in pruned_dirs:
                shutil.rmtree(job_dir, ignore_errors=True)
            with self._lock:
                self._active_job_id = None
                self._active_process = None
                self._active_stop_requested = False
            self._signal.set()

    def _write_derived_config(self, job: dict[str, Any]) -> None:
        raw = tomllib.loads(self._config_path.read_text(encoding="utf-8"))
        dataset_raw = raw.setdefault("dataset", {})
        loader_raw = raw.setdefault("loader", {})
        model_raw = raw.setdefault("model", {})
        training_raw = raw.setdefault("training", {})
        state_inputs_raw = raw.setdefault("state_inputs", {})
        if job.get("epochs") is not None:
            training_raw["epochs"] = int(job["epochs"])
        if job.get("learningRate") is not None:
            training_raw["learning_rate"] = float(job["learningRate"])
        if job.get("widthMultiplier") is not None:
            model_raw["width_multiplier"] = float(job["widthMultiplier"])
        if job.get("trainRunIds") is not None:
            dataset_raw["train_run_ids"] = [str(value) for value in job["trainRunIds"]]
        if job.get("valRunIds") is not None:
            dataset_raw["val_run_ids"] = [str(value) for value in job["valRunIds"]]
        if job.get("lossWeights"):
            training_raw["loss_weights"] = {key: float(value) for key, value in dict(job["lossWeights"]).items()}
        if job.get("consistency"):
            training_raw["consistency"] = {key: float(value) for key, value in dict(job["consistency"]).items()}
        if job.get("turnOversampling"):
            turn_oversampling = _normalize_turn_oversampling_thresholds(dict(job["turnOversampling"]))
            loader_raw["turn_oversampling"] = {
                key: (bool(value) if key == "enabled" else float(value))
                for key, value in turn_oversampling.items()
            }
        if job.get("yawLossWeighting"):
            training_raw["yaw_loss_weighting"] = {
                key: (bool(value) if key == "enabled" else float(value))
                for key, value in dict(job["yawLossWeighting"]).items()
            }
        if job.get("stateInputs"):
            state_input_overrides = dict(job["stateInputs"])
            for definition in STATE_INPUT_DEFINITIONS:
                item = state_input_overrides.get(definition.camel_key)
                if not isinstance(item, dict) or not item:
                    continue
                serialized: dict[str, Any] = {}
                if "enabled" in item:
                    serialized["enabled"] = bool(item["enabled"])
                if "heads" in item:
                    serialized["heads"] = [str(value) for value in item["heads"]]
                if definition.default_cap is not None and "cap" in item:
                    serialized["cap"] = float(item["cap"])
                state_inputs_raw[definition.key] = serialized
        encoded = _dump_toml(raw)
        Path(str(job["configPath"])).write_text(encoded, encoding="utf-8")

    def _persist_job(self, job: dict[str, Any]) -> None:
        job_dir = Path(str(job["jobDir"]))
        job_dir.mkdir(parents=True, exist_ok=True)
        payload = {"job": _clone_job(job)}
        (job_dir / "job.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _load_existing_jobs(self) -> None:
        if not self._jobs_dir.is_dir():
            return
        for child in self._jobs_dir.iterdir():
            if not child.is_dir():
                continue
            job_path = child / "job.json"
            if not job_path.is_file():
                continue
            try:
                payload = json.loads(job_path.read_text(encoding="utf-8"))
                job = payload.get("job", {})
                if not isinstance(job, dict):
                    continue
                if not _is_terminal_status(str(job.get("status", ""))):
                    job["status"] = "failed"
                    job["error"] = "python model server restarted before training job completed"
                    job["finishedAt"] = _iso_now()
                    job["lastUpdatedAt"] = job["finishedAt"]
                    (child / "job.json").write_text(json.dumps({"job": job}, indent=2) + "\n", encoding="utf-8")
                job_id = str(job.get("id", "")).strip()
                if job_id:
                    self._jobs[job_id] = job
            except Exception:
                continue
        pruned_dirs = self._prune_terminal_history_locked()
        for job_dir in pruned_dirs:
            shutil.rmtree(job_dir, ignore_errors=True)

    def _prune_terminal_history_locked(self) -> list[Path]:
        terminal_jobs = _sort_jobs_by_created_desc([
            _clone_job(job)
            for job in self._jobs.values()
            if _is_terminal_status(str(job.get("status", "")))
        ])
        if len(terminal_jobs) <= self._recent_limit:
            return []

        removable = terminal_jobs[self._recent_limit:]
        removable_ids = {str(job["id"]) for job in removable}
        removable_dirs: list[Path] = []
        for job_id in removable_ids:
            job = self._jobs.pop(job_id, None)
            if job and job.get("jobDir"):
                removable_dirs.append(Path(str(job["jobDir"])))
        return removable_dirs
