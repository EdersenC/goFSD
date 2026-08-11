from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from pathlib import Path
from typing import Any

from config import environment_data_root, resolve_data_root_child
from control_contract import STOP_SIGN_CONTROL_TARGET_NAMES
from state_inputs import (
    DEFAULT_WIDTH_MULTIPLIER,
    STATE_INPUT_DEFINITIONS,
    STATE_INPUT_DEFINITIONS_BY_CAMEL,
    state_input_config_from_metadata,
    state_input_definitions_metadata,
)


ALLOWED_LOSS_WEIGHT_KEYS = [
    *STOP_SIGN_CONTROL_TARGET_NAMES,
    "expert_throttle",
    "expert_brake",
    "actual_brake_pressure",
]

ALLOWED_EARLY_STOPPING_METRICS = [
    "stop_sign_score",
    "val_loss",
    "control_loss",
    "aux_loss",
    "control_mae_overall",
    "aux_mae_overall",
    "future_speed_mps_loss",
    "stop_intent_loss",
    "expert_throttle_loss",
    "expert_brake_loss",
    "actual_brake_pressure_loss",
    "future_speed_mps_mae",
    "stop_intent_mae",
    "expert_throttle_mae",
    "expert_brake_mae",
    "actual_brake_pressure_mae",
]

JOB_ID_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,126}[a-z0-9])?")
TRAINING_EVENT_PREFIX = "training_event="
JOB_SPEC_KEYS = (
    "name",
    "notes",
    "epochs",
    "learningRate",
    "widthMultiplier",
    "smoothL1Beta",
    "earlyStoppingMetric",
    "trainRunIds",
    "valRunIds",
    "lossWeights",
    "stateInputs",
)


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


class TrainingJobDuplicateError(TrainingJobError):
    pass


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time() % 1) * 1_000_000_000):09d}Z"


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


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


def _is_safe_run_id(value: str) -> bool:
    return (
        bool(value)
        and len(value) <= 255
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
        and "\x00" not in value
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _is_link_like(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction()) if callable(is_junction) else False
    except OSError:
        return True


def _job_spec_identity(spec: dict[str, Any]) -> str:
    payload = {key: copy.deepcopy(spec.get(key)) for key in JOB_SPEC_KEYS}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _training_event(line: str) -> dict[str, Any] | None:
    if not line.startswith(TRAINING_EVENT_PREFIX):
        return None
    try:
        payload = json.loads(line[len(TRAINING_EVENT_PREFIX):])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _last_log_detail(log_path: Path, *, max_length: int = 500) -> str:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    details = [line.strip() for line in lines if line.strip()]
    for detail in reversed(details):
        if detail.startswith("[stderr]"):
            return detail[-max_length:]
    return details[-1][-max_length:] if details else ""


def _terminate_process(process: subprocess.Popen[str], *, timeout_s: float = 3.0) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except OSError:
        return
    try:
        process.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        process.kill()
    except OSError:
        return
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return


def _close_process_pipes(process: subprocess.Popen[str]) -> None:
    for stream in (process.stdout, process.stderr):
        if stream is None:
            continue
        try:
            stream.close()
        except OSError:
            pass


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


def _resolve_jobs_dir(config_path: Path) -> Path:
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    backend_training = raw.get("backend", {}).get("training", {})
    jobs_dir_raw = str(backend_training.get("jobs_dir", "")).strip()
    if environment_data_root():
        return Path(resolve_data_root_child(jobs_dir_raw, "training_jobs"))
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


def _resolve_training_runs_dir(config_path: Path) -> Path:
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    output_raw = raw.get("output", {})
    configured_value = output_raw.get("base_dir") if isinstance(output_raw, dict) else None
    if environment_data_root():
        return Path(resolve_data_root_child(configured_value, "training_runs"))
    if configured_value is None or not str(configured_value).strip():
        return config_path.resolve().parent.parent / "training_runs"
    runs_dir = _normalize_runtime_path(str(configured_value))
    if runs_dir.is_absolute():
        return runs_dir
    return (config_path.resolve().parent.parent / runs_dir).resolve()


def _load_page_defaults(config_path: Path, jobs_dir: Path) -> dict[str, Any]:
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    dataset_raw = raw.get("dataset", {})
    loader_raw = raw.get("loader", {})
    training_raw = raw.get("training", {})
    model_raw = raw.get("model", {})
    loss_weights = training_raw.get("target_loss_weights")
    if not isinstance(loss_weights, dict):
        loss_weights = training_raw.get("loss_weights", {})
    optimizer_raw = training_raw.get("optimizer", {}) if isinstance(training_raw.get("optimizer"), dict) else {}
    scheduler_raw = training_raw.get("scheduler", {}) if isinstance(training_raw.get("scheduler"), dict) else {}
    ema_raw = training_raw.get("ema", {}) if isinstance(training_raw.get("ema"), dict) else {}
    has_optimizer_config = isinstance(training_raw.get("optimizer"), dict)
    has_scheduler_config = isinstance(training_raw.get("scheduler"), dict)
    has_ema_config = isinstance(training_raw.get("ema"), dict)
    learning_rate = float(optimizer_raw.get("lr", training_raw.get("learning_rate", 0.0)) or 0.0)
    optimizer_name = str(optimizer_raw.get("name", "adamw" if has_optimizer_config else "adam") or "").strip().lower() or "adam"
    optimizer_weight_decay = float(optimizer_raw.get("weight_decay", 0.0001 if has_optimizer_config else 0.0) or 0.0)
    grad_clip_default = 1.0 if has_optimizer_config else None
    grad_clip_raw = optimizer_raw.get("grad_clip_norm", grad_clip_default)
    scheduler_name = str(scheduler_raw.get("name", "cosine" if has_scheduler_config else "none") or "none").strip().lower()
    if scheduler_name in {"off", "disabled"}:
        scheduler_name = "none"
    warmup_fraction = float(scheduler_raw.get("warmup_fraction", 0.05 if has_scheduler_config else 0.0) or 0.0)
    min_lr_ratio = float(scheduler_raw.get("min_lr_ratio", 0.05 if has_scheduler_config else 0.0) or 0.0)
    ema_enabled = bool(ema_raw.get("enabled", True if has_ema_config else False))
    ema_decay = float(ema_raw.get("decay", 0.999) or 0.999)
    state_inputs = raw.get("state_inputs", {})
    state_input_config = state_input_config_from_metadata(state_inputs)
    state_inputs_payload: dict[str, Any] = {}

    def float_from_map(source: dict[str, Any], key: str, default: float) -> float:
        if key not in source:
            return default
        value = source.get(key)
        return float(default if value is None else value)

    for definition in STATE_INPUT_DEFINITIONS:
        spec = state_input_config.spec(definition.key)
        payload = {
            "enabled": bool(spec.enabled),
        }
        if definition.default_cap is not None:
            payload["cap"] = float(spec.cap if spec.cap is not None else definition.default_cap)
        state_inputs_payload[definition.camel_key] = payload
    return {
        "configPath": str(config_path),
        "pythonBin": sys.executable,
        "trainScript": str((Path(__file__).resolve().parent / "train.py").resolve()),
        "jobsDir": str(jobs_dir),
        "epochs": int(training_raw.get("epochs", 0) or 0),
        "learningRate": learning_rate,
        "lossFunction": str(training_raw.get("loss_function", "smooth_l1") or "smooth_l1"),
        "smoothL1Beta": float(training_raw.get("smooth_l1_beta", 0.1) or 0.1),
        "earlyStoppingMetric": str(
            training_raw.get("early_stopping_metric", "stop_sign_score")
            or "stop_sign_score"
        ),
        "optimizer": {
            "name": optimizer_name,
            "weightDecay": optimizer_weight_decay,
            "gradClipNorm": grad_clip_raw,
        },
        "scheduler": {
            "name": scheduler_name,
            "warmupFraction": warmup_fraction,
            "minLrRatio": min_lr_ratio,
            "stepFrequency": "per_step" if scheduler_name == "cosine" else "none",
        },
        "ema": {
            "enabled": ema_enabled,
            "decay": ema_decay,
        },
        "widthMultiplier": float(model_raw.get("width_multiplier", DEFAULT_WIDTH_MULTIPLIER) or DEFAULT_WIDTH_MULTIPLIER),
        "trainRunIds": [str(value) for value in dataset_raw.get("train_run_ids", []) if str(value).strip()],
        "valRunIds": [str(value) for value in dataset_raw.get("val_run_ids", []) if str(value).strip()],
        "lossWeights": {key: float_from_map(loss_weights, key, 1.0) for key in ALLOWED_LOSS_WEIGHT_KEYS},
        "stateInputs": state_inputs_payload,
        "stateInputDefinitions": state_input_definitions_metadata(),
        "allowedLossWeightKeys": list(ALLOWED_LOSS_WEIGHT_KEYS),
        "allowedEarlyStoppingMetrics": list(ALLOWED_EARLY_STOPPING_METRICS),
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
        removed_keys = sorted(
            set(spec)
            & {
                "consistency",
                "turnOversampling",
                "yawLossWeighting",
                "yaw_loss_weighting",
            }
        )
        if removed_keys:
            raise TrainingJobRequestError(
                "removed training job field(s): "
                f"{', '.join(removed_keys)}. Configure only supported trainer fields."
            )
        name = str(spec.get("name", "") or "").strip()
        notes = str(spec.get("notes", "") or "").strip()
        if len(name) > 80:
            raise TrainingJobRequestError("name must be 80 characters or fewer")
        if len(notes) > 2_000:
            raise TrainingJobRequestError("notes must be 2000 characters or fewer")
        learning_rate = spec.get("learningRate")
        if learning_rate is not None:
            if not _is_finite_number(learning_rate) or float(learning_rate) <= 0:
                raise TrainingJobRequestError("learningRate must be a positive finite number")
            learning_rate = float(learning_rate)
        epochs = spec.get("epochs")
        if epochs is not None:
            if (
                not _is_finite_number(epochs)
                or not float(epochs).is_integer()
                or int(epochs) < 1
            ):
                raise TrainingJobRequestError("epochs must be an integer >= 1")
            epochs = int(epochs)
        width_multiplier = spec.get("widthMultiplier")
        if width_multiplier is not None:
            if not _is_finite_number(width_multiplier) or float(width_multiplier) <= 0:
                raise TrainingJobRequestError("widthMultiplier must be a positive finite number")
            width_multiplier = float(width_multiplier)
        smooth_l1_beta = spec.get("smoothL1Beta")
        if smooth_l1_beta is not None:
            if not _is_finite_number(smooth_l1_beta) or float(smooth_l1_beta) <= 0:
                raise TrainingJobRequestError("smoothL1Beta must be a positive finite number")
            smooth_l1_beta = float(smooth_l1_beta)
        early_stopping_metric = spec.get("earlyStoppingMetric")
        if early_stopping_metric is not None:
            early_stopping_metric = str(early_stopping_metric).strip()
            if not early_stopping_metric:
                raise TrainingJobRequestError("earlyStoppingMetric must be a non-empty string")
            if early_stopping_metric not in ALLOWED_EARLY_STOPPING_METRICS:
                raise TrainingJobRequestError(
                    f"earlyStoppingMetric must be one of {', '.join(ALLOWED_EARLY_STOPPING_METRICS)}"
                )

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
            seen: set[str] = set()
            for value in raw_list:
                if not isinstance(value, str):
                    raise TrainingJobRequestError(f"{field_name} must only contain string run ids")
                item = value.strip()
                if not item:
                    continue
                if not _is_safe_run_id(item):
                    raise TrainingJobRequestError(
                        f"{field_name} contains an unsafe run id; use a run folder name, not a path"
                    )
                if item in seen:
                    raise TrainingJobRequestError(f"{field_name} must not contain duplicate run ids")
                seen.add(item)
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
                    raise TrainingJobRequestError(
                        "stateInputs.*.heads has been removed; state inputs are fused into the temporal planner"
                    )
                normalized[source_name] = source_config
            return normalized

        loss_weight_overrides = normalize_float_map(spec.get("lossWeights"), "lossWeights")
        unknown_loss_weights = sorted(set(loss_weight_overrides) - set(ALLOWED_LOSS_WEIGHT_KEYS))
        if unknown_loss_weights:
            raise TrainingJobRequestError(
                f"lossWeights only supports {', '.join(ALLOWED_LOSS_WEIGHT_KEYS)}; "
                f"unknown: {', '.join(unknown_loss_weights)}"
            )
        negative_loss_weights = sorted(key for key, value in loss_weight_overrides.items() if value < 0.0)
        if negative_loss_weights:
            raise TrainingJobRequestError(f"lossWeights must be >= 0 for {', '.join(negative_loss_weights)}")

        train_run_ids = normalize_string_list(spec.get("trainRunIds"), "trainRunIds")
        val_run_ids = normalize_string_list(spec.get("valRunIds"), "valRunIds")
        if train_run_ids is not None and val_run_ids is not None:
            overlapping_run_ids = sorted(set(train_run_ids) & set(val_run_ids))
            if overlapping_run_ids:
                raise TrainingJobRequestError(
                    "trainRunIds and valRunIds must be separate; overlap: "
                    f"{', '.join(overlapping_run_ids)}"
                )

        normalized_specs.append({
            "name": name,
            "notes": notes,
            "epochs": epochs,
            "learningRate": learning_rate,
            "widthMultiplier": width_multiplier,
            "smoothL1Beta": smooth_l1_beta,
            "earlyStoppingMetric": early_stopping_metric,
            "trainRunIds": train_run_ids,
            "valRunIds": val_run_ids,
            "lossWeights": loss_weight_overrides,
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
    def __init__(self, config_path: Path, *, start_worker: bool = True) -> None:
        self._config_path = config_path.resolve()
        self._project_root = self._config_path.parent.parent.resolve()
        self._train_script = (Path(__file__).resolve().parent / "train.py").resolve()
        self._jobs_dir = _resolve_jobs_dir(self._config_path).resolve()
        self._runs_dir = _resolve_training_runs_dir(self._config_path).resolve()
        self._page_config = _load_page_defaults(self._config_path, self._jobs_dir)
        self._lock = threading.Lock()
        self._signal = threading.Event()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._queue: list[str] = []
        self._active_job_id: str | None = None
        self._active_process: subprocess.Popen[str] | None = None
        self._active_stop_requested = False
        self._closing = False
        self._recent_limit = 100
        self._recovery_warnings: list[str] = []

        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing_jobs()
        self._thread: threading.Thread | None = None
        if start_worker:
            self._thread = threading.Thread(target=self._run_loop, name="training-queue", daemon=True)
            self._thread.start()
            if self._queue:
                self._signal.set()

    def page_config(self) -> dict[str, Any]:
        payload = copy.deepcopy(self._page_config)
        payload["historyLimit"] = self._recent_limit
        return payload

    def state(self) -> dict[str, Any]:
        with self._lock:
            queued_jobs = [_clone_job(self._jobs[job_id]) for job_id in self._queue if job_id in self._jobs]
            active_job = (
                _clone_job(self._jobs[self._active_job_id])
                if self._active_job_id and self._active_job_id in self._jobs
                else None
            )
            active_job_id = self._active_job_id
            recent_jobs = _sort_jobs_by_created_desc([
                _clone_job(job)
                for job in self._jobs.values()
                if _is_terminal_status(str(job.get("status", "")))
            ])[:self._recent_limit]
        return {
            "activeJobId": active_job_id,
            "queuedCount": len(queued_jobs),
            "running": active_job_id is not None,
            "queuedJobs": queued_jobs,
            "activeJob": active_job,
            "recentJobs": recent_jobs,
            "jobsDirectory": str(self._jobs_dir),
            "recoveryWarnings": list(self._recovery_warnings),
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
        log_path = self._job_dir_for_id(str(job["id"])) / "train.log"
        if not log_path.is_file():
            return ""
        text = log_path.read_text(encoding="utf-8", errors="replace")
        return _tail_text(text, tail_lines)

    def enqueue(self, payload: Any) -> list[dict[str, Any]]:
        specs = _parse_job_specs(payload)
        created_at = _iso_now()
        created: list[dict[str, Any]] = []
        with self._lock:
            if self._closing:
                raise TrainingJobRequestError("the model service is shutting down and cannot accept new training jobs")
            requested_identities: set[str] = set()
            active_identities = {
                _job_spec_identity(job)
                for job in self._jobs.values()
                if not _is_terminal_status(str(job.get("status", "")))
            }
            for spec in specs:
                identity = _job_spec_identity(spec)
                if identity in requested_identities:
                    raise TrainingJobDuplicateError("the request contains an identical training job more than once")
                if identity in active_identities:
                    raise TrainingJobDuplicateError(
                        "an identical training job is already queued or running; wait for it to finish or stop it first"
                    )
                requested_identities.add(identity)
            for index, spec in enumerate(specs, start=1):
                job = self._create_queued_job_locked(spec, created_at=created_at, index=index)
                try:
                    self._persist_job(job)
                except Exception:
                    for rollback_job in [*created, job]:
                        rollback_id = str(rollback_job["id"])
                        self._jobs.pop(rollback_id, None)
                        self._queue = [queued_id for queued_id in self._queue if queued_id != rollback_id]
                        rollback_dir = self._job_dir_for_id(rollback_id)
                        if rollback_dir.exists() and not _is_link_like(rollback_dir):
                            shutil.rmtree(rollback_dir, ignore_errors=True)
                    raise
                self._jobs[str(job["id"])] = job
                self._queue.append(str(job["id"]))
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

    def _job_dir_for_id(self, job_id: str) -> Path:
        normalized = str(job_id).strip()
        if JOB_ID_PATTERN.fullmatch(normalized) is None:
            raise TrainingJobError("training job has an invalid internal id")
        job_dir = self._jobs_dir / normalized
        if job_dir.parent != self._jobs_dir:
            raise AssertionError("training job directory escaped the configured jobs directory")
        return job_dir

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
        job_dir = self._job_dir_for_id(job_id)
        if _is_link_like(job_dir):
            raise TrainingJobError("training job directory cannot be a symbolic link")
        job_dir.mkdir(parents=True, exist_ok=False)
        total_epochs = int(spec["epochs"] or self._page_config.get("epochs") or 0)
        return {
            "id": job_id,
            "name": spec["name"],
            "notes": spec["notes"],
            "status": "queued",
            "phase": "waiting",
            "epochs": spec["epochs"],
            "learningRate": spec["learningRate"],
            "widthMultiplier": spec["widthMultiplier"],
            "smoothL1Beta": spec["smoothL1Beta"],
            "earlyStoppingMetric": spec["earlyStoppingMetric"],
            "trainRunIds": list(spec["trainRunIds"]) if spec["trainRunIds"] is not None else None,
            "valRunIds": list(spec["valRunIds"]) if spec["valRunIds"] is not None else None,
            "lossWeights": dict(spec["lossWeights"]),
            "stateInputs": copy.deepcopy(spec["stateInputs"]),
            "createdAt": created_at,
            "queueOrder": time.time_ns() * 1_000 + index,
            "lastUpdatedAt": created_at,
            "configPath": str(job_dir / "derived_train_config.toml"),
            "logPath": str(job_dir / "train.log"),
            "jobDir": str(job_dir),
            "runDir": "",
            "runMetricsPath": "",
            "currentEpoch": 0,
            "totalEpochs": total_epochs,
            "progressPercent": 0.0,
            "latestMetrics": {},
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
            "smoothL1Beta": float(job["smoothL1Beta"]) if job.get("smoothL1Beta") is not None else None,
            "earlyStoppingMetric": str(job.get("earlyStoppingMetric") or "") or None,
            "trainRunIds": list(job["trainRunIds"]) if job.get("trainRunIds") is not None else None,
            "valRunIds": list(job["valRunIds"]) if job.get("valRunIds") is not None else None,
            "lossWeights": dict(job.get("lossWeights") or {}),
            "stateInputs": copy.deepcopy(job.get("stateInputs") or {}),
        }

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(str(job_id).strip())
            if job is None:
                raise TrainingJobNotFoundError("training job not found")
            if job.get("status") != "queued" or job.get("id") == self._active_job_id:
                raise TrainingJobNotPendingError("training job is not pending")
            original_job = _clone_job(job)
            original_queue = list(self._queue)
            job["status"] = "canceled"
            job["phase"] = "finished"
            job["cancelRequested"] = True
            job["finishedAt"] = _iso_now()
            job["lastUpdatedAt"] = job["finishedAt"]
            self._queue = [queued_id for queued_id in self._queue if queued_id != job["id"]]
            try:
                self._persist_job(job)
            except Exception:
                self._jobs[str(original_job["id"])] = original_job
                self._queue = original_queue
                raise
            return _clone_job(job)

    def stop(self, job_id: str) -> dict[str, Any]:
        process: subprocess.Popen[str] | None = None
        persistence_error: Exception | None = None
        with self._lock:
            job = self._jobs.get(str(job_id).strip())
            if job is None:
                raise TrainingJobNotFoundError("training job not found")
            if job.get("id") != self._active_job_id:
                raise TrainingJobNotActiveError("training job is not active")
            if job.get("stopRequested"):
                return _clone_job(job)
            job["stopRequested"] = True
            job["phase"] = "stopping"
            job["lastUpdatedAt"] = _iso_now()
            self._active_stop_requested = True
            process = self._active_process
            try:
                self._persist_job(job)
            except Exception as exc:
                persistence_error = exc

        if process is not None:
            _terminate_process(process)
        if persistence_error is not None:
            raise TrainingJobError(f"could not persist the stop request: {persistence_error}") from persistence_error
        return self.get_job(job_id)

    def delete_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(str(job_id).strip())
            if job is None:
                raise TrainingJobNotFoundError("training job not found")
            if not _is_terminal_status(str(job.get("status", ""))):
                raise TrainingJobNotTerminalError("training job must be terminal before it can be deleted")
            deleted = _clone_job(job)
            job_dir = self._job_dir_for_id(str(job["id"]))
            if _is_link_like(job_dir):
                raise TrainingJobError("refusing to delete a symbolic-link training job directory")
            if job_dir.exists():
                shutil.rmtree(job_dir)
            self._jobs.pop(job["id"], None)
            self._queue = [queued_id for queued_id in self._queue if queued_id != job["id"]]
        return deleted

    def clear_history(self) -> dict[str, Any]:
        deleted_jobs: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        with self._lock:
            for job_id, job in list(self._jobs.items()):
                if not _is_terminal_status(str(job.get("status", ""))):
                    continue
                job_dir = self._job_dir_for_id(str(job["id"]))
                if _is_link_like(job_dir):
                    failures.append({
                        "jobId": job_id,
                        "error": "refusing to delete a symbolic-link training job directory",
                    })
                    continue
                try:
                    if job_dir.exists():
                        shutil.rmtree(job_dir)
                except OSError as exc:
                    failures.append({"jobId": job_id, "error": str(exc)})
                    continue
                deleted_jobs.append(_clone_job(job))
                self._jobs.pop(job_id, None)
        return {
            "status": "cleared",
            "deletedCount": len(deleted_jobs),
            "failedCount": len(failures),
            "failures": failures,
            "jobs": deleted_jobs,
        }

    def _resolve_reported_artifacts(self, run_dir_raw: Any, metrics_path_raw: Any) -> tuple[Path, Path] | None:
        run_dir_text = str(run_dir_raw or "").strip().strip("\"'")
        if not run_dir_text:
            return None
        run_dir = _normalize_runtime_path(run_dir_text)
        if not run_dir.is_absolute():
            run_dir = self._project_root / run_dir
        run_dir = run_dir.resolve(strict=False)
        if run_dir.parent != self._runs_dir:
            return None

        metrics_text = str(metrics_path_raw or "").strip().strip("\"'")
        metrics_path = _normalize_runtime_path(metrics_text) if metrics_text else run_dir / "run_metrics.json"
        if not metrics_path.is_absolute():
            metrics_path = self._project_root / metrics_path
        metrics_path = metrics_path.resolve(strict=False)
        if metrics_path != run_dir / "run_metrics.json":
            return None
        return run_dir, metrics_path

    def _record_training_output(self, job_id: str, line: str) -> None:
        event = _training_event(line)
        if event is None:
            self._record_legacy_run_dir(job_id, line)
            return

        event_type = str(event.get("type", "")).strip().lower()
        if event_type == "artifacts":
            artifacts = self._resolve_reported_artifacts(event.get("runDir"), event.get("runMetricsPath"))
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                if artifacts is None:
                    job["artifactWarning"] = (
                        "Trainer reported an output path outside the configured training-runs directory."
                    )
                else:
                    run_dir, metrics_path = artifacts
                    job["runDir"] = str(run_dir)
                    job["runMetricsPath"] = str(metrics_path)
                    job.pop("artifactWarning", None)
                job["lastUpdatedAt"] = _iso_now()
                self._persist_job(job)
            return

        if event_type != "epoch":
            return

        raw_epoch = event.get("epoch")
        raw_total_epochs = event.get("totalEpochs")
        if isinstance(raw_epoch, bool) or not isinstance(raw_epoch, int) or raw_epoch < 1:
            return
        if isinstance(raw_total_epochs, bool) or not isinstance(raw_total_epochs, int) or raw_total_epochs < 1:
            return

        metrics = event.get("metrics")
        latest_metrics: dict[str, float] = {}
        if isinstance(metrics, dict):
            latest_metrics = {
                str(key): float(value)
                for key, value in metrics.items()
                if _is_finite_number(value)
            }

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            current_epoch = min(raw_epoch, raw_total_epochs)
            job["currentEpoch"] = current_epoch
            job["totalEpochs"] = raw_total_epochs
            job["progressPercent"] = round(100.0 * current_epoch / raw_total_epochs, 1)
            job["latestMetrics"] = latest_metrics
            if not job.get("stopRequested"):
                job["phase"] = "training"
            raw_best_epoch = event.get("bestEpoch")
            if isinstance(raw_best_epoch, int) and not isinstance(raw_best_epoch, bool) and raw_best_epoch >= 0:
                job["bestEpoch"] = raw_best_epoch
            raw_elapsed_seconds = event.get("elapsedSeconds")
            if _is_finite_number(raw_elapsed_seconds) and float(raw_elapsed_seconds) >= 0:
                job["elapsedSeconds"] = float(raw_elapsed_seconds)
            job["lastUpdatedAt"] = _iso_now()
            self._persist_job(job)

    def _record_legacy_run_dir(self, job_id: str, line: str) -> None:
        match = re.search(r"(?:^|\s)run_dir=(?P<path>[^\s]+)", line)
        if match is None:
            return
        artifacts = self._resolve_reported_artifacts(match.group("path"), None)
        if artifacts is None:
            return
        run_dir, metrics_path = artifacts
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.get("runDir"):
                return
            job["runDir"] = str(run_dir)
            job["runMetricsPath"] = str(metrics_path)
            job["lastUpdatedAt"] = _iso_now()
            self._persist_job(job)

    def close(self) -> None:
        process: subprocess.Popen[str] | None = None
        persistence_error: Exception | None = None
        with self._lock:
            if self._closing:
                return
            self._closing = True
            if self._active_job_id is not None:
                self._active_stop_requested = True
                job = self._jobs.get(self._active_job_id)
                if job is not None:
                    job["stopRequested"] = True
                    job["phase"] = "stopping"
                    job["lastUpdatedAt"] = _iso_now()
                    try:
                        self._persist_job(job)
                    except Exception as exc:
                        persistence_error = exc
            process = self._active_process
        self._signal.set()
        if process is not None:
            _terminate_process(process)
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5.0)
        if persistence_error is not None:
            raise TrainingJobError(
                f"could not persist the shutdown stop request: {persistence_error}"
            ) from persistence_error

    def _run_loop(self) -> None:
        while True:
            self._signal.wait()
            self._signal.clear()
            while True:
                with self._lock:
                    if self._closing:
                        return
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
            job["status"] = "running"
            job["phase"] = "starting"
            job["startedAt"] = started_at
            job["lastUpdatedAt"] = started_at
            job["stopRequested"] = False
            job["cancelRequested"] = False
            self._persist_job(job)

        process: subprocess.Popen[str] | None = None
        reader_threads: list[threading.Thread] = []
        try:
            self._write_derived_config(job)
            with self._lock:
                if self._active_stop_requested:
                    job = self._jobs[job_id]
                    job["status"] = "stopped"
                    job["phase"] = "finished"
                    job["error"] = "Training was stopped before the process started."
                    job["finishedAt"] = _iso_now()
                    job["lastUpdatedAt"] = job["finishedAt"]
                    self._persist_job(job)
                    self._prune_terminal_history_locked()
                    return

            log_lock = threading.Lock()
            reader_errors: list[str] = []
            command = [sys.executable, "-u", str(self._train_script), "--config", str(job["configPath"])]
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
                stop_requested_after_start = self._active_stop_requested
                job["status"] = "running"
                job["phase"] = "stopping" if stop_requested_after_start else "training"
                job["command"] = list(command)
                job["processId"] = int(process.pid)
                job["lastUpdatedAt"] = _iso_now()
                self._persist_job(job)

            if stop_requested_after_start and process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass

            log_path = Path(str(job["logPath"]))
            with log_path.open("w", encoding="utf-8") as log_file:
                def consume(stream: Any, prefix: str) -> None:
                    assert stream is not None
                    try:
                        for line in stream:
                            clean = line.rstrip("\n")
                            with log_lock:
                                log_file.write(f"[{prefix}] {clean}\n")
                                log_file.flush()
                            self._record_training_output(job_id, clean)
                    except Exception as exc:
                        with log_lock:
                            reader_errors.append(str(exc))
                            log_file.write(f"[{prefix}] Training output reader failed: {exc}\n")
                            log_file.flush()

                reader_threads = [
                    threading.Thread(target=consume, args=(process.stdout, "stdout"), daemon=True),
                    threading.Thread(target=consume, args=(process.stderr, "stderr"), daemon=True),
                ]
                for reader_thread in reader_threads:
                    reader_thread.start()
                exit_code = process.wait()
                for reader_thread in reader_threads:
                    reader_thread.join()
                _close_process_pipes(process)
                if reader_errors:
                    raise TrainingJobError(reader_errors[0])

            with self._lock:
                job = self._jobs[job_id]
                job["exitCode"] = int(exit_code)
                job["processId"] = None
                job["finishedAt"] = _iso_now()
                job["lastUpdatedAt"] = job["finishedAt"]
                job["phase"] = "finished"
                if exit_code == 0 and not self._active_stop_requested:
                    job["status"] = "completed"
                    job["error"] = ""
                    job["progressPercent"] = 100.0
                elif self._active_stop_requested:
                    job["status"] = "stopped"
                    job["error"] = "Training was stopped. You can use Try again to start a fresh job."
                else:
                    job["status"] = "failed"
                    detail = _last_log_detail(log_path)
                    job["error"] = f"Training exited with status {exit_code}."
                    if detail:
                        job["error"] += f" Last output: {detail}"
                self._persist_job(job)
                self._prune_terminal_history_locked()
        except Exception as exc:
            if process is not None:
                _terminate_process(process)
                for reader_thread in reader_threads:
                    reader_thread.join(timeout=1.0)
                _close_process_pipes(process)
            with self._lock:
                job = self._jobs[job_id]
                job["finishedAt"] = _iso_now()
                job["lastUpdatedAt"] = job["finishedAt"]
                job["processId"] = None
                job["phase"] = "finished"
                job["status"] = "stopped" if self._active_stop_requested else "failed"
                if self._active_stop_requested:
                    job["error"] = "Training was stopped. You can use Try again to start a fresh job."
                else:
                    job["error"] = f"Training could not run: {exc}"
                self._persist_job(job)
                self._prune_terminal_history_locked()
        finally:
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
            resolved_lr = float(job["learningRate"])
            training_raw["learning_rate"] = resolved_lr
            optimizer_raw = training_raw.get("optimizer")
            if isinstance(optimizer_raw, dict):
                optimizer_raw["lr"] = resolved_lr
        if job.get("widthMultiplier") is not None:
            model_raw["width_multiplier"] = float(job["widthMultiplier"])
        if job.get("smoothL1Beta") is not None:
            training_raw["smooth_l1_beta"] = float(job["smoothL1Beta"])
        if job.get("earlyStoppingMetric"):
            training_raw["early_stopping_metric"] = str(job["earlyStoppingMetric"])
        if job.get("trainRunIds") is not None:
            dataset_raw["train_run_ids"] = [str(value) for value in job["trainRunIds"]]
        if job.get("valRunIds") is not None:
            dataset_raw["val_run_ids"] = [str(value) for value in job["valRunIds"]]
        if job.get("lossWeights"):
            training_raw["target_loss_weights"] = {
                key: float(value)
                for key, value in dict(job["lossWeights"]).items()
            }
            training_raw.pop("loss_weights", None)
        training_raw.pop("consistency", None)
        training_raw.pop("yaw_loss_weighting", None)
        loader_raw.pop("turn_oversampling", None)
        if job.get("stateInputs"):
            state_input_overrides = dict(job["stateInputs"])
            for definition in STATE_INPUT_DEFINITIONS:
                item = state_input_overrides.get(definition.camel_key)
                if not isinstance(item, dict) or not item:
                    continue
                serialized: dict[str, Any] = {}
                if "enabled" in item:
                    serialized["enabled"] = bool(item["enabled"])
                if definition.default_cap is not None and "cap" in item:
                    serialized["cap"] = float(item["cap"])
                state_inputs_raw[definition.key] = serialized
        encoded = _dump_toml(raw)
        _atomic_write_text(Path(str(job["configPath"])), encoded)

    def _persist_job(self, job: dict[str, Any]) -> None:
        job_id = str(job.get("id", "")).strip()
        job_dir = self._job_dir_for_id(job_id)
        if _is_link_like(job_dir):
            raise TrainingJobError("training job directory cannot be a symbolic link")
        job_dir.mkdir(parents=True, exist_ok=True)
        job["jobDir"] = str(job_dir)
        job["configPath"] = str(job_dir / "derived_train_config.toml")
        job["logPath"] = str(job_dir / "train.log")
        payload = {"job": _clone_job(job)}
        _atomic_write_text(job_dir / "job.json", json.dumps(payload, indent=2) + "\n")

    def _load_existing_jobs(self) -> None:
        if not self._jobs_dir.is_dir():
            return
        recovered_queue: list[tuple[str, int, str]] = []
        for child in sorted(self._jobs_dir.iterdir(), key=lambda item: item.name):
            if _is_link_like(child):
                self._add_recovery_warning(f"Skipped symbolic-link job directory: {child.name}")
                continue
            if not child.is_dir():
                continue
            job_path = child / "job.json"
            if _is_link_like(job_path):
                self._add_recovery_warning(f"Skipped symbolic-link job state: {child.name}")
                continue
            if not job_path.is_file():
                continue
            try:
                payload = json.loads(job_path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("job state is not an object")
                job = payload.get("job", {})
                if not isinstance(job, dict):
                    raise ValueError("job payload is not an object")
                job_id = str(job.get("id", "")).strip()
                if JOB_ID_PATTERN.fullmatch(job_id) is None or job_id != child.name:
                    raise ValueError("job id does not match its safe directory name")

                status = str(job.get("status", "")).strip()
                job.setdefault("currentEpoch", 0)
                job.setdefault("totalEpochs", int(job.get("epochs") or self._page_config.get("epochs") or 0))
                job.setdefault("progressPercent", 100.0 if status == "completed" else 0.0)
                job.setdefault("latestMetrics", {})
                job["processId"] = None

                if status == "queued" and not job.get("cancelRequested"):
                    job["phase"] = "waiting"
                    job["stopRequested"] = False
                    job["recoveredAt"] = _iso_now()
                    job["recoveryMessage"] = "Restored after the model service restarted; this job is waiting to run."
                    raw_queue_order = job.get("queueOrder")
                    queue_order = (
                        int(raw_queue_order)
                        if isinstance(raw_queue_order, int) and not isinstance(raw_queue_order, bool)
                        else 0
                    )
                    recovered_queue.append((str(job.get("createdAt", "")), queue_order, job_id))
                elif status == "queued":
                    job["status"] = "canceled"
                    job["phase"] = "finished"
                    job["finishedAt"] = _iso_now()
                    job["lastUpdatedAt"] = job["finishedAt"]
                elif not _is_terminal_status(status):
                    job["status"] = "failed"
                    job["phase"] = "finished"
                    job["error"] = (
                        "The model service restarted while this job was active. "
                        "Use Try again to start a fresh training process."
                    )
                    job["finishedAt"] = _iso_now()
                    job["lastUpdatedAt"] = job["finishedAt"]
                    job["interruptedAt"] = job["finishedAt"]
                else:
                    job["phase"] = "finished"

                self._persist_job(job)
                self._jobs[job_id] = job
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TrainingJobError) as exc:
                self._add_recovery_warning(f"Could not restore training job {child.name}: {exc}")

        self._queue = [job_id for _, _, job_id in sorted(recovered_queue)]
        self._prune_terminal_history_locked()

    def _add_recovery_warning(self, message: str) -> None:
        if len(self._recovery_warnings) < 20:
            self._recovery_warnings.append(message)

    def _prune_terminal_history_locked(self) -> None:
        terminal_jobs = _sort_jobs_by_created_desc([
            _clone_job(job)
            for job in self._jobs.values()
            if _is_terminal_status(str(job.get("status", "")))
        ])
        if len(terminal_jobs) <= self._recent_limit:
            return

        removable = terminal_jobs[self._recent_limit:]
        removable_ids = {str(job["id"]) for job in removable}
        for job_id in removable_ids:
            job_dir = self._job_dir_for_id(job_id)
            if _is_link_like(job_dir):
                self._add_recovery_warning(f"Could not prune symbolic-link training job directory: {job_id}")
                continue
            try:
                if job_dir.exists():
                    shutil.rmtree(job_dir)
            except OSError as exc:
                self._add_recovery_warning(f"Could not prune training job {job_id}: {exc}")
                continue
            self._jobs.pop(job_id, None)
