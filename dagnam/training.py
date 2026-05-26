"""Local training metrics reporter for generated training scripts.

Generated training code imports this module to emit structured JSON events to a
local metrics file. It never makes network calls and never touches credentials.

The metrics path resolves in this order:

1. ``DAGNAM_METRICS_PATH`` environment variable;
2. ``~/.dagnam/config.json`` key ``training_metrics_path``;
3. ``./dagnam_metrics.jsonl`` in the current working directory.

All public functions catch exceptions internally and never raise. Metrics
reporting must not crash a training script.
"""

from __future__ import annotations

import atexit
import datetime
import json
import os
import sys

_DEFAULT_METRICS_PATH = "./dagnam_metrics.jsonl"
_metrics_path: str | None = None
_file = None
_using_fallback_path = False
_fallback_warning_emitted = False


def _configured_metrics_path() -> str | None:
    """Return the persistent SDK metrics path, if configured."""
    try:
        config_mod = __import__("dagnam._core.config", fromlist=["get_config_value"])
        value = config_mod.get_config_value("training_metrics_path")
    except Exception:
        return None
    return value if isinstance(value, str) and value else None


def _resolve_metrics_path() -> tuple[str, bool]:
    env_path = os.environ.get("DAGNAM_METRICS_PATH")
    if env_path:
        return env_path, False

    config_path = _configured_metrics_path()
    if config_path:
        return config_path, False

    return _DEFAULT_METRICS_PATH, True


def _warn_fallback_once() -> None:
    global _fallback_warning_emitted
    if _fallback_warning_emitted:
        return
    _fallback_warning_emitted = True
    message = (
        "Dagnam local training metrics path is not configured; writing to "
        "./dagnam_metrics.jsonl. To view local training progress in Dagnam, run: "
        "dagnam config set training_metrics_path ./dagnam_metrics.jsonl\n"
    )
    try:
        sys.stderr.write(message)
    except Exception:
        pass


def _get_file():
    """Return the open file handle, opening it lazily on first call."""
    global _file, _metrics_path, _using_fallback_path
    if _metrics_path is None:
        _metrics_path, _using_fallback_path = _resolve_metrics_path()
    if _file is None or _file.closed:
        if _using_fallback_path:
            _warn_fallback_once()
        parent = os.path.dirname(os.path.abspath(_metrics_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        _file = open(_metrics_path, "a", encoding="utf-8")  # noqa: SIM115
        atexit.register(_close_file)
    return _file


def _close_file() -> None:
    """Close the metrics file handle if open."""
    global _file
    if _file is not None and not _file.closed:
        try:
            _file.close()
        except Exception:
            pass
    _file = None


def _write_event(event: dict) -> None:
    """Serialize *event* as a JSON line, write it, and flush."""
    try:
        line = json.dumps(event, default=str)
        f = _get_file()
        f.write(line + "\n")
        f.flush()
    except Exception:
        pass


def _utcnow_iso() -> str:
    """Return a naive UTC ISO timestamp without the deprecated utcnow call."""
    return datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat()


def report_metric(epoch: int, step: int, metrics: dict) -> None:
    """Write a metric event with training or validation metrics."""
    _write_event(
        {
            "type": "metric",
            "timestamp": _utcnow_iso(),
            "epoch": epoch,
            "step": step,
            "metrics": metrics,
        }
    )


def report_progress(epoch: int, total_epochs: int, step: int, total_steps: int) -> None:
    """Write a progress event with epoch and step counters."""
    _write_event(
        {
            "type": "progress",
            "timestamp": _utcnow_iso(),
            "epoch": epoch,
            "total_epochs": total_epochs,
            "step": step,
            "total_steps": total_steps,
        }
    )


def report_system(
    gpu_utilization: float | None = None,
    gpu_memory_used: int | None = None,
    gpu_memory_total: int | None = None,
    cpu_percent: float | None = None,
) -> None:
    """Write a system metrics event with GPU and CPU statistics."""
    event: dict = {"type": "system", "timestamp": _utcnow_iso()}
    if gpu_utilization is not None:
        event["gpu_utilization"] = gpu_utilization
    if gpu_memory_used is not None:
        event["gpu_memory_used"] = gpu_memory_used
    if gpu_memory_total is not None:
        event["gpu_memory_total"] = gpu_memory_total
    if cpu_percent is not None:
        event["cpu_percent"] = cpu_percent
    _write_event(event)


def report_log(level: str, message: str) -> None:
    """Write a log event with a severity level and message."""
    _write_event({"type": "log", "timestamp": _utcnow_iso(), "level": level, "message": message})


def report_error(
    category: str,
    technical_summary: str,
    epoch: int | None = None,
    step: int | None = None,
    traceback: str | None = None,
) -> None:
    """Write a structured training failure event before re-raising."""
    event = {
        "type": "error",
        "timestamp": _utcnow_iso(),
        "category": category,
        "technical_summary": technical_summary[:500],
    }
    if epoch is not None:
        event["epoch"] = epoch
    if step is not None:
        event["step"] = step
    if traceback is not None:
        event["traceback"] = traceback[:8192]
    _write_event(event)


def write_training_state(
    epoch: int,
    step: int,
    latest_checkpoint_path: str | None,
    latest_checkpoint_id: str | None = None,
) -> None:
    """Atomically write the crash-recovery heartbeat sidecar."""
    training_dir = os.environ.get("DAGNAM_TRAINING_DIR")
    if not training_dir:
        return

    try:
        state_path = os.path.join(training_dir, ".training_state.json")
        tmp_path = state_path + ".tmp"
        payload = {
            "epoch": epoch,
            "step": step,
            "latest_checkpoint_path": latest_checkpoint_path,
            "latest_checkpoint_id": latest_checkpoint_id,
            "last_update_iso": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload))
        os.replace(tmp_path, state_path)
    except Exception:
        pass


def _reset() -> None:
    """Reset module state and re-read the metrics path for tests."""
    global _metrics_path, _file, _using_fallback_path, _fallback_warning_emitted
    _close_file()
    _metrics_path, _using_fallback_path = _resolve_metrics_path()
    _fallback_warning_emitted = False
