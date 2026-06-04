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
import threading
from typing import Any

_DEFAULT_METRICS_PATH = "./dagnam_metrics.jsonl"
_metrics_path: str | None = None
_file = None
_using_fallback_path = False
_fallback_warning_emitted = False
SCHEMA_VERSION = "1"
_project_id: str | None = None
_schema_version: str | None = None
_uploader_thread: threading.Thread | None = None
_uploader_stop: threading.Event | None = None
_stream_finalized = False
_run_failed = False
_finalize_registered = False


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


def _write_event(event: dict[str, Any]) -> None:
    """Serialize *event* as a JSON line, write it, and flush."""
    try:
        if _schema_version is not None and "schema_version" not in event:
            event = {**event, "schema_version": _schema_version}
        line = json.dumps(event, default=str)
        f = _get_file()
        f.write(line + "\n")
        f.flush()
    except Exception:
        pass


def _utcnow_iso() -> str:
    """Return a naive UTC ISO timestamp without the deprecated utcnow call."""
    return datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat()


def report_metric(epoch: int, step: int, metrics: dict[str, float]) -> None:
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
    event: dict[str, Any] = {"type": "system", "timestamp": _utcnow_iso()}
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
    global _run_failed
    _run_failed = True
    event: dict[str, Any] = {
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


def _online_context() -> bool:
    """Return whether this local process can register an online run."""
    if os.environ.get("DAGNAM_INTERNAL") or not _project_id:
        return False
    try:
        auth_mod = __import__("dagnam._core.auth", fromlist=["get_api_key"])
        auth_mod.get_api_key()
    except Exception:
        return False
    return True


def _sdk_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("dagnam")
        except PackageNotFoundError:
            return "0+unknown"
    except Exception:
        return "0+unknown"


def _start_uploader(project_id: str, framework: str, name: str) -> None:
    """Register a local run and start its daemon uploader."""
    global _stream_finalized, _uploader_stop, _uploader_thread, _finalize_registered

    auth_mod = __import__("dagnam._core.auth", fromlist=["get_api_key", "get_api_url"])
    client_mod = __import__("dagnam._core.client", fromlist=["DagnamClient"])
    uploader_mod = __import__(
        "dagnam._core.metrics_uploader",
        fromlist=["HTTPSink", "run_upload_loop"],
    )
    client_class = client_mod.DagnamClient

    if _uploader_thread is not None and _uploader_thread.is_alive():
        return

    api_url = auth_mod.get_api_url()
    machine_client = client_class(api_url, auth_mod.get_api_key())
    # The producer doesn't know the real hyperparameters — generated train.py
    # only emits metrics. These are schema-satisfying placeholders for the
    # backend's TrainingConfig (learning_rate must be >0, batch_size >=1); the
    # platform never trains a local run, so they have no compute effect.
    config = {
        "epochs": int(os.environ.get("DAGNAM_TOTAL_EPOCHS", "1") or 1),
        "batch_size": 1,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "loss_function": "unknown",
        "dataset_config": {
            "training_dataset_id": os.environ.get(
                "DAGNAM_DATASET_ID",
                "00000000-0000-0000-0000-000000000000",
            ),
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1,
        },
        "run_name": name,
    }
    try:
        run = machine_client.register_local_run(
            project_id=project_id,
            framework=framework,
            config=config,
        )
        run_id = str(run["id"])
        token = str(machine_client.mint_run_token(run_id)["token"])
    except Exception as exc:
        sys.stderr.write(
            f"Dagnam: could not register local run ({exc}); "
            "streaming disabled, metrics still saved locally.\n"
        )
        return

    stop = threading.Event()
    _uploader_stop = stop
    _stream_finalized = False

    def _loop() -> None:
        def _refresh_upload_client():
            refreshed_token = str(machine_client.mint_run_token(run_id)["token"])
            return client_class(api_url, refreshed_token)

        upload_client = client_class(api_url, token)
        sink = uploader_mod.HTTPSink(
            upload_client,
            run_id,
            source={
                "kind": "local_stream",
                "sdk_version": _sdk_version(),
                "schema_version": SCHEMA_VERSION,
            },
            refresh_client=_refresh_upload_client,
        )
        try:
            uploader_mod.run_upload_loop(
                path=_metrics_path,
                job_id=run_id,
                sink=sink,
                should_continue=lambda: not stop.is_set(),
                replay_existing=True,
            )
        except Exception as exc:
            # A terminal ingest response (e.g. 409 after the run is cancelled or
            # already finished) is the platform's "stop streaming" signal — exit
            # quietly. Anything else is unexpected; surface it on stderr so a real
            # failure is never silently swallowed (metrics are still on disk).
            if not uploader_mod.is_terminal_upload_error(exc):
                sys.stderr.write(
                    f"Dagnam: live metrics streaming stopped unexpectedly ({exc!r}); "
                    "metrics are still saved locally.\n"
                )

    _uploader_thread = threading.Thread(target=_loop, name="dagnam-uploader", daemon=True)
    _uploader_thread.start()
    sys.stdout.write(f"Dagnam: streaming local run '{name}' live to the platform.\n")
    # Register the at-exit terminal flush only once per process; re-running init()
    # (e.g. after a previous uploader finished) must not stack duplicate handlers.
    if not _finalize_registered:
        atexit.register(_finalize_stream)
        _finalize_registered = True


def _finalize_stream() -> None:
    """Emit one local terminal event, then stop after the uploader drains."""
    global _stream_finalized
    if os.environ.get("DAGNAM_INTERNAL") or _uploader_thread is None or _stream_finalized:
        return
    _stream_finalized = True
    _write_event({"type": "failed" if _run_failed else "complete", "timestamp": _utcnow_iso()})
    if _uploader_stop is not None:
        _uploader_stop.set()
    _uploader_thread.join(timeout=30.0)


def _generated_name() -> str:
    naming_mod = __import__("dagnam._core.naming", fromlist=["generate_run_name"])
    return naming_mod.generate_run_name()


def init(
    project_id: str,
    *,
    framework: str = "pytorch",
    name: str | None = None,
    mode: str = "auto",
) -> None:
    """Bind generated training code to a project and stream when logged in."""
    global _project_id, _schema_version
    if mode not in {"auto", "off"}:
        raise ValueError("mode must be 'auto' or 'off'")
    _project_id = os.environ.get("DAGNAM_PROJECT_ID") or project_id
    _schema_version = SCHEMA_VERSION
    _get_file()
    if mode == "auto" and _online_context():
        _start_uploader(_project_id, framework, name or _generated_name())


def _reset() -> None:  # pyright: ignore[reportUnusedFunction]
    """Reset module state and re-read the metrics path for tests."""
    global _metrics_path, _file, _using_fallback_path, _fallback_warning_emitted
    global _project_id, _schema_version, _uploader_thread, _uploader_stop
    global _stream_finalized, _run_failed
    _close_file()
    _metrics_path, _using_fallback_path = _resolve_metrics_path()
    _fallback_warning_emitted = False
    _project_id = None
    _schema_version = None
    _uploader_thread = None
    _uploader_stop = None
    _stream_finalized = False
    _run_failed = False
