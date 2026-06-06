"""Authenticated local training metrics attach/upload helpers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from dagnam._core.auth import get_api_key, get_api_url
from dagnam._core.client import DagnamClient
from dagnam._core.config import get_config_value

DEFAULT_ATTACH_METRICS_PATH = "dagnam_metrics.jsonl"


class MetricsJsonlTailer:
    """Read complete JSONL events from a metrics file without losing partial lines."""

    def __init__(self, path: str | os.PathLike[str], *, replay: bool = False) -> None:
        self.path = Path(path)
        self._offset = 0
        self._partial = b""
        if self.path.exists() and not replay:
            self._offset = self.path.stat().st_size

    def read_available(self) -> Iterator[dict[str, Any]]:
        """Yield newly available complete JSONL object events."""
        if not self.path.exists():
            return

        with self.path.open("rb") as fh:
            fh.seek(self._offset)
            data = fh.read()

        if not data:
            return

        base_offset = self._offset - len(self._partial)
        combined = self._partial + data
        self._partial = b""
        cursor = base_offset

        parts = combined.split(b"\n")
        complete_parts = parts[:-1]
        if parts[-1]:
            self._partial = parts[-1]
        else:
            self._partial = b""

        for raw_bytes in complete_parts:
            line_len = len(raw_bytes) + 1
            event_offset = cursor
            cursor += line_len
            if event_offset == 0:
                raw_bytes = raw_bytes.removeprefix(b"\xef\xbb\xbf")
            raw = raw_bytes.rstrip(b"\r").decode("utf-8", errors="replace")
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                sys.stderr.write(
                    f"Skipping malformed metrics JSONL line at offset {event_offset}\n"
                )
                continue
            if isinstance(event, dict):
                yield {"offset": event_offset, "event": event}
            else:
                sys.stderr.write(
                    f"Skipping non-object metrics JSONL line at offset {event_offset}\n"
                )

        self._offset = cursor + len(self._partial)


def resolve_attach_metrics_path(
    explicit: str | None = None,
    *,
    require_existing_source: bool = False,
) -> Path:
    """Resolve attach metrics path using CLI/env/config/default rules."""
    if explicit:
        return Path(explicit)

    env_path = os.environ.get("DAGNAM_METRICS_PATH")
    if env_path:
        return Path(env_path)

    config_path = get_config_value("training_metrics_path")
    if isinstance(config_path, str) and config_path:
        return Path(config_path)

    if require_existing_source:
        raise FileNotFoundError(
            "Metrics path is not configured. Run: dagnam training attach <job-id> "
            "--metrics-path ./dagnam_metrics.jsonl"
        )

    path = Path.cwd() / DEFAULT_ATTACH_METRICS_PATH
    sys.stdout.write(f"Using local metrics path: {path}\n")
    return path


def event_with_id(job_id: str, item: dict[str, Any]) -> dict[str, Any]:
    """Return an uploaded event with a stable file-offset identifier."""
    event = dict(item["event"])
    event.setdefault("event_id", f"{job_id}:{item['offset']}")
    return event


def run_training_attach(
    *,
    job_id: str,
    metrics_path: str | os.PathLike[str] | None,
    command: Sequence[str] | None = None,
    replay: bool = False,
    client: Any | None = None,
    poll_interval: float = 1.0,
    batch_size: int = 50,
    retry_initial_interval: float = 1.0,
    retry_max_interval: float = 30.0,
    max_pending: int = 10000,
    final_upload_attempts: int = 3,
    child_shutdown_timeout: float = 5.0,
) -> int:
    """Run an attach session, optionally launching a child training command."""
    command = list(command or [])
    path = (
        Path(metrics_path)
        if metrics_path is not None
        else resolve_attach_metrics_path(None, require_existing_source=not command)
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    from dagnam._core.metrics_uploader import (
        HTTPSink,
        UploadRetriesExhaustedError,
        run_upload_loop,
    )

    resolved_client = client or DagnamClient(get_api_url(), get_api_key())
    sink = HTTPSink(resolved_client, job_id)
    process = None
    if command:
        env = dict(os.environ)
        env["DAGNAM_METRICS_PATH"] = str(path)
        process = subprocess.Popen(command, env=env)  # noqa: S603

    def terminate_child() -> None:
        if process is None:
            return  # pragma: no cover -- unreachable: sole caller (L220) guards `process is not None`; `process` is assigned only at L147/151
        if os.name == "nt" and getattr(process, "pid", None) is not None:
            subprocess.run(  # noqa: S603
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],  # noqa: S607
                capture_output=True,
                check=False,
            )
            wait = getattr(process, "wait", None)
            if wait is not None:
                wait()
            return
        process.terminate()
        wait = getattr(process, "wait", None)
        if wait is None:
            return
        try:
            wait(timeout=child_shutdown_timeout)
        except TypeError:
            wait()
        except subprocess.TimeoutExpired:
            process.kill()
            wait()

    process_reaped = False
    try:
        if process is None:
            sys.stdout.write(
                f"Watching {path} for Dagnam training metrics. Press Ctrl+C to stop.\n"
            )
            run_upload_loop(
                path=path,
                job_id=job_id,
                sink=sink,
                should_continue=lambda: True,
                replay=replay,
                poll_interval=poll_interval,
                batch_size=batch_size,
                retry_initial_interval=retry_initial_interval,
                retry_max_interval=retry_max_interval,
                max_pending=max_pending,
                final_upload_attempts=final_upload_attempts,
            )
            return 0

        run_upload_loop(
            path=path,
            job_id=job_id,
            sink=sink,
            should_continue=lambda: process.poll() is None,
            poll_interval=poll_interval,
            batch_size=batch_size,
            retry_initial_interval=retry_initial_interval,
            retry_max_interval=retry_max_interval,
            max_pending=max_pending,
            final_upload_attempts=final_upload_attempts,
        )
        exit_code = int(process.wait())
        process_reaped = True
        return exit_code
    except UploadRetriesExhaustedError:
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        if process is not None and not process_reaped:
            terminate_child()
