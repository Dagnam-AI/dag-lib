"""Authenticated local training metrics attach/upload helpers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from dagnam._core.auth import get_api_key, get_api_url
from dagnam._core.client import DagnamClient
from dagnam._core.config import get_config_value
from dagnam._core.exceptions import APIError, AuthError, TrainingJobNotFoundError

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
            raw = raw_bytes.rstrip(b"\r").decode("utf-8", errors="replace")
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                sys.stderr.write(f"Skipping malformed metrics JSONL line at offset {event_offset}\n")
                continue
            if isinstance(event, dict):
                yield {"offset": event_offset, "event": event}
            else:
                sys.stderr.write(f"Skipping non-object metrics JSONL line at offset {event_offset}\n")

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


def _event_with_id(job_id: str, item: dict[str, Any]) -> dict[str, Any]:
    event = dict(item["event"])
    event.setdefault("event_id", f"{job_id}:{item['offset']}")
    return event


def _upload_batch(client: Any, job_id: str, batch: list[dict[str, Any]]) -> None:
    if batch:
        client.upload_training_events(job_id, batch)


def _is_terminal_upload_error(exc: Exception) -> bool:
    """Return True for upload failures that retrying cannot fix.

    Auth (401), job-not-found (404), and other client errors (4xx except 429
    Too Many Requests) will keep failing identically, so the attach session
    should surface them and exit rather than loop forever. Connection errors
    and timeouts surface as ``APIError(0, ...)`` and 5xx/429 stay transient.
    """
    if isinstance(exc, (AuthError, TrainingJobNotFoundError)):
        return True
    if isinstance(exc, APIError):
        return 400 <= exc.status_code < 500 and exc.status_code != 429
    return False


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
) -> int:
    """Run an attach session, optionally launching a child training command."""
    command = list(command or [])
    path = Path(metrics_path) if metrics_path is not None else resolve_attach_metrics_path(
        None, require_existing_source=not command
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    resolved_client = client or DagnamClient(get_api_url(), get_api_key())
    tailer = MetricsJsonlTailer(path, replay=replay)
    process = None
    if command:
        env = dict(os.environ)
        env["DAGNAM_METRICS_PATH"] = str(path)
        process = subprocess.Popen(command, env=env)  # noqa: S603

    pending: list[dict[str, Any]] = []
    upload_backoff = retry_initial_interval
    overflow_warning_emitted = False

    def cap_pending() -> None:
        # Bound memory during a sustained upload outage: the tailer keeps
        # appending newly-written events while uploads retry, so drop the
        # oldest backlog once it exceeds max_pending and keep the newest.
        nonlocal overflow_warning_emitted
        overflow = len(pending) - max_pending
        if overflow <= 0:
            return
        del pending[:overflow]
        if not overflow_warning_emitted:
            overflow_warning_emitted = True
            sys.stderr.write(
                f"Dagnam metrics upload backlog exceeded {max_pending} events; "
                "dropping oldest events to bound memory while retries continue.\n"
            )

    def drain_once() -> None:
        for item in tailer.read_available():
            pending.append(_event_with_id(job_id, item))
            cap_pending()
            if len(pending) >= batch_size:
                upload_pending()

    def upload_pending() -> None:
        nonlocal upload_backoff
        if not pending:
            return
        try:
            _upload_batch(resolved_client, job_id, pending.copy())
        except Exception as exc:
            if _is_terminal_upload_error(exc):
                # Auth/not-found/4xx won't recover on retry: stop tailing and
                # let the error propagate so the user sees why it failed.
                raise
            sys.stderr.write(f"Failed to upload Dagnam training metrics; retrying: {exc}\n")
            if upload_backoff > 0:
                time.sleep(upload_backoff)
            upload_backoff = min(max(upload_backoff * 2, retry_initial_interval), retry_max_interval)
            return
        pending.clear()
        upload_backoff = retry_initial_interval

    process_reaped = False
    try:
        if process is None:
            sys.stdout.write(f"Watching {path} for Dagnam training metrics. Press Ctrl+C to stop.\n")
            while True:
                drain_once()
                upload_pending()
                time.sleep(poll_interval)

        while process.poll() is None:
            drain_once()
            upload_pending()
            time.sleep(poll_interval)

        drain_once()
        upload_pending()
        exit_code = int(process.wait())
        process_reaped = True
        return exit_code
    except KeyboardInterrupt:
        drain_once()
        upload_pending()
        return 130
    finally:
        # Never orphan the child: terminate it on Ctrl+C or when a terminal
        # upload error propagates out. Skip only when it already exited and
        # was reaped on the normal path.
        if process is not None and not process_reaped:
            process.terminate()
