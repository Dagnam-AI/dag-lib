"""Shared JSONL tail, batch, and retry engine for training metrics."""

from __future__ import annotations

from collections.abc import Callable
import os
import sys
import time
from typing import Any, Protocol

from dagnam._core.exceptions import APIError, AuthError, TrainingJobNotFoundError
from dagnam.training_attach import MetricsJsonlTailer, event_with_id


class Sink(Protocol):
    """Transport boundary for a batch of metric events."""

    def send(self, events: list[dict[str, Any]]) -> None: ...


class ListSink:
    """In-memory sink used by focused tests."""

    def __init__(self) -> None:
        self.batches: list[list[dict[str, Any]]] = []

    def send(self, events: list[dict[str, Any]]) -> None:
        if events:
            self.batches.append([dict(event) for event in events])


class HTTPSink:
    """Send batches through a configured ``DagnamClient``."""

    def __init__(
        self,
        client: Any,
        job_id: str,
        *,
        source: dict[str, Any] | None = None,
        refresh_client: Callable[[], Any] | None = None,
    ) -> None:
        self._client = client
        self._job_id = job_id
        self._source = source
        self._refresh_client = refresh_client

    def send(self, events: list[dict[str, Any]]) -> None:
        if events:
            try:
                self._send(events)
            except AuthError:
                if self._refresh_client is None:
                    raise
                self._client = self._refresh_client()
                self._send(events)

    def _send(self, events: list[dict[str, Any]]) -> None:
        """Send one batch with the current client."""
        if events:
            if self._source is None:
                self._client.upload_training_events(self._job_id, events)
            else:
                self._client.upload_training_events(self._job_id, events, source=self._source)


class UploadRetriesExhaustedError(Exception):
    """Final metric drain still has pending events after bounded retries."""


def is_terminal_upload_error(exc: Exception) -> bool:
    """Return whether retrying an upload cannot recover."""
    if isinstance(exc, (AuthError, TrainingJobNotFoundError)):
        return True
    return isinstance(exc, APIError) and 400 <= exc.status_code < 500 and exc.status_code != 429


def run_upload_loop(
    *,
    path: str | os.PathLike[str],
    job_id: str,
    sink: Sink,
    should_continue: Callable[[], bool],
    replay: bool = False,
    replay_existing: bool = False,
    poll_interval: float = 1.0,
    batch_size: int = 50,
    retry_initial_interval: float = 1.0,
    retry_max_interval: float = 30.0,
    max_pending: int = 10000,
    final_upload_attempts: int = 3,
) -> int:
    """Tail complete JSONL events and upload batches until told to stop."""
    tailer = MetricsJsonlTailer(path, replay=replay or replay_existing)
    pending: list[dict[str, Any]] = []
    upload_backoff = retry_initial_interval
    overflow_warned = False
    sent = 0

    def cap_pending() -> None:
        nonlocal overflow_warned
        overflow = len(pending) - max_pending
        if overflow <= 0:
            return
        del pending[:overflow]
        if not overflow_warned:
            overflow_warned = True
            sys.stderr.write(
                f"Dagnam metrics upload backlog exceeded {max_pending} events; "
                "dropping oldest events to bound memory while retries continue.\n"
            )

    def flush_once() -> bool:
        nonlocal sent, upload_backoff
        if not pending:
            return True
        try:
            sink.send(pending.copy())
        except Exception as exc:
            if is_terminal_upload_error(exc):
                raise
            sys.stderr.write(f"Failed to upload Dagnam training metrics; retrying: {exc}\n")
            if upload_backoff > 0:
                time.sleep(upload_backoff)
            upload_backoff = min(
                max(upload_backoff * 2, retry_initial_interval),
                retry_max_interval,
            )
            return False
        sent += len(pending)
        pending.clear()
        upload_backoff = retry_initial_interval
        return True

    def flush_final() -> bool:
        for _ in range(final_upload_attempts):
            if flush_once():
                return True
        return not pending

    def drain_once() -> None:
        for item in tailer.read_available():
            pending.append(event_with_id(job_id, item))
            cap_pending()
            if len(pending) >= batch_size:
                flush_once()

    if replay:
        drain_once()
        if not flush_final():
            raise UploadRetriesExhaustedError
        return sent

    try:
        while should_continue():
            drain_once()
            flush_once()
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        drain_once()
        flush_once()
        raise

    drain_once()
    if not flush_final():
        raise UploadRetriesExhaustedError
    return sent


def drain_jsonl_to_sink(
    *,
    path: str | os.PathLike[str],
    job_id: str,
    sink: Sink,
    replay: bool = True,
) -> int:
    """Drain existing JSONL content once."""
    return run_upload_loop(
        path=path,
        job_id=job_id,
        sink=sink,
        should_continue=lambda: False,
        replay=replay,
    )
