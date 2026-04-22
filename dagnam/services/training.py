"""SSE client for live training job events.

Wraps ``sseclient-py`` around the authenticated backend stream so Python
callers can iterate training events just like the frontend EventSource.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import requests

from dagnam._core._resolver import resolve_client
from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import StreamError

_TERMINAL_EVENTS = {"complete", "failed", "cancelled", "stream_end"}
_MAX_RECONNECTS = 5
_BACKOFF_BASE = 1.0  # seconds


@dataclass
class TrainingEvent:
    """Parsed SSE event from a training stream."""

    event: str  # e.g. "metric", "log", "progress", "heartbeat", "complete"
    data: dict | str  # JSON-decoded payload if possible, else raw string
    id: Optional[str] = None
    retry: Optional[int] = None


def _parse_event(raw) -> TrainingEvent:
    event_type = getattr(raw, "event", None) or "message"
    data_str = getattr(raw, "data", "") or ""
    try:
        data = json.loads(data_str) if data_str else {}
    except (json.JSONDecodeError, ValueError):
        data = data_str

    event_id = getattr(raw, "id", None)
    retry_attr = getattr(raw, "retry", None)
    try:
        retry = int(retry_attr) if retry_attr is not None else None
    except (TypeError, ValueError):
        retry = None

    return TrainingEvent(event=event_type, data=data, id=event_id, retry=retry)


def stream_training(
    job_id: str,
    *,
    last_event_id: Optional[str] = None,
    include_heartbeats: bool = False,
    max_reconnects: int = _MAX_RECONNECTS,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> Iterator[TrainingEvent]:
    """Yield training events for ``job_id`` until the job terminates.

    Transparently reconnects on transport errors using the last seen event
    ID, up to ``max_reconnects`` times with exponential backoff.

    >>> for ev in dagnam.stream_training("job_xyz"):
    ...     if ev.event == "metric":
    ...         print(ev.data["epoch"], ev.data["loss"])
    """
    try:
        import sseclient  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sseclient-py is required for dagnam.stream_training. "
            "Install with: pip install 'dagnam[streaming]' or pip install sseclient-py"
        ) from exc

    resolved = resolve_client(client, api_key, api_url)
    attempts = 0
    cursor = last_event_id

    while True:
        response = resolved.open_training_stream(job_id, last_event_id=cursor)
        try:
            sse = sseclient.SSEClient(response)
            for raw in sse.events():
                ev = _parse_event(raw)
                if ev.id:
                    cursor = ev.id
                if ev.event == "heartbeat" and not include_heartbeats:
                    continue
                yield ev
                if ev.event in _TERMINAL_EVENTS:
                    return
            # Stream closed without a terminal event — fall through to reconnect.
        except (requests.exceptions.RequestException, ConnectionError, OSError):
            # Transport dropped; fall through to reconnect.
            pass
        finally:
            try:
                response.close()
            except Exception:
                pass

        attempts += 1
        if attempts > max_reconnects:
            raise StreamError(
                f"Training stream for job {job_id} dropped after "
                f"{max_reconnects} reconnect attempts"
            )
        time.sleep(_BACKOFF_BASE * (2 ** (attempts - 1)))
