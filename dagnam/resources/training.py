"""SSE client for live training job events.

Wraps ``sseclient-py`` around the authenticated backend stream so Python
callers can iterate training events just like the frontend EventSource.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from importlib import import_module
import json
import random
import time
from typing import Optional, Protocol, cast

import requests

from dagnam._types import JsonObject, ensure_json_object
from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import StreamError
from dagnam._core.resolver import resolve_client

_TERMINAL_EVENTS = {"complete", "failed", "cancelled", "stream_end"}
_MAX_RECONNECTS = 5
_BACKOFF_BASE = 1.0  # seconds


@dataclass
class TrainingEvent:
    """Parsed SSE event from a training stream."""

    event: str  # e.g. "metric", "log", "progress", "heartbeat", "complete"
    data: JsonObject | str  # JSON-decoded payload if possible, else raw string
    id: Optional[str] = None
    retry: Optional[int] = None


class RawSSEEvent(Protocol):
    event: object
    data: object
    id: object
    retry: object


class SSEClientInstance(Protocol):
    def events(self) -> Iterator[RawSSEEvent]: ...


class SSEClientFactory(Protocol):
    def __call__(self, event_source: object) -> SSEClientInstance: ...


class SSEClientModule(Protocol):
    SSEClient: SSEClientFactory


def parse_event(raw: RawSSEEvent) -> TrainingEvent:
    raw_event_type = raw.event or "message"
    event_type = raw_event_type if isinstance(raw_event_type, str) else str(raw_event_type)
    raw_data = raw.data or ""
    data_str = raw_data if isinstance(raw_data, str) else str(raw_data)
    try:
        loaded: object = cast(object, json.loads(data_str)) if data_str else cast(object, {})
        data = ensure_json_object(cast(object, loaded)) if isinstance(loaded, dict) else data_str
    except (json.JSONDecodeError, ValueError):
        data = data_str

    raw_event_id = raw.id
    event_id = raw_event_id if isinstance(raw_event_id, str) else None
    retry_attr = raw.retry
    if isinstance(retry_attr, str | bytes | int | float | bool):
        try:
            retry = int(retry_attr)
        except ValueError:
            retry = None
    else:
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
        sseclient = cast(SSEClientModule, import_module("sseclient"))
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
                ev = parse_event(raw)
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
        delay = _BACKOFF_BASE * (2 ** (attempts - 1))
        time.sleep(delay + random.uniform(0, delay * 0.1))
