"""Shared SSE iteration + reconnect logic.

Used by ``dagnam.training.stream_training`` and ``dagnam.deployments.stream_logs``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from importlib import import_module
import json
import random
import time
from typing import Callable, Optional, Protocol, cast

import requests

from dagnam._types import JsonObject, ensure_json_object
from dagnam._core.exceptions import StreamError

DEFAULT_MAX_RECONNECTS = 5
DEFAULT_BACKOFF_BASE = 1.0

TERMINAL_TRAINING_EVENTS = frozenset({"complete", "failed", "cancelled", "stream_end"})
TERMINAL_DEPLOYMENT_EVENTS = frozenset({"deployment_ready", "deployment_failed", "stream_end"})


@dataclass
class SSEEvent:
    """A parsed server-sent event."""

    event: str
    data: JsonObject | str
    id: Optional[str] = None
    retry: Optional[int] = None


class RawSSEEvent(Protocol):
    """Event object returned by sseclient."""

    event: object
    data: object
    id: object
    retry: object


class ClosableResponse(Protocol):
    """Response-like stream object that can be closed after iteration."""

    def close(self) -> None: ...


class SSEClientInstance(Protocol):
    """sseclient instance surface used by this module."""

    def events(self) -> Iterator[RawSSEEvent]: ...


class SSEClientFactory(Protocol):
    """sseclient.SSEClient constructor surface."""

    def __call__(self, event_source: object) -> SSEClientInstance: ...


class SSEClientModule(Protocol):
    """sseclient module surface used by this module."""

    SSEClient: SSEClientFactory


def parse_raw_event(raw: object) -> SSEEvent:
    raw_event_type = getattr(raw, "event", None) or "message"
    event_type = raw_event_type if isinstance(raw_event_type, str) else str(raw_event_type)
    raw_data = getattr(raw, "data", None) or ""
    data_str = raw_data if isinstance(raw_data, str) else str(raw_data)
    try:
        loaded: object = cast(object, json.loads(data_str)) if data_str else cast(object, {})
        data = ensure_json_object(cast(object, loaded)) if isinstance(loaded, dict) else data_str
    except (json.JSONDecodeError, ValueError):
        data = data_str

    raw_event_id = getattr(raw, "id", None)
    event_id = raw_event_id if isinstance(raw_event_id, str) else None
    retry_attr = getattr(raw, "retry", None)
    if isinstance(retry_attr, str | bytes | int | float | bool):
        try:
            retry = int(retry_attr)
        except ValueError:
            retry = None
    else:
        retry = None

    return SSEEvent(event=event_type, data=data, id=event_id, retry=retry)


def iter_with_reconnect(
    open_stream: Callable[[Optional[str]], ClosableResponse],
    *,
    terminal_events: frozenset[str],
    include_heartbeats: bool = False,
    max_reconnects: int = DEFAULT_MAX_RECONNECTS,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    resource_label: str = "SSE stream",
    last_event_id: Optional[str] = None,
) -> Iterator[SSEEvent]:
    """Yield parsed SSE events, transparently reconnecting on transport errors.

    ``open_stream(cursor)`` must return a streaming ``requests.Response`` with
    an open SSE body; the caller owns HTTP errors (it should raise inside
    ``open_stream`` before returning).
    """
    try:
        sseclient = cast(SSEClientModule, import_module("sseclient"))
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sseclient-py is required for SSE streams. "
            "Install with: pip install 'dagnam[streaming]' or pip install sseclient-py"
        ) from exc

    attempts = 0
    cursor = last_event_id

    while True:
        response = open_stream(cursor)
        try:
            sse = sseclient.SSEClient(response)
            for raw in sse.events():
                ev = parse_raw_event(raw)
                if ev.id:
                    cursor = ev.id
                if ev.event == "heartbeat" and not include_heartbeats:
                    continue
                yield ev
                if ev.event in terminal_events:
                    return
        except (requests.exceptions.RequestException, ConnectionError, OSError):
            pass
        finally:
            try:
                response.close()
            except Exception:
                pass

        attempts += 1
        if attempts > max_reconnects:
            raise StreamError(f"{resource_label} dropped after {max_reconnects} reconnect attempts")
        delay = backoff_base * (2 ** (attempts - 1))
        time.sleep(delay + random.uniform(0, delay * 0.1))
