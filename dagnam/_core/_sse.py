"""Shared SSE iteration + reconnect logic.

Used by ``dagnam.training.stream_training`` and ``dagnam.deployments.stream_logs``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import requests

from dagnam._core.exceptions import StreamError

DEFAULT_MAX_RECONNECTS = 5
DEFAULT_BACKOFF_BASE = 1.0

TERMINAL_TRAINING_EVENTS = frozenset(
    {"complete", "failed", "cancelled", "stream_end"}
)
TERMINAL_DEPLOYMENT_EVENTS = frozenset(
    {"deployment_ready", "deployment_failed", "stream_end"}
)


@dataclass
class SSEEvent:
    """A parsed server-sent event."""

    event: str
    data: dict | str
    id: Optional[str] = None
    retry: Optional[int] = None


def parse_raw_event(raw) -> SSEEvent:
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

    return SSEEvent(event=event_type, data=data, id=event_id, retry=retry)


def iter_with_reconnect(
    open_stream: Callable[[Optional[str]], requests.Response],
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
        import sseclient  # type: ignore[import]
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
            raise StreamError(
                f"{resource_label} dropped after {max_reconnects} reconnect attempts"
            )
        time.sleep(backoff_base * (2 ** (attempts - 1)))
