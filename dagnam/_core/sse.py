"""Shared SSE iteration + reconnect logic.

Used by ``dagnam.training.stream_training`` and ``dagnam.deployments.stream_logs``.

Note on event size: a single SSE ``data:`` field is not length-bounded at this
layer — the sync ``sseclient`` and async ``httpx_sse`` decoders yield whatever
the server sends before a newline, and only then does :func:`parse_raw_event`
``json.loads`` it. A hostile or compromised streaming endpoint could therefore
send one very large unterminated event to pressure client memory. Callers that
stream from an untrusted endpoint should treat that as the trust boundary; the
authoritative mitigation is server-side (bounded event framing), not this
client, which cannot cap a line the upstream decoder has already buffered.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from importlib import import_module
import json
import random
import time
from typing import Callable, Optional, Protocol, cast

import requests

from dagnam._core.exceptions import StreamError
from dagnam._types import JsonObject, ensure_json_object

DEFAULT_MAX_RECONNECTS = 50
DEFAULT_BACKOFF_BASE = 1.0

TERMINAL_TRAINING_EVENTS = frozenset({"complete", "failed", "cancelled", "stream_end"})
TERMINAL_DEPLOYMENT_EVENTS = frozenset({"deployment_ready", "deployment_failed", "stream_end"})
TERMINAL_INFERENCE_EVENTS = frozenset({"complete", "error"})


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
        loaded: object = cast("object", json.loads(data_str)) if data_str else cast("object", {})
        data = ensure_json_object(cast("object", loaded)) if isinstance(loaded, dict) else data_str
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
        sseclient = cast("SSEClientModule", import_module("sseclient"))
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sseclient-py is required for SSE streams. "
            "Install with: pip install 'dagnam[streaming]' or pip install sseclient-py"
        ) from exc

    attempts = 0
    cursor = last_event_id

    while True:
        made_progress = False
        response = open_stream(cursor)
        try:
            sse = sseclient.SSEClient(response)
            for raw in sse.events():
                ev = parse_raw_event(raw)
                if ev.id:
                    cursor = ev.id
                made_progress = True
                attempts = 0
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

        if not made_progress:
            attempts += 1
            if attempts > max_reconnects:
                raise StreamError(
                    f"{resource_label} dropped after {max_reconnects} reconnect attempts"
                )
        delay = backoff_base * (2 ** (min(attempts, 6) - 1)) if attempts else backoff_base
        if delay:
            time.sleep(delay + random.uniform(0, delay * 0.1))


async def aiter_with_reconnect(
    open_stream: Callable[[Optional[str]], AsyncIterator[SSEEvent]],
    *,
    terminal_events: frozenset[str],
    transient_errors: tuple[type[BaseException], ...] = (ConnectionError, OSError),
    include_heartbeats: bool = False,
    max_reconnects: int = DEFAULT_MAX_RECONNECTS,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    resource_label: str = "SSE stream",
    last_event_id: Optional[str] = None,
) -> AsyncIterator[SSEEvent]:
    """Async counterpart of :func:`iter_with_reconnect`.

    ``open_stream(cursor)`` returns an async iterator over one connection's
    events. A connect-time failure it raises (a 404/auth error, or an already
    translated ``APIError``) surfaces immediately; a mid-stream drop — one of
    ``transient_errors``, or the iterator simply ending without a terminal
    event — triggers a reconnect. Each reconnect re-invokes ``open_stream``,
    re-minting the short-lived stream token and preserving the Last-Event-ID
    cursor, so a stream outlives its token's TTL. After ``max_reconnects``
    consecutive no-progress attempts a :class:`StreamError` is raised, so a
    dropped stream is never silently mistaken for completion.
    """
    attempts = 0
    cursor = last_event_id

    while True:
        made_progress = False
        try:
            async for ev in open_stream(cursor):
                if ev.id:
                    cursor = ev.id
                made_progress = True
                attempts = 0
                if ev.event == "heartbeat" and not include_heartbeats:
                    continue
                yield ev
                if ev.event in terminal_events:
                    return
        except transient_errors:
            pass

        if not made_progress:
            attempts += 1
            if attempts > max_reconnects:
                raise StreamError(
                    f"{resource_label} dropped after {max_reconnects} reconnect attempts"
                )
        delay = backoff_base * (2 ** (min(attempts, 6) - 1)) if attempts else backoff_base
        if delay:
            await asyncio.sleep(delay + random.uniform(0, delay * 0.1))


def iter_sse_once(
    open_stream: Callable[[], ClosableResponse],
    *,
    terminal_events: frozenset[str],
    include_heartbeats: bool = False,
    resource_label: str = "SSE stream",
) -> Iterator[SSEEvent]:
    """Yield parsed SSE events from a single connection — never reconnects.

    For non-resumable streams (streaming inference): a reconnect would replay
    generation from scratch, so a mid-stream drop or an end-of-body without a
    terminal event raises :class:`StreamError` instead.
    """
    try:
        sseclient = cast("SSEClientModule", import_module("sseclient"))
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sseclient-py is required for SSE streams. "
            "Install with: pip install 'dagnam[streaming]' or pip install sseclient-py"
        ) from exc

    response = open_stream()
    try:
        sse = sseclient.SSEClient(response)
        for raw in sse.events():
            ev = parse_raw_event(raw)
            if ev.event == "heartbeat" and not include_heartbeats:
                continue
            yield ev
            if ev.event in terminal_events:
                return
    except (requests.exceptions.RequestException, ConnectionError, OSError) as exc:
        raise StreamError(f"{resource_label} dropped mid-stream: {exc}") from exc
    finally:
        try:
            response.close()
        except Exception:
            pass
    raise StreamError(f"{resource_label} ended without a terminal event")


async def aiter_sse_once(
    open_stream: Callable[[], AsyncIterator[SSEEvent]],
    *,
    terminal_events: frozenset[str],
    transient_errors: tuple[type[BaseException], ...] = (ConnectionError, OSError),
    include_heartbeats: bool = False,
    resource_label: str = "SSE stream",
) -> AsyncIterator[SSEEvent]:
    """Async counterpart of :func:`iter_sse_once` (single connection, no retry)."""
    try:
        async for ev in open_stream():
            if ev.event == "heartbeat" and not include_heartbeats:
                continue
            yield ev
            if ev.event in terminal_events:
                return
    except transient_errors as exc:
        raise StreamError(f"{resource_label} dropped mid-stream: {exc}") from exc
    raise StreamError(f"{resource_label} ended without a terminal event")
