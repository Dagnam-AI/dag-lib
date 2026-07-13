"""Coverage for ``dagnam._core.sse`` — event parsing + reconnect loop."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import requests
from tests.typing_helpers import PytestMonkeyPatch

from dagnam._core import sse as sse_mod
from dagnam._core.exceptions import StreamError
from dagnam._core.sse import (
    TERMINAL_INFERENCE_EVENTS,
    SSEEvent,
    aiter_sse_once,
    aiter_with_reconnect,
    iter_sse_once,
    iter_with_reconnect,
    parse_raw_event,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeRawEvent:
    def __init__(
        self,
        *,
        event: object = None,
        data: object = "",
        id: object = None,
        retry: object = None,
    ) -> None:
        self.event = event
        self.data = data
        self.id = id
        self.retry = retry


RawScriptEntry = _FakeRawEvent | requests.exceptions.ConnectionError


def test_parse_raw_event_with_json_payload() -> None:
    ev = parse_raw_event(_FakeRawEvent(event="progress", data='{"step": 7}', id="e1", retry="2000"))
    assert ev == SSEEvent(event="progress", data={"step": 7}, id="e1", retry=2000)


def test_parse_raw_event_defaults_to_message_and_empty_dict() -> None:
    ev = parse_raw_event(_FakeRawEvent())
    assert ev.event == "message"
    assert ev.data == {}
    assert ev.id is None
    assert ev.retry is None


def test_parse_raw_event_falls_back_to_string_on_invalid_json() -> None:
    ev = parse_raw_event(_FakeRawEvent(event="log", data="not json"))
    assert ev.data == "not json"


def test_parse_raw_event_swallows_bad_retry() -> None:
    ev = parse_raw_event(_FakeRawEvent(event="x", data="{}", retry="abc"))
    assert ev.retry is None


def test_parse_raw_event_handles_object_without_attributes() -> None:
    """getattr fallbacks for raw events that don't expose all SSE fields."""
    ev = parse_raw_event(SimpleNamespace())
    assert ev.event == "message"
    assert ev.data == {}


class _FakeResponse:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _ExplodingResponse:
    def close(self) -> None:  # pragma: no cover - exercised via finally
        raise RuntimeError("close blew up")


class _FakeSSE:
    """Minimal sseclient.SSEClient stand-in driven by a scripted event list."""

    def __init__(self, raw_events: Sequence[RawScriptEntry]) -> None:
        self._raw_events = raw_events

    def events(self) -> Iterator[_FakeRawEvent]:
        for entry in self._raw_events:
            if isinstance(entry, Exception):
                raise entry
            yield entry


def _install_fake_sseclient(
    monkeypatch: PytestMonkeyPatch, scripts: Sequence[Sequence[RawScriptEntry]]
) -> None:
    """scripts: iterable of lists of raw-event-or-exception sequences, one per connection."""

    iterator = iter(scripts)

    class _SSEClient:
        def __init__(self, response: object) -> None:  # response unused
            self._client = _FakeSSE(next(iterator))

        def events(self) -> Iterator[_FakeRawEvent]:
            return self._client.events()

    fake_module = SimpleNamespace(SSEClient=_SSEClient)
    monkeypatch.setitem(__import__("sys").modules, "sseclient", fake_module)


def test_iter_with_reconnect_yields_events_and_stops_on_terminal(
    monkeypatch: PytestMonkeyPatch,
) -> None:
    _install_fake_sseclient(
        monkeypatch,
        [
            [
                _FakeRawEvent(event="heartbeat", data="{}"),
                _FakeRawEvent(event="progress", data='{"step": 1}', id="e1"),
                _FakeRawEvent(event="complete", data="{}"),
                _FakeRawEvent(event="extra", data="{}"),  # never reached
            ]
        ],
    )

    responses: list[_FakeResponse] = []

    def open_stream(cursor: str | None) -> _FakeResponse:
        assert cursor is None or isinstance(cursor, str)
        r = _FakeResponse()
        responses.append(r)
        return r

    events = list(
        iter_with_reconnect(
            open_stream,
            terminal_events=frozenset({"complete"}),
        )
    )
    assert [e.event for e in events] == ["progress", "complete"]
    assert responses[0].closed is True


def test_iter_with_reconnect_emits_heartbeats_when_requested(
    monkeypatch: PytestMonkeyPatch,
) -> None:
    _install_fake_sseclient(
        monkeypatch,
        [
            [
                _FakeRawEvent(event="heartbeat", data="{}"),
                _FakeRawEvent(event="stream_end", data="{}"),
            ]
        ],
    )

    def open_stream(_cursor: str | None) -> _FakeResponse:
        return _FakeResponse()

    events = list(
        iter_with_reconnect(
            open_stream,
            terminal_events=frozenset({"stream_end"}),
            include_heartbeats=True,
        )
    )
    assert [e.event for e in events] == ["heartbeat", "stream_end"]


def test_iter_with_reconnect_reconnects_after_transporterror(
    monkeypatch: PytestMonkeyPatch,
) -> None:
    _install_fake_sseclient(
        monkeypatch,
        [
            [
                _FakeRawEvent(event="progress", data='{"step": 1}', id="cursor-1"),
                requests.exceptions.ConnectionError("boom"),
            ],
            [
                _FakeRawEvent(event="complete", data="{}"),
            ],
        ],
    )

    def sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(sse_mod.time, "sleep", sleep)

    cursors_seen: list[str | None] = []

    def open_stream(cursor: str | None) -> _FakeResponse:
        cursors_seen.append(cursor)
        return _FakeResponse()

    events = list(
        iter_with_reconnect(
            open_stream,
            terminal_events=frozenset({"complete"}),
        )
    )
    assert [e.event for e in events] == ["progress", "complete"]
    # First open with no cursor, second open with the captured event id.
    assert cursors_seen == [None, "cursor-1"]


def test_iter_with_reconnect_swallows_closeerrors(monkeypatch: PytestMonkeyPatch) -> None:
    _install_fake_sseclient(
        monkeypatch,
        [
            [_FakeRawEvent(event="complete", data="{}")],
        ],
    )

    def open_stream(_cursor: str | None) -> _ExplodingResponse:
        return _ExplodingResponse()

    events = list(
        iter_with_reconnect(
            open_stream,
            terminal_events=frozenset({"complete"}),
        )
    )
    assert [e.event for e in events] == ["complete"]


def test_iter_with_reconnect_gives_up_after_max_attempts(monkeypatch: PytestMonkeyPatch) -> None:
    _install_fake_sseclient(
        monkeypatch,
        [
            [requests.exceptions.ConnectionError("boom")],
            [requests.exceptions.ConnectionError("boom")],
            [requests.exceptions.ConnectionError("boom")],
        ],
    )

    def sleep(_seconds: float) -> None:
        return None

    def uniform(_start: float, _stop: float) -> float:
        return 0.0

    monkeypatch.setattr(sse_mod.time, "sleep", sleep)
    monkeypatch.setattr(sse_mod.random, "uniform", uniform)

    def open_stream(_cursor: str | None) -> _FakeResponse:
        return _FakeResponse()

    with pytest.raises(StreamError, match="dropped after 2 reconnect attempts"):
        list(
            iter_with_reconnect(
                open_stream,
                terminal_events=frozenset({"complete"}),
                max_reconnects=2,
                backoff_base=0.01,
            )
        )


class TestAiterWithReconnect:
    """The async reconnect loop mirrors iter_with_reconnect."""

    @pytest.mark.anyio
    async def test_skips_heartbeats_and_stops_at_terminal(self) -> None:
        async def open_stream(_cursor: str | None) -> AsyncIterator[SSEEvent]:
            yield SSEEvent(event="heartbeat", data={})
            yield SSEEvent(event="progress", data={}, id="2")
            yield SSEEvent(event="complete", data="done")
            yield SSEEvent(event="progress", data={})  # pragma: no cover - after terminal

        events = [
            e
            async for e in aiter_with_reconnect(
                open_stream, terminal_events=frozenset({"complete"})
            )
        ]
        assert [e.event for e in events] == ["progress", "complete"]

    @pytest.mark.anyio
    async def test_recovers_from_transient_error(self, monkeypatch: PytestMonkeyPatch) -> None:
        monkeypatch.setattr(sse_mod.asyncio, "sleep", AsyncMock())
        scripts: list[object] = [RuntimeError("drop"), [SSEEvent(event="complete", data="x")]]

        async def open_stream(_cursor: str | None) -> AsyncIterator[SSEEvent]:
            step = scripts.pop(0)
            if isinstance(step, Exception):
                raise step
            for ev in step:  # type: ignore[union-attr]
                yield ev

        events = [
            e
            async for e in aiter_with_reconnect(
                open_stream,
                terminal_events=frozenset({"complete"}),
                transient_errors=(RuntimeError,),
            )
        ]
        assert [e.event for e in events] == ["complete"]

    @pytest.mark.anyio
    async def test_gives_up_after_max_reconnects(self) -> None:
        async def open_stream(_cursor: str | None) -> AsyncIterator[SSEEvent]:
            return
            yield  # pragma: no cover - unreachable sentinel making this an async generator

        with pytest.raises(StreamError, match="dropped after 2 reconnect"):
            _ = [
                e
                async for e in aiter_with_reconnect(
                    open_stream,
                    terminal_events=frozenset({"x"}),
                    max_reconnects=2,
                    backoff_base=0,
                )
            ]


# ------------------------------------------------------------- single-shot


def test_iter_sse_once_yields_until_terminal_and_closes(
    monkeypatch: PytestMonkeyPatch,
) -> None:
    _install_fake_sseclient(
        monkeypatch,
        [
            [
                _FakeRawEvent(event="token", data='{"token": "he"}'),
                _FakeRawEvent(event="token", data='{"token": "llo"}'),
                _FakeRawEvent(event="complete", data='{"done": true}'),
                _FakeRawEvent(event="token", data='{"token": "never"}'),
            ]
        ],
    )
    response = _FakeResponse()
    events = list(
        iter_sse_once(
            lambda: response,
            terminal_events=TERMINAL_INFERENCE_EVENTS,
        )
    )
    assert [ev.event for ev in events] == ["token", "token", "complete"]
    assert response.closed


def test_iter_sse_once_skips_heartbeats_by_default(monkeypatch: PytestMonkeyPatch) -> None:
    _install_fake_sseclient(
        monkeypatch,
        [
            [
                _FakeRawEvent(event="heartbeat", data="{}"),
                _FakeRawEvent(event="complete", data="{}"),
            ]
        ],
    )
    events = list(iter_sse_once(lambda: _FakeResponse(), terminal_events=frozenset({"complete"})))
    assert [ev.event for ev in events] == ["complete"]


def test_iter_sse_once_includes_heartbeats_when_requested(
    monkeypatch: PytestMonkeyPatch,
) -> None:
    _install_fake_sseclient(
        monkeypatch,
        [[_FakeRawEvent(event="heartbeat", data="{}"), _FakeRawEvent(event="complete", data="{}")]],
    )
    events = list(
        iter_sse_once(
            lambda: _FakeResponse(),
            terminal_events=frozenset({"complete"}),
            include_heartbeats=True,
        )
    )
    assert [ev.event for ev in events] == ["heartbeat", "complete"]


def test_iter_sse_once_raises_streamerror_on_transport_drop(
    monkeypatch: PytestMonkeyPatch,
) -> None:
    _install_fake_sseclient(
        monkeypatch,
        [
            [
                _FakeRawEvent(event="token", data='{"token": "a"}'),
                requests.exceptions.ConnectionError("boom"),
            ]
        ],
    )
    response = _FakeResponse()
    it = iter_sse_once(
        lambda: response,
        terminal_events=TERMINAL_INFERENCE_EVENTS,
        resource_label="inference stream dep-1",
    )
    assert next(it).event == "token"
    with pytest.raises(StreamError, match="inference stream dep-1 dropped"):
        list(it)
    assert response.closed


def test_iter_sse_once_raises_streamerror_when_no_terminal(
    monkeypatch: PytestMonkeyPatch,
) -> None:
    _install_fake_sseclient(monkeypatch, [[_FakeRawEvent(event="token", data='{"token": "a"}')]])
    with pytest.raises(StreamError, match="ended without a terminal event"):
        list(iter_sse_once(lambda: _FakeResponse(), terminal_events=TERMINAL_INFERENCE_EVENTS))


def test_iter_sse_once_swallows_close_errors(monkeypatch: PytestMonkeyPatch) -> None:
    _install_fake_sseclient(monkeypatch, [[_FakeRawEvent(event="complete", data="{}")]])
    events = list(
        iter_sse_once(lambda: _ExplodingResponse(), terminal_events=frozenset({"complete"}))
    )
    assert [ev.event for ev in events] == ["complete"]


@pytest.mark.anyio
async def test_aiter_sse_once_yields_until_terminal() -> None:
    async def open_stream() -> AsyncIterator[SSEEvent]:
        yield SSEEvent(event="token", data={"token": "h"})
        yield SSEEvent(event="heartbeat", data={})
        yield SSEEvent(event="complete", data={"done": True})
        yield SSEEvent(event="token", data={"token": "never"})

    got = [
        ev
        async for ev in aiter_sse_once(
            lambda: open_stream(), terminal_events=TERMINAL_INFERENCE_EVENTS
        )
    ]
    assert [ev.event for ev in got] == ["token", "complete"]


@pytest.mark.anyio
async def test_aiter_sse_once_heartbeats_when_requested() -> None:
    async def open_stream() -> AsyncIterator[SSEEvent]:
        yield SSEEvent(event="heartbeat", data={})
        yield SSEEvent(event="complete", data={})

    got = [
        ev
        async for ev in aiter_sse_once(
            lambda: open_stream(),
            terminal_events=frozenset({"complete"}),
            include_heartbeats=True,
        )
    ]
    assert [ev.event for ev in got] == ["heartbeat", "complete"]


@pytest.mark.anyio
async def test_aiter_sse_once_raises_on_transient_drop() -> None:
    async def open_stream() -> AsyncIterator[SSEEvent]:
        yield SSEEvent(event="token", data={"token": "a"})
        raise ConnectionError("dropped")

    with pytest.raises(StreamError, match="dropped mid-stream"):
        _ = [
            ev
            async for ev in aiter_sse_once(
                lambda: open_stream(), terminal_events=TERMINAL_INFERENCE_EVENTS
            )
        ]


@pytest.mark.anyio
async def test_aiter_sse_once_raises_when_no_terminal() -> None:
    async def open_stream() -> AsyncIterator[SSEEvent]:
        yield SSEEvent(event="token", data={"token": "a"})

    with pytest.raises(StreamError, match="ended without a terminal event"):
        _ = [
            ev
            async for ev in aiter_sse_once(
                lambda: open_stream(), terminal_events=TERMINAL_INFERENCE_EVENTS
            )
        ]


# ---------------------------------------------------------------------------
# Task 9: dagnam.sse reconnect logging
# ---------------------------------------------------------------------------


def test_iter_with_reconnect_logs_debug_per_reconnect_attempt(
    monkeypatch: PytestMonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Connection 1 drops mid-iteration; connection 2 delivers the terminal event.
    _install_fake_sseclient(
        monkeypatch,
        [
            [requests.exceptions.ConnectionError("boom")],
            [_FakeRawEvent(event="stream_end", data="{}")],
        ],
    )

    def open_stream(_cursor: str | None) -> _FakeResponse:
        return _FakeResponse()

    with caplog.at_level(logging.DEBUG, logger="dagnam.sse"):
        events = list(
            iter_with_reconnect(
                open_stream,
                terminal_events=frozenset({"stream_end"}),
                backoff_base=0.0,
                max_reconnects=5,
            )
        )
    assert [e.event for e in events] == ["stream_end"]
    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert debug_records
    assert "reconnect" in debug_records[0].message.lower()


def test_iter_with_reconnect_logs_warning_when_exhausted(
    monkeypatch: PytestMonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Every connection drops immediately; with max_reconnects=2 the loop gives up.
    _install_fake_sseclient(
        monkeypatch,
        [
            [requests.exceptions.ConnectionError("boom")],
            [requests.exceptions.ConnectionError("boom")],
            [requests.exceptions.ConnectionError("boom")],
        ],
    )

    def open_stream(_cursor: str | None) -> _FakeResponse:
        return _FakeResponse()

    with caplog.at_level(logging.DEBUG, logger="dagnam.sse"), pytest.raises(StreamError):
        list(
            iter_with_reconnect(
                open_stream,
                terminal_events=frozenset({"stream_end"}),
                backoff_base=0.0,
                max_reconnects=2,
            )
        )
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning_records
    assert "giving up" in warning_records[0].message.lower()


# ---------------------------------------------------------------------------
# Task 3b: dagnam.sse reconnect logging — async mirror (aiter_with_reconnect)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_aiter_with_reconnect_logs_debug_per_reconnect_attempt(
    monkeypatch: PytestMonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Connection 1 drops mid-iteration; connection 2 delivers the terminal event.
    monkeypatch.setattr(sse_mod.asyncio, "sleep", AsyncMock())
    scripts: list[object] = [RuntimeError("drop"), [SSEEvent(event="stream_end", data="{}")]]

    async def open_stream(_cursor: str | None) -> AsyncIterator[SSEEvent]:
        step = scripts.pop(0)
        if isinstance(step, Exception):
            raise step
        for ev in step:  # type: ignore[union-attr]
            yield ev

    with caplog.at_level(logging.DEBUG, logger="dagnam.sse"):
        events = [
            e
            async for e in aiter_with_reconnect(
                open_stream,
                terminal_events=frozenset({"stream_end"}),
                transient_errors=(RuntimeError,),
                max_reconnects=5,
            )
        ]
    assert [e.event for e in events] == ["stream_end"]
    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert debug_records
    assert "reconnect" in debug_records[0].message.lower()


@pytest.mark.anyio
async def test_aiter_with_reconnect_logs_warning_when_exhausted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Every connection ends without a terminal event; max_reconnects=2 -> give up.
    async def open_stream(_cursor: str | None) -> AsyncIterator[SSEEvent]:
        return
        yield  # pragma: no cover - unreachable sentinel making this an async generator

    with caplog.at_level(logging.DEBUG, logger="dagnam.sse"), pytest.raises(StreamError):
        _ = [
            e
            async for e in aiter_with_reconnect(
                open_stream,
                terminal_events=frozenset({"stream_end"}),
                backoff_base=0.0,
                max_reconnects=2,
            )
        ]
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning_records
    assert "giving up" in warning_records[0].message.lower()
