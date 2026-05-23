"""Coverage for ``dagnam._core.sse`` — event parsing + reconnect loop."""

from __future__ import annotations
from collections.abc import Iterator, Sequence
from tests.typing_helpers import PytestMonkeyPatch


from types import SimpleNamespace

import pytest
import requests

from dagnam._core import sse as sse_mod
from dagnam._core.exceptions import StreamError
from dagnam._core.sse import (
    SSEEvent,
    iter_with_reconnect,
    parse_raw_event,
)


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


def test_iter_with_reconnect_yields_events_and_stops_on_terminal(monkeypatch: PytestMonkeyPatch) -> None:
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


def test_iter_with_reconnect_emits_heartbeats_when_requested(monkeypatch: PytestMonkeyPatch) -> None:
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


def test_iter_with_reconnect_reconnects_after_transporterror(monkeypatch: PytestMonkeyPatch) -> None:
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
