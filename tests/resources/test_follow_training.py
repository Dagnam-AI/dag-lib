"""``follow_training``: a training stream that ends when the job is paused, however it says so.

A pause reaches a client three ways: a ``paused`` event, a ``status`` event whose ``new_status`` is
``paused``, or a ``stream_end`` whose ``reason`` is ``paused``. A stream opened on an already paused
job may say nothing at all and send heartbeats for ever, so the job's status is read when the
follower starts and then at most once per ``check_every`` seconds while events (heartbeats
included) arrive. That is the stop condition: it needs no particular event from the platform.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import itertools
from typing import TYPE_CHECKING

import pytest

from dagnam._core.exceptions import APIError
from dagnam._core.sse import SSEEvent, is_pause
from dagnam._types import JsonObject
from dagnam.resources import training as training_mod

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch

PAUSE = "Paused at epoch 3: there weren't enough credits."


def _ev(name: str, data: JsonObject | str | None = None) -> SSEEvent:
    return SSEEvent(event=name, data={} if data is None else data)


class _World:
    """A fake platform: a stream of events and a job record that can change over time."""

    def __init__(self, monkeypatch: PytestMonkeyPatch) -> None:
        self.now = 0.0
        self.statuses: Callable[[int], object] = lambda _n: {"status": "running"}
        self.status_calls = 0
        self.events: Iterator[SSEEvent] = iter(())
        self.opened = 0
        self.closed = 0
        self.seconds_per_event = 30.0
        monkeypatch.setattr(training_mod, "stream_training", self._stream)
        monkeypatch.setattr(training_mod, "get_training_job", self._job)

    def clock(self) -> float:
        return self.now

    def _job(self, job_id: str, **_auth: object) -> object:
        self.status_calls += 1
        value = self.statuses(self.status_calls)
        if isinstance(value, Exception):
            raise value
        return value

    def _stream(
        self, job_id: str, *, include_heartbeats: bool, **_auth: object
    ) -> Iterator[SSEEvent]:
        assert include_heartbeats, "the follower must see heartbeats to check on the job"
        self.opened += 1
        try:
            for event in self.events:
                self.now += self.seconds_per_event
                yield event
        finally:
            self.closed += 1

    def follow(self, *, include_heartbeats: bool = False) -> Iterator[SSEEvent]:
        return training_mod.follow_training(
            "job-1", include_heartbeats=include_heartbeats, clock=self.clock
        )


@pytest.fixture
def world(monkeypatch: PytestMonkeyPatch) -> _World:
    return _World(monkeypatch)


def _names(events: Iterator[SSEEvent]) -> list[str]:
    return [event.event for event in events]


# ------------------------------------------------------------------- is_pause


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (_ev("paused"), True),
        (_ev("paused", "plain text"), True),
        (_ev("status", {"new_status": "paused"}), True),
        (_ev("status", {"new_status": "running"}), False),
        (_ev("status", "paused"), False),
        (_ev("stream_end", {"reason": "paused"}), True),
        (_ev("stream_end", {"reason": "complete"}), False),
        (_ev("stream_end", {}), False),
        (_ev("progress", {"new_status": "paused"}), False),
        (_ev("complete"), False),
    ],
)
def test_is_pause_reads_each_way_the_platform_says_it(event: SSEEvent, expected: bool) -> None:
    assert is_pause(event) is expected


# ------------------------------------------------------------ already paused


def test_an_already_paused_job_is_reported_without_opening_the_stream(world: _World) -> None:
    world.statuses = lambda _n: {"status": "paused", "error_message": PAUSE}
    world.events = itertools.repeat(_ev("heartbeat"))

    [event] = list(world.follow())

    assert (event.event, event.data) == ("paused", {"message": PAUSE})
    assert is_pause(event)
    assert world.opened == 0


def test_a_paused_job_without_a_message_still_reports_a_pause(world: _World) -> None:
    world.statuses = lambda _n: {"status": "paused", "error_message": None}
    [event] = list(world.follow())
    assert event.data == {"message": ""}


# ------------------------------------------------- heartbeats and nothing else


def test_a_stream_of_nothing_but_heartbeats_ends_when_the_job_turns_out_paused(
    world: _World,
) -> None:
    world.statuses = lambda n: (
        {"status": "paused", "error_message": PAUSE} if n >= 3 else {"status": "running"}
    )
    world.events = itertools.repeat(_ev("heartbeat"))

    events = list(world.follow(include_heartbeats=True))

    # Connect (check 1), then one check per 60 s of 30 s heartbeats: bounded, not for ever.
    assert world.status_calls == 3
    assert _names(iter(events)) == ["heartbeat"] * 4 + ["paused"]
    assert world.closed == 1


def test_heartbeats_are_hidden_unless_asked_for_but_still_drive_the_checks(world: _World) -> None:
    world.statuses = lambda n: (
        {"status": "paused", "error_message": PAUSE} if n >= 2 else {"status": "running"}
    )
    world.events = itertools.repeat(_ev("heartbeat"))

    assert _names(world.follow()) == ["paused"]
    assert world.status_calls == 2


def test_a_running_job_is_checked_once_a_minute_not_on_every_event(world: _World) -> None:
    world.events = iter([_ev("heartbeat")] * 9 + [_ev("complete")])

    assert _names(world.follow()) == ["complete"]

    # Connect, then at 60, 120, 180, 240 and 300 s of the 300 s the ten events took.
    assert world.status_calls == 6


def test_a_job_whose_status_cannot_be_read_does_not_break_the_stream(world: _World) -> None:
    world.statuses = lambda _n: APIError(503, "down")
    world.events = iter([_ev("metric"), _ev("heartbeat"), _ev("heartbeat"), _ev("complete")])

    assert _names(world.follow()) == ["metric", "complete"]


# ------------------------------------------------------ the events themselves


def test_a_paused_event_ends_the_stream_at_once(world: _World) -> None:
    world.events = iter([_ev("metric"), _ev("paused", {"message": PAUSE}), _ev("metric")])
    assert _names(world.follow()) == ["metric", "paused"]
    assert world.closed == 1


def test_a_status_event_that_says_paused_ends_the_stream(world: _World) -> None:
    world.events = iter(
        [
            _ev("status", {"new_status": "running"}),
            _ev("status", {"new_status": "paused"}),
            _ev("metric"),
        ]
    )
    assert _names(world.follow()) == ["status", "status"]


def test_a_stream_end_that_says_paused_ends_the_stream(world: _World) -> None:
    world.events = iter([_ev("stream_end", {"reason": "paused"}), _ev("metric")])
    [event] = list(world.follow())
    assert is_pause(event)


def test_a_stream_that_ends_on_its_own_ends_the_follower(world: _World) -> None:
    world.events = iter([_ev("metric"), _ev("stream_end", {})])
    assert _names(world.follow()) == ["metric", "stream_end"]
