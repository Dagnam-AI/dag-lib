"""Unit tests for dagnam.lro.LongRunningOperation."""

from __future__ import annotations

from itertools import count
from typing import Callable

import pytest

from dagnam._core.exceptions import LROFailedError, LROTimeoutError
from dagnam._core.lro import LongRunningOperation


def _stepped_poller(states: list[str]) -> Callable[[], dict]:
    """Return a poll fn that walks through ``states`` (repeating the last)."""
    it = iter(states)
    last = {"status": states[-1]}

    def _poll() -> dict:
        nonlocal last
        try:
            last = {"status": next(it)}
        except StopIteration:
            pass
        return last

    return _poll


class FakeClock:
    """A deterministic monotonic clock with a manual advance() step."""

    def __init__(self) -> None:
        self._t = 0.0

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        self._t += float(seconds)

    def advance(self, seconds: float) -> None:
        self._t += float(seconds)


class TestTerminalResolution:
    def test_already_terminal_on_first_poll(self):
        op = LongRunningOperation(
            poll=lambda: {"status": "running"},
            success_states={"running"},
        )
        clk = FakeClock()
        result = op.wait(timeout=60, sleep=clk.sleep, now=clk.now).result()
        assert result["status"] == "running"
        assert op.done()

    def test_waits_through_transient_states(self):
        poll = _stepped_poller(["deploying", "deploying", "running"])
        op = LongRunningOperation(
            poll=poll,
            success_states={"running"},
            poll_min=0.01,
            poll_max=0.01,
        )
        clk = FakeClock()
        op.wait(timeout=60, sleep=clk.sleep, now=clk.now)
        assert op.result()["status"] == "running"

    def test_failure_raises_with_error_message(self):
        op = LongRunningOperation(
            poll=lambda: {"status": "failed", "error_message": "OOM"},
            success_states={"running"},
        )
        clk = FakeClock()
        with pytest.raises(LROFailedError) as excinfo:
            op.wait(timeout=1, sleep=clk.sleep, now=clk.now).result()
        assert excinfo.value.state == "failed"
        assert excinfo.value.detail == "OOM"


class TestTimeout:
    def test_timeout_when_never_terminal(self):
        op = LongRunningOperation(
            poll=lambda: {"status": "deploying"},
            success_states={"running"},
            poll_min=0.1,
            poll_max=0.1,
        )
        clk = FakeClock()
        with pytest.raises(LROTimeoutError):
            op.wait(timeout=0.5, sleep=clk.sleep, now=clk.now)

    def test_result_before_wait_raises(self):
        op = LongRunningOperation(
            poll=lambda: {"status": "deploying"},
            success_states={"running"},
        )
        with pytest.raises(LROTimeoutError):
            op.result()


class TestCustomStateKey:
    def test_alternate_state_key(self):
        op = LongRunningOperation(
            poll=lambda: {"task_status": "SUCCESS", "data": 42},
            success_states={"SUCCESS"},
            state_key="task_status",
        )
        clk = FakeClock()
        result = op.wait(timeout=60, sleep=clk.sleep, now=clk.now).result()
        assert result["data"] == 42

    def test_initial_payload_is_preserved(self):
        op = LongRunningOperation(
            poll=lambda: {"status": "running"},
            success_states={"running"},
            initial={"status": "deploying", "id": "dep-1"},
        )
        assert op.initial()["id"] == "dep-1"
        assert op.done() is False  # deploying is not terminal

    def test_requires_success_state(self):
        with pytest.raises(ValueError):
            LongRunningOperation(poll=lambda: {}, success_states=[])
