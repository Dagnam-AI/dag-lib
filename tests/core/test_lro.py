"""Unit tests for dagnam.lro.LongRunningOperation."""

from __future__ import annotations

import logging
from typing import Callable

import pytest

from dagnam._core.exceptions import APIError, LROFailedError, LROTimeoutError
from dagnam._core.lro import LongRunningOperation
from dagnam._types import JsonMapping


def _stepped_poller(states: list[str]) -> Callable[[], JsonMapping]:
    """Return a poll fn that walks through ``states`` (repeating the last)."""
    it = iter(states)
    last: JsonMapping = {"status": states[-1]}

    def _poll() -> JsonMapping:
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


class TestTransientPollErrors:
    """A transient poll failure (network / 429 / 5xx) must not abort wait()."""

    def test_retries_transient_api_error_then_succeeds(self) -> None:
        calls = {"n": 0}

        def poll() -> JsonMapping:
            calls["n"] += 1
            if calls["n"] <= 2:
                raise APIError(503, "temporarily unavailable")
            return {"status": "running"}

        op = LongRunningOperation(
            poll=poll, success_states={"running"}, poll_min=0.01, poll_max=0.01
        )
        clk = FakeClock()
        op.wait(timeout=60, sleep=clk.sleep, now=clk.now)
        assert op.result()["status"] == "running"
        assert calls["n"] == 3

    def test_retries_network_error_status_zero(self) -> None:
        calls = {"n": 0}

        def poll() -> JsonMapping:
            calls["n"] += 1
            if calls["n"] == 1:
                raise APIError(0, "Connection failed")
            return {"status": "running"}

        op = LongRunningOperation(
            poll=poll, success_states={"running"}, poll_min=0.01, poll_max=0.01
        )
        clk = FakeClock()
        op.wait(timeout=60, sleep=clk.sleep, now=clk.now)
        assert op.result()["status"] == "running"

    def test_propagates_non_transient_api_error_immediately(self) -> None:
        calls = {"n": 0}

        def poll() -> JsonMapping:
            calls["n"] += 1
            raise APIError(404, "not found")

        op = LongRunningOperation(poll=poll, success_states={"running"})
        clk = FakeClock()
        with pytest.raises(APIError) as exc:
            op.wait(timeout=60, sleep=clk.sleep, now=clk.now)
        assert exc.value.status_code == 404
        assert calls["n"] == 1  # not retried

    def test_gives_up_on_persistent_transient_error_at_deadline(self) -> None:
        def poll() -> JsonMapping:
            raise APIError(503, "down")

        op = LongRunningOperation(poll=poll, success_states={"running"}, poll_min=0.5, poll_max=0.5)
        clk = FakeClock()
        with pytest.raises(APIError) as exc:
            op.wait(timeout=1.0, sleep=clk.sleep, now=clk.now)
        assert exc.value.status_code == 503


class TestTerminalResolution:
    def test_already_terminal_on_first_poll(self) -> None:
        op = LongRunningOperation(
            poll=lambda: {"status": "running"},
            success_states={"running"},
        )
        clk = FakeClock()
        result = op.wait(timeout=60, sleep=clk.sleep, now=clk.now).result()
        assert result["status"] == "running"
        assert op.done()

    def test_waits_through_transient_states(self) -> None:
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

    def test_failure_raises_witherror_message(self) -> None:
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
    def test_timeout_when_never_terminal(self) -> None:
        op = LongRunningOperation(
            poll=lambda: {"status": "deploying"},
            success_states={"running"},
            poll_min=0.1,
            poll_max=0.1,
        )
        clk = FakeClock()
        with pytest.raises(LROTimeoutError):
            op.wait(timeout=0.5, sleep=clk.sleep, now=clk.now)

    def test_result_before_wait_raises(self) -> None:
        op = LongRunningOperation(
            poll=lambda: {"status": "deploying"},
            success_states={"running"},
        )
        with pytest.raises(LROTimeoutError):
            op.result()

    def test_result_after_timeout_reports_non_terminal_state(self) -> None:
        # After wait() times out, _latest holds a non-terminal payload; calling
        # result() then raises the "still in non-terminal state" timeout (lro.py:181).
        op = LongRunningOperation(
            poll=lambda: {"status": "deploying"},
            success_states={"running"},
            poll_min=0.1,
            poll_max=0.1,
        )
        clk = FakeClock()
        with pytest.raises(LROTimeoutError):
            op.wait(timeout=0.5, sleep=clk.sleep, now=clk.now)
        with pytest.raises(LROTimeoutError, match="non-terminal state"):
            op.result()

    def test_zero_timeout_skips_poll_loop(self) -> None:
        # timeout=0 → deadline already reached after the first non-terminal poll,
        # so the `while now() < deadline` loop is never entered (branch 141->155).
        op = LongRunningOperation(
            poll=lambda: {"status": "deploying"},
            success_states={"running"},
        )
        clk = FakeClock()
        with pytest.raises(LROTimeoutError):
            op.wait(timeout=0, sleep=clk.sleep, now=clk.now)

    def test_failure_without_detail_omits_suffix(self) -> None:
        # Failure payload with no error_message → LROFailedError detail is None and
        # the message carries no ": <detail>" suffix (exceptions.py branch 141->143).
        op = LongRunningOperation(
            poll=lambda: {"status": "failed"},
            success_states={"running"},
        )
        clk = FakeClock()
        with pytest.raises(LROFailedError) as excinfo:
            op.wait(timeout=1, sleep=clk.sleep, now=clk.now).result()
        assert excinfo.value.detail is None
        assert str(excinfo.value) == "Operation entered failure state 'failed'"


class TestCustomStateKey:
    def test_alternate_state_key(self) -> None:
        op = LongRunningOperation(
            poll=lambda: {"task_status": "SUCCESS", "data": 42},
            success_states={"SUCCESS"},
            state_key="task_status",
        )
        clk = FakeClock()
        result = op.wait(timeout=60, sleep=clk.sleep, now=clk.now).result()
        assert result["data"] == 42

    def test_initial_payload_is_preserved(self) -> None:
        initial: JsonMapping = {"status": "deploying", "id": "dep-1"}
        op = LongRunningOperation(
            poll=lambda: {"status": "running"},
            success_states={"running"},
            initial=initial,
        )
        initial_payload = op.initial()
        assert initial_payload is not None
        assert initial_payload["id"] == "dep-1"
        assert op.done() is False  # deploying is not terminal

    def test_requires_success_state(self) -> None:
        with pytest.raises(ValueError):
            LongRunningOperation(poll=lambda: {}, success_states=[])


# ---------------------------------------------------------------------------
# Task 8: LRO error-retry unified onto the shared jittered-backoff policy
# ---------------------------------------------------------------------------


def _recording_sleep(clk: FakeClock, slept: list[float]) -> Callable[[float], None]:
    def _sleep(seconds: float) -> None:
        slept.append(seconds)
        clk.sleep(seconds)

    return _sleep


def test_lro_uses_shared_transient_status() -> None:
    from dagnam._core import lro
    from dagnam._core._retry import TRANSIENT_STATUS

    assert lro.TRANSIENT_STATUS is TRANSIENT_STATUS


def test_lro_error_retry_uses_jittered_backoff_bounds() -> None:
    """The error-retry path (NOT the normal poll cadence) now goes through the
    shared compute_backoff jitter policy: rng() * min(poll_max, poll_min * 2**attempt)."""
    calls = {"n": 0}
    slept: list[float] = []

    def poll() -> JsonMapping:
        calls["n"] += 1
        if calls["n"] == 1:
            raise APIError(503, "down")
        return {"status": "running"}

    op = LongRunningOperation(poll=poll, success_states={"running"}, poll_min=2.0, poll_max=10.0)
    clk = FakeClock()
    op.wait(
        timeout=60,
        sleep=_recording_sleep(clk, slept),
        now=clk.now,
        rng=lambda: 1.0,  # pin jitter to the full window
    )
    # compute_backoff(attempt=0, base=poll_min=2.0, cap=poll_max=10.0, rng=1.0)
    #   == 1.0 * min(10.0, 2.0 * 2**0) == 2.0
    assert slept == [2.0]


def test_lro_error_retry_honors_retry_after_header_over_backoff() -> None:
    calls = {"n": 0}
    slept: list[float] = []

    def poll() -> JsonMapping:
        calls["n"] += 1
        if calls["n"] == 1:
            exc = APIError(429, "rate limited")
            exc.retry_after_header = "3"
            raise exc
        return {"status": "running"}

    op = LongRunningOperation(poll=poll, success_states={"running"}, poll_min=2.0, poll_max=10.0)
    clk = FakeClock()
    op.wait(
        timeout=60,
        sleep=_recording_sleep(clk, slept),
        now=clk.now,
        rng=lambda: 1.0,
    )
    assert slept == [3.0]  # header (3.0) wins over computed backoff (2.0)


def test_lro_logs_debug_per_retry_and_warning_on_giveup(caplog: pytest.LogCaptureFixture) -> None:
    def poll() -> JsonMapping:
        raise APIError(503, "down")

    op = LongRunningOperation(poll=poll, success_states={"running"}, poll_min=0.1, poll_max=0.1)
    clk = FakeClock()
    with caplog.at_level(logging.DEBUG, logger="dagnam.lro"):
        with pytest.raises(APIError):
            op.wait(timeout=0.25, sleep=clk.sleep, now=clk.now, rng=lambda: 1.0)
    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert debug_records
    assert warning_records
    assert "giving up" in warning_records[0].message


def test_configure_polling_clamps_nonpositive_min_interval() -> None:
    from dagnam._core.lro import LongRunningOperation

    op = LongRunningOperation(
        poll=lambda: {"status": "running"},
        state_key="status",
        success_states={"done"},
    )
    op.configure_polling(min_interval=0.0, max_interval=-1.0)
    # 0/negative would enable a sleep(0) tight-spin under a 429/503 flood; floored.
    assert op._poll_min == 0.1
    assert op._poll_max >= op._poll_min
