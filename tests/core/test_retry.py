"""Unit tests for the shared transient-failure retry policy (dagnam._core._retry)."""

from __future__ import annotations

import logging

import pytest

from dagnam._core._retry import (
    DEFAULT_BUDGET_MAX,
    DEFAULT_BUDGET_RETRY_COST,
    TRANSIENT_STATUS,
    RetryBudget,
    compute_backoff,
    parse_retry_after,
    run_with_retry,
)
from dagnam._core.exceptions import APIError


def test_transient_status_set():
    assert frozenset({0, 429, 500, 502, 503, 504}) == TRANSIENT_STATUS


def test_budget_withdraw_until_exhausted():
    budget = RetryBudget(max_tokens=10.0, retry_cost=5.0)
    assert budget.try_withdraw() is True  # 10 -> 5
    assert budget.try_withdraw() is True  # 5 -> 0
    assert budget.try_withdraw() is False  # 0 < 5


def test_budget_deposit_is_capped():
    budget = RetryBudget(max_tokens=10.0, retry_cost=5.0)
    budget.try_withdraw()  # 10 -> 5
    budget.deposit()  # 5 -> 6
    budget.deposit()  # 6 -> 7
    for _ in range(100):
        budget.deposit()  # never exceeds 10
    assert budget.try_withdraw() is True  # 10 -> 5
    assert budget.try_withdraw() is True  # 5 -> 0
    assert budget.try_withdraw() is False


def test_budget_defaults():
    assert DEFAULT_BUDGET_MAX == 100.0
    assert DEFAULT_BUDGET_RETRY_COST == 5.0


def test_compute_backoff_equal_jitter_scales_with_attempt():
    # rng pinned to 1.0 -> full window; base*2**attempt until capped
    assert compute_backoff(0, base=0.5, cap=10.0, rng=lambda: 1.0) == 0.5
    assert compute_backoff(1, base=0.5, cap=10.0, rng=lambda: 1.0) == 1.0
    assert compute_backoff(2, base=0.5, cap=10.0, rng=lambda: 1.0) == 2.0


def test_compute_backoff_is_capped():
    assert compute_backoff(20, base=0.5, cap=10.0, rng=lambda: 1.0) == 10.0


def test_compute_backoff_applies_jitter_fraction():
    assert compute_backoff(3, base=1.0, cap=100.0, rng=lambda: 0.25) == 2.0  # 0.25 * 8


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2", 2.0),
        ("0", 0.0),
        ("999", 10.0),  # capped
        (None, None),
        ("soon", None),  # junk -> None
        ("-3", None),  # negative delta -> None
    ],
)
def test_parse_retry_after_delta_seconds(value, expected):
    assert parse_retry_after(value, cap=10.0) == expected


def test_parse_retry_after_http_date_form(monkeypatch):
    import time as _time

    # Pin "now" so the HTTP-date's remaining-seconds computation is deterministic.
    fixed_now = 1_700_000_000.0
    monkeypatch.setattr(_time, "time", lambda: fixed_now)
    # 5 seconds after fixed_now, formatted as an RFC 7231 HTTP-date.
    from email.utils import formatdate

    http_date = formatdate(fixed_now + 5.0, usegmt=True)
    assert parse_retry_after(http_date, cap=10.0) == pytest.approx(5.0, abs=1.0)


def test_parse_retry_after_past_http_date_floors_at_zero(monkeypatch):
    import time as _time

    fixed_now = 1_700_000_000.0
    monkeypatch.setattr(_time, "time", lambda: fixed_now)
    from email.utils import formatdate

    http_date = formatdate(fixed_now - 3600.0, usegmt=True)  # an hour in the past
    assert parse_retry_after(http_date, cap=10.0) == 0.0


def test_parse_retry_after_naive_http_date_assumes_utc(monkeypatch):
    from datetime import UTC, datetime
    from email.utils import format_datetime
    import time as _time

    fixed_now = 1_700_000_000.0
    monkeypatch.setattr(_time, "time", lambda: fixed_now)
    # A parseable date with a "-0000" (unknown) timezone parses back as a *naive*
    # datetime, exercising the tzinfo-is-None branch that assumes UTC.
    target = datetime.fromtimestamp(fixed_now + 4.0, tz=UTC).replace(tzinfo=None)
    http_date = format_datetime(target)  # yields a "-0000" suffix -> naive on parse
    assert parse_retry_after(http_date, cap=10.0) == pytest.approx(4.0, abs=1.0)


def test_parse_retry_after_garbage_http_date_like_string_is_none():
    assert parse_retry_after("Not, a Date at all", cap=10.0) is None


def _api_error(status: int, retry_after: str | None = None) -> APIError:
    exc = APIError(status, "boom")
    exc.retry_after_header = retry_after
    return exc


def test_retries_transient_then_succeeds():
    calls = {"n": 0}
    slept: list[float] = []

    def call() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise _api_error(503)
        return "ok"

    result = run_with_retry(
        call,
        retryable=True,
        budget=RetryBudget(),
        sleep=slept.append,
        rng=lambda: 1.0,
        logger=logging.getLogger("dagnam.http"),
        label="GET /x",
    )
    assert result == "ok"
    assert calls["n"] == 3
    assert len(slept) == 2  # two retries slept


def test_non_transient_status_not_retried():
    def call() -> str:
        raise _api_error(404)

    with pytest.raises(APIError) as ei:
        run_with_retry(
            call,
            retryable=True,
            budget=RetryBudget(),
            sleep=lambda _s: None,
            rng=lambda: 1.0,
            logger=logging.getLogger("dagnam.http"),
        )
    assert ei.value.status_code == 404


def test_non_retryable_method_not_retried():
    calls = {"n": 0}

    def call() -> str:
        calls["n"] += 1
        raise _api_error(503)

    with pytest.raises(APIError):
        run_with_retry(
            call,
            retryable=False,
            budget=RetryBudget(),
            sleep=lambda _s: None,
            rng=lambda: 1.0,
            logger=logging.getLogger("dagnam.http"),
        )
    assert calls["n"] == 1  # issued once, never retried


def test_stops_at_max_retries():
    calls = {"n": 0}

    def call() -> str:
        calls["n"] += 1
        raise _api_error(500)

    with pytest.raises(APIError):
        run_with_retry(
            call,
            retryable=True,
            budget=RetryBudget(),
            sleep=lambda _s: None,
            rng=lambda: 1.0,
            logger=logging.getLogger("dagnam.http"),
            max_retries=3,
        )
    assert calls["n"] == 4  # 1 initial + 3 retries


def test_budget_exhaustion_stops_retry():
    calls = {"n": 0}
    budget = RetryBudget(max_tokens=1.0, retry_cost=5.0)  # cannot afford a retry

    def call() -> str:
        calls["n"] += 1
        raise _api_error(502)

    with pytest.raises(APIError):
        run_with_retry(
            call,
            retryable=True,
            budget=budget,
            sleep=lambda _s: None,
            rng=lambda: 1.0,
            logger=logging.getLogger("dagnam.http"),
        )
    assert calls["n"] == 1  # starts capped at 1.0; entry deposit() re-caps at
    # min(1.0, 1.0+1.0)=1.0 (never 2); withdraw(5) fails


def test_retry_after_header_overrides_backoff():
    calls = {"n": 0}
    slept: list[float] = []

    def call() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _api_error(429, retry_after="2")
        return "ok"

    run_with_retry(
        call,
        retryable=True,
        budget=RetryBudget(),
        sleep=slept.append,
        rng=lambda: 1.0,
        logger=logging.getLogger("dagnam.http"),
        backoff_cap=10.0,
    )
    assert slept == [2.0]  # honored the header, not computed backoff


def test_409_with_idempotency_key_retries_then_replays_success():
    calls = {"n": 0}
    slept: list[float] = []

    def call() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _api_error(409)
        return "replayed"

    result = run_with_retry(
        call,
        retryable=False,
        budget=RetryBudget(),
        sleep=slept.append,
        rng=lambda: 1.0,
        logger=logging.getLogger("dagnam.http"),
        idempotency_key="idem-1",
    )
    assert result == "replayed"
    assert calls["n"] == 2
    assert len(slept) == 1  # one conflict-retry slept


def test_409_exhaustion_raises():
    calls = {"n": 0}

    def call() -> str:
        calls["n"] += 1
        raise _api_error(409)

    with pytest.raises(APIError) as ei:
        run_with_retry(
            call,
            retryable=False,
            budget=RetryBudget(),
            sleep=lambda _s: None,
            rng=lambda: 1.0,
            logger=logging.getLogger("dagnam.http"),
            idempotency_key="idem-2",
            conflict_max_retries=3,
        )
    assert ei.value.status_code == 409
    assert calls["n"] == 4  # 1 initial + 3 conflict retries


def test_409_without_idempotency_key_not_retried():
    calls = {"n": 0}

    def call() -> str:
        calls["n"] += 1
        raise _api_error(409)

    with pytest.raises(APIError):
        run_with_retry(
            call,
            retryable=True,
            budget=RetryBudget(),
            sleep=lambda _s: None,
            rng=lambda: 1.0,
            logger=logging.getLogger("dagnam.http"),
            idempotency_key=None,
        )
    assert calls["n"] == 1  # 409 is not transient and has no key to scope a conflict-retry


def test_409_conflict_retry_charges_the_shared_budget():
    calls = {"n": 0}
    budget = RetryBudget(
        max_tokens=1.0, retry_cost=5.0
    )  # deposit(+1)=2 tokens, can't afford a 5-cost retry

    def call() -> str:
        calls["n"] += 1
        raise _api_error(409)

    with pytest.raises(APIError):
        run_with_retry(
            call,
            retryable=False,
            budget=budget,
            sleep=lambda _s: None,
            rng=lambda: 1.0,
            logger=logging.getLogger("dagnam.http"),
            idempotency_key="idem-3",
        )
    assert calls["n"] == 1  # budget exhausted before the first conflict-retry
