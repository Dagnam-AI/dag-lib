"""Async twin of the retry-driver policy tests (``run_with_retry_async``)."""

from __future__ import annotations

import logging

import pytest

from dagnam._core._resume import IN_PROGRESS_TEXT
from dagnam._core._retry import RetryBudget, run_with_retry_async
from dagnam._core.exceptions import APIError

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _api_error(status: int, retry_after: str | None = None) -> APIError:
    exc = APIError(status, "boom")
    exc.retry_after_header = retry_after
    return exc


def _in_progress() -> APIError:
    """The 409 the server's idempotency layer sends while the first request runs."""
    return APIError(409, IN_PROGRESS_TEXT)


async def test_async_retries_transient_then_succeeds() -> None:
    calls = {"n": 0}
    slept: list[float] = []

    async def _sleep(d: float) -> None:
        slept.append(d)

    async def call() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise _api_error(502)
        return "ok"

    result = await run_with_retry_async(
        call,
        retryable=True,
        budget=RetryBudget(),
        sleep=_sleep,
        rng=lambda: 1.0,
        logger=logging.getLogger("dagnam.http"),
        label="GET /x",
    )
    assert result == "ok"
    assert calls["n"] == 3
    assert len(slept) == 2


async def test_async_404_not_retried() -> None:
    async def _sleep(_d: float) -> None: ...

    async def call() -> str:
        raise _api_error(404)

    with pytest.raises(APIError) as ei:
        await run_with_retry_async(
            call,
            retryable=True,
            budget=RetryBudget(),
            sleep=_sleep,
            rng=lambda: 1.0,
            logger=logging.getLogger("dagnam.http"),
        )
    assert ei.value.status_code == 404


async def test_async_not_retryable_flag_raises_immediately() -> None:
    calls = {"n": 0}

    async def _sleep(_d: float) -> None: ...

    async def call() -> str:
        calls["n"] += 1
        raise _api_error(503)

    with pytest.raises(APIError):
        await run_with_retry_async(
            call,
            retryable=False,
            budget=RetryBudget(),
            sleep=_sleep,
            rng=lambda: 1.0,
            logger=logging.getLogger("dagnam.http"),
        )
    assert calls["n"] == 1


async def test_async_max_retries_exhausted_raises() -> None:
    calls = {"n": 0}

    async def _sleep(_d: float) -> None: ...

    async def call() -> str:
        calls["n"] += 1
        raise _api_error(503)

    with pytest.raises(APIError):
        await run_with_retry_async(
            call,
            retryable=True,
            budget=RetryBudget(),
            sleep=_sleep,
            rng=lambda: 1.0,
            logger=logging.getLogger("dagnam.http"),
            max_retries=2,
        )
    assert calls["n"] == 3  # 1 initial + 2 retries


async def test_async_budget_exhaustion_stops_retry() -> None:
    calls = {"n": 0}
    budget = RetryBudget(max_tokens=1.0, retry_cost=5.0)

    async def _sleep(_d: float) -> None: ...

    async def call() -> str:
        calls["n"] += 1
        raise _api_error(503)

    with pytest.raises(APIError):
        await run_with_retry_async(
            call,
            retryable=True,
            budget=budget,
            sleep=_sleep,
            rng=lambda: 1.0,
            logger=logging.getLogger("dagnam.http"),
        )
    assert calls["n"] == 1


async def test_async_honors_retry_after_header() -> None:
    calls = {"n": 0}
    slept: list[float] = []

    async def _sleep(d: float) -> None:
        slept.append(d)

    async def call() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise _api_error(503, retry_after="7")
        return "ok"

    result = await run_with_retry_async(
        call,
        retryable=True,
        budget=RetryBudget(),
        sleep=_sleep,
        rng=lambda: 1.0,
        logger=logging.getLogger("dagnam.http"),
    )
    assert result == "ok"
    assert slept == [7.0]  # Retry-After overrode the computed backoff


async def test_async_409_with_idempotency_key_retries_then_replays_success() -> None:
    calls = {"n": 0}
    slept: list[float] = []

    async def _sleep(d: float) -> None:
        slept.append(d)

    async def call() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _in_progress()
        return "replayed"

    result = await run_with_retry_async(
        call,
        retryable=False,
        budget=RetryBudget(),
        sleep=_sleep,
        rng=lambda: 1.0,
        logger=logging.getLogger("dagnam.http"),
        idempotency_key="idem-1",
    )
    assert result == "replayed"
    assert calls["n"] == 2
    assert len(slept) == 1


async def test_async_409_exhaustion_raises() -> None:
    calls = {"n": 0}

    async def _sleep(_d: float) -> None:
        return None

    async def call() -> str:
        calls["n"] += 1
        raise _in_progress()

    with pytest.raises(APIError) as ei:
        await run_with_retry_async(
            call,
            retryable=False,
            budget=RetryBudget(),
            sleep=_sleep,
            rng=lambda: 1.0,
            logger=logging.getLogger("dagnam.http"),
            idempotency_key="idem-2",
            conflict_max_retries=3,
        )
    assert ei.value.status_code == 409
    assert calls["n"] == 4


async def test_async_409_without_idempotency_key_not_retried() -> None:
    calls = {"n": 0}

    async def _sleep(_d: float) -> None:
        return None

    async def call() -> str:
        calls["n"] += 1
        raise _in_progress()

    with pytest.raises(APIError):
        await run_with_retry_async(
            call,
            retryable=True,
            budget=RetryBudget(),
            sleep=_sleep,
            rng=lambda: 1.0,
            logger=logging.getLogger("dagnam.http"),
            idempotency_key=None,
        )
    assert calls["n"] == 1


async def test_async_409_conflict_retry_charges_the_shared_budget() -> None:
    calls = {"n": 0}
    budget = RetryBudget(max_tokens=1.0, retry_cost=5.0)

    async def _sleep(_d: float) -> None:
        return None

    async def call() -> str:
        calls["n"] += 1
        raise _in_progress()

    with pytest.raises(APIError):
        await run_with_retry_async(
            call,
            retryable=False,
            budget=budget,
            sleep=_sleep,
            rng=lambda: 1.0,
            logger=logging.getLogger("dagnam.http"),
            idempotency_key="idem-3",
        )
    assert calls["n"] == 1


async def test_async_409_that_is_not_the_idempotency_conflict_is_final_even_with_a_key() -> None:
    calls = {"n": 0}

    async def _sleep(_d: float) -> None:
        return None

    async def call() -> str:
        calls["n"] += 1
        raise APIError(409, "Active deployment limit reached")

    with pytest.raises(APIError):
        await run_with_retry_async(
            call,
            retryable=True,
            budget=RetryBudget(),
            sleep=_sleep,
            rng=lambda: 1.0,
            logger=logging.getLogger("dagnam.http"),
            idempotency_key="idem-4",
        )
    assert calls["n"] == 1
