"""Shared transient-failure retry policy for the sync and async clients.

One policy — imported by the sync ``_request``, the async ``_request`` (Plan
03), and the LRO poller — so retry behavior never diverges. Retries are bounded
by attempt count, a per-client token-bucket budget, and exponential backoff with
equal jitter; a ``Retry-After`` header (when present) overrides the computed
backoff.
"""

from __future__ import annotations

from datetime import UTC
import email.utils
import logging
import threading
import time
from typing import Awaitable, Callable

from dagnam._core.exceptions import APIError

__all__ = [
    "DEFAULT_BACKOFF_BASE",
    "DEFAULT_BACKOFF_CAP",
    "DEFAULT_BUDGET_MAX",
    "DEFAULT_BUDGET_RETRY_COST",
    "DEFAULT_CONFLICT_BACKOFF_BASE",
    "DEFAULT_CONFLICT_BACKOFF_CAP",
    "DEFAULT_CONFLICT_MAX_RETRIES",
    "DEFAULT_MAX_RETRIES",
    "TRANSIENT_STATUS",
    "RetryBudget",
    "compute_backoff",
    "parse_retry_after",
    "run_with_retry",
    "run_with_retry_async",
]

TRANSIENT_STATUS: frozenset[int] = frozenset({0, 429, 500, 502, 503, 504})

DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 0.5
DEFAULT_BACKOFF_CAP = 10.0
DEFAULT_BUDGET_MAX = 100.0
DEFAULT_BUDGET_RETRY_COST = 5.0

# 409 conflict-retry (idempotency keys). Deliberately separate from the
# transient policy: a 409 is retried ONLY when an idempotency key is present (a
# slow original resolves into a replay), with its own short, capped backoff and
# its own attempt budget. 409 is never added to TRANSIENT_STATUS.
DEFAULT_CONFLICT_MAX_RETRIES = 3
DEFAULT_CONFLICT_BACKOFF_BASE = 1.0
DEFAULT_CONFLICT_BACKOFF_CAP = 5.0


class RetryBudget:
    """Token bucket bounding the retry-to-request ratio across a client.

    Each request deposits one token; each retry withdraws ``retry_cost``. When
    the bucket cannot cover a retry, the driver stops retrying and surfaces the
    underlying error — capping retry amplification during a broad outage.
    """

    def __init__(
        self,
        max_tokens: float = DEFAULT_BUDGET_MAX,
        retry_cost: float = DEFAULT_BUDGET_RETRY_COST,
    ) -> None:
        self._max = max_tokens
        self._cost = retry_cost
        self._tokens = max_tokens
        self._lock = threading.Lock()

    def deposit(self) -> None:
        with self._lock:
            self._tokens = min(self._max, self._tokens + 1.0)

    def try_withdraw(self) -> bool:
        with self._lock:
            if self._tokens >= self._cost:
                self._tokens -= self._cost
                return True
            return False


def compute_backoff(
    attempt: int,
    *,
    base: float = DEFAULT_BACKOFF_BASE,
    cap: float = DEFAULT_BACKOFF_CAP,
    rng: Callable[[], float],
) -> float:
    """Exponential backoff with equal jitter: ``rng() * min(cap, base*2**attempt)``."""
    window = min(cap, base * (2**attempt))
    return rng() * window


def parse_retry_after(value: str | None, *, cap: float) -> float | None:
    """Parse a ``Retry-After`` header, capped at ``cap``.

    Supports both RFC 7231 forms: delta-seconds (``"120"``) and the HTTP-date
    form (``"Wed, 21 Oct 2026 07:28:00 GMT"``, via
    ``email.utils.parsedate_to_datetime``). Returns ``None`` for an absent or
    unparseable value, or a negative delta-seconds value, so the caller falls
    back to computed backoff. A past HTTP-date floors at ``0.0`` (the retry is
    already due) rather than going negative.
    """
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        pass
    else:
        if seconds < 0:
            return None
        return min(seconds, cap)

    try:
        dt = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    remaining = max(0.0, dt.timestamp() - time.time())
    return min(remaining, cap)


def run_with_retry[T](
    call: Callable[[], T],
    *,
    retryable: bool,
    budget: RetryBudget,
    sleep: Callable[[float], None],
    rng: Callable[[], float],
    logger: logging.Logger,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    backoff_cap: float = DEFAULT_BACKOFF_CAP,
    label: str = "",
    idempotency_key: str | None = None,
    conflict_max_retries: int = DEFAULT_CONFLICT_MAX_RETRIES,
    conflict_backoff_base: float = DEFAULT_CONFLICT_BACKOFF_BASE,
    conflict_backoff_cap: float = DEFAULT_CONFLICT_BACKOFF_CAP,
) -> T:
    """Issue ``call()``, retrying transient ``APIError``s per the shared policy.

    A 409 is handled separately from ``TRANSIENT_STATUS``: it is retried ONLY
    when ``idempotency_key`` is set (a slow original resolves into a replay),
    with its own short, capped backoff and its own attempt budget, charged
    against the same ``RetryBudget``. A 409 with no idempotency key raises
    immediately — 409 is never added to ``TRANSIENT_STATUS``.
    """
    budget.deposit()
    attempt = 0
    conflict_attempt = 0
    while True:
        try:
            return call()
        except APIError as exc:
            if exc.status_code == 409 and idempotency_key is not None:
                if conflict_attempt >= conflict_max_retries or not budget.try_withdraw():
                    raise
                delay = compute_backoff(
                    conflict_attempt, base=conflict_backoff_base, cap=conflict_backoff_cap, rng=rng
                )
                logger.debug(
                    "retrying %s after 409 conflict (attempt %d/%d, sleeping %.2fs)",
                    label or "request",
                    conflict_attempt + 1,
                    conflict_max_retries,
                    delay,
                )
                sleep(delay)
                conflict_attempt += 1
                continue
            if not retryable or exc.status_code not in TRANSIENT_STATUS:
                raise
            if attempt >= max_retries:
                raise
            if not budget.try_withdraw():
                logger.debug("retry budget exhausted for %s; surfacing error", label or "request")
                raise
            delay = parse_retry_after(exc.retry_after_header, cap=backoff_cap)
            if delay is None:
                delay = compute_backoff(attempt, base=backoff_base, cap=backoff_cap, rng=rng)
            logger.debug(
                "retrying %s after status=%s (attempt %d/%d, sleeping %.2fs)",
                label or "request",
                exc.status_code,
                attempt + 1,
                max_retries,
                delay,
            )
            sleep(delay)
            attempt += 1


async def run_with_retry_async[T](
    call: Callable[[], Awaitable[T]],
    *,
    retryable: bool,
    budget: RetryBudget,
    sleep: Callable[[float], Awaitable[None]],
    rng: Callable[[], float],
    logger: logging.Logger,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    backoff_cap: float = DEFAULT_BACKOFF_CAP,
    label: str = "",
    idempotency_key: str | None = None,
    conflict_max_retries: int = DEFAULT_CONFLICT_MAX_RETRIES,
    conflict_backoff_base: float = DEFAULT_CONFLICT_BACKOFF_BASE,
    conflict_backoff_cap: float = DEFAULT_CONFLICT_BACKOFF_CAP,
) -> T:
    """Async twin of :func:`run_with_retry` — same policy, awaiting call/sleep.

    Carries the identical scoped 409 conflict-retry: retried only when
    ``idempotency_key`` is set, with its own short backoff and attempt budget
    charged against the shared ``RetryBudget``.
    """
    budget.deposit()
    attempt = 0
    conflict_attempt = 0
    while True:
        try:
            return await call()
        except APIError as exc:
            if exc.status_code == 409 and idempotency_key is not None:
                if conflict_attempt >= conflict_max_retries or not budget.try_withdraw():
                    raise
                delay = compute_backoff(
                    conflict_attempt, base=conflict_backoff_base, cap=conflict_backoff_cap, rng=rng
                )
                logger.debug(
                    "retrying %s after 409 conflict (attempt %d/%d, sleeping %.2fs)",
                    label or "request",
                    conflict_attempt + 1,
                    conflict_max_retries,
                    delay,
                )
                await sleep(delay)
                conflict_attempt += 1
                continue
            if not retryable or exc.status_code not in TRANSIENT_STATUS:
                raise
            if attempt >= max_retries:
                raise
            if not budget.try_withdraw():
                logger.debug("retry budget exhausted for %s; surfacing error", label or "request")
                raise
            delay = parse_retry_after(exc.retry_after_header, cap=backoff_cap)
            if delay is None:
                delay = compute_backoff(attempt, base=backoff_base, cap=backoff_cap, rng=rng)
            logger.debug(
                "retrying %s after status=%s (attempt %d/%d, sleeping %.2fs)",
                label or "request",
                exc.status_code,
                attempt + 1,
                max_retries,
                delay,
            )
            await sleep(delay)
            attempt += 1
