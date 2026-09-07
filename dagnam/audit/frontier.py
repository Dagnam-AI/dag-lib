"""Replay the holdout through a served candidate, and pick the frontier's winner (spec U4, D3).

``replay_holdout`` speaks the platform's OpenAI-compatible route with the
deployment-scoped key: ``model``, ``messages`` and nothing else, because the
route refuses generation parameters rather than ignoring them. Latency is the
client's own clock over the successful calls. ``frontier`` is the one rule of
the report: the cheapest candidate whose agreement *lower bound* clears the floor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import time
from typing import Any

import requests

from dagnam._core._retry import parse_retry_after
from dagnam._types import JsonValue
from dagnam.audit.candidates import CandidateKind

CHAT_TIMEOUT_SECONDS = 180.0
"""Per-call ceiling; a serverless replica's cold start is inside it."""
RATE_LIMIT_RETRIES = 8
"""Attempts one replay call gets; the gateway's 429s are not the candidate's failure."""
RATE_LIMIT_SLEEP_SECONDS = 1.0
"""Wait before retrying a 429 whose ``Retry-After`` is absent or unparseable."""
RATE_LIMIT_SLEEP_MAX_SECONDS = 65.0
"""Cap on an honored ``Retry-After``: one full per-minute window, and a little slack."""


@dataclass(frozen=True, slots=True)
class Endpoint:
    """A served candidate: the API root, the deployment id (the ``model``), its key."""

    base_url: str
    model: str
    api_key: str


@dataclass(frozen=True, slots=True)
class Latency:
    """Client-side percentiles over the successful calls; ``None`` when none succeeded."""

    p50_ms: float | None
    p95_ms: float | None
    calls: int
    errors: int

    def to_json(self) -> dict[str, JsonValue]:
        """The ``latency`` object the state and the report carry."""
        return {"p50": self.p50_ms, "p95": self.p95_ms, "calls": self.calls, "errors": self.errors}


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile of ``values`` (``q`` in 0..1); ``None`` for no values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def _completion(
    endpoint: Endpoint, messages: JsonValue, timeout: float
) -> tuple[str | None, float]:
    """The answer, and the milliseconds the round trip that produced it took.

    A 429 is the gateway's rate limit refusing the *replay*, not the candidate
    failing it, so the same request is retried after the ``Retry-After`` the
    refusal names; only an exhausted attempt budget is an error. The reported
    time is that one successful attempt's -- the waits are nobody's latency.
    """
    for attempt in range(RATE_LIMIT_RETRIES):
        started = time.perf_counter()
        try:
            response = requests.post(
                f"{endpoint.base_url}/v1/chat/completions",
                json={"model": endpoint.model, "messages": messages},
                headers={"Authorization": f"Bearer {endpoint.api_key}"},
                timeout=timeout,
            )
            if response.status_code == 429:
                if attempt + 1 < RATE_LIMIT_RETRIES:
                    wait = parse_retry_after(
                        response.headers.get("Retry-After"), cap=RATE_LIMIT_SLEEP_MAX_SECONDS
                    )
                    time.sleep(RATE_LIMIT_SLEEP_SECONDS if wait is None else wait)
                continue
            response.raise_for_status()
            content: Any = response.json()["choices"][0]["message"]["content"]
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
            return None, 0.0
        elapsed_ms = (time.perf_counter() - started) * 1000
        return (content if isinstance(content, str) else None), elapsed_ms
    return None, 0.0


def replay_holdout(
    endpoint: Endpoint,
    rows: Sequence[Mapping[str, Any]],
    *,
    concurrency: int = 4,
    timeout: float = CHAT_TIMEOUT_SECONDS,
) -> tuple[list[str | None], Latency]:
    """Send each row's ``messages`` to the endpoint; ``None`` where the call failed.

    Results keep the rows' order. At most ``concurrency`` calls are in flight,
    so the replay never becomes a load test of a serverless replica.
    """

    def call(row: Mapping[str, Any]) -> tuple[str | None, float]:
        return _completion(endpoint, row["messages"], timeout)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        results = list(pool.map(call, rows))
    answers = [content for content, _ in results]
    timings = [ms for content, ms in results if content is not None]
    latency = Latency(
        p50_ms=percentile(timings, 0.5),
        p95_ms=percentile(timings, 0.95),
        calls=len(results),
        errors=len(results) - len(timings),
    )
    return answers, latency


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """One scored point on the frontier: its interval, monthly cost and tail latency."""

    kind: CandidateKind
    ci: tuple[float, float]
    cost_usd_month: float
    p95_ms: float | None


@dataclass(frozen=True, slots=True)
class Winner:
    """The frontier's choice and the numbers it was chosen on."""

    kind: CandidateKind
    cost_usd_month: float
    agreement_lo: float


def frontier(points: Sequence[CandidateResult], *, floor: float) -> Winner | None:
    """The cheapest point whose ``ci`` lower bound is at least ``floor``; ``None`` if none clears it.

    Ties on cost go to the higher lower bound, so a more certain candidate wins
    a dead heat.
    """
    passing = [p for p in points if p.ci[0] >= floor]
    if not passing:
        return None
    best = min(passing, key=lambda p: (p.cost_usd_month, -p.ci[0]))
    return Winner(kind=best.kind, cost_usd_month=best.cost_usd_month, agreement_lo=best.ci[0])


__all__ = [
    "CHAT_TIMEOUT_SECONDS",
    "RATE_LIMIT_RETRIES",
    "RATE_LIMIT_SLEEP_MAX_SECONDS",
    "RATE_LIMIT_SLEEP_SECONDS",
    "CandidateResult",
    "Endpoint",
    "Latency",
    "Winner",
    "frontier",
    "percentile",
    "replay_holdout",
]
