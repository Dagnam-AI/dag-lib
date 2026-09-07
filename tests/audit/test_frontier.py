"""Replay through a fake OpenAI-compatible endpoint, and the frontier's one rule."""

from __future__ import annotations

import json
import time
from typing import Any

import pytest
import requests
from tests.typing_helpers import PytestMonkeyPatch, RequestsMocker

from dagnam.audit.candidates import CandidateKind
from dagnam.audit.frontier import (
    RATE_LIMIT_RETRIES,
    RATE_LIMIT_SLEEP_MAX_SECONDS,
    RATE_LIMIT_SLEEP_SECONDS,
    CandidateResult,
    Endpoint,
    Latency,
    Winner,
    frontier,
    percentile,
    replay_holdout,
)

ENDPOINT = Endpoint(base_url="https://x", model="dep-1", api_key="dk-secret")


def _rows(n: int) -> list[dict[str, Any]]:
    return [{"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(n)]


def _serve(requests_mock: RequestsMocker, *, fail_on: str | None = None) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def answer(request: Any, context: Any) -> dict[str, Any]:
        body = json.loads(request.text)
        seen.append({"headers": dict(request.headers), "body": body})
        content = body["messages"][-1]["content"]
        if content == fail_on:
            context.status_code = 500
            return {"error": "boom"}
        return {"choices": [{"message": {"role": "assistant", "content": f"a:{content}"}}]}

    requests_mock.post("https://x/v1/chat/completions", json=answer)
    return seen


def test_replay_keeps_order_counts_errors_and_sends_only_model_and_messages(
    requests_mock: RequestsMocker,
) -> None:
    seen = _serve(requests_mock, fail_on="q1")
    answers, latency = replay_holdout(ENDPOINT, _rows(5), concurrency=2)

    assert answers == ["a:q0", None, "a:q2", "a:q3", "a:q4"]
    assert latency.calls == 5
    assert latency.errors == 1
    assert latency.p50_ms is not None
    assert latency.p95_ms is not None
    assert latency.p50_ms <= latency.p95_ms
    assert len(seen) == 5
    assert all(set(s["body"]) == {"model", "messages"} for s in seen)
    assert all(s["body"]["model"] == "dep-1" for s in seen)
    assert all(s["headers"]["Authorization"] == "Bearer dk-secret" for s in seen)
    assert latency.to_json() == {
        "p50": latency.p50_ms,
        "p95": latency.p95_ms,
        "calls": 5,
        "errors": 1,
    }


def test_replay_treats_malformed_bodies_and_transport_errors_as_none(
    requests_mock: RequestsMocker,
) -> None:
    responses = [
        {"json": {"choices": []}},
        {"json": {"choices": [{"message": {"content": None}}]}},
        {"text": "not json"},
        {"exc": requests.ConnectionError("down")},
        {"json": {"choices": [{"message": {"content": "ok"}}]}},
    ]
    requests_mock.post("https://x/v1/chat/completions", responses)
    answers, latency = replay_holdout(ENDPOINT, _rows(5), concurrency=1)
    assert answers == [None, None, None, None, "ok"]
    assert (latency.calls, latency.errors) == (5, 4)


def test_replay_with_nothing_succeeding_has_no_latency(requests_mock: RequestsMocker) -> None:
    requests_mock.post("https://x/v1/chat/completions", status_code=503)
    answers, latency = replay_holdout(ENDPOINT, _rows(2), concurrency=0)
    assert answers == [None, None]
    assert latency == Latency(p50_ms=None, p95_ms=None, calls=2, errors=2)


def test_percentile_is_nearest_rank() -> None:
    assert percentile([], 0.5) is None
    assert percentile([5.0], 0.95) == 5.0
    assert percentile([3.0, 1.0, 2.0, 4.0], 0.5) == 2.0
    assert percentile([3.0, 1.0, 2.0, 4.0], 0.95) == 4.0
    assert percentile([1.0, 2.0], 0.0) == 1.0


def _point(kind: CandidateKind, lo: float, cost: float) -> CandidateResult:
    return CandidateResult(kind=kind, ci=(lo, min(1.0, lo + 0.02)), cost_usd_month=cost, p95_ms=100)


def test_frontier_picks_cheapest_above_floor_on_the_lower_bound() -> None:
    points = [
        _point(CandidateKind.HOSTED_FLOOR, 0.99, 900.0),
        _point(CandidateKind.HEAD_TUNE, 0.975, 40.0),
        _point(CandidateKind.SFT_SMALL, 0.969, 10.0),  # cheapest, but its lower bound misses
    ]
    assert frontier(points, floor=0.97) == Winner(CandidateKind.HEAD_TUNE, 40.0, 0.975)
    assert frontier(points, floor=0.95) == Winner(CandidateKind.SFT_SMALL, 10.0, 0.969)


def test_frontier_is_none_when_nothing_clears_the_floor() -> None:
    assert frontier([_point(CandidateKind.HEAD_TUNE, 0.9, 1.0)], floor=0.97) is None
    assert frontier([], floor=0.5) is None


def test_frontier_breaks_a_cost_tie_on_certainty() -> None:
    points = [
        _point(CandidateKind.HEAD_TUNE, 0.98, 5.0),
        _point(CandidateKind.SFT_SMALL, 0.99, 5.0),
    ]
    winner = frontier(points, floor=0.97)
    assert winner is not None
    assert winner.kind is CandidateKind.SFT_SMALL


@pytest.mark.parametrize("concurrency", [1, 4])
def test_replay_concurrency_does_not_change_results(
    requests_mock: RequestsMocker, concurrency: int
) -> None:
    _serve(requests_mock)
    answers, _ = replay_holdout(ENDPOINT, _rows(9), concurrency=concurrency)
    assert answers == [f"a:q{i}" for i in range(9)]


CHAT_URL = "https://x/v1/chat/completions"
OK_RESPONSE = {"json": {"choices": [{"message": {"content": "ok"}}]}}


@pytest.fixture
def slept(monkeypatch: PytestMonkeyPatch) -> list[float]:
    """Every second the replay waits, without waiting any of them."""
    recorded: list[float] = []
    # ``frontier`` calls ``time.sleep`` through this same module object.
    monkeypatch.setattr(time, "sleep", recorded.append)
    return recorded


def _rate_limited(retry_after: str | None = None) -> dict[str, Any]:
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return {"status_code": 429, "headers": headers, "json": {"error": "rate limited"}}


def test_a_429_is_waited_out_and_the_same_request_retried(
    requests_mock: RequestsMocker, slept: list[float]
) -> None:
    requests_mock.post(CHAT_URL, [_rate_limited("2"), OK_RESPONSE])

    answers, latency = replay_holdout(ENDPOINT, _rows(1), concurrency=1)

    assert answers == ["ok"]
    assert (latency.calls, latency.errors) == (1, 0)
    assert slept == [2.0]
    sent = [json.loads(r.text or "") for r in requests_mock.request_history]
    assert sent == [{"model": "dep-1", "messages": [{"role": "user", "content": "q0"}]}] * 2


def test_a_429_without_a_usable_retry_after_waits_the_default(
    requests_mock: RequestsMocker, slept: list[float]
) -> None:
    requests_mock.post(CHAT_URL, [_rate_limited(), OK_RESPONSE, _rate_limited("soon"), OK_RESPONSE])

    answers, latency = replay_holdout(ENDPOINT, _rows(2), concurrency=1)

    assert answers == ["ok", "ok"]
    assert latency.errors == 0
    assert slept == [RATE_LIMIT_SLEEP_SECONDS, RATE_LIMIT_SLEEP_SECONDS]


def test_an_outsized_retry_after_is_capped(
    requests_mock: RequestsMocker, slept: list[float]
) -> None:
    requests_mock.post(CHAT_URL, [_rate_limited("600"), OK_RESPONSE])

    answers, _ = replay_holdout(ENDPOINT, _rows(1), concurrency=1)

    assert answers == ["ok"]
    assert slept == [RATE_LIMIT_SLEEP_MAX_SECONDS]


def test_a_429_on_every_attempt_exhausts_the_budget_and_counts_one_error(
    requests_mock: RequestsMocker, slept: list[float]
) -> None:
    requests_mock.post(CHAT_URL, [_rate_limited("1")] * RATE_LIMIT_RETRIES)

    answers, latency = replay_holdout(ENDPOINT, _rows(1), concurrency=1)

    assert answers == [None]
    assert (latency.calls, latency.errors) == (1, 1)
    assert requests_mock.call_count == RATE_LIMIT_RETRIES
    # No wait after the attempt that gives up.
    assert slept == [1.0] * (RATE_LIMIT_RETRIES - 1)


def test_a_non_429_http_error_fails_the_call_without_waiting(
    requests_mock: RequestsMocker, slept: list[float]
) -> None:
    requests_mock.post(CHAT_URL, status_code=500)

    answers, latency = replay_holdout(ENDPOINT, _rows(1), concurrency=1)

    assert answers == [None]
    assert (latency.calls, latency.errors) == (1, 1)
    assert requests_mock.call_count == 1
    assert slept == []
