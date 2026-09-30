"""Replay through a fake OpenAI-compatible endpoint, and the frontier's one rule."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sys
import threading
import time
from typing import Any, ClassVar, override

import pytest
import requests
from tests.typing_helpers import PytestMonkeyPatch, RequestsMocker

from dagnam.audit.candidates import CandidateKind
from dagnam.audit.frontier import (
    RATE_LIMIT_RETRIES,
    RATE_LIMIT_SLEEP_MAX_SECONDS,
    RATE_LIMIT_SLEEP_SECONDS,
    TRANSIENT_STATUSES,
    CandidateResult,
    Endpoint,
    Latency,
    Winner,
    frontier,
    latency_of,
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
    requests_mock.post("https://x/v1/chat/completions", status_code=500)
    answers, latency = replay_holdout(ENDPOINT, _rows(2), concurrency=0)
    assert answers == [None, None]
    assert latency == Latency(p50_ms=None, p95_ms=None, calls=2, errors=2)


def test_replay_reports_each_row_as_it_lands(requests_mock: RequestsMocker) -> None:
    _serve(requests_mock, fail_on="q1")
    landed: list[tuple[int, str | None]] = []
    replay_holdout(
        ENDPOINT, _rows(3), concurrency=1, on_result=lambda i, a, _ms: landed.append((i, a))
    )
    assert landed == [(0, "a:q0"), (1, None), (2, "a:q2")]


def test_an_interrupted_replay_raises_after_reporting_what_already_landed(
    requests_mock: RequestsMocker,
) -> None:
    """B9: a Ctrl+C mid-replay must not take the answers already paid for with it."""

    def answer(request: Any, context: Any) -> dict[str, Any]:
        if json.loads(request.text)["messages"][-1]["content"] == "q1":
            raise KeyboardInterrupt
        return {"choices": [{"message": {"content": "ok"}}]}

    requests_mock.post(CHAT_URL, json=answer)
    landed: list[int] = []
    with pytest.raises(KeyboardInterrupt):
        replay_holdout(
            ENDPOINT, _rows(3), concurrency=1, on_result=lambda i, _a, _ms: landed.append(i)
        )
    assert landed == [0]


def test_latency_is_over_the_answered_calls_only() -> None:
    assert latency_of([("a", 10.0), (None, 0.0), ("b", 30.0)]) == Latency(
        p50_ms=10.0, p95_ms=30.0, calls=3, errors=1
    )


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


def _refused(status: int, retry_after: str | None = None) -> dict[str, Any]:
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return {"status_code": status, "headers": headers, "json": {"error": "refused"}}


def _rate_limited(retry_after: str | None = None) -> dict[str, Any]:
    return _refused(429, retry_after)


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
    # Each call starts its own backoff, so both first refusals wait the base.
    assert slept == [RATE_LIMIT_SLEEP_SECONDS, RATE_LIMIT_SLEEP_SECONDS]


def test_consecutive_429s_without_a_header_double_the_wait(
    requests_mock: RequestsMocker, slept: list[float]
) -> None:
    requests_mock.post(CHAT_URL, [_rate_limited()] * 4 + [OK_RESPONSE])

    answers, latency = replay_holdout(ENDPOINT, _rows(1), concurrency=1)

    assert answers == ["ok"]
    assert (latency.calls, latency.errors) == (1, 0)
    assert slept == [1.0, 2.0, 4.0, 8.0]


def test_the_computed_backoff_is_capped_too(
    requests_mock: RequestsMocker, slept: list[float], monkeypatch: PytestMonkeyPatch
) -> None:
    # Not by dotted path: ``dagnam.audit`` re-exports the ``frontier`` *function*
    # over its own submodule, so the attribute walk lands on the wrong object.
    module = sys.modules["dagnam.audit.frontier"]
    monkeypatch.setattr(module, "RATE_LIMIT_SLEEP_SECONDS", 100.0)
    requests_mock.post(CHAT_URL, [_rate_limited(), OK_RESPONSE])

    answers, _ = replay_holdout(ENDPOINT, _rows(1), concurrency=1)

    assert answers == ["ok"]
    assert slept == [RATE_LIMIT_SLEEP_MAX_SECONDS]


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
    requests_mock.post(CHAT_URL, [_rate_limited()] * RATE_LIMIT_RETRIES)

    answers, latency = replay_holdout(ENDPOINT, _rows(1), concurrency=1)

    assert answers == [None]
    assert (latency.calls, latency.errors) == (1, 1)
    assert requests_mock.call_count == RATE_LIMIT_RETRIES
    # The full doubling sequence, and no wait after the attempt that gives up.
    assert slept == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    assert len(slept) == RATE_LIMIT_RETRIES - 1
    assert max(slept) <= RATE_LIMIT_SLEEP_MAX_SECONDS


def test_a_non_transient_http_error_fails_the_call_without_waiting(
    requests_mock: RequestsMocker, slept: list[float]
) -> None:
    # A 500 is the candidate answering badly, not the gateway refusing to ask it.
    assert 500 not in TRANSIENT_STATUSES
    requests_mock.post(CHAT_URL, status_code=500)

    answers, latency = replay_holdout(ENDPOINT, _rows(1), concurrency=1)

    assert answers == [None]
    assert (latency.calls, latency.errors) == (1, 1)
    assert requests_mock.call_count == 1
    assert slept == []


def test_a_transient_502_is_waited_out_and_the_same_request_retried(
    requests_mock: RequestsMocker, slept: list[float]
) -> None:
    requests_mock.post(CHAT_URL, [_refused(502), OK_RESPONSE])

    answers, latency = replay_holdout(ENDPOINT, _rows(1), concurrency=1)

    assert answers == ["ok"]
    assert (latency.calls, latency.errors) == (1, 0)
    assert slept == [RATE_LIMIT_SLEEP_SECONDS]
    sent = [json.loads(r.text or "") for r in requests_mock.request_history]
    assert sent == [{"model": "dep-1", "messages": [{"role": "user", "content": "q0"}]}] * 2


def test_a_503_that_names_a_retry_after_waits_exactly_that_long(
    requests_mock: RequestsMocker, slept: list[float]
) -> None:
    requests_mock.post(CHAT_URL, [_refused(503, "3"), OK_RESPONSE])

    answers, latency = replay_holdout(ENDPOINT, _rows(1), concurrency=1)

    assert answers == ["ok"]
    assert (latency.calls, latency.errors) == (1, 0)
    assert slept == [3.0]


def test_a_504_on_every_attempt_exhausts_the_budget_and_counts_one_error(
    requests_mock: RequestsMocker, slept: list[float]
) -> None:
    requests_mock.post(CHAT_URL, [_refused(504)] * RATE_LIMIT_RETRIES)

    answers, latency = replay_holdout(ENDPOINT, _rows(1), concurrency=1)

    assert answers == [None]
    assert (latency.calls, latency.errors) == (1, 1)
    assert requests_mock.call_count == RATE_LIMIT_RETRIES
    # The full doubling sequence, and no wait after the attempt that gives up.
    assert slept == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    assert max(slept) <= RATE_LIMIT_SLEEP_MAX_SECONDS


class _ChatHandler(BaseHTTPRequestHandler):
    """A keep-alive OpenAI-compatible endpoint that records which connection each call used."""

    protocol_version = "HTTP/1.1"
    connections: ClassVar[set[tuple[str, int]]] = set()

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        content = json.loads(self.rfile.read(length))["messages"][-1]["content"]
        type(self).connections.add(self.client_address)
        body = json.dumps({"choices": [{"message": {"content": f"a:{content}"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @override
    def log_message(self, format: str, *args: Any) -> None:
        return  # the test output is not an access log


def test_the_replay_reuses_its_connections() -> None:
    # R3-20: a new TCP+TLS connection per call; the latency the report put beside the
    # vendor's was mostly the handshake. One pool: at most `concurrency` connections.
    _ChatHandler.connections = set()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = Endpoint(f"http://127.0.0.1:{server.server_address[1]}", "dep-1", "dk")
        answers, latency = replay_holdout(endpoint, _rows(12), concurrency=2)
    finally:
        server.shutdown()
        server.server_close()

    assert answers == [f"a:q{i}" for i in range(12)]
    assert latency.errors == 0
    assert len(_ChatHandler.connections) <= 2
