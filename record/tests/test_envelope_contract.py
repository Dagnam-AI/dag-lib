"""The envelope contract: every line built from any input is valid, or building it raises."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
import json
from typing import Any, Callable, Dict

import pytest

from dagnam_record._envelope import LineTooLargeError, build_line

JS_SAFE = 2**53 - 1
BASE: Dict[str, Any] = {
    "provider": "openai",
    "endpoint": "chat.completions",
    "request": {"messages": [{"role": "user", "content": "Hi"}]},
    "response": {"choices": [{"message": {"content": "Hello"}}]},
    "latency_ms": 1.0,
    "now": 1_791_797_400.0,
}


def _build(overrides: Dict[str, Any]) -> str:
    return build_line(**{**BASE, **overrides})


def _cycle() -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    body["me"] = body
    return body


def _deep(depth: int) -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    node = body
    for _ in range(depth):
        node["x"] = {}
        node = node["x"]
    return body


def _media(blob: object) -> Dict[str, Any]:
    return {"messages": [{"role": "user", "content": [{"type": "image", "blob": blob}]}]}


REJECTED: Dict[str, Dict[str, Any]] = {
    "latency-nan": {"latency_ms": float("nan")},
    "latency-inf": {"latency_ms": float("inf")},
    "latency--inf": {"latency_ms": float("-inf")},
    "latency-bool": {"latency_ms": True},
    "latency-str": {"latency_ms": "5"},
    "latency-none": {"latency_ms": None},
    "latency-decimal": {"latency_ms": Decimal(1)},
    "status-zero": {"status": 0},
    "status-99": {"status": 99},
    "status-600": {"status": 600},
    "status-negative": {"status": -1},
    "status-bool": {"status": True},
    "status-str": {"status": "200"},
    "status-float": {"status": 200.5},
    "status-huge": {"status": 10**30},
    "trace-empty": {"trace_id": ""},
    "trace-int": {"trace_id": 5},
    "trace-surrogate": {"trace_id": "\ud800"},
    "now-nan": {"now": float("nan")},
    "now-inf": {"now": float("inf")},
    "now--inf": {"now": float("-inf")},
    "now-huge": {"now": 1e300},
    "now--huge": {"now": -1e300},
    "now-year-10000": {"now": 253_402_300_800},
    "now-bool": {"now": True},
    "now-str": {"now": "noon"},
    "provider-none": {"provider": None},
    "provider-int": {"provider": 5},
    "provider-unknown": {"provider": "gemini"},
    "endpoint-unknown": {"endpoint": "completions"},
    "request-none": {"request": None},
    "request-list": {"request": []},
    "request-str": {"request": "text"},
    "response-none": {"response": None},
    "response-list": {"response": []},
    "response-int": {"response": 5},
    "request-datetime": {"request": {"when": datetime(2026, 1, 1)}},
    "request-bytes": {"request": {"blob": b"abc"}},
    "request-set": {"request": {"tags": {1, 2}}},
    "request-object": {"request": {"sdk": object()}},
    "request-cycle": {"request": _cycle()},
    "request-deep": {"request": _deep(100_000)},
    "request-bigint": {"request": {"n": 10**5000}},
    "request-nan": {"request": {"n": float("nan")}},
    "request-surrogate-key": {"request": {"\ud800": 1}},
    "request-tuple-key": {"request": {(1, 2): 1}},
    "request-oversize": {"request": {"text": "x" * 1_100_000}},
    "request-media-object": {"request": _media(object())},
    "response-cycle": {"response": _cycle()},
    "response-bigint": {"response": {"n": 10**5000}},
    "response-oversize": {"response": {"text": "x" * 1_100_000}},
    "tool-arguments-object": {
        "response": {
            "choices": [
                {"message": {"tool_calls": [{"function": {"name": "f", "arguments": object()}}]}}
            ]
        }
    },
}

ACCEPTED: Dict[str, Dict[str, Any]] = {
    "latency-negative": {"latency_ms": -5},
    "latency-negative-zero": {"latency_ms": -0.0},
    "latency-zero": {"latency_ms": 0},
    "latency-int": {"latency_ms": 412},
    "latency-huge-float": {"latency_ms": 1e308},
    "latency-huge-int": {"latency_ms": 10**400},
    "latency-just-over": {"latency_ms": 2**53},
    "status-none": {"status": None},
    "status-100": {"status": 100},
    "status-404": {"status": 404},
    "status-599": {"status": 599},
    "trace-long": {"trace_id": "t" * 1000},
    "now-year-1": {"now": -62_135_596_800},
    "now-zero": {"now": 0},
    "now-int": {"now": 1_791_797_400},
    "tuple-values": {"request": {"messages": [], "pair": (1, 2)}},
    "odd-keys": {"request": {1: "a", None: "b", 2.5: "c", False: "d"}},
    "unicode": {"request": {"messages": [{"role": "user", "content": "caf\u00e9 \U0001f600"}]}},
    "big-int-in-body": {"request": {"n": 10**300}},
    "empty-bodies": {"request": {}, "response": {}},
    "usage-huge": {
        "response": {
            "usage": {
                "prompt_tokens": 10**30,
                "completion_tokens": -1,
                "prompt_tokens_details": {"cached_tokens": 1.5},
            }
        }
    },
    "usage-types": {"response": {"usage": {"prompt_tokens": True, "completion_tokens": "5"}}},
    "anthropic-usage-sum": {
        "provider": "anthropic",
        "endpoint": "messages",
        "response": {
            "usage": {
                "input_tokens": JS_SAFE,
                "cache_read_input_tokens": JS_SAFE,
                "cache_creation_input_tokens": JS_SAFE,
                "output_tokens": 10**30,
            }
        },
    },
    "responses-usage": {
        "endpoint": "responses",
        "request": {"input": "Hi"},
        "response": {"usage": {"input_tokens": 10**20}},
    },
    "other-custom": {"provider": "other", "endpoint": "custom"},
    "media-only": {"request": _media("AAAA"), "no_raw": True},
}


@pytest.mark.parametrize("overrides", REJECTED.values(), ids=list(REJECTED))
def test_bad_input_raises_value_error_and_nothing_else(overrides: Dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"\w"):
        _build(overrides)


@pytest.mark.parametrize("overrides", ACCEPTED.values(), ids=list(ACCEPTED))
def test_every_line_built_from_odd_input_passes_the_strict_validator(
    overrides: Dict[str, Any], validate_line: Callable[[str], None]
) -> None:
    validate_line(_build(overrides))


def test_a_line_over_the_contract_limit_raises_line_too_large() -> None:
    with pytest.raises(LineTooLargeError, match="1000000"):
        _build({"request": {"text": "x" * 1_100_000}})
    assert issubclass(LineTooLargeError, ValueError)


def test_latency_is_clamped_normalised_and_capped() -> None:
    def latency_of(value: Any) -> str:
        text = _build({"latency_ms": value})
        return text.split('"latency_ms":')[1].split(",")[0]

    assert latency_of(-5) == "0.0"
    assert latency_of(-0.0) == "0.0"
    assert latency_of(412) == "412.0"
    assert latency_of(812.5) == "812.5"
    assert float(latency_of(1e30)) == JS_SAFE
    assert float(latency_of(10**400)) == JS_SAFE


def test_a_call_with_no_http_response_is_recorded_as_status_599() -> None:
    assert json.loads(_build({"status": None}))["status"] == 599
    assert json.loads(_build({}))["status"] == 200


def test_token_counts_are_capped_to_what_a_javascript_reader_holds() -> None:
    norm = json.loads(_build(ACCEPTED["anthropic-usage-sum"]))["norm"]
    assert norm["prompt_tokens"] == JS_SAFE
    assert norm["completion_tokens"] == JS_SAFE
    assert norm["cached_prompt_tokens"] == JS_SAFE
    huge = json.loads(_build(ACCEPTED["usage-huge"]))["norm"]
    assert (huge["prompt_tokens"], huge["completion_tokens"], huge["cached_prompt_tokens"]) == (
        JS_SAFE,
        0,
        0,
    )


@pytest.mark.parametrize("field", ["v", "norm_v"])
@pytest.mark.parametrize("bad", [1.0, True, 2, "1", None])
def test_the_strict_validator_wants_the_integer_one_like_the_real_parser(
    validate_line: Callable[[str], None], field: str, bad: Any
) -> None:
    line = json.loads(_build({}))
    validate_line(json.dumps(line))
    line[field] = bad
    with pytest.raises(ValueError, match=r"\w"):
        validate_line(json.dumps(line))
