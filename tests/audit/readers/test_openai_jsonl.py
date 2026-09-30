"""OpenAI batch/stored-completion JSONL -> TraceRecord."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import pytest

from dagnam.audit import read_traces
from dagnam.audit.readers import openai_jsonl

# Row counts of ``fixtures/openai_sample.jsonl`` (see fixtures/README.md).
LINES = 12
RECORDS = 12
PER_SESSION = 4
SESSIONS = 3


def test_reader_yields_request_response_pairs(fixtures_dir: Path) -> None:
    records, stats = read_traces(fixtures_dir / "openai_sample.jsonl", source="openai")
    rows = list(records)

    assert len(rows) == RECORDS
    assert (stats.rows_seen, stats.rows_kept, stats.rows_malformed) == (LINES, RECORDS, 0)
    for row in rows:
        assert row.system
        assert row.messages
        assert all(m.role == "user" for m in row.messages)
        assert row.response
        assert row.prompt_tokens > 0
        assert row.latency_ms == 0.0  # the batch API records no latency
        assert row.cost_usd is None  # nor a cost
        assert row.ts == datetime.fromtimestamp(row.ts.timestamp(), tz=UTC)
        assert row.model == "gpt-4o-mini-2024-07-18"
    by_session = Counter(r.session_id for r in rows)
    assert len(by_session) == SESSIONS
    assert set(by_session.values()) == {PER_SESSION}
    assert rows[0].ts == datetime(2026, 8, 1, 9, 0, tzinfo=UTC)
    assert rows[0].trace_id == "chatcmpl-ticket-1-intent"


def _pair() -> dict[str, object]:
    return {
        "custom_id": "req-1",
        "body": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        "response": {
            "status_code": 200,
            "body": {
                "id": "chatcmpl-1",
                "created": 1_785_000_000,
                "model": "gpt-4o-mini-2024-07-18",
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        },
        "error": None,
    }


def test_skips_failed_requests() -> None:
    assert openai_jsonl.to_record({**_pair(), "error": {"code": "rate_limit"}}) is None
    failed = _pair()
    failed["response"] = {"status_code": 429, "body": {"error": {"message": "slow down"}}}
    assert openai_jsonl.to_record(failed) is None


def test_bare_request_and_completion_shape() -> None:
    row = {
        "request": {
            "model": "m",
            "messages": [{"role": "developer", "content": "sys"}, {"role": "user", "content": "q"}],
        },
        "response": {
            "id": "chatcmpl-2",
            "created": 1_785_000_000,
            "model": "m",
            "choices": [
                {"message": {"role": "assistant", "content": None, "tool_calls": [{"id": "c"}]}}
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 0},
            "metadata": {"session_id": "s", "workload": "w"},
        },
        "latency_ms": 321,
    }

    record = openai_jsonl.to_record(row)

    assert record is not None
    assert record.trace_id == "chatcmpl-2"
    assert record.system == "sys"
    assert record.response == ""
    assert record.response_tool_calls == ({"id": "c"},)
    assert record.latency_ms == 321.0
    assert (record.session_id, record.workload_hint) == ("s", "w")


def test_flat_completion_row_uses_custom_id_when_no_completion_id() -> None:
    row = {
        "custom_id": "flat-1",
        "messages": [{"role": "user", "content": "q"}],
        "choices": [{"message": {"role": "assistant", "content": "a"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        "created": 1_785_000_000,
        "model": "m",
    }
    record = openai_jsonl.to_record(row)
    assert record is not None
    assert record.trace_id == "flat-1"
    assert record.session_id is None


def _completion(**changes: Any) -> dict[str, Any]:
    """``_pair()`` with ``changes`` merged into its completion body."""
    body: dict[str, Any] = {
        "id": "chatcmpl-1",
        "created": 1_785_000_000,
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [{"message": {"role": "assistant", "content": "hello"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    }
    return {**_pair(), "response": {"status_code": 200, "body": {**body, **changes}}}


@pytest.mark.parametrize(
    "bad",
    [
        {"choices": []},  # an empty choice list
        {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "p"}}]},
    ],
    ids=["empty-choices", "stream-chunk"],
)
def test_one_unexpected_completion_is_counted_malformed_not_fatal(
    tmp_path: Path, bad: dict[str, Any]
) -> None:
    # 1 bad line among 99 good ones used to abort the scan with an IndexError/KeyError.
    rows = [_completion() for _ in range(99)] + [_completion(**bad)]
    path = tmp_path / "a.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    records, stats = read_traces(path, source="openai")

    assert len(list(records)) == 99
    assert (stats.rows_malformed, stats.first_malformed) == (1, (99,))


def test_cached_prompt_tokens_are_read() -> None:
    usage = {"prompt_tokens": 10_000, "completion_tokens": 1}
    row = _completion(usage={**usage, "prompt_tokens_details": {"cached_tokens": 9_900}})

    record = openai_jsonl.to_record(row)

    assert record is not None
    assert (record.prompt_tokens, record.cached_prompt_tokens) == (10_000, 9_900)


def _responses_row(i: int, output: list[dict[str, Any]]) -> dict[str, Any]:
    """A ``{request, response}`` line of the Responses API (the Agents SDK's default)."""
    return {
        "request": {"model": "gpt-5-mini", "instructions": "Classify.", "input": f"ticket {i}"},
        "response": {
            "id": f"resp_{i}",
            "object": "response",
            "created_at": 1_790_000_000 + i,
            "model": "gpt-5-mini",
            "output": output,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 1,
                "input_tokens_details": {"cached_tokens": 4},
            },
        },
    }


def test_responses_api_lines_are_read(tmp_path: Path) -> None:
    # Every Responses line was malformed ("missing messages") and the scan aborted.
    message = {"type": "message", "content": [{"type": "output_text", "text": "billing"}]}
    reasoning = {"type": "reasoning", "id": "rs", "summary": []}
    rows = [_responses_row(i, [reasoning, message]) for i in range(50)]
    path = tmp_path / "responses.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    records, stats = read_traces(path, source="openai")
    first, *_ = list(records)

    assert stats.rows_malformed == 0
    assert first.system == "Classify."
    assert [(m.role, m.content) for m in first.messages] == [("user", "ticket 0")]
    assert first.response == "billing"
    assert first.ts == datetime.fromtimestamp(1_790_000_000, tz=UTC)
    assert (first.prompt_tokens, first.completion_tokens, first.cached_prompt_tokens) == (10, 1, 4)
    call = {"type": "function_call", "call_id": "c", "name": "route", "arguments": '{"team": "a"}'}
    tooled = openai_jsonl.to_record(_responses_row(0, [reasoning, call]))
    assert tooled is not None
    assert tooled.response_tool_calls == (call,)


def test_inline_reasoning_is_not_part_of_the_answer() -> None:
    # DeepSeek-R1 served without a reasoning parser: a binary label became 300
    # distinct free-text outputs, so the workload was never audited.
    think = "<think>The review sounds happy overall, mentions fast delivery.</think>\npos"
    record = openai_jsonl.to_record(
        _completion(choices=[{"message": {"role": "assistant", "content": think}}])
    )
    assert record is not None
    assert record.response == "pos"


def test_the_response_schema_is_the_request_signature_and_the_tools_are_not() -> None:
    fmt = {"type": "json_schema", "json_schema": {"name": "invoice", "schema": {}}}
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    row = _completion()
    row["body"] = {**row["body"], "response_format": fmt, "tools": tools}
    record = openai_jsonl.to_record(row)
    assert record is not None
    assert record.signature == "schema=invoice"  # B2-2: the offered tools never key it


def test_a_text_outcome_in_the_metadata_is_ignored() -> None:
    messages = [{"role": "user", "content": "hi"}]
    body = {"model": "gpt-4o-mini", "messages": messages, "metadata": {"outcome": "accepted"}}
    record = openai_jsonl.to_record({**_pair(), "body": body})
    assert record is not None
    assert record.outcome is None
