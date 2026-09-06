"""OpenAI batch/stored-completion JSONL -> TraceRecord."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

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
