"""Langfuse observation export -> TraceRecord."""

from __future__ import annotations

from collections import Counter
from datetime import UTC
import json
from pathlib import Path

from dagnam.audit import read_traces
from dagnam.audit.readers import langfuse

# Row counts of ``fixtures/langfuse_sample.jsonl`` (see fixtures/README.md).
LINES = 15
RECORDS = 12
PER_SESSION = 4
SESSIONS = 3


def test_reader_yields_generations(fixtures_dir: Path) -> None:
    records, stats = read_traces(fixtures_dir / "langfuse_sample.jsonl", source="langfuse")
    rows = list(records)

    assert len(rows) == RECORDS
    assert (stats.rows_seen, stats.rows_kept, stats.rows_malformed) == (LINES, RECORDS, 0)
    assert stats.first_malformed == ()
    for row in rows:
        assert row.system
        assert row.messages
        assert all(m.role == "user" for m in row.messages)
        assert row.response
        assert row.latency_ms > 0
        assert row.prompt_tokens > 0
        assert row.ts.tzinfo is UTC
        assert row.model == "gpt-4o-mini"
        assert row.workload_hint in {"intent", "urgency", "extract", "reply"}
    assert sum(r.cost_usd is not None for r in rows) / len(rows) >= 0.99
    by_session = Counter(r.session_id for r in rows)
    assert len(by_session) == SESSIONS
    assert set(by_session.values()) == {PER_SESSION}
    assert len({r.trace_id for r in rows}) == RECORDS


def test_workload_hint_is_none_without_metadata(fixtures_dir: Path, tmp_path: Path) -> None:
    stripped = tmp_path / "stripped.jsonl"
    with (fixtures_dir / "langfuse_sample.jsonl").open() as src, stripped.open("w") as dst:
        for line in src:
            row = json.loads(line)
            row.pop("metadata", None)
            dst.write(json.dumps(row) + "\n")

    records, _ = read_traces(stripped, source="langfuse")
    assert {r.workload_hint for r in records} == {None}


def _generation() -> dict[str, object]:
    return {
        "id": "gen-1",
        "traceId": "trace-1",
        "type": "GENERATION",
        "startTime": "2026-08-01T09:00:00.000Z",
        "endTime": "2026-08-01T09:00:00.250Z",
        "model": "gpt-4o-mini",
        "input": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        "output": {"role": "assistant", "content": "hello"},
        "usage": {"input": 5, "output": 1},
        "calculatedTotalCost": 0.001,
    }


def test_skips_non_generation_and_empty_output() -> None:
    assert langfuse.to_record({**_generation(), "type": "SPAN"}) is None
    assert langfuse.to_record({**_generation(), "output": None}) is None


def test_legacy_token_and_latency_spellings() -> None:
    row = _generation()
    del row["usage"], row["endTime"]
    row.update({"promptTokens": 7, "completionTokens": 2, "latency": 1.5, "sessionId": "s-9"})

    record = langfuse.to_record(row)

    assert record is not None
    assert (record.prompt_tokens, record.completion_tokens) == (7, 2)
    assert record.latency_ms == 1500.0
    assert record.session_id == "s-9"


def test_usage_details_and_cost_details_spellings() -> None:
    row = _generation()
    del row["usage"], row["calculatedTotalCost"]
    row.update({"usageDetails": {"input": 9, "output": 3}, "costDetails": {"total": 0.5}})

    record = langfuse.to_record(row)

    assert record is not None
    assert (record.prompt_tokens, record.completion_tokens, record.cost_usd) == (9, 3, 0.5)


def test_string_output_and_tool_calls() -> None:
    plain = langfuse.to_record({**_generation(), "output": "just text"})
    assert plain is not None
    assert plain.response == "just text"
    assert plain.response_tool_calls == ()

    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    tooled = langfuse.to_record(
        {**_generation(), "output": {"role": "assistant", "content": None, "tool_calls": [call]}}
    )
    assert tooled is not None
    assert tooled.response == ""
    assert tooled.response_tool_calls == (call,)


def test_missing_tokens_and_cost_are_defaulted_not_malformed() -> None:
    row = _generation()
    del row["usage"], row["calculatedTotalCost"]

    record = langfuse.to_record(row)

    assert record is not None
    assert (record.prompt_tokens, record.completion_tokens) == (0, 0)
    assert record.cost_usd is None
    assert record.outcome is None


def test_outcome_from_metadata() -> None:
    record = langfuse.to_record({**_generation(), "metadata": {"outcome": "0.5"}})
    assert record is not None
    assert record.outcome == 0.5
