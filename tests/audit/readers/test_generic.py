"""Generic JSONL/CSV exports read through a ``{target: source}`` column map."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from dagnam.audit import Message, UnsupportedExportError, read_traces
from dagnam.audit.readers import generic

# Row counts of ``fixtures/generic_sample.{jsonl,csv}`` (see fixtures/README.md).
LINES = 12
RECORDS = 12
PER_SESSION = 4
SESSIONS = 3

COLUMN_MAP = {
    "trace_id": "call_id",
    "ts": "at",
    "system": "sys",
    "messages": "prompt",
    "response": "out",
    "prompt_tokens": "in_tok",
    "completion_tokens": "out_tok",
    "latency_ms": "ms",
    "cost_usd": "usd",
    "session_id": "ticket",
    "outcome": "score",
    "workload_hint": "kind",
}


@pytest.mark.parametrize("name", ["generic_sample.jsonl", "generic_sample.csv"])
def test_reader_maps_columns(fixtures_dir: Path, name: str) -> None:
    source = "csv" if name.endswith(".csv") else "jsonl"
    records, stats = read_traces(fixtures_dir / name, source=source, column_map=COLUMN_MAP)
    rows = list(records)

    assert len(rows) == RECORDS
    assert (stats.rows_seen, stats.rows_kept, stats.rows_malformed) == (LINES, RECORDS, 0)
    for row in rows:
        assert row.system
        assert [m.role for m in row.messages] == ["user"]
        assert row.response
        assert row.latency_ms > 0
        assert row.prompt_tokens > 0
        assert row.cost_usd is not None
        assert row.outcome in {1.0, 0.8}
        assert row.model == "gpt-4o-mini"  # identity mapping: unmapped targets read their own name
        assert row.ts.tzinfo is UTC
        assert row.response_tool_calls == ()
    by_session = Counter(r.session_id for r in rows)
    assert len(by_session) == SESSIONS
    assert set(by_session.values()) == {PER_SESSION}


def test_unknown_target_raises_before_any_row_is_read(tmp_path: Path) -> None:
    missing_file = tmp_path / "never-created.jsonl"
    with pytest.raises(ValueError, match="unknown column_map target 'responses'"):
        read_traces(missing_file, source="jsonl", column_map={"responses": "out"})


def test_missing_required_source_column_raises_unsupported_export(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"call_id": "1", "at": "2026-08-01T00:00:00Z", "out": "x"}) + "\n")

    records, stats = read_traces(
        path, source="jsonl", column_map={"trace_id": "call_id", "ts": "at"}
    )
    with pytest.raises(UnsupportedExportError) as info:
        next(records)

    assert info.value.source == "jsonl"
    assert info.value.missing == ("messages", "response")
    assert "--map" in str(info.value)
    assert stats.rows_kept == 0


def test_optional_targets_default_and_json_strings_are_decoded() -> None:
    to_record = generic.bind(None).to_record
    row = {
        "trace_id": "1",
        "ts": "2026-08-01T00:00:00+02:00",
        "messages": json.dumps(
            [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        ),
        "response": "r",
        "response_tool_calls": json.dumps([{"id": "c"}]),
    }

    record = to_record(row)

    assert record is not None
    assert record.ts == datetime(2026, 7, 31, 22, tzinfo=UTC)
    assert record.system == "s"
    assert record.messages == (Message("user", "u"),)
    assert record.response_tool_calls == ({"id": "c"},)
    assert (record.model, record.prompt_tokens, record.completion_tokens) == ("unknown", 0, 0)
    assert record.latency_ms == 0.0
    assert (record.cost_usd, record.session_id, record.outcome, record.workload_hint) == (None,) * 4


def test_explicit_system_column_wins_over_system_turn() -> None:
    to_record = generic.bind({"system": "sys"}).to_record
    row = {
        "trace_id": "1",
        "ts": "2026-08-01T00:00:00Z",
        "sys": "explicit",
        "messages": [
            {"role": "system", "content": "in-messages"},
            {"role": "user", "content": "u"},
        ],
        "response": "r",
    }
    record = to_record(row)
    assert record is not None
    assert record.system == "explicit"


def test_unparseable_tool_calls_are_malformed() -> None:
    to_record = generic.bind(None).to_record
    row = {"trace_id": "1", "ts": "2026-08-01T00:00:00Z", "messages": "hi", "response": "r"}
    with pytest.raises(generic.MalformedRowError):
        to_record({**row, "response_tool_calls": "not json"})
    with pytest.raises(generic.MalformedRowError):
        to_record({**row, "response_tool_calls": '"a string"'})
    with pytest.raises(generic.MalformedRowError, match="not JSON"):
        to_record({**row, "response_tool_calls": "[not json"})


def test_missing_required_value_is_malformed() -> None:
    to_record = generic.bind(None).to_record
    row = {"trace_id": "1", "ts": "2026-08-01T00:00:00Z", "messages": "hi", "response": ""}
    with pytest.raises(generic.MalformedRowError, match="missing response"):
        to_record(row)  # an empty CSV cell is an absent value
