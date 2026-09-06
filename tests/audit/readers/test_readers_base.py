"""Streaming row iteration, malformed accounting and the shared field helpers."""

from __future__ import annotations

from datetime import UTC, datetime
import gzip
import json
from pathlib import Path

import polars as pl
import pytest

from dagnam.audit import MALFORMED_FATAL_SHARE, MalformedExportError, read_traces
from dagnam.audit.readers import Source, base

# A well-formed generic row; ``messages`` is deliberately a plain string.
ROW = {"trace_id": "t", "ts": "2026-08-01T00:00:00Z", "messages": "hi", "response": "ok"}
# Fixture size for the share tests: 3/50 = 6% (fatal), 2/50 = 4% (counted).
TOTAL = 50
FATAL_MALFORMED = 3
COUNTED_MALFORMED = 2


def _write(path: Path, rows: list[object]) -> Path:
    with path.open("w") as handle:
        for row in rows:
            handle.write((row if isinstance(row, str) else json.dumps(row)) + "\n")
    return path


def _rows_with_malformed(n_bad: int) -> list[object]:
    rows: list[object] = [{**ROW, "trace_id": str(i)} for i in range(TOTAL)]
    for i in range(n_bad):
        rows[10 + i] = {"trace_id": "bad", "ts": "not a date", "messages": "x", "response": "y"}
    return rows


def test_iter_rows_reads_jsonl_gzip_parquet_and_csv(tmp_path: Path) -> None:
    jsonl = _write(tmp_path / "a.jsonl", [ROW, ROW])
    with gzip.open(tmp_path / "a.jsonl.gz", "wt") as handle:
        handle.write(jsonl.read_text())
    pl.DataFrame([ROW, ROW]).write_parquet(tmp_path / "a.parquet")
    pl.DataFrame([ROW, ROW]).write_csv(tmp_path / "a.csv")

    for name in ("a.jsonl", "a.jsonl.gz", "a.parquet", "a.csv"):
        assert list(base.iter_rows(tmp_path / name)) == [ROW, ROW], name


def test_iter_rows_rejects_unknown_suffix(tmp_path: Path) -> None:
    (tmp_path / "a.xlsx").write_bytes(b"")
    with pytest.raises(ValueError, match=r"unsupported export file type '\.xlsx'"):
        next(base.iter_rows(tmp_path / "a.xlsx"))


def test_iter_rows_skips_blank_lines_and_flags_unparseable_ones(tmp_path: Path) -> None:
    path = _write(tmp_path / "a.jsonl", [ROW, "", "not json", "[1, 2]", ROW])
    assert list(base.iter_rows(path)) == [ROW, None, None, ROW]


def test_iter_rows_batches_by_thousand(tmp_path: Path) -> None:
    path = _write(tmp_path / "a.jsonl", [ROW] * (base.BATCH_ROWS + 1))
    assert sum(1 for _ in base.iter_rows(path)) == base.BATCH_ROWS + 1


@pytest.mark.parametrize("source", ["jsonl", "csv"])
def test_empty_file_yields_nothing(tmp_path: Path, source: Source) -> None:
    path = tmp_path / f"empty.{source}"
    path.write_text("")
    records, stats = read_traces(path, source=source)
    assert list(records) == []
    assert (stats.rows_seen, stats.rows_kept, stats.rows_malformed) == (0, 0, 0)


def test_stats_are_final_only_after_exhaustion(tmp_path: Path) -> None:
    path = _write(tmp_path / "a.jsonl", _rows_with_malformed(COUNTED_MALFORMED))
    records, stats = read_traces(path, source="jsonl")

    assert stats.rows_seen == 0
    first = next(records)
    assert first.trace_id == "0"
    assert stats.rows_seen == 1
    rest = list(records)
    assert len(rest) == TOTAL - COUNTED_MALFORMED - 1
    assert (stats.rows_seen, stats.rows_kept, stats.rows_malformed) == (
        TOTAL,
        TOTAL - COUNTED_MALFORMED,
        COUNTED_MALFORMED,
    )
    assert stats.first_malformed == (10, 11)
    assert stats.rows_malformed / stats.rows_seen < MALFORMED_FATAL_SHARE


def test_malformed_share_above_threshold_is_fatal_after_the_file_is_read(tmp_path: Path) -> None:
    path = _write(tmp_path / "a.jsonl", _rows_with_malformed(FATAL_MALFORMED))
    records, stats = read_traces(path, source="jsonl")

    kept: list[str] = []
    with pytest.raises(MalformedExportError) as info:
        for record in records:
            kept.append(record.trace_id)

    assert len(kept) == TOTAL - FATAL_MALFORMED
    assert info.value.source == "jsonl"
    assert info.value.stats is stats
    assert len(stats.first_malformed) == 3
    assert stats.first_malformed == (10, 11, 12)
    assert "3 of 50" in str(info.value)
    assert "[10, 11, 12]" in str(info.value)


def test_all_malformed_file_is_fatal_and_keeps_three_indices(tmp_path: Path) -> None:
    path = _write(tmp_path / "a.jsonl", ["nope"] * 12)
    records, stats = read_traces(path, source="jsonl")
    with pytest.raises(MalformedExportError):
        list(records)
    assert (stats.rows_seen, stats.rows_kept, stats.rows_malformed) == (12, 0, 12)
    assert stats.first_malformed == (0, 1, 2)


def test_control_characters_are_stripped_from_every_string(tmp_path: Path) -> None:
    hostile = "\x1b]8;;http://evil.example\x07click\x1b]8;;\x07 done\x9b2J"
    row = {
        **ROW,
        "trace_id": "id\x1b[31m",
        "model": "m\x07",
        "system": hostile,
        "messages": [{"role": "user\x00", "content": [{"type": "text", "text": hostile}]}],
        "response": hostile,
    }
    records, _ = read_traces(_write(tmp_path / "a.jsonl", [row]), source="jsonl")
    record = next(records)

    texts = [record.trace_id, record.model, record.system or "", record.response]
    texts += [m.role + m.content for m in record.messages]
    for value in texts:
        assert not set(value) & {"\x1b", "\x07", "\x9b", "\x00"}
    # Printable content survives; only the control bytes that made it a live escape are gone.
    assert record.response == "]8;;http://evil.exampleclick]8;; done2J"
    assert record.messages[0].content == record.response
    assert record.messages[0].role == "user"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-08-01T09:00:00Z", datetime(2026, 8, 1, 9, tzinfo=UTC)),
        ("2026-08-01T09:00:00.250000", datetime(2026, 8, 1, 9, 0, 0, 250_000, tzinfo=UTC)),
        ("2026-08-01T11:00:00+02:00", datetime(2026, 8, 1, 9, tzinfo=UTC)),
        (1_785_000_000, datetime.fromtimestamp(1_785_000_000, tz=UTC)),
        (datetime(2026, 8, 1, 9), datetime(2026, 8, 1, 9, tzinfo=UTC)),
    ],
)
def test_parse_ts_normalises_to_utc(value: object, expected: datetime) -> None:
    assert base.parse_ts(value) == expected


@pytest.mark.parametrize("value", ["yesterday", None, 1e18, {"t": 1}])
def test_parse_ts_rejects_garbage(value: object) -> None:
    with pytest.raises(base.MalformedRowError):
        base.parse_ts(value)


def test_numeric_coercion() -> None:
    assert base.as_int("12") == 12
    assert base.as_int(12.9) == 12
    assert base.as_float("0.5") == 0.5
    assert base.optional_float(None) is None
    assert base.optional_float("") is None
    for bad in ("nan", "x", [1], None):
        with pytest.raises(base.MalformedRowError):
            base.as_float(bad)


def test_text_flattens_containers_and_content_parts() -> None:
    assert base.text({"a": [1, "b"]}) == '{"a": [1, "b"]}'
    assert base.text(None) == ""
    assert base.content_text([{"type": "text", "text": "a"}, {"type": "image_url"}, "b"]) == "ab"
    assert base.content_text("plain") == "plain"


def test_split_prompt_separates_system_from_turns() -> None:
    system, turns = base.split_prompt(
        [
            {"role": "system", "content": "one"},
            {"role": "developer", "content": "two"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "prior answer"},
            {"role": "tool", "content": "42"},
        ]
    )
    assert system == "one\ntwo"
    assert [(m.role, m.content) for m in turns] == [("user", "q"), ("tool", "42")]
    assert base.split_prompt("bare prompt") == (None, (base.Message("user", "bare prompt"),))
    assert base.split_prompt({"messages": [{"role": "user", "content": "x"}]})[1][0].content == "x"
    assert base.split_prompt({"input": "x"}) == (None, (base.Message("user", '{"input": "x"}'),))
    with pytest.raises(base.MalformedRowError):
        base.split_prompt([{"role": "user"}, "loose string"])
    with pytest.raises(base.MalformedRowError):
        base.split_prompt([])


def test_get_walks_dotted_aliases_in_order() -> None:
    row = {"usage": {"input": 3}, "promptTokens": 9, "flat": None}
    assert base.get(row, "usage.input", "promptTokens") == 3
    assert base.get(row, "usage.output", "promptTokens") == 9
    assert base.get(row, "usage.input.deeper", "flat") is None
    with pytest.raises(base.MalformedRowError, match=r"missing usage\.output"):
        base.require(row, "usage.output")


def test_tool_calls_accepts_only_a_list_of_objects() -> None:
    assert base.tool_calls(None) == ()
    assert base.tool_calls([{"id": 1}]) == ({"id": 1},)
    with pytest.raises(base.MalformedRowError):
        base.tool_calls([{"id": 1}, "x"])
