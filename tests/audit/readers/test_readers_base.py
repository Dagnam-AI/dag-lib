"""Streaming row iteration, malformed accounting and the shared field helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import gzip
import json
from pathlib import Path

import polars as pl
import pytest

from dagnam.audit import MALFORMED_FATAL_SHARE, MalformedExportError, TraceRecord, read_traces
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
    with gzip.open(tmp_path / "a.csv.gz", "wb") as handle:
        handle.write((tmp_path / "a.csv").read_bytes())

    for name in ("a.jsonl", "a.jsonl.gz", "a.parquet", "a.csv", "a.csv.gz"):
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
        # m3: a basic-format ISO date is all digits, and was read as a 1970 epoch.
        ("20260801", datetime(2026, 8, 1, tzinfo=UTC)),
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


def test_text_stringifies_containers() -> None:
    assert base.text({"a": [1, "b"]}) == '{"a": [1, "b"]}'
    assert base.text(None) == ""


def test_get_walks_dotted_aliases_in_order() -> None:
    row = {"usage": {"input": 3}, "promptTokens": 9, "flat": None}
    assert base.get(row, "usage.input", "promptTokens") == 3
    assert base.get(row, "usage.output", "promptTokens") == 9
    assert base.get(row, "usage.input.deeper", "flat") is None
    with pytest.raises(base.MalformedRowError, match=r"missing usage\.output"):
        base.require(row, "usage.output")


def test_parse_ts_rejects_implausible_timestamps() -> None:
    # ``created: 0`` among one day of calls stretched the window to 20,697 days.
    tomorrow_and_more = (datetime.now(UTC) + timedelta(days=2)).isoformat()
    for value in (0, "1970-01-01T00:00:00Z", "2019-12-31T23:59:59Z", tomorrow_and_more):
        with pytest.raises(base.MalformedRowError, match="implausible"):
            base.parse_ts(value)
    assert base.parse_ts("2020-01-01T00:00:00Z") == base.EARLIEST_TS


def test_a_row_shape_the_reader_did_not_expect_is_counted_malformed(tmp_path: Path) -> None:
    def to_record(row: base.Row) -> TraceRecord:
        return row["missing"]  # a KeyError, not a MalformedRowError

    reader = base.Reader((), to_record)
    records, stats = base.read_records(_write(tmp_path / "a.jsonl", [ROW]), "x", reader)
    with pytest.raises(MalformedExportError):
        list(records)
    assert (stats.rows_malformed, stats.first_malformed) == (1, (0,))


def test_gemini_thinking_tokens_are_completion_tokens() -> None:
    usage = {"promptTokenCount": 1_000, "candidatesTokenCount": 200, "thoughtsTokenCount": 800}
    assert base.completion_tokens({"usageMetadata": usage}, "usageMetadata") == 1_000
    snake = {"candidates_token_count": 2, "thoughts_token_count": 3}
    assert base.completion_tokens({"u": snake}, "u") == 5
    assert base.completion_tokens({"u": {"candidates_token_count": 2}}, "u") == 2
    # LangChain's ``output_tokens`` already sums every output type, reasoning included.
    langchain = {
        "usage_metadata": {"output_tokens": 1_000, "output_token_details": {"reasoning": 800}},
        "outputs": {"usage_metadata": {"thoughts_token_count": 800}},
    }
    assert base.completion_tokens(langchain, "usage_metadata", "outputs.usage_metadata") == 1_000


def test_cached_prompt_tokens_in_every_vendor_spelling() -> None:
    openai = {"usage": {"prompt_tokens": 10_000, "prompt_tokens_details": {"cached_tokens": 9_900}}}
    assert base.cached_prompt_tokens(openai, "usage") == 9_900
    anthropic = {"usage": {"input_tokens": 100, "cache_read_input_tokens": 800}}
    assert base.cached_prompt_tokens(anthropic, "usage") == 800
    langchain = {"usage_metadata": {"input_tokens": 900, "input_token_details": {"cache_read": 7}}}
    assert base.cached_prompt_tokens(langchain, "usage_metadata") == 7
    gemini = {"usageMetadata": {"promptTokenCount": 10, "cachedContentTokenCount": 4}}
    assert base.cached_prompt_tokens(gemini, "usageMetadata") == 4
    assert base.cached_prompt_tokens({"cached_content_token_count": 3}, "") == 3
    assert base.cached_prompt_tokens({}, "usage", "") == 0


def test_anthropic_cache_counts_are_added_only_to_the_block_that_excludes_them() -> None:
    # LangChain's ``usage_metadata.input_tokens`` already includes the cache; the
    # raw Anthropic usage beside it does not. Summing across blocks doubled it.
    row = {
        "usage_metadata": {"input_tokens": 10_000, "output_tokens": 5},
        "outputs": {"usage": {"input_tokens": 100, "cache_read_input_tokens": 9_900}},
    }
    assert base.prompt_tokens(row, "usage_metadata", "outputs.usage") == 10_000
    assert base.prompt_tokens(row, "outputs.usage") == 10_000


@pytest.mark.parametrize(
    "value",
    [
        1_790_000_000,  # seconds
        "1790000000",  # seconds in a CSV cell
        1_790_000_000.25,
        "1790000000.25",
        1_790_000_000_000,  # milliseconds (Helicone, PostHog)
        "1790000000000",
        1_790_000_000_000_000,  # microseconds
        1_790_000_000_000_000_000,  # nanoseconds (OTel)
        "1790000000000000000",
    ],
)
def test_parse_ts_reads_epoch_seconds_ms_us_and_ns_as_numbers_or_strings(value: object) -> None:
    # An epoch-timestamped CSV used to abort the scan: every cell is a string.
    assert base.parse_ts(value).replace(microsecond=0) == datetime.fromtimestamp(
        1_790_000_000, tz=UTC
    )


def test_an_epoch_csv_export_reads(tmp_path: Path) -> None:
    for name, stamp in (("flat_s.csv", "1790000000"), ("flat_ms.csv", "1790000000000")):
        path = tmp_path / name
        path.write_text(
            "trace_id,ts,messages,response,model\n"
            f'r1,{stamp},"[{{""role"":""user"",""content"":""hi""}}]",a,gpt-4o\n'
        )
        records, stats = read_traces(path, source="csv")
        (record,) = list(records)
        assert stats.rows_malformed == 0
        assert record.ts == datetime.fromtimestamp(1_790_000_000, tz=UTC)


def test_a_json_array_export_reads_like_jsonl(tmp_path: Path) -> None:
    # The UI "Export JSON" is one array; `.json` used to be rejected outright.
    array = tmp_path / "ui_export.json"
    array.write_text(json.dumps([ROW, {**ROW, "trace_id": "u"}, "not an object"]))
    assert list(base.iter_rows(array)) == [ROW, {**ROW, "trace_id": "u"}, None]
    wrapped = tmp_path / "api_page.json"
    wrapped.write_text(json.dumps({"data": [ROW], "meta": {"page": 1}}))
    assert list(base.iter_rows(wrapped)) == [ROW]
    lines = tmp_path / "really_jsonl.json"
    lines.write_text(json.dumps(ROW) + "\n\n" + json.dumps(ROW) + "\n")
    assert list(base.iter_rows(lines)) == [ROW, ROW]
    other = tmp_path / "scalar.json"
    other.write_text("42")
    assert list(base.iter_rows(other)) == [None]
    with gzip.open(tmp_path / "ui_export.json.gz", "wt") as handle:
        handle.write(json.dumps([ROW]))
    assert list(base.iter_rows(tmp_path / "ui_export.json.gz")) == [ROW]
    array.write_text(json.dumps([ROW, {**ROW, "trace_id": "u"}]))
    records, _ = read_traces(array, source="jsonl")
    assert [r.trace_id for r in records] == ["t", "u"]


def test_jsonl_under_a_json_name_splits_on_newlines_only(tmp_path: Path) -> None:
    # m5: U+2028, U+2029 and U+0085 may sit unescaped inside a JSON string, and
    # str.splitlines() cut such a row in two, both halves malformed.
    row = {**ROW, "response": "a\u2028b\u2029c\x85d"}
    lines = tmp_path / "really_jsonl.json"
    lines.write_text(
        json.dumps(row, ensure_ascii=False) + "\r\n" + json.dumps(ROW) + "\n", encoding="utf-8"
    )
    assert list(base.iter_rows(lines)) == [row, ROW]


def test_gzipped_jsonl_streams_without_inflating_to_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A 5 GB gzip of a 50 GB export needed 50 GB of temp space before one row was read.
    def no_temp_dir(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("inflated to disk")

    monkeypatch.setattr(base.tempfile, "TemporaryDirectory", no_temp_dir)
    with gzip.open(tmp_path / "a.jsonl.gz", "wt") as handle:
        handle.write(json.dumps(ROW) + "\n\nnot json\n" + json.dumps(ROW) + "\n")
    assert list(base.iter_rows(tmp_path / "a.jsonl.gz")) == [ROW, None, ROW]


def test_a_text_outcome_is_ignored_not_malformed(tmp_path: Path) -> None:
    # R1-N11: ``outcome`` is read and never used, so "good" must never fail a row.
    rows: list[object] = [
        {**ROW, "trace_id": str(i), "outcome": "good" if i % 2 else "0.5"} for i in range(4)
    ]
    records, stats = read_traces(_write(tmp_path / "a.jsonl", rows), source="jsonl")
    assert [r.outcome for r in records] == [0.5, None, 0.5, None]
    assert stats.rows_malformed == 0
    assert base.optional_outcome(True) is None
    assert base.optional_outcome(float("nan")) is None
