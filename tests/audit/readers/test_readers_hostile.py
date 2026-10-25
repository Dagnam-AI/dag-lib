"""Files a customer's tool can really write: a lone surrogate escape, a byte order mark.

An uploaded export is untrusted, so each case is a real file on disk, read through the public
entry points (``read_traces`` and ``scan_export``), never a hand-built row.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import cast

import pytest
from tests.audit.readers.test_generic import COLUMN_MAP

from dagnam.audit import ReadStats, TraceRecord, read_traces, scan_export
from dagnam.audit.readers import ExportInputError, Source, UnsupportedExportError
from dagnam.audit.readers.base import MalformedRowError
from dagnam.audit.readers.recorder import to_record

# JSON text of a lone high surrogate; a tool that cuts a string between UTF-16 halves writes it.
LONE = "\\ud800"
SYSTEM = "support triage assistant"
SOURCES = [
    ("openai", "openai_sample.jsonl"),
    ("langfuse", "langfuse_sample.jsonl"),
    ("langsmith", "langsmith_sample.jsonl"),
    ("jsonl", "generic_sample.jsonl"),
    ("dagnam", "recorder_openai_chat.jsonl"),
]
# Where the scan hashes a string: the system prompt (template id) and the reply (output digest).
POISONS = [
    *((source, name, SYSTEM) for source, name in SOURCES),
    ("openai", "openai_sample.jsonl", '"content": "medium"'),
    ("jsonl", "generic_sample.jsonl", '"out": "medium"'),
]


def _map(source: str) -> dict[str, str] | None:
    return COLUMN_MAP if source == "jsonl" else None


def _read(path: Path, source: str) -> tuple[list[TraceRecord], ReadStats]:
    records, stats = read_traces(path, source=cast("Source", source), column_map=_map(source))
    return list(records), stats


def _poisoned(fixtures_dir: Path, name: str, where: str) -> tuple[str, str]:
    """``(clean, poisoned)``: the fixture, and the same text with the second line holding ``where``
    carrying a lone surrogate escape inside it."""
    clean = (fixtures_dir / name).read_text(encoding="utf-8")
    lines = clean.splitlines(keepends=True)
    target = [i for i, line in enumerate(lines) if where in line][1]
    lines[target] = lines[target].replace(where, where[:-2] + LONE + where[-2:], 1)
    return clean, "".join(lines)


@pytest.mark.parametrize(("source", "name", "where"), POISONS)
def test_a_lone_surrogate_row_is_one_malformed_row_not_a_crashed_scan(
    fixtures_dir: Path, tmp_path: Path, source: str, name: str, where: str
) -> None:
    clean, poisoned = _poisoned(fixtures_dir, name, where)
    (tmp_path / "clean.jsonl").write_text(clean * 4, encoding="utf-8")
    assert LONE in poisoned
    path = tmp_path / "poisoned.jsonl"
    path.write_text(poisoned + clean * 3, encoding="utf-8")

    # What the reader let through used to raise in the scan, at the hash of that string.
    scan = scan_export(path, source=cast("Source", source), column_map=_map(source))
    whole, _ = _read(tmp_path / "clean.jsonl", source)
    kept, stats = _read(path, source)

    assert scan.stats.rows_malformed == stats.rows_malformed == 1
    assert len(kept) == len(whole) - 1


def test_a_lone_surrogate_in_a_json_document_drops_that_row_only(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    lines = (fixtures_dir / "openai_sample.jsonl").read_text("utf-8").splitlines() * 4
    rows = [json.loads(line) for line in lines]  # distinct objects: only one is poisoned
    rows[1]["body"]["messages"][0]["content"] = "support \ud800 assistant"  # escaped when dumped
    path = tmp_path / "export.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    assert LONE in path.read_text(encoding="utf-8")

    kept, stats = _read(path, "openai")

    assert (stats.rows_seen, stats.rows_malformed) == (len(rows), 1)
    assert len(kept) == len(rows) - 1


def test_a_lone_surrogate_in_a_gzipped_export_drops_that_row_only(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    clean, poisoned = _poisoned(fixtures_dir, "openai_sample.jsonl", SYSTEM)
    path = tmp_path / "export.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(poisoned + clean * 3)

    _, stats = _read(path, "openai")

    assert stats.rows_malformed == 1


def test_a_paired_surrogate_escape_is_an_ordinary_character(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    clean = (fixtures_dir / "openai_sample.jsonl").read_text(encoding="utf-8")
    path = tmp_path / "pair.jsonl"
    path.write_text(clean.replace(SYSTEM, "support \\ud83d\\ude00 assistant"), encoding="utf-8")

    _, stats = _read(path, "openai")

    assert stats.rows_malformed == 0


@pytest.mark.parametrize(("source", "name"), SOURCES)
def test_a_byte_order_mark_does_not_cost_the_first_row(
    fixtures_dir: Path, tmp_path: Path, source: str, name: str
) -> None:
    text = (fixtures_dir / name).read_text(encoding="utf-8")
    plain = tmp_path / "plain.jsonl"
    plain.write_text(text, encoding="utf-8")
    marked = tmp_path / "marked.jsonl"
    marked.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))  # what Windows tools save

    expected, expected_stats = _read(plain, source)
    kept, stats = _read(marked, source)

    assert [r.trace_id for r in kept] == [r.trace_id for r in expected]
    assert stats.rows_malformed == expected_stats.rows_malformed == 0


def test_a_byte_order_mark_on_a_json_document_and_a_gzipped_export(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    text = (fixtures_dir / "openai_sample.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines()]
    document = tmp_path / "export.json"
    document.write_bytes(b"\xef\xbb\xbf" + json.dumps(rows).encode("utf-8"))
    zipped = tmp_path / "export.jsonl.gz"
    zipped.write_bytes(gzip.compress(b"\xef\xbb\xbf" + text.encode("utf-8")))

    for path in (document, zipped):
        kept, stats = _read(path, "openai")
        assert (len(kept), stats.rows_malformed) == (len(rows), 0), path.name


SECRET = "SECRET-customer-text " + "zzz" * 40
HEADER = {"v": 2, "trace_id": "t", "ts": "2026-10-12T00:00:00Z"}


@pytest.mark.parametrize(
    "fields",
    [
        {"norm_v": SECRET},
        {"norm_v": {"note": SECRET}},
        {"norm_v": [SECRET]},
        {"norm_v": True},
        {"v": 10**3000},  # an integer, but not a version anyone wrote
    ],
)
def test_a_newer_export_refusal_never_quotes_a_field_of_the_file(
    tmp_path: Path, fields: dict[str, object]
) -> None:
    path = tmp_path / "newer.jsonl"
    path.write_text(json.dumps(HEADER | fields) + "\n", encoding="utf-8")

    with pytest.raises(UnsupportedExportError) as refused:
        _read(path, "dagnam")
    with pytest.raises(MalformedRowError) as row:
        to_record(json.loads(path.read_text(encoding="utf-8")))

    for error in (refused.value, row.value):
        assert "SECRET" not in str(error)
        assert len(str(error)) < 200
    assert "newer recorder" in str(refused.value)


def test_a_newer_export_refusal_still_names_the_versions_it_can() -> None:
    row = HEADER | {"norm_v": 7}
    with pytest.raises(MalformedRowError, match=r"\(v=2, norm_v=7\)"):
        to_record(row)


@pytest.mark.parametrize(
    ("name", "body"),
    [
        # polars quotes the cell it could not parse, and the bytes of a file it cannot open.
        ("quote.csv", f'trace_id,ts,messages,response\nt1,2026-08-01T00:00:00Z,"{SECRET} ok\n'),
        ("ragged.csv", f"trace_id,ts\nt1,2026-08-01T00:00:00Z,{SECRET},extra\n"),
        ("footer.parquet", f"PAR1{SECRET}PAR1"),
        ("bytes.csv", f"trace_id,ts,messages,response\nt1,\xff{SECRET},hi,ok\n"),
    ],
)
def test_a_table_the_reader_cannot_open_is_refused_without_quoting_it(
    tmp_path: Path, name: str, body: str
) -> None:
    path = tmp_path / name
    path.write_bytes(body.encode("latin-1"))

    with pytest.raises(ExportInputError) as refused:
        _read(path, "csv" if name.endswith(".csv") else "jsonl")

    assert "SECRET" not in str(refused.value)
    assert str(refused.value).startswith(f"the {path.suffix.lstrip('.')} export could not be read")


def test_a_json_document_nested_too_deep_is_one_malformed_row_not_a_crash(tmp_path: Path) -> None:
    depth = 100_000
    path = tmp_path / "deep.json"
    path.write_text("[" * depth + "]" * depth + "\n" + '{"trace_id": "t"}\n', encoding="utf-8")

    with pytest.raises(UnsupportedExportError):  # the second line is read, the first is not
        _read(path, "jsonl")
