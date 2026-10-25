"""The ``dagnam`` source reads a directory of the recorder's day files; no other source does."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import cast

import pytest

from dagnam.audit import NoTracesError, read_traces, scan_export
from dagnam.audit.readers import ExportInputError, Source

CHAT = "recorder_openai_chat.jsonl"


def _lines(fixtures_dir: Path) -> list[str]:
    return (fixtures_dir / CHAT).read_text(encoding="utf-8").splitlines(keepends=True)


def _ids(path: Path) -> list[str]:
    records, _ = read_traces(path, source="dagnam")
    return [record.trace_id for record in records]


def test_a_directory_reads_its_day_files_in_name_order(fixtures_dir: Path, tmp_path: Path) -> None:
    lines = _lines(fixtures_dir)
    (tmp_path / "2026-10-25.jsonl").write_text("".join(lines[:2]), encoding="utf-8")
    (tmp_path / "2026-10-24.jsonl").write_text("".join(lines[2:4]), encoding="utf-8")
    with gzip.open(tmp_path / "2026-10-26.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write("".join(lines[4:6]))
    (tmp_path / ".disabled").write_text("", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not a trace file", encoding="utf-8")
    (tmp_path / "sub.jsonl").mkdir()  # a directory is not a day file

    records, stats = read_traces(tmp_path, source="dagnam")

    expected = [
        *_ids(tmp_path / "2026-10-24.jsonl"),
        *_ids(tmp_path / "2026-10-25.jsonl"),
        *_ids(tmp_path / "2026-10-26.jsonl.gz"),
    ]
    assert expected[:2] == ["rec_chat_3", "rec_chat_4"]
    assert [record.trace_id for record in records] == expected
    assert (stats.rows_seen, stats.rows_malformed) == (6, 0)


def test_a_torn_last_line_does_not_swallow_the_first_line_of_the_next_day(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    lines = _lines(fixtures_dir)
    torn = lines[0][: len(lines[0]) // 2]  # the recorder died mid-write: no newline, half a line
    (tmp_path / "2026-10-24.jsonl").write_text("".join(lines[1:20]) + torn, encoding="utf-8")
    (tmp_path / "2026-10-25.jsonl").write_text("".join(lines), encoding="utf-8")

    records, stats = read_traces(tmp_path, source="dagnam")

    kept = list(records)
    whole = len(_ids(tmp_path / "2026-10-25.jsonl"))
    assert stats.rows_malformed == 1  # the torn line alone, not the next day's first line too
    assert stats.rows_kept == len(kept) == 2 * whole - 1  # the torn day lost its first line only


def test_an_empty_directory_has_no_traces(tmp_path: Path) -> None:
    with pytest.raises(NoTracesError):
        scan_export(tmp_path, source="dagnam")


def test_a_directory_is_scanned_like_the_file_it_holds(fixtures_dir: Path, tmp_path: Path) -> None:
    (tmp_path / "day.jsonl").write_text("".join(_lines(fixtures_dir)), encoding="utf-8")

    from_directory = scan_export(tmp_path, source="dagnam")
    from_file = scan_export(tmp_path / "day.jsonl", source="dagnam")

    assert from_directory.workloads == from_file.workloads


@pytest.mark.parametrize("source", ["openai", "langfuse", "langsmith", "jsonl", "csv"])
def test_every_other_source_refuses_a_directory(tmp_path: Path, source: str) -> None:
    (tmp_path / "day.jsonl").write_text("{}\n", encoding="utf-8")
    records, _ = read_traces(tmp_path, source=cast("Source", source))

    with pytest.raises(ExportInputError, match="is a directory: name the export file"):
        list(records)
