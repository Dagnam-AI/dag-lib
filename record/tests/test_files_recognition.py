"""What counts as the recorder's file, and the bounded reading of a data file."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Optional

from _api_support import envelope, write_data
import pytest

from dagnam_record import _files
from dagnam_record._files import NotRecorderDirectoryError
from dagnam_record._writer import read_status

NOW = 1_791_797_400.0
LINE = envelope("x", NOW)
# 2026-10-12 in Arabic-Indic digits: ``\\d`` matches them, ``[0-9]`` does not.
ARABIC_DIGITS = "".join(
    chr(0x660 + int(digit)) if digit.isdigit() else digit for digit in "2026-10-12"
)
FOREIGN = "just somebody else's notes"


def names(directory: Path) -> List[str]:
    return sorted(path.name for path in directory.iterdir())


def status(directory: Path, text: Optional[str] = None) -> None:
    body = text if text is not None else json.dumps({"counts": {"written": 1}, "started": NOW})
    (directory / ".status.json").write_text(body, encoding="utf-8")


def test_dated_names_with_foreign_content_and_no_status_are_not_the_recorders(
    tmp_path: Path,
) -> None:
    write_data(tmp_path, "2026-10-12.jsonl", [FOREIGN])
    write_data(tmp_path, "2026-10-13.jsonl", ['{"v": 1, "trace_id": "spaced"}'])
    assert _files.find(str(tmp_path)) == _files.Found([], None)
    with pytest.raises(NotRecorderDirectoryError):
        _files.purge(str(tmp_path))
    with pytest.raises(NotRecorderDirectoryError):
        _files.summarize(str(tmp_path))
    assert _files.prune(str(tmp_path), 14, NOW + 99 * 86400) == []
    assert names(tmp_path) == ["2026-10-12.jsonl", "2026-10-13.jsonl"]


@pytest.mark.parametrize("garbage", ["", "not json", "[]", "{}", "5", "null", '{"a": 1'])
def test_a_status_file_that_does_not_parse_as_an_object_proves_nothing(
    garbage: str, tmp_path: Path
) -> None:
    write_data(tmp_path, "2026-10-12.jsonl", [FOREIGN])
    status(tmp_path, garbage)
    assert _files.find(str(tmp_path)) == _files.Found([], None)
    with pytest.raises(NotRecorderDirectoryError):
        _files.purge(str(tmp_path))
    assert names(tmp_path) == [".status.json", "2026-10-12.jsonl"]


def test_a_status_file_does_not_vouch_for_a_dated_file_that_is_not_an_envelope(
    tmp_path: Path,
) -> None:
    write_data(tmp_path, "2026-10-12.jsonl", ["my log"])
    write_data(tmp_path, "2026-10-13.jsonl", [LINE])
    (tmp_path / "2026-10-14.jsonl").write_text("", encoding="utf-8")
    status(tmp_path)
    found = _files.find(str(tmp_path))
    assert [os.path.basename(path) for path in found.data] == ["2026-10-13.jsonl"]
    assert found.status == str(tmp_path / ".status.json")
    assert _files.purge(str(tmp_path)).deleted == ["2026-10-13.jsonl", ".status.json"]
    assert names(tmp_path) == ["2026-10-12.jsonl", "2026-10-14.jsonl"]


@pytest.mark.parametrize(
    "body", ['{"status": "ok"}', '{"counts": []}', '{"counts": "x"}', '{"counts": null}', "[]"]
)
def test_a_status_file_without_counts_is_not_the_recorders_and_vouches_for_nothing(
    body: str, tmp_path: Path
) -> None:
    write_data(tmp_path, "2026-02-02.jsonl", ["my log"])
    status(tmp_path, body)
    assert _files.find(str(tmp_path)) == _files.Found([], None)
    with pytest.raises(NotRecorderDirectoryError):
        _files.purge(str(tmp_path))
    assert names(tmp_path) == [".status.json", "2026-02-02.jsonl"]


def test_a_real_status_file_alone_is_a_recorder_directory_and_leaves_foreign_files(
    tmp_path: Path,
) -> None:
    write_data(tmp_path, "2026-02-02.jsonl", ["my log"])
    status(tmp_path)
    assert _files.purge(str(tmp_path)).deleted == [".status.json"]
    assert names(tmp_path) == ["2026-02-02.jsonl"]


def test_prune_never_deletes_a_dated_file_that_does_not_start_like_an_envelope(
    tmp_path: Path,
) -> None:
    status(tmp_path)
    write_data(tmp_path, "2020-01-01.jsonl", ["my log"])
    write_data(tmp_path, "2020-01-02.jsonl", [LINE])
    (tmp_path / "2020-01-03.jsonl").write_text("", encoding="utf-8")
    removed = _files.prune(str(tmp_path), 14, NOW)
    assert [os.path.basename(path) for path in removed] == ["2020-01-02.jsonl"]
    assert names(tmp_path) == [".status.json", "2020-01-01.jsonl", "2020-01-03.jsonl"]


def test_a_status_file_over_a_megabyte_is_treated_as_unreadable(tmp_path: Path) -> None:
    write_data(tmp_path, "2026-10-12.jsonl", [LINE])
    status(tmp_path, '{"counts": {"written": 1}}' + " " * 1_000_001)
    assert read_status(str(tmp_path)) == {}
    assert _files.summarize(str(tmp_path)).counts is None
    status(tmp_path, '{"counts": {"written": 1}}' + " " * 100)
    assert read_status(str(tmp_path)) == {"counts": {"written": 1}}


def test_without_a_status_each_file_must_start_like_an_envelope_line(tmp_path: Path) -> None:
    write_data(tmp_path, "2026-10-11.jsonl", [FOREIGN])
    write_data(tmp_path, "2026-10-12.jsonl", [LINE])
    write_data(tmp_path, "2026-10-13.jsonl", ['{"v":2,"trace_id":"x"}'])
    write_data(tmp_path, "2026-10-14.jsonl", ["", LINE])
    write_data(tmp_path, "2026-10-15.jsonl", ['{"v":1,"trace_i'])
    (tmp_path / "2026-10-16.jsonl").write_text("", encoding="utf-8")
    status(tmp_path, "not json")  # an unreadable status file is deleted with the rest
    found = _files.find(str(tmp_path))
    assert [os.path.basename(path) for path in found.data] == ["2026-10-12.jsonl"]
    assert found.status == str(tmp_path / ".status.json")
    result = _files.purge(str(tmp_path))
    assert result.deleted == ["2026-10-12.jsonl", ".status.json"]
    assert names(tmp_path) == [
        "2026-10-11.jsonl",
        "2026-10-13.jsonl",
        "2026-10-14.jsonl",
        "2026-10-15.jsonl",
        "2026-10-16.jsonl",
    ]


def test_a_dated_name_must_be_a_real_date_written_in_ascii_digits(tmp_path: Path) -> None:
    status(tmp_path)
    for name in (
        "2026-13-45.jsonl",
        "2026-02-30.jsonl",
        "0000-00-00.jsonl",
        ARABIC_DIGITS + ".jsonl",
        "2026-1-12.jsonl",
    ):
        write_data(tmp_path, name, [LINE])
    write_data(tmp_path, "2026-10-12.jsonl", [LINE])
    found = _files.find(str(tmp_path))
    assert [os.path.basename(path) for path in found.data] == ["2026-10-12.jsonl"]


def test_a_path_swapped_for_a_symlink_or_a_directory_does_not_vouch_for_itself(
    tmp_path: Path,
) -> None:
    real = write_data(tmp_path / "elsewhere", "real.jsonl", [LINE])
    (tmp_path / "link.jsonl").symlink_to(real)
    (tmp_path / "dir.jsonl").mkdir()
    assert _files.starts_like_a_line(str(real)) is True
    assert _files.starts_like_a_line(str(tmp_path / "link.jsonl")) is False
    assert _files.starts_like_a_line(str(tmp_path / "dir.jsonl")) is False
    assert _files.starts_like_a_line(str(tmp_path / "absent.jsonl")) is False


def test_a_dry_run_lists_what_would_go_and_deletes_nothing(tmp_path: Path) -> None:
    write_data(tmp_path, "2026-10-12.jsonl", [LINE])
    status(tmp_path)
    result = _files.purge(str(tmp_path), dry_run=True)
    assert result == _files.Purged(["2026-10-12.jsonl", ".status.json"], [])
    assert names(tmp_path) == [".status.json", "2026-10-12.jsonl"]
    (tmp_path / "empty").mkdir()
    with pytest.raises(NotRecorderDirectoryError):
        _files.purge(str(tmp_path / "empty"), dry_run=True)


# --- bounded reading ---------------------------------------------------------------------------


class Spy:
    """A file handle that records how much each ``readline`` was allowed to read."""

    def __init__(self, handle: Any, sizes: List[int]) -> None:
        self.handle = handle
        self.sizes = sizes

    def __enter__(self) -> Spy:
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the real handle."""
        self.handle.close()

    def readline(self, size: int = -1) -> bytes:
        """Note the size, then read."""
        self.sizes.append(size)
        return bytes(self.handle.readline(size))


def test_a_huge_line_is_read_in_pieces_never_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status(tmp_path)
    huge = "z" * 200_000
    write_data(tmp_path, "2026-10-12.jsonl", [LINE, huge, huge])
    sizes: List[int] = []
    real = os.fdopen
    monkeypatch.setattr(_files, "_CHUNK", 4096)

    def spying(fd: int, mode: str = "r", **kwargs: Any) -> Any:
        return Spy(real(fd, mode), sizes) if mode == "rb" else real(fd, mode, **kwargs)

    monkeypatch.setattr(os, "fdopen", spying)
    summary = _files.summarize(str(tmp_path))
    assert summary.lines == 3
    assert sizes
    assert set(sizes) == {4096}
    assert summary.first == summary.last == "2026-10-12T09:30:00.000Z"


def test_lines_across_chunk_boundaries_are_counted_once_and_their_stamps_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status(tmp_path)
    monkeypatch.setattr(_files, "_CHUNK", 7)
    write_data(
        tmp_path,
        "2026-10-12.jsonl",
        [envelope("a", NOW), "", "   ", envelope("b", NOW + 60), envelope("c", NOW + 120)],
    )
    summary = _files.summarize(str(tmp_path))
    assert summary.lines == 3
    assert summary.first == "2026-10-12T09:30:00.000Z"
    assert summary.last == "2026-10-12T09:32:00.000Z"


def test_the_first_stamp_is_the_first_one_that_parses_and_the_last_is_the_last(
    tmp_path: Path,
) -> None:
    status(tmp_path)
    write_data(
        tmp_path,
        "2026-10-12.jsonl",
        ['{"v":1,"trace_id":"a","ts":"TOP SECRET"}', "garbage", envelope("b", NOW), "garbage"],
    )
    write_data(tmp_path, "2026-10-13.jsonl", [envelope("c", NOW + 86_400), "garbage"])
    summary = _files.summarize(str(tmp_path))
    assert summary.lines == 6
    assert summary.first == "2026-10-12T09:30:00.000Z"
    assert summary.last == "2026-10-13T09:30:00.000Z"


def test_a_last_line_without_a_newline_is_counted_and_stamped(tmp_path: Path) -> None:
    status(tmp_path)
    (tmp_path / "2026-10-12.jsonl").write_text(LINE, encoding="utf-8")
    summary = _files.summarize(str(tmp_path))
    assert (summary.lines, summary.first, summary.last) == (
        1,
        "2026-10-12T09:30:00.000Z",
        "2026-10-12T09:30:00.000Z",
    )


def test_only_a_stamp_in_the_envelope_position_is_ever_reported(tmp_path: Path) -> None:
    status(tmp_path)
    forged = '{"v":1,"trace_id":"a","note":"x","ts":"2026-01-01T00:00:00Z"}'
    write_data(tmp_path, "2026-10-12.jsonl", [forged])
    summary = _files.summarize(str(tmp_path))
    assert (summary.lines, summary.first, summary.last) == (1, None, None)
