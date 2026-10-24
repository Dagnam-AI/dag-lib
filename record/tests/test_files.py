"""Pruning, summarizing and purging the recorder's files: only its own, never through a symlink."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from _api_support import envelope, write_data
import pytest

from dagnam_record import _files
from dagnam_record._files import NotRecorderDirectoryError
from dagnam_record._writer import COUNT_KEYS

POSIX_ONLY = pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs POSIX files")
NOW = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc).timestamp()


def at(day: str, clock: str = "09:30:00") -> float:
    return datetime.fromisoformat(f"{day}T{clock}+00:00").timestamp()


def names(directory: Path) -> List[str]:
    return sorted(path.name for path in directory.iterdir())


def status_file(directory: Path, **fields: Any) -> None:
    data: Dict[str, Any] = {
        "directory": str(directory),
        "started": at("2026-10-12"),
        "expired": False,
    }
    data["counts"] = dict.fromkeys(COUNT_KEYS, 0)
    data.update(fields)
    (directory / ".status.json").write_text(json.dumps(data), encoding="utf-8")


# --- prune -----------------------------------------------------------------------------------


def test_prune_deletes_only_dated_data_files_older_than_the_window(tmp_path: Path) -> None:
    for day in ("2026-10-08", "2026-09-30", "2026-10-09", "2026-10-10", "2026-10-23", "2026-11-30"):
        write_data(tmp_path, f"{day}.jsonl", ["x"])
    removed = _files.prune(str(tmp_path), 14, NOW)
    assert sorted(os.path.basename(path) for path in removed) == [
        "2026-09-30.jsonl",
        "2026-10-08.jsonl",
    ]
    # The 9th is exactly 14 days back: kept. Future dates are kept too.
    assert names(tmp_path) == [
        "2026-10-09.jsonl",
        "2026-10-10.jsonl",
        "2026-10-23.jsonl",
        "2026-11-30.jsonl",
    ]


@pytest.mark.parametrize(("window", "kept"), [(1, 2), (2, 3), (7, 8), (8, 9), (30, 9)])
def test_the_boundary_moves_with_the_window(window: int, kept: int, tmp_path: Path) -> None:
    for back in range(0, 9):
        day = datetime.fromtimestamp(NOW - back * 86400, timezone.utc).strftime("%Y-%m-%d")
        write_data(tmp_path, f"{day}.jsonl", ["x"])
    _files.prune(str(tmp_path), window, NOW)
    assert len(names(tmp_path)) == kept  # today and ``window`` days back stay


def test_the_cutoff_is_the_utc_date_not_the_time_of_day(tmp_path: Path) -> None:
    write_data(tmp_path, "2026-10-09.jsonl", ["x"])
    midnight = at("2026-10-23", "00:00:00")
    last_second = at("2026-10-22", "23:59:59")
    assert _files.prune(str(tmp_path), 14, last_second) == []
    assert _files.prune(str(tmp_path), 14, midnight) == []
    assert len(_files.prune(str(tmp_path), 14, at("2026-10-24", "00:00:00"))) == 1


@POSIX_ONLY
def test_prune_never_touches_anything_that_is_not_a_dated_data_file(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    target = write_data(outside, "keep.txt", ["precious"])
    box = tmp_path / "box"
    box.mkdir()
    old = "2020-01-01"
    status_file(box)
    write_data(box, "notes.jsonl", ["x"])
    write_data(box, f"{old}.txt", ["x"])
    write_data(box, f"{old}.jsonl.bak", ["x"])
    write_data(box, f"{old}.jsonl.gz", ["x"])
    write_data(box, "2020-13-45.jsonl", ["x"])
    write_data(box, "abcd-ef-gh.jsonl", ["x"])
    (box / "2020-02-02.jsonl").mkdir()
    (box / "2020-03-03.jsonl").symlink_to(target)
    os.mkfifo(box / "2020-04-04.jsonl")
    before = names(box)
    assert _files.prune(str(box), 14, NOW) == []
    assert names(box) == before
    assert target.read_text(encoding="utf-8") == "precious\n"


def test_prune_of_a_missing_directory_or_a_symlink_to_one_does_nothing(tmp_path: Path) -> None:
    assert _files.prune(str(tmp_path / "absent"), 14, NOW) == []
    real = tmp_path / "real"
    write_data(real, "2020-01-01.jsonl", ["x"])
    link = tmp_path / "link"
    link.symlink_to(real)
    assert _files.prune(str(link), 14, NOW) == []
    assert names(real) == ["2020-01-01.jsonl"]


def test_prune_survives_a_file_it_cannot_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_data(tmp_path, "2020-01-01.jsonl", ["x"])
    write_data(tmp_path, "2020-01-02.jsonl", ["x"])

    def refuse(path: str) -> None:
        raise PermissionError(path)

    monkeypatch.setattr(os, "unlink", refuse)
    assert _files.prune(str(tmp_path), 14, NOW) == []
    monkeypatch.undo()
    assert len(names(tmp_path)) == 2


# --- summarize -------------------------------------------------------------------------------


def test_summarize_counts_files_lines_and_timestamps(tmp_path: Path) -> None:
    write_data(
        tmp_path,
        "2026-10-12.jsonl",
        [
            envelope("a", at("2026-10-12", "09:30:00")),
            "",
            envelope("b", at("2026-10-12", "23:00:00")),
        ],
    )
    write_data(tmp_path, "2026-10-13.jsonl", [envelope("c", at("2026-10-13", "01:02:03"))])
    write_data(tmp_path, "2026-10-14.jsonl", [])
    status_file(
        tmp_path,
        counts={
            "written": 3,
            "dropped_full": 4,
            "dropped_large": 5,
            "dropped_error": 6,
            "dropped_expired": 7,
            "dropped_partial": 8,
        },
        expired=True,
    )
    summary = _files.summarize(str(tmp_path))
    assert (summary.files, summary.lines) == (3, 3)
    assert summary.first == "2026-10-12T09:30:00.000Z"
    assert summary.last == "2026-10-13T01:02:03.000Z"
    assert summary.counts == {
        "written": 3,
        "dropped_full": 4,
        "dropped_large": 5,
        "dropped_error": 6,
        "dropped_expired": 7,
        "dropped_partial": 8,
    }
    assert summary.counts is not None
    assert tuple(summary.counts) == COUNT_KEYS
    assert summary.started == "2026-10-12T09:30:00Z"
    assert summary.expired is True


def test_summarize_prints_only_well_formed_timestamps(tmp_path: Path) -> None:
    write_data(tmp_path, "2026-10-12.jsonl", ['{"ts": "TOP SECRET TEXT"}', "not json", "[1]"])
    write_data(tmp_path, "2026-10-13.jsonl", [envelope("x", at("2026-10-13"))])
    summary = _files.summarize(str(tmp_path))
    assert summary.lines == 4
    assert (
        summary.first == "2026-10-13T09:30:00.000Z"
    )  # the first file's first line has no real stamp
    assert summary.last == "2026-10-13T09:30:00.000Z"
    write_data(tmp_path, "2026-10-14.jsonl", ["not json"])
    assert _files.summarize(str(tmp_path)).last == "2026-10-13T09:30:00.000Z"


def test_summarize_without_a_status_file_or_with_a_bad_one(tmp_path: Path) -> None:
    write_data(tmp_path, "2026-10-12.jsonl", [envelope("a", at("2026-10-12"))])
    plain = _files.summarize(str(tmp_path))
    assert (plain.counts, plain.started, plain.expired) == (None, None, None)
    (tmp_path / ".status.json").write_text("[]", encoding="utf-8")
    assert _files.summarize(str(tmp_path)).counts is None
    status_file(
        tmp_path,
        counts={"written": -1, "dropped_full": "x", "dropped_large": True},
        started="x",
        expired="no",
    )
    bad = _files.summarize(str(tmp_path))
    assert bad.counts == dict.fromkeys(COUNT_KEYS, 0)
    assert (bad.started, bad.expired) == (None, None)


def test_summarize_a_status_file_alone_is_a_recorder_directory(tmp_path: Path) -> None:
    status_file(tmp_path)
    summary = _files.summarize(str(tmp_path))
    assert (summary.files, summary.lines, summary.first, summary.last) == (0, 0, None, None)


def test_summarize_refuses_what_is_not_a_recorder_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _files.summarize(str(tmp_path / "absent"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(NotRecorderDirectoryError):
        _files.summarize(str(empty))
    write_data(tmp_path / "other", "notes.jsonl", ["x"])
    with pytest.raises(NotRecorderDirectoryError):
        _files.summarize(str(tmp_path / "other"))
    real = tmp_path / "real"
    write_data(real, "2026-10-12.jsonl", [envelope("a", at("2026-10-12"))])
    (tmp_path / "link").symlink_to(real)
    with pytest.raises(NotADirectoryError):
        _files.summarize(str(tmp_path / "link"))
    afile = tmp_path / "afile"
    afile.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        _files.summarize(str(afile))


@POSIX_ONLY
def test_summarize_never_follows_a_symlink_or_waits_on_a_fifo(tmp_path: Path) -> None:
    outside = write_data(tmp_path / "outside", "secret.txt", [envelope("leak", at("2026-10-12"))])
    box = tmp_path / "box"
    write_data(box, "2026-10-12.jsonl", [envelope("a", at("2026-10-12"))])
    (box / "2026-10-13.jsonl").symlink_to(outside)
    os.mkfifo(box / "2026-10-14.jsonl")
    (box / ".status.json").symlink_to(outside)
    summary = _files.summarize(str(box))
    assert (summary.files, summary.lines) == (1, 1)
    assert summary.counts is None  # the symlinked status file is not read


@POSIX_ONLY
def test_a_data_file_swapped_for_a_fifo_after_the_listing_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    os.mkfifo(tmp_path / "2026-10-12.jsonl")
    write_data(tmp_path, "2026-10-13.jsonl", [envelope("a", at("2026-10-13"))])
    listed = _files.Found(
        [str(tmp_path / "2026-10-12.jsonl"), str(tmp_path / "2026-10-13.jsonl")], None
    )
    monkeypatch.setattr(_files, "find", lambda directory: listed)
    summary = _files.summarize(str(tmp_path))
    assert (summary.files, summary.lines) == (2, 1)


def test_a_listed_path_that_is_now_a_symlink_or_a_directory_is_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = write_data(tmp_path / "outside", "secret.txt", [envelope("leak", at("2026-10-12"))])
    (tmp_path / "box").mkdir()
    (tmp_path / "box" / "2026-10-12.jsonl").symlink_to(outside)
    (tmp_path / "box" / "2026-10-13.jsonl").mkdir()
    listed = _files.Found(
        [str(tmp_path / "box" / "2026-10-12.jsonl"), str(tmp_path / "box" / "2026-10-13.jsonl")],
        None,
    )
    monkeypatch.setattr(_files, "find", lambda directory: listed)
    summary = _files.summarize(str(tmp_path / "box"))
    assert (summary.files, summary.lines, summary.first) == (2, 0, None)


def test_a_file_that_vanishes_after_the_listing_counts_as_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listed = _files.Found([str(tmp_path / "2026-10-12.jsonl")], None)
    monkeypatch.setattr(_files, "find", lambda directory: listed)
    assert _files.summarize(str(tmp_path)).lines == 0


# --- purge -----------------------------------------------------------------------------------


def test_purge_deletes_exactly_the_recorder_files(tmp_path: Path) -> None:
    write_data(tmp_path, "2026-10-12.jsonl", ["x"])
    write_data(tmp_path, "2026-10-13.jsonl", ["x"])
    status_file(tmp_path)
    for keep in (
        "notes.jsonl",
        "2026-10-12.txt",
        "2026-10-12.jsonl.gz",
        "data.json",
        ".status.json.1.tmp",
    ):
        write_data(tmp_path, keep, ["mine"])
    (tmp_path / "2026-10-14.jsonl").mkdir()
    result = _files.purge(str(tmp_path))
    assert result.deleted == ["2026-10-12.jsonl", "2026-10-13.jsonl", ".status.json"]
    assert result.failed == []
    assert names(tmp_path) == [
        ".status.json.1.tmp",
        "2026-10-12.jsonl.gz",
        "2026-10-12.txt",
        "2026-10-14.jsonl",
        "data.json",
        "notes.jsonl",
    ]


def test_purge_refuses_a_directory_without_a_recorder_file(tmp_path: Path) -> None:
    write_data(tmp_path, "notes.jsonl", ["mine"])
    write_data(tmp_path, "report.txt", ["mine"])
    with pytest.raises(NotRecorderDirectoryError):
        _files.purge(str(tmp_path))
    assert names(tmp_path) == ["notes.jsonl", "report.txt"]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(NotRecorderDirectoryError):
        _files.purge(str(empty))


def test_purge_of_a_missing_directory_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _files.purge(str(tmp_path / "absent"))


@POSIX_ONLY
def test_purge_never_follows_a_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    write_data(real, "2026-10-12.jsonl", ["precious"])
    (tmp_path / "link").symlink_to(real)
    with pytest.raises(NotADirectoryError):
        _files.purge(str(tmp_path / "link"))
    assert names(real) == ["2026-10-12.jsonl"]
    box = tmp_path / "box"
    mine = write_data(tmp_path / "mine", "diary.txt", ["precious"])
    write_data(box, "2026-10-13.jsonl", ["x"])
    (box / "2026-10-14.jsonl").symlink_to(mine)
    (box / ".status.json").symlink_to(mine)
    os.mkfifo(box / "2026-10-15.jsonl")
    assert _files.purge(str(box)).deleted == ["2026-10-13.jsonl"]
    assert mine.read_text(encoding="utf-8") == "precious\n"
    assert names(box) == [".status.json", "2026-10-14.jsonl", "2026-10-15.jsonl"]


def test_purge_reports_files_it_could_not_delete_and_ignores_ones_already_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("2026-10-11.jsonl", "2026-10-12.jsonl", "2026-10-13.jsonl"):
        write_data(tmp_path, name, ["x"])
    real_unlink = os.unlink

    def flaky(path: str) -> None:
        if path.endswith("10-11.jsonl"):
            raise PermissionError(path)
        if path.endswith("10-12.jsonl"):
            real_unlink(path)
            raise FileNotFoundError(path)
        real_unlink(path)

    monkeypatch.setattr(os, "unlink", flaky)
    result = _files.purge(str(tmp_path))
    assert (result.deleted, result.failed) == (["2026-10-13.jsonl"], ["2026-10-11.jsonl"])
