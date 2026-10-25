"""The trace directory: private, ours to tighten only, and measured by the data it holds."""

from __future__ import annotations

import os
from pathlib import Path
import stat
from typing import Any

import pytest

from dagnam_record._private_dir import ensure_private_dir, recorded_bytes

POSIX_ONLY = pytest.mark.skipif(not hasattr(os, "O_DIRECTORY"), reason="needs O_DIRECTORY")


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@POSIX_ONLY
def test_an_empty_directory_that_is_too_open_is_tightened(tmp_path: Path) -> None:
    directory = tmp_path / "traces"
    directory.mkdir()
    directory.chmod(0o755)
    ensure_private_dir(str(directory))
    assert mode(directory) == 0o700


@POSIX_ONLY
@pytest.mark.parametrize("name", [".status.json", "2026-10-12.jsonl"])
def test_a_directory_that_holds_a_recorder_file_is_tightened(tmp_path: Path, name: str) -> None:
    directory = tmp_path / "traces"
    directory.mkdir()
    (directory / name).write_text("{}", encoding="utf-8")
    directory.chmod(0o750)
    ensure_private_dir(str(directory))
    assert mode(directory) == 0o700


@POSIX_ONLY
@pytest.mark.parametrize("opened", [0o750, 0o755, 0o770, 0o701], ids=oct)
def test_a_shared_directory_with_other_files_is_refused_and_left_as_it_was(
    tmp_path: Path, opened: int
) -> None:
    directory = tmp_path / "logs"
    directory.mkdir()
    (directory / "app.log").write_text("someone else's file", encoding="utf-8")
    directory.chmod(opened)
    with pytest.raises(PermissionError, match="shared"):
        ensure_private_dir(str(directory))
    assert mode(directory) == opened  # a log shipper's group can still read its logs


@POSIX_ONLY
def test_a_private_directory_with_other_files_is_fine_as_it_is(tmp_path: Path) -> None:
    directory = tmp_path / "mine"
    directory.mkdir(mode=0o700)
    (directory / "notes.txt").write_text("x", encoding="utf-8")
    ensure_private_dir(str(directory))
    assert mode(directory) == 0o700


def test_the_bytes_recorded_are_those_of_the_dated_data_files_only(tmp_path: Path) -> None:
    (tmp_path / "2026-10-11.jsonl").write_bytes(b"a" * 100)
    (tmp_path / "2026-10-12.jsonl").write_bytes(b"b" * 50)
    (tmp_path / ".status.json").write_bytes(b"c" * 999)
    (tmp_path / "notes.jsonl").write_bytes(b"d" * 999)
    (tmp_path / "2026-10-13.jsonl").mkdir()
    assert recorded_bytes(str(tmp_path)) == 150


def test_a_directory_that_cannot_be_listed_holds_no_bytes(tmp_path: Path) -> None:
    assert recorded_bytes(str(tmp_path / "missing")) == 0


@POSIX_ONLY
def test_a_dated_name_that_is_a_symlink_is_not_counted(tmp_path: Path) -> None:
    target: Any = tmp_path / "elsewhere"
    target.write_bytes(b"x" * 500)
    (tmp_path / "traces").mkdir()
    (tmp_path / "traces" / "2026-10-12.jsonl").symlink_to(target)
    assert recorded_bytes(str(tmp_path / "traces")) == 0
