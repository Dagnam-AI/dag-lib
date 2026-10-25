"""The writer and unsafe paths: FIFOs, symlinks, foreign owners and file permissions."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import sys
import threading
from typing import Any, Callable, List

import pytest

POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes and symlinks")
FIFO_ONLY = pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs os.mkfifo")


def lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def bounded(call: Callable[[], Any]) -> Any:
    """Run ``call`` in a thread and fail the test if it has not returned within 5 seconds."""
    result: List[Any] = []
    worker = threading.Thread(target=lambda: result.append(call()), daemon=True)
    worker.start()
    worker.join(5)
    assert not worker.is_alive(), "the call blocked"
    return result[0] if result else None


@POSIX_ONLY
def test_the_directory_is_0700_and_the_file_0600(make: Any, tmp_path: Path) -> None:
    writer = make(tmp_path / "a" / "b" / "traces")
    writer.submit("{}")
    writer.drain()
    directory = tmp_path / "a" / "b" / "traces"
    assert mode(directory) == 0o700
    assert mode(directory / "2026-10-12.jsonl") == 0o600


@POSIX_ONLY
def test_an_existing_directory_we_own_is_tightened_to_0700(make: Any, tmp_path: Path) -> None:
    directory = tmp_path / "traces"
    directory.mkdir()
    os.chmod(directory, int("750", 8))  # group-readable: looser than 0700
    writer = make()
    writer.submit("{}")
    writer.drain()
    assert mode(directory) == 0o700
    assert writer.counts["written"] == 1


@POSIX_ONLY
def test_an_existing_private_directory_is_left_alone(make: Any, tmp_path: Path) -> None:
    directory = tmp_path / "traces"
    directory.mkdir()
    os.chmod(directory, 0o700)
    writer = make()
    writer.submit("{}")
    writer.drain()
    assert mode(directory) == 0o700
    assert writer.counts["written"] == 1


@POSIX_ONLY
def test_a_file_that_is_too_open_is_made_private(make: Any, tmp_path: Path) -> None:
    directory = tmp_path / "traces"
    directory.mkdir(mode=0o700)
    existing = directory / "2026-10-12.jsonl"
    existing.write_text("old\n", encoding="utf-8")
    os.chmod(existing, 0o644)
    writer = make()
    writer.submit("new")
    writer.drain()
    assert mode(existing) == 0o600
    assert lines(existing) == ["old", "new"]


@POSIX_ONLY
def test_a_directory_owned_by_someone_else_is_refused(
    make: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "traces").mkdir()
    monkeypatch.setattr("dagnam_record._private_dir._OWNER", os.getuid() + 1)
    writer = make()
    writer.submit("secret")
    writer.drain()
    assert writer.counts["dropped_error"] == 2  # one when the writer was made, one for the line
    assert list((tmp_path / "traces").iterdir()) == []


@POSIX_ONLY
def test_a_symlinked_directory_is_refused(make: Any, tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    target.mkdir()
    (tmp_path / "traces").symlink_to(target, target_is_directory=True)
    writer = make()
    writer.submit("secret")
    writer.drain()
    assert writer.counts["dropped_error"] == 2
    assert list(target.iterdir()) == []


def test_a_directory_path_that_is_a_file_is_refused(make: Any, tmp_path: Path) -> None:
    (tmp_path / "traces").write_text("not a directory", encoding="utf-8")
    writer = make()
    writer.submit("x")
    writer.drain()
    assert writer.counts["dropped_error"] == 2


@POSIX_ONLY
def test_a_symlinked_data_file_is_not_followed(make: Any, tmp_path: Path) -> None:
    directory = tmp_path / "traces"
    directory.mkdir(mode=0o700)
    victim = tmp_path / "victim.txt"
    victim.write_text("keep\n", encoding="utf-8")
    (directory / "2026-10-12.jsonl").symlink_to(victim)
    writer = make()
    writer.submit("planted")
    writer.drain()
    assert writer.counts["dropped_error"] == 1
    assert victim.read_text(encoding="utf-8") == "keep\n"


@POSIX_ONLY
def test_a_directory_is_tightened_through_its_descriptor_not_by_path(
    make: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "traces").mkdir()
    os.chmod(tmp_path / "traces", int("750", 8))

    def no_path_chmod(path: str, mode: int) -> None:
        raise AssertionError("a path can be swapped for a symlink between the check and the chmod")

    monkeypatch.setattr(os, "chmod", no_path_chmod)
    make()
    assert mode(tmp_path / "traces") == 0o700


def test_without_o_directory_a_real_directory_still_works(
    make: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("dagnam_record._private_dir.DIRECTORY", 0)
    (tmp_path / "traces").mkdir()
    writer = make()
    writer.submit("a")
    writer.drain()
    assert writer.counts["written"] == 1
    assert writer.counts["dropped_error"] == 0


@POSIX_ONLY
def test_without_o_directory_a_symlinked_directory_is_still_refused(
    make: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("dagnam_record._private_dir.DIRECTORY", 0)
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "traces").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    writer = make()
    assert writer.counts["dropped_error"] == 1
    assert list((tmp_path / "elsewhere").iterdir()) == []


def test_without_o_directory_a_file_is_still_refused(
    make: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("dagnam_record._private_dir.DIRECTORY", 0)
    (tmp_path / "traces").write_text("a file", encoding="utf-8")
    assert make().counts["dropped_error"] == 1


@FIFO_ONLY
def test_a_fifo_where_the_status_file_belongs_does_not_hang_the_writer(
    make: Any, tmp_path: Path
) -> None:
    directory = tmp_path / "traces"
    directory.mkdir(mode=0o700)
    os.mkfifo(directory / ".status.json")
    writer = bounded(make)  # opening a FIFO for reading would wait for a writer forever
    writer.submit("x")
    bounded(writer.drain)
    bounded(writer.close)
    assert writer.counts["written"] == 1
    assert writer.counts["dropped_error"] == 0


@POSIX_ONLY
def test_a_symlinked_status_file_is_not_followed(make: Any, tmp_path: Path) -> None:
    directory = tmp_path / "traces"
    directory.mkdir(mode=0o700)
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text('{"started": 1.0, "counts": {"written": 7}}', encoding="utf-8")
    (directory / ".status.json").symlink_to(elsewhere)
    writer = make()
    assert writer.counts["written"] == 0
    writer.submit("x")
    writer.drain()
    assert not writer.expired  # the planted start time (long ago) was not believed


@FIFO_ONLY
def test_a_fifo_with_no_reader_where_the_data_file_belongs_is_an_error_not_a_stall(
    make: Any, tmp_path: Path
) -> None:
    directory = tmp_path / "traces"
    directory.mkdir(mode=0o700)
    os.mkfifo(directory / "2026-10-12.jsonl")
    writer = make()
    writer.submit("x")
    bounded(writer.drain)
    bounded(writer.close)
    assert writer.counts["dropped_error"] == 1
    assert writer.counts["written"] == 0


@FIFO_ONLY
def test_a_fifo_with_a_reader_where_the_data_file_belongs_is_refused_too(
    make: Any, tmp_path: Path
) -> None:
    directory = tmp_path / "traces"
    directory.mkdir(mode=0o700)
    fifo = directory / "2026-10-12.jsonl"
    os.mkfifo(fifo)
    reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    try:
        writer = make(start=True)
        writer.submit("x")
        bounded(writer.close)  # the thread must stop even though the open succeeded
        assert writer.counts["dropped_error"] == 1
        assert os.read(reader, 100) == b""  # nothing was written into the pipe
    finally:
        os.close(reader)


@FIFO_ONLY
def test_a_status_pipe_with_data_in_it_is_not_believed(make: Any, tmp_path: Path) -> None:
    directory = tmp_path / "traces"
    directory.mkdir(mode=0o700)
    fifo = directory / ".status.json"
    os.mkfifo(fifo)
    reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    feeder = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
    try:
        os.write(feeder, b'{"started": 1.0, "counts": {"written": 7}}')
        writer = bounded(make)
        assert writer.counts["written"] == 0
        writer.submit("x")
        bounded(writer.drain)
        assert not writer.expired  # the start time in the pipe (long ago) was not used
    finally:
        os.close(feeder)
        os.close(reader)
