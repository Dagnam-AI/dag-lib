"""``DirectoryTar``: a directory as an uncompressed tar, sized up front, never held in memory."""

from __future__ import annotations

import errno
import io
import os
from pathlib import Path
import tarfile

import pytest

from dagnam._core.tar_stream import DirectoryTar, FileChangedError, FileStream


def _tar_bytes(stream: DirectoryTar) -> bytes:
    return b"".join(stream)


def _members(blob: bytes) -> dict[str, tarfile.TarInfo]:
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        return {member.name: member for member in tar.getmembers()}


def _tree(root: Path) -> Path:
    (root / "model" / "shards").mkdir(parents=True)
    (root / "model" / "config.json").write_text("{}")
    (root / "model" / "shards" / "a.bin").write_bytes(b"A" * 700)
    (root / "model" / "shards" / "b.bin").write_bytes(b"")
    (root / "weights.pth").write_bytes(b"W" * 512)
    return root


def test_nested_files_round_trip_under_relative_names(tmp_path: Path) -> None:
    stream = DirectoryTar(_tree(tmp_path / "ckpt"))
    blob = _tar_bytes(stream)

    assert len(blob) == len(stream)
    members = _members(blob)
    assert sorted(members) == [
        "model/config.json",
        "model/shards/a.bin",
        "model/shards/b.bin",
        "weights.pth",
    ]
    assert all(member.isreg() for member in members.values())

    out = tmp_path / "out"
    out.mkdir()
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        tar.extractall(out, filter="data")
    assert (out / "model" / "shards" / "a.bin").read_bytes() == b"A" * 700
    assert (out / "model" / "shards" / "b.bin").read_bytes() == b""
    assert (out / "model" / "config.json").read_text() == "{}"
    assert (out / "weights.pth").read_bytes() == b"W" * 512


def test_the_length_matches_the_bytes_for_awkward_names(tmp_path: Path) -> None:
    root = tmp_path / "ckpt"
    long_dir = root / ("d" * 90)
    long_dir.mkdir(parents=True)
    (long_dir / ("f" * 90 + ".bin")).write_bytes(b"x" * 1025)
    (root / "naïve-ünï.txt").write_bytes(b"y")

    stream = DirectoryTar(root)
    blob = _tar_bytes(stream)

    assert len(blob) == len(stream)
    assert sorted(_members(blob)) == sorted(["d" * 90 + "/" + "f" * 90 + ".bin", "naïve-ünï.txt"])


def test_an_empty_directory_is_just_the_end_of_archive_marker(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    stream = DirectoryTar(root)
    assert len(stream) == 1024
    assert _tar_bytes(stream) == bytes(1024)
    assert _members(_tar_bytes(stream)) == {}


def test_it_can_be_iterated_twice(tmp_path: Path) -> None:
    stream = DirectoryTar(_tree(tmp_path / "ckpt"))
    assert _tar_bytes(stream) == _tar_bytes(stream)


def test_symlinks_are_never_followed_out_of_the_directory(tmp_path: Path) -> None:
    secret_dir = tmp_path / "outside"
    secret_dir.mkdir()
    (secret_dir / "secret.txt").write_text("TOP-SECRET")
    root = _tree(tmp_path / "ckpt")
    (root / "to-file").symlink_to(secret_dir / "secret.txt")
    (root / "to-dir").symlink_to(secret_dir, target_is_directory=True)
    (root / "to-inside").symlink_to(root / "weights.pth")
    (root / "dangling").symlink_to(tmp_path / "nowhere")
    (root / "model" / "up").symlink_to("../../outside/secret.txt")

    stream = DirectoryTar(root)
    blob = _tar_bytes(stream)

    assert b"TOP-SECRET" not in blob
    assert sorted(_members(blob)) == [
        "model/config.json",
        "model/shards/a.bin",
        "model/shards/b.bin",
        "weights.pth",
    ]
    assert stream.skipped == 5
    assert len(blob) == len(stream)


def test_special_files_are_skipped(tmp_path: Path) -> None:
    root = _tree(tmp_path / "ckpt")
    os.mkfifo(root / "pipe")
    stream = DirectoryTar(root)
    assert "pipe" not in _members(_tar_bytes(stream))
    assert stream.skipped == 1


def test_every_member_name_is_relative_and_stays_inside(tmp_path: Path) -> None:
    root = _tree(tmp_path / "ckpt")
    (root / "..sneaky").write_bytes(b"1")
    (root / "model" / "...").write_bytes(b"2")
    names = _members(_tar_bytes(DirectoryTar(root)))
    for name in names:
        assert not name.startswith("/")
        assert ".." not in name.split("/")
        assert "." not in name.split("/")
        assert name == name.strip()
    assert {"..sneaky", "model/..."} <= set(names)


def _ckpt(tmp_path: Path, data: bytes) -> tuple[Path, Path]:
    root = tmp_path / "ckpt"
    root.mkdir()
    victim = root / "w.bin"
    victim.write_bytes(data)
    return root, victim


def test_a_file_that_shrinks_after_planning_aborts_the_stream(tmp_path: Path) -> None:
    root, victim = _ckpt(tmp_path, b"S" * 3000)
    stream = DirectoryTar(root)
    victim.write_bytes(b"S" * 10)

    with pytest.raises(FileChangedError):
        _tar_bytes(stream)

    assert isinstance(stream.failure, FileChangedError)


def test_a_file_that_grows_after_planning_aborts_the_stream(tmp_path: Path) -> None:
    root, victim = _ckpt(tmp_path, b"G" * 100)
    stream = DirectoryTar(root)
    victim.write_bytes(b"G" * 5000)

    with pytest.raises(FileChangedError):
        _tar_bytes(stream)


def test_a_file_that_shrinks_mid_read_aborts_the_stream(tmp_path: Path) -> None:
    size = 3 * 1024 * 1024
    root, victim = _ckpt(tmp_path, b"S" * size)
    parts = iter(DirectoryTar(root))
    next(parts)  # the header
    next(parts)  # the first slice of the file
    with victim.open("r+b") as shrinking:
        shrinking.truncate(1024)

    with pytest.raises(FileChangedError, match="shrank"):
        list(parts)


def test_a_file_that_grows_mid_read_aborts_the_stream(tmp_path: Path) -> None:
    size = 3 * 1024 * 1024
    root, victim = _ckpt(tmp_path, b"G" * size)
    parts = iter(DirectoryTar(root))
    next(parts)
    next(parts)
    with victim.open("ab") as growing:
        growing.write(b"more")

    with pytest.raises(FileChangedError, match="grew"):
        list(parts)


def test_a_file_rewritten_in_place_to_the_same_size_aborts_the_stream(tmp_path: Path) -> None:
    root, victim = _ckpt(tmp_path, b"R" * 100)
    stream = DirectoryTar(root)
    parts = iter(stream)
    next(parts)
    next(parts)
    os.utime(victim, ns=(1, 1))

    with pytest.raises(FileChangedError, match="changed"):
        list(parts)


def test_a_file_that_vanishes_after_planning_aborts_the_stream(tmp_path: Path) -> None:
    root, victim = _ckpt(tmp_path, b"x")
    stream = DirectoryTar(root)
    victim.unlink()

    with pytest.raises(FileNotFoundError):
        _tar_bytes(stream)

    assert isinstance(stream.failure, FileNotFoundError)


def test_a_file_swapped_for_a_symlink_after_planning_is_not_read_through_it(
    tmp_path: Path,
) -> None:
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"TOP-SECRET")
    root, victim = _ckpt(tmp_path, b"x" * 10)
    stream = DirectoryTar(root)
    victim.unlink()
    victim.symlink_to(secret)

    with pytest.raises(OSError) as caught:
        _tar_bytes(stream)
    assert caught.value.errno == errno.ELOOP


def test_a_file_swapped_for_a_fifo_after_planning_does_not_block(tmp_path: Path) -> None:
    root, victim = _ckpt(tmp_path, b"x" * 10)
    stream = DirectoryTar(root)
    victim.unlink()
    os.mkfifo(victim)

    with pytest.raises(OSError, match="not a regular file"):
        _tar_bytes(stream)


def test_a_file_added_to_the_directory_after_planning_aborts_the_stream(tmp_path: Path) -> None:
    root, _ = _ckpt(tmp_path, b"x" * 10)
    stream = DirectoryTar(root)
    parts = iter(stream)
    next(parts)
    next(parts)
    (root / "w-00002.bin").write_bytes(b"late shard")

    with pytest.raises(FileChangedError, match="set of files"):
        list(parts)

    assert isinstance(stream.failure, FileChangedError)


def test_a_file_added_in_a_new_subdirectory_after_planning_aborts_the_stream(
    tmp_path: Path,
) -> None:
    root, _ = _ckpt(tmp_path, b"x" * 10)
    stream = DirectoryTar(root)
    (root / "late").mkdir()
    (root / "late" / "shard.bin").write_bytes(b"s")

    with pytest.raises(FileChangedError, match="set of files"):
        _tar_bytes(stream)


def test_a_file_removed_after_it_was_sent_aborts_the_stream(tmp_path: Path) -> None:
    root = tmp_path / "ckpt"
    root.mkdir()
    (root / "a.bin").write_bytes(b"a" * 10)
    (root / "b.bin").write_bytes(b"b" * 10)
    parts = iter(DirectoryTar(root))
    next(parts)  # a.bin's header
    next(parts)  # a.bin's bytes
    next(parts)  # a.bin's padding
    (root / "a.bin").unlink()
    next(parts)  # b.bin's header
    next(parts)
    next(parts)

    with pytest.raises(FileChangedError, match="set of files"):
        next(parts)


def test_a_link_or_a_pipe_added_after_planning_does_not_change_what_is_sent(
    tmp_path: Path,
) -> None:
    root, _ = _ckpt(tmp_path, b"x" * 10)
    stream = DirectoryTar(root)
    (root / "link").symlink_to(tmp_path)
    os.mkfifo(root / "pipe")

    assert len(_tar_bytes(stream)) == len(stream)


def test_a_failure_is_forgotten_by_the_next_pass(tmp_path: Path) -> None:
    root, victim = _ckpt(tmp_path, b"x" * 10)
    stream = DirectoryTar(root)
    victim.unlink()
    with pytest.raises(FileNotFoundError):
        _tar_bytes(stream)
    victim.write_bytes(b"y" * 10)
    assert len(_tar_bytes(stream)) == len(stream)
    assert stream.failure is None


def test_it_reports_when_there_is_nothing_to_send(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "only-a-link").symlink_to(tmp_path)
    assert DirectoryTar(empty).files == 0
    assert DirectoryTar(_tree(tmp_path / "ckpt")).files == 4


# ------------------------------------------------------------------ a single file


def test_a_file_is_streamed_exactly(tmp_path: Path) -> None:
    path = tmp_path / "w.pth"
    path.write_bytes(b"W" * (2 * 1024 * 1024 + 5))
    stream = FileStream(path)
    assert len(stream) == 2 * 1024 * 1024 + 5
    assert b"".join(stream) == path.read_bytes()
    assert b"".join(stream) == path.read_bytes()
    assert stream.failure is None


@pytest.mark.parametrize("make", ["fifo", "directory"])
def test_only_a_regular_file_can_be_streamed(tmp_path: Path, make: str) -> None:
    target = tmp_path / "x"
    if make == "fifo":
        os.mkfifo(target)
    else:
        target.mkdir()
    with pytest.raises(OSError, match="not a regular file"):
        FileStream(target)


def test_a_missing_file_cannot_be_streamed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FileStream(tmp_path / "missing")


def test_a_file_that_changes_size_while_being_sent_aborts_the_stream(tmp_path: Path) -> None:
    path = tmp_path / "w.pth"
    path.write_bytes(b"W" * 100)
    stream = FileStream(path)
    path.write_bytes(b"W" * 101)
    with pytest.raises(FileChangedError):
        b"".join(stream)
    assert isinstance(stream.failure, FileChangedError)


def test_a_file_swapped_for_a_symlink_is_not_read_through_it(tmp_path: Path) -> None:
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"T" * 10)
    path = tmp_path / "w.pth"
    path.write_bytes(b"W" * 10)
    stream = FileStream(path)
    path.unlink()
    path.symlink_to(secret)
    with pytest.raises(OSError) as caught:
        b"".join(stream)
    assert caught.value.errno == errno.ELOOP
