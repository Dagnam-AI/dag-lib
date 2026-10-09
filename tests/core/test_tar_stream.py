"""``DirectoryTar``: a directory as an uncompressed tar, sized up front, never held in memory."""

from __future__ import annotations

import io
import os
from pathlib import Path
import tarfile

import pytest

from dagnam._core.tar_stream import DirectoryTar


def _tar_bytes(stream: DirectoryTar) -> bytes:
    return b"".join(stream)


def _members(blob: bytes) -> dict[str, tarfile.TarInfo]:
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        return {member.name: member for member in tar.getmembers()}


def _file(blob: bytes, name: str) -> bytes:
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        member = tar.extractfile(name)
        assert member is not None
        return member.read()


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


def test_a_file_that_shrinks_after_planning_is_zero_padded_to_the_declared_size(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ckpt"
    root.mkdir()
    victim = root / "w.bin"
    victim.write_bytes(b"S" * 3000)
    stream = DirectoryTar(root)
    victim.write_bytes(b"S" * 10)

    blob = _tar_bytes(stream)

    assert len(blob) == len(stream)
    assert _file(blob, "w.bin") == b"S" * 10 + bytes(2990)


def test_a_file_that_grows_after_planning_is_cut_at_the_declared_size(tmp_path: Path) -> None:
    root = tmp_path / "ckpt"
    root.mkdir()
    victim = root / "w.bin"
    victim.write_bytes(b"G" * 100)
    stream = DirectoryTar(root)
    victim.write_bytes(b"G" * 5000)

    blob = _tar_bytes(stream)

    assert len(blob) == len(stream)
    assert _members(blob)["w.bin"].size == 100


def test_a_file_that_vanishes_after_planning_raises_while_streaming(tmp_path: Path) -> None:
    root = tmp_path / "ckpt"
    root.mkdir()
    victim = root / "w.bin"
    victim.write_bytes(b"x")
    stream = DirectoryTar(root)
    victim.unlink()

    with pytest.raises(FileNotFoundError):
        _tar_bytes(stream)
