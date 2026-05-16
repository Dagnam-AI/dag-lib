"""Coverage for dagnam.data.loaders.media — archive + folder discovery."""

from __future__ import annotations

import io
import tarfile
import zipfile

import pytest

from dagnam.data.loaders.media import (
    _MAX_ARCHIVE_MEMBERS,
    _safe_extract_tar,
    _safe_extract_zip,
    _validate_archive_size,
    discover_class_folders,
    ensure_extracted,
    split_indices,
)

# ---------------------------------------------------------------- discover_class_folders


def test_discover_class_folders_missing_root(tmp_path):
    layout = discover_class_folders(tmp_path / "does-not-exist")
    assert layout.class_names == []
    assert not layout.has_explicit_splits


def test_discover_class_folders_unsplit_layout(tmp_path):
    (tmp_path / "cat").mkdir()
    (tmp_path / "cat" / "a.jpg").write_bytes(b"x")
    (tmp_path / "dog").mkdir()
    (tmp_path / "dog" / "b.jpg").write_bytes(b"x")
    # An empty dir at root shouldn't count as a class
    (tmp_path / "empty").mkdir()
    layout = discover_class_folders(tmp_path)
    assert layout.has_explicit_splits is False
    assert layout.class_names == ["cat", "dog"]


def test_discover_class_folders_split_layout(tmp_path):
    for split in ("train", "val"):
        for cls in ("cat", "dog"):
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            (d / "x.jpg").write_bytes(b"x")
    layout = discover_class_folders(tmp_path)
    assert layout.has_explicit_splits is True
    assert layout.splits == ["train", "val"]
    assert layout.class_names == ["cat", "dog"]


def test_discover_class_folders_split_dirs_without_classes_fall_through(tmp_path):
    # train/ exists but is empty — falls back to unsplit detection on root subdirs.
    (tmp_path / "train").mkdir()
    (tmp_path / "alpha").mkdir()
    (tmp_path / "alpha" / "x.txt").write_bytes(b"x")
    layout = discover_class_folders(tmp_path)
    assert layout.has_explicit_splits is False
    assert layout.class_names == ["alpha"]


# ---------------------------------------------------------------- split_indices


def test_split_indices_partitions_completely():
    train, val, test = split_indices(100, val_ratio=0.2, test_ratio=0.1, seed=0)
    assert len(train) + len(val) + len(test) == 100
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)


# ---------------------------------------------------------------- ensure_extracted


def test_ensure_extracted_no_archive_returns_data_dir(tmp_path):
    assert ensure_extracted(tmp_path) == tmp_path


def test_ensure_extracted_already_extracted(tmp_path):
    e = tmp_path / "_extracted"
    e.mkdir()
    (e / "data").write_bytes(b"x")
    assert ensure_extracted(tmp_path) == e


def test_ensure_extracted_zip_with_single_top_level_dir(tmp_path):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("root/file.txt", "hi")
        zf.writestr("root/sub/file2.txt", "hi")
    result = ensure_extracted(tmp_path)
    assert result.name == "root"
    assert (result / "file.txt").read_text() == "hi"


def test_ensure_extracted_zip_multiple_top_level(tmp_path):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a/x.txt", "a")
        zf.writestr("b/y.txt", "b")
    result = ensure_extracted(tmp_path)
    assert result == tmp_path / "_extracted"


def test_ensure_extracted_tar(tmp_path):
    archive = tmp_path / "data.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        data = b"hello"
        info = tarfile.TarInfo(name="root/file.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    result = ensure_extracted(tmp_path)
    assert (result / "file.txt").read_text() == "hello"


def test_ensure_extracted_unknown_archive_returns_data_dir(tmp_path):
    # File matches the glob but is neither zip nor tar.
    (tmp_path / "stray.zip").write_bytes(b"not really a zip")
    assert ensure_extracted(tmp_path) == tmp_path


# ---------------------------------------------------------------- archive safety


def test_validate_archive_size_too_many_members():
    with pytest.raises(ValueError, match="too many"):
        _validate_archive_size((1 for _ in range(_MAX_ARCHIVE_MEMBERS + 1)), _MAX_ARCHIVE_MEMBERS + 1)


def test_validate_archive_size_too_large():
    big = 10 * 1024 * 1024 * 1024  # 10 GB single member
    with pytest.raises(ValueError, match="too large"):
        _validate_archive_size((big,), 1)


def test_safe_extract_zip_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("../escape.txt", "evil")
    with zipfile.ZipFile(archive_path, "r") as zf:
        with pytest.raises(ValueError, match="Unsafe archive member path"):
            _safe_extract_zip(zf, tmp_path / "out")


def test_safe_extract_tar_rejects_symlink_member(tmp_path):
    archive_path = tmp_path / "evil.tar"
    with tarfile.open(archive_path, "w") as tf:
        info = tarfile.TarInfo(name="link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    with tarfile.open(archive_path, "r") as tf:
        out = tmp_path / "out"
        out.mkdir()
        with pytest.raises(ValueError, match="Unsafe archive member link"):
            _safe_extract_tar(tf, out)


def test_safe_extract_tar_rejects_special_member(tmp_path):
    archive_path = tmp_path / "weird.tar"
    with tarfile.open(archive_path, "w") as tf:
        info = tarfile.TarInfo(name="fifo")
        info.type = tarfile.FIFOTYPE
        tf.addfile(info)
    with tarfile.open(archive_path, "r") as tf:
        out = tmp_path / "out"
        out.mkdir()
        with pytest.raises(ValueError, match="Unsafe archive member type"):
            _safe_extract_tar(tf, out)
