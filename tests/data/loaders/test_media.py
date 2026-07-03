"""Coverage for dagnam.data.loaders.media — archive + folder discovery."""

from __future__ import annotations

import io
from pathlib import Path
import tarfile
import zipfile

import pytest

from dagnam.data.loaders.media import (
    _MAX_ARCHIVE_MEMBERS,
    _safe_extract_tar,
    _validate_archive_size,
    discover_class_folders,
    ensure_extracted,
    resolve_split_dir,
    safe_extract_zip,
    select_split_indices,
    split_indices,
)

# ---------------------------------------------------------------- discover_class_folders


def test_discover_class_folders_missing_root(tmp_path: Path) -> None:
    layout = discover_class_folders(tmp_path / "does-not-exist")
    assert layout.class_names == []
    assert not layout.has_explicit_splits


def test_discover_class_folders_unsplit_layout(tmp_path: Path) -> None:
    (tmp_path / "cat").mkdir()
    (tmp_path / "cat" / "a.jpg").write_bytes(b"x")
    (tmp_path / "dog").mkdir()
    (tmp_path / "dog" / "b.jpg").write_bytes(b"x")
    # An empty dir at root shouldn't count as a class
    (tmp_path / "empty").mkdir()
    layout = discover_class_folders(tmp_path)
    assert layout.has_explicit_splits is False
    assert layout.class_names == ["cat", "dog"]


def test_discover_class_folders_split_layout(tmp_path: Path) -> None:
    for split in ("train", "val"):
        for cls in ("cat", "dog"):
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            (d / "x.jpg").write_bytes(b"x")
    layout = discover_class_folders(tmp_path)
    assert layout.has_explicit_splits is True
    assert layout.splits == ["train", "val"]
    assert layout.class_names == ["cat", "dog"]


def test_discover_class_folders_split_dirs_without_classes_fall_through(tmp_path: Path) -> None:
    # train/ exists but is empty — falls back to unsplit detection on root subdirs.
    (tmp_path / "train").mkdir()
    (tmp_path / "alpha").mkdir()
    (tmp_path / "alpha" / "x.txt").write_bytes(b"x")
    layout = discover_class_folders(tmp_path)
    assert layout.has_explicit_splits is False
    assert layout.class_names == ["alpha"]


# ---------------------------------------------------------------- split_indices


def test_split_indices_partitions_completely() -> None:
    train, val, test = split_indices(100, val_ratio=0.2, test_ratio=0.1, seed=0)
    assert len(train) + len(val) + len(test) == 100
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)


# ---------------------------------------------------------------- ensure_extracted


def test_ensure_extracted_no_archive_returns_data_dir(tmp_path: Path) -> None:
    assert ensure_extracted(tmp_path) == tmp_path


def test_ensure_extracted_already_extracted(tmp_path: Path) -> None:
    e = tmp_path / "_extracted"
    e.mkdir()
    (e / "data").write_bytes(b"x")
    assert ensure_extracted(tmp_path) == e


def test_ensure_extracted_zip_with_single_top_level_dir(tmp_path: Path) -> None:
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("root/file.txt", "hi")
        zf.writestr("root/sub/file2.txt", "hi")
    result = ensure_extracted(tmp_path)
    assert result.name == "root"
    assert (result / "file.txt").read_text() == "hi"


def test_ensure_extracted_zip_multiple_top_level(tmp_path: Path) -> None:
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a/x.txt", "a")
        zf.writestr("b/y.txt", "b")
    result = ensure_extracted(tmp_path)
    assert result == tmp_path / "_extracted"


def test_ensure_extracted_tar(tmp_path: Path) -> None:
    archive = tmp_path / "data.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        data = b"hello"
        info = tarfile.TarInfo(name="root/file.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    result = ensure_extracted(tmp_path)
    assert (result / "file.txt").read_text() == "hello"


def test_ensure_extracted_unknown_archive_returns_data_dir(tmp_path: Path) -> None:
    # File matches the glob but is neither zip nor tar.
    (tmp_path / "stray.zip").write_bytes(b"not really a zip")
    assert ensure_extracted(tmp_path) == tmp_path


# ---------------------------------------------------------------- archive safety


def test_validate_archive_size_too_many_members() -> None:
    with pytest.raises(ValueError, match="too many"):
        _validate_archive_size(
            (1 for _ in range(_MAX_ARCHIVE_MEMBERS + 1)), _MAX_ARCHIVE_MEMBERS + 1
        )


def test_validate_archive_size_too_large() -> None:
    big = 10 * 1024 * 1024 * 1024  # 10 GB single member
    with pytest.raises(ValueError, match="too large"):
        _validate_archive_size((big,), 1)


def test_safe_extract_zip_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("../escape.txt", "evil")
    with zipfile.ZipFile(archive_path, "r") as zf:
        with pytest.raises(ValueError, match="Unsafe archive member path"):
            safe_extract_zip(zf, tmp_path / "out")


def test_safe_extract_tar_rejects_symlink_member(tmp_path: Path) -> None:
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


def test_safe_extract_tar_rejects_special_member(tmp_path: Path) -> None:
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


# ---------------------------------------------------------------- safe extract directory members


def test_safe_extract_zip_creates_directory_members(tmp_path: Path) -> None:
    """Explicit directory entries are created and skipped (no file copy)."""
    archive_path = tmp_path / "dirs.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("sub/", "")  # explicit directory entry
        zf.writestr("sub/file.txt", "hi")
    out = tmp_path / "out"
    with zipfile.ZipFile(archive_path, "r") as zf:
        safe_extract_zip(zf, out)
    assert (out / "sub").is_dir()
    assert (out / "sub" / "file.txt").read_text() == "hi"


def test_safe_extract_tar_creates_directory_members(tmp_path: Path) -> None:
    """Tar directory members are created and skipped before file members."""
    archive_path = tmp_path / "dirs.tar"
    with tarfile.open(archive_path, "w") as tf:
        dir_info = tarfile.TarInfo(name="sub")
        dir_info.type = tarfile.DIRTYPE
        tf.addfile(dir_info)
        data = b"hello"
        file_info = tarfile.TarInfo(name="sub/file.txt")
        file_info.size = len(data)
        tf.addfile(file_info, io.BytesIO(data))
    out = tmp_path / "out"
    out.mkdir()
    with tarfile.open(archive_path, "r") as tf:
        _safe_extract_tar(tf, out)
    assert (out / "sub").is_dir()
    assert (out / "sub" / "file.txt").read_text() == "hello"


def test_safe_extract_tar_raises_when_extractfile_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A file member that yields no stream from extractfile is rejected."""
    archive_path = tmp_path / "data.tar"
    with tarfile.open(archive_path, "w") as tf:
        data = b"content"
        info = tarfile.TarInfo(name="file.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    out = tmp_path / "out"
    out.mkdir()
    with tarfile.open(archive_path, "r") as tf:
        monkeypatch.setattr(tf, "extractfile", lambda _member: None)
        with pytest.raises(ValueError, match="Unable to extract archive member"):
            _safe_extract_tar(tf, out)


def test_safe_extract_zip_rejects_symlink_member(tmp_path: Path) -> None:
    """A zip entry whose external_attr marks it a symlink is rejected."""
    import stat

    archive_path = tmp_path / "link.zip"
    info = zipfile.ZipInfo("link")
    # Encode the symlink file-type bits in the high 16 bits of external_attr.
    info.external_attr = stat.S_IFLNK << 16
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(info, "/etc/passwd")
    out = tmp_path / "out"
    with zipfile.ZipFile(archive_path, "r") as zf:
        with pytest.raises(ValueError, match="Unsafe archive member link"):
            safe_extract_zip(zf, out)


# ---------------------------------------------------------------- resolve_split_dir


def test_resolve_split_dir_direct_match(tmp_path: Path) -> None:
    assert resolve_split_dir(tmp_path, "train", ["train", "val"]) == tmp_path / "train"


def test_resolve_split_dir_aliases(tmp_path: Path) -> None:
    assert resolve_split_dir(tmp_path, "val", ["train", "validation"]) == tmp_path / "validation"
    assert resolve_split_dir(tmp_path, "validation", ["train", "val"]) == tmp_path / "val"
    assert resolve_split_dir(tmp_path, "test", ["train", "dev"]) == tmp_path / "dev"


def test_resolve_split_dir_fallback_to_train(tmp_path: Path) -> None:
    assert resolve_split_dir(tmp_path, "val", ["train"]) == tmp_path / "train"


def test_resolve_split_dir_raises_when_no_match(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No directory found"):
        resolve_split_dir(tmp_path, "val", ["other"])


# ---------------------------------------------------------------- select_split_indices


def test_select_split_indices_partitions_disjointly() -> None:
    train = select_split_indices(100, "train", val_ratio=0.1, test_ratio=0.1, seed=42)
    val = select_split_indices(100, "val", val_ratio=0.1, test_ratio=0.1, seed=42)
    test = select_split_indices(100, "test", val_ratio=0.1, test_ratio=0.1, seed=42)
    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10
    assert set(train) | set(val) | set(test) == set(range(100))
    assert not (set(train) & set(val))
    assert not (set(val) & set(test))


def test_select_split_indices_deterministic_by_seed() -> None:
    assert select_split_indices(50, "train", seed=7) == select_split_indices(50, "train", seed=7)
    assert select_split_indices(50, "train", seed=7) != select_split_indices(50, "train", seed=8)
