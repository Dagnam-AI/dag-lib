"""Media utilities for image and audio dataset loaders.

Provides archive extraction, class-folder discovery, and deterministic
split helpers used by image_folder_loader and audio_loader.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import random
import shutil
import stat
import tarfile
import zipfile

# Standard split folder names recognized by the discovery logic.
_SPLIT_NAMES = frozenset({"train", "val", "validation", "test", "dev"})

# Image extensions recognized during folder scanning.
IMAGE_EXTENSIONS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
    }
)

# Audio extensions recognized during folder scanning.
AUDIO_EXTENSIONS = frozenset(
    {
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".aac",
        ".wma",
        ".m4a",
    }
)
_MAX_ARCHIVE_UNCOMPRESSED_BYTES = 8 * 1024 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 200_000


@dataclass(frozen=True)
class FolderLayout:
    """Immutable description of a class-folder dataset layout.

    Attributes:
        has_explicit_splits: True if the root contains split subdirectories
            (e.g. train/, val/, test/) each with class folders inside.
        class_names: Sorted list of discovered class names.
        splits: List of split names found (empty if unsplit).
        root: Root directory of the dataset.
    """

    has_explicit_splits: bool
    class_names: list[str]
    splits: list[str]
    root: Path


def discover_class_folders(root: Path) -> FolderLayout:
    """Discover the class-folder layout of a dataset directory.

    Handles two layouts:
    1. **Split layout**: ``root/{split}/{class}/*.ext``
    2. **Unsplit layout**: ``root/{class}/*.ext``

    Returns a :class:`FolderLayout` describing what was found.
    """
    if not root.exists() or not root.is_dir():
        return FolderLayout(
            has_explicit_splits=False,
            class_names=[],
            splits=[],
            root=root,
        )

    # Check for split-based layout
    subdirs = sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))

    # Normalize split names for matching
    split_dirs = [d for d in subdirs if d.lower() in _SPLIT_NAMES]

    if split_dirs:
        # Verify at least one split dir has class subdirectories
        all_classes: set[str] = set()
        valid_splits: list[str] = []

        for split_name in split_dirs:
            split_path = root / split_name
            class_dirs = sorted(
                d.name for d in split_path.iterdir() if d.is_dir() and not d.name.startswith(".")
            )
            if class_dirs:
                all_classes.update(class_dirs)
                valid_splits.append(split_name)

        if valid_splits:
            return FolderLayout(
                has_explicit_splits=True,
                class_names=sorted(all_classes),
                splits=sorted(valid_splits),
                root=root,
            )

    # Unsplit layout: root/{class}/*.ext
    class_dirs = sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name.lower() not in _SPLIT_NAMES
    )

    # Verify these are actual class folders (contain files, not just subdirs)
    valid_classes: list[str] = []
    for cls_name in class_dirs:
        cls_path = root / cls_name
        has_files = any(f.is_file() and not f.name.startswith(".") for f in cls_path.iterdir())
        if has_files:
            valid_classes.append(cls_name)

    return FolderLayout(
        has_explicit_splits=False,
        class_names=sorted(valid_classes),
        splits=[],
        root=root,
    )


def scan_class_samples(root: Path) -> tuple[list[tuple[Path, int]], list[str]]:
    """Enumerate ``(image_path, class_idx)`` pairs and the sorted class names.

    The expensive per-class file walk is cached per ``(root, signature)`` so
    repeated loader calls over an unchanged tree do not re-walk every file. The
    signature covers the root *and* each class subdirectory's mtime, so adding or
    removing a file *inside* an existing class folder — which bumps that folder's
    mtime, not the root's — invalidates the entry. Computing the signature is a
    cheap top-level ``iterdir`` + per-class ``stat`` that never descends into the
    files, so the cache still saves the costly inner scan.
    """
    samples, classes = _scan_class_samples_cached(str(root), _scan_signature(root))
    return list(samples), list(classes)


def _scan_signature(root: Path) -> tuple[tuple[str, int], ...]:
    """Freshness key covering the root and every immediate class subdirectory.

    A class folder's mtime — not the root's — changes when a file is added to or
    removed from it, so each class dir's ``st_mtime_ns`` must be part of the key
    for the scan cache to invalidate on intra-class content changes; the leading
    root entry catches added/removed class folders. Nanosecond precision avoids
    the whole-second blind spot of a bare ``int(st_mtime)``.
    """
    signature: list[tuple[str, int]] = [("", root.stat().st_mtime_ns)]
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            signature.append((entry.name, entry.stat().st_mtime_ns))
    return tuple(signature)


@lru_cache(maxsize=64)
def _scan_class_samples_cached(
    root_str: str, _signature: tuple[tuple[str, int], ...]
) -> tuple[tuple[tuple[Path, int], ...], tuple[str, ...]]:
    """Cached single-pass scan of a class-folder directory.

    ``_signature`` participates only in the cache key (see ``_scan_signature``);
    a change to the root or any class folder yields a new key and forces a fresh
    scan.
    """
    root = Path(root_str)
    classes = sorted(
        entry.name for entry in root.iterdir() if entry.is_dir() and not entry.name.startswith(".")
    )
    class_to_idx = {name: index for index, name in enumerate(classes)}
    samples: list[tuple[Path, int]] = []
    for class_name in classes:
        class_dir = root / class_name
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, class_to_idx[class_name]))
    return tuple(samples), tuple(classes)


@lru_cache(maxsize=128)
def _shuffled_indices(n: int, seed: int) -> tuple[int, ...]:
    """Memoize the deterministic shuffle of ``range(n)`` for a given seed.

    The same ``(n, seed)`` permutation backs all three splits, so caching it
    avoids re-shuffling once per split when loaders request train/val/test in
    sequence.
    """
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    return tuple(indices)


def _split_bounds(n: int, val_ratio: float, test_ratio: float) -> tuple[int, int]:
    """Return the ``(train_end, val_end)`` boundaries of the deterministic split."""
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test
    return n_train, n_train + n_val


def split_indices(
    n: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Deterministically split indices into train/val/test sets.

    Uses a seeded random shuffle to ensure reproducibility across runs.

    Args:
        n: Total number of samples.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    indices = _shuffled_indices(n, seed)
    train_end, val_end = _split_bounds(n, val_ratio, test_ratio)
    return (
        list(indices[:train_end]),
        list(indices[train_end:val_end]),
        list(indices[val_end:]),
    )


def resolve_split_dir(root: Path, split: str, available: list[str]) -> Path:
    """Resolve the directory for a requested split, honoring common aliases.

    Direct match wins; otherwise ``val``/``validation``/``test`` fall back to
    their aliases (``validation``/``dev``); failing that, ``train`` is used if
    present. Raises :class:`FileNotFoundError` when nothing matches.
    """
    if split in available:
        return root / split

    aliases = {
        "val": ["validation", "dev"],
        "validation": ["val"],
        "test": ["dev"],
    }
    for alias in aliases.get(split, []):
        if alias in available:
            return root / alias

    if "train" in available:
        return root / "train"

    raise FileNotFoundError(
        f"No directory found for split '{split}' in {root}. Available splits: {available}"
    )


def select_split_indices(
    n: int,
    split: str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> list[int]:
    """Return only the requested split's indices from a deterministic partition.

    Thin selector over the memoized permutation so loaders never rebuild the
    ``{"train": ..., "val": ..., "test": ...}[split]`` map by hand and never
    materialize the two splits they did not ask for.
    """
    indices = _shuffled_indices(n, seed)
    train_end, val_end = _split_bounds(n, val_ratio, test_ratio)
    spans = {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, n)}
    start, stop = spans[split]
    return list(indices[start:stop])


def ensure_extracted(data_dir: Path) -> Path:
    """Extract an archive in data_dir if one exists and return the extraction root.

    Supports .zip, .tar.gz, .tar.bz2, and .tar archives. Extracts into
    a ``_extracted/`` subdirectory to avoid polluting the cache folder.

    If no archive is found or extraction already happened, returns the
    data_dir itself (or the _extracted dir if it exists).
    """
    extracted_dir = data_dir / "_extracted"
    if extracted_dir.exists() and any(extracted_dir.iterdir()):
        return extracted_dir

    # Look for archive files
    archives = list(data_dir.glob("*.zip")) + list(data_dir.glob("*.tar*"))
    if not archives:
        return data_dir

    archive_path = archives[0]
    extracted_dir.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            _safe_extract_zip(zf, extracted_dir)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            _safe_extract_tar(tf, extracted_dir)
    else:
        # Not a recognized archive — just use data_dir
        return data_dir

    # If extraction produced a single top-level directory, use that as root
    top_level = [d for d in extracted_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if len(top_level) == 1:
        return top_level[0]

    return extracted_dir


def _is_within_directory(base: Path, target: Path) -> bool:
    """Return True when target resolves inside base on Python 3.9+."""
    base_resolved = base.resolve()
    target_resolved = target.resolve()
    try:
        target_resolved.relative_to(base_resolved)
        return True
    except ValueError:
        return False


def _validate_archive_target(destination: Path, member_name: str) -> None:
    target = destination / member_name
    if Path(member_name).is_absolute() or not _is_within_directory(destination, target):
        raise ValueError(f"Unsafe archive member path: {member_name}")


def _safe_extract_zip(archive: zipfile.ZipFile, destination: Path) -> None:
    members = archive.infolist()
    _validate_archive_size((member.file_size for member in members), len(members))
    for member in members:
        _validate_archive_target(destination, member.filename)
        if _zip_member_is_symlink(member):
            raise ValueError(f"Unsafe archive member link: {member.filename}")

    for member in members:
        target = destination / member.filename
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member, "r") as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)


def _safe_extract_tar(archive: tarfile.TarFile, destination: Path) -> None:
    members = archive.getmembers()
    _validate_archive_size((member.size for member in members), len(members))
    for member in members:
        _validate_archive_target(destination, member.name)
        if member.issym() or member.islnk():
            raise ValueError(f"Unsafe archive member link: {member.name}")
        if not (member.isdir() or member.isfile()):
            raise ValueError(f"Unsafe archive member type: {member.name}")

    for member in members:
        target = destination / member.name
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        src = archive.extractfile(member)
        if src is None:
            raise ValueError(f"Unable to extract archive member: {member.name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)


def _zip_member_is_symlink(member: zipfile.ZipInfo) -> bool:
    file_type = (member.external_attr >> 16) & 0o170000
    return file_type == stat.S_IFLNK


def _validate_archive_size(member_sizes: Iterable[int], member_count: int) -> None:
    if member_count > _MAX_ARCHIVE_MEMBERS:
        raise ValueError(f"Archive has too many members: {member_count}")

    total_size = 0
    for size in member_sizes:
        total_size += max(0, int(size or 0))
        if total_size > _MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise ValueError(
                "Archive is too large after decompression "
                f"({total_size} bytes > {_MAX_ARCHIVE_UNCOMPRESSED_BYTES} bytes)"
            )
