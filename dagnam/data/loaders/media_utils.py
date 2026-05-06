"""Media utilities for image and audio dataset loaders.

Provides archive extraction, class-folder discovery, and deterministic
split helpers used by image_folder_loader and audio_loader.
"""

from __future__ import annotations

import random
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# Standard split folder names recognized by the discovery logic.
_SPLIT_NAMES = frozenset({"train", "val", "validation", "test", "dev"})

# Image extensions recognized during folder scanning.
IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp",
})

# Audio extensions recognized during folder scanning.
AUDIO_EXTENSIONS = frozenset({
    ".wav", ".mp3", ".flac", ".ogg", ".aac", ".wma", ".m4a",
})


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
    class_names: List[str]
    splits: List[str]
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
    subdirs = sorted(
        d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )

    # Normalize split names for matching
    split_dirs = [d for d in subdirs if d.lower() in _SPLIT_NAMES]

    if split_dirs:
        # Verify at least one split dir has class subdirectories
        all_classes: set[str] = set()
        valid_splits: list[str] = []

        for split_name in split_dirs:
            split_path = root / split_name
            class_dirs = sorted(
                d.name
                for d in split_path.iterdir()
                if d.is_dir() and not d.name.startswith(".")
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
        has_files = any(
            f.is_file() and not f.name.startswith(".")
            for f in cls_path.iterdir()
        )
        if has_files:
            valid_classes.append(cls_name)

    return FolderLayout(
        has_explicit_splits=False,
        class_names=sorted(valid_classes),
        splits=[],
        root=root,
    )


def split_indices(
    n: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
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
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    train = indices[:n_train]
    val = indices[n_train: n_train + n_val]
    test = indices[n_train + n_val:]

    return train, val, test


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
    top_level = [
        d for d in extracted_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]
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
    for member in archive.infolist():
        _validate_archive_target(destination, member.filename)
    archive.extractall(destination)


def _safe_extract_tar(archive: tarfile.TarFile, destination: Path) -> None:
    for member in archive.getmembers():
        _validate_archive_target(destination, member.name)
        if member.issym() or member.islnk():
            raise ValueError(f"Unsafe archive member link: {member.name}")
    archive.extractall(destination)
