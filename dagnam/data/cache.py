"""Local cache management for the dagnam library.

Manages the local dataset cache at ~/.dagnam/datasets/.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import time

DEFAULT_CACHE_DIR: Path = Path.home() / ".dagnam" / "datasets"
DEFAULT_MAX_CACHE_BYTES: int = 10 * 1024 * 1024 * 1024  # 10 GB


def get_cache_dir(dataset_id: str, base_dir: Path | None = None) -> Path:
    """Returns ~/.dagnam/datasets/{dataset_id}/ (or custom base).

    Creates the directory (including parents) if it doesn't exist.
    """
    base = base_dir if base_dir is not None else DEFAULT_CACHE_DIR
    cache_dir = base / dataset_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def is_cached(dataset_id: str, server_checksum: str, base_dir: Path | None = None) -> bool:
    """True if .checksum file exists and matches server_checksum."""
    cache_dir = get_cache_dir(dataset_id, base_dir)
    checksum_file = cache_dir / ".checksum"
    if not checksum_file.exists():
        return False
    local_checksum = checksum_file.read_text(encoding="utf-8").strip()
    matched = local_checksum == server_checksum
    if matched:
        touch_cache(dataset_id, base_dir)
    return matched


def touch_cache(dataset_id: str, base_dir: Path | None = None) -> None:
    """Update the .last_access timestamp for a cached dataset."""
    cache_dir = get_cache_dir(dataset_id, base_dir)
    access_file = cache_dir / ".last_access"
    access_file.write_text(str(time.time()), encoding="utf-8")


def get_cache_size(base_dir: Path | None = None) -> int:
    """Calculate total size of the cache directory in bytes."""
    base = base_dir if base_dir is not None else DEFAULT_CACHE_DIR
    if not base.exists():
        return 0
    return sum(f.stat().st_size for f in base.rglob("*") if f.is_file())


def get_cache_info(base_dir: Path | None = None) -> list[dict]:
    """Return info about each cached dataset."""
    base = base_dir if base_dir is not None else DEFAULT_CACHE_DIR
    if not base.exists():
        return []

    entries = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        size = sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
        access_file = child / ".last_access"
        last_access: float | None = None
        if access_file.exists():
            try:
                last_access = float(access_file.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                pass
        entries.append(
            {
                "dataset_id": child.name,
                "size_bytes": size,
                "last_access": last_access,
            }
        )
    return entries


def evict_lru(max_size_bytes: int | None = None, base_dir: Path | None = None) -> list[str]:
    """Evict least-recently-used datasets until cache is under max_size_bytes.

    Returns list of evicted dataset IDs.
    """
    if max_size_bytes is None:
        from dagnam._core.config import get_config_value

        max_size_bytes = get_config_value("max_cache_size", DEFAULT_MAX_CACHE_BYTES)

    base = base_dir if base_dir is not None else DEFAULT_CACHE_DIR
    if not base.exists():
        return []

    total = get_cache_size(base)
    if total <= max_size_bytes:
        return []

    # Get all datasets sorted by last_access (oldest first, None treated as 0)
    entries = get_cache_info(base)
    entries.sort(key=lambda e: e["last_access"] or 0)

    evicted: list[str] = []
    for entry in entries:
        if total <= max_size_bytes:
            break
        ds_dir = base / entry["dataset_id"]
        if ds_dir.exists():
            shutil.rmtree(ds_dir)
            total -= entry["size_bytes"]
            evicted.append(entry["dataset_id"])

    return evicted


def save_metadata(dataset_id: str, meta: dict, base_dir: Path | None = None) -> None:
    """Write meta.json to cache directory."""
    cache_dir = get_cache_dir(dataset_id, base_dir)
    meta_file = cache_dir / "meta.json"
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_metadata(dataset_id: str, base_dir: Path | None = None) -> dict:
    """Read meta.json from cache directory. Returns empty dict if file doesn't exist."""
    cache_dir = get_cache_dir(dataset_id, base_dir)
    meta_file = cache_dir / "meta.json"
    if not meta_file.exists():
        return {}
    return json.loads(meta_file.read_text(encoding="utf-8"))


def save_checksum(dataset_id: str, checksum: str, base_dir: Path | None = None) -> None:
    """Write .checksum file after successful download."""
    cache_dir = get_cache_dir(dataset_id, base_dir)
    checksum_file = cache_dir / ".checksum"
    checksum_file.write_text(checksum, encoding="utf-8")


def compute_file_checksum(file_path: Path) -> str:
    """SHA256 of file, read in 8KB chunks. Returns hex digest."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()
