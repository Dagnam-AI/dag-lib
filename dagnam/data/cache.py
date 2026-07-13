"""Local cache management for the dagnam library.

Manages the local dataset cache at ~/.dagnam/datasets/.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import time
from typing import TypedDict
from urllib.parse import quote

import filelock

from dagnam._types import JsonObject, ensure_json_object

DEFAULT_CACHE_DIR: Path = Path.home() / ".dagnam" / "datasets"
DEFAULT_MAX_CACHE_BYTES: int = 10 * 1024 * 1024 * 1024  # 10 GB
DEFAULT_LOCK_TIMEOUT: float = 60.0
# Reserved sibling directory (under the cache base) that holds in-progress,
# not-yet-promoted downloads. It is NOT a dataset, so cache enumeration and
# eviction must skip it — otherwise eviction could rmtree a peer process's
# live staging download.
STAGING_DIR_NAME = ".staging"
_CACHE_LOG = logging.getLogger("dagnam.cache")


class CacheInfo(TypedDict):
    """Metadata for one cached dataset directory."""

    dataset_id: str
    size_bytes: int
    last_access: float | None


def cache_dir_name(dataset_id: str) -> str:
    """Return a single filesystem-safe cache directory name for a dataset key."""
    raw = str(dataset_id)
    if raw == "":
        raise ValueError("dataset_id must not be empty")
    encoded = quote(raw, safe="-_.@")
    if encoded in {".", ".."}:
        encoded = encoded.replace(".", "%2E")
    return encoded


def get_cache_dir(dataset_id: str, base_dir: Path | None = None) -> Path:
    """Returns ~/.dagnam/datasets/{dataset_id}/ (or custom base).

    Creates the directory (including parents) if it doesn't exist.
    """
    base = base_dir if base_dir is not None else DEFAULT_CACHE_DIR
    cache_dir = base / cache_dir_name(dataset_id)
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


def _is_managed_cache_entry(child: Path) -> bool:
    """True if *child* is a dagnam-managed cache directory.

    A managed entry carries a dagnam marker dotfile: ``.last_access`` (written by
    :func:`touch_cache` for every dataset *and* checkpoint entry enrolled in LRU
    accounting) or ``.checksum`` (written by :func:`save_checksum` after a
    verified dataset download). Eviction requires one of these markers so it can
    NEVER delete an unrelated directory that merely happens to sit next to the
    cache root (e.g. when ``base_dir`` is a user-supplied path such as the
    current working directory). The checkpoint cache has no ``.checksum`` for
    S3-presigned downloads, so ``.last_access`` is the marker both caches share.
    """
    return (child / ".last_access").is_file() or (child / ".checksum").is_file()


def _dir_size(root: Path) -> int:
    """Total size in bytes of the regular files under ``root``.

    A file may be evicted by a concurrent process between enumeration and the
    ``stat`` read (a TOCTOU race); such a vanished file is skipped rather than
    aborting the whole scan with ``FileNotFoundError``/``PermissionError``.
    """
    total = 0
    for f in root.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue
    return total


def get_cache_size(base_dir: Path | None = None) -> int:
    """Calculate total size of the cache directory in bytes."""
    base = base_dir if base_dir is not None else DEFAULT_CACHE_DIR
    if not base.exists():
        return 0
    return _dir_size(base)


def get_cache_info(base_dir: Path | None = None) -> list[CacheInfo]:
    """Return info about each cached dataset."""
    base = base_dir if base_dir is not None else DEFAULT_CACHE_DIR
    if not base.exists():
        return []

    entries: list[CacheInfo] = []
    for child in base.iterdir():
        if not child.is_dir() or child.name == STAGING_DIR_NAME:
            continue
        size = _dir_size(child)
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

        configured_size = get_config_value("max_cache_size", DEFAULT_MAX_CACHE_BYTES)
        max_size_bytes = (
            configured_size if isinstance(configured_size, int) else DEFAULT_MAX_CACHE_BYTES
        )

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
        if not ds_dir.exists():
            # Reported by get_cache_info but removed by a concurrent process
            # before we reached it (a TOCTOU race); nothing to evict.
            continue
        if not _is_managed_cache_entry(ds_dir):
            # Not a dagnam cache directory (no .checksum marker). Never delete
            # an unrelated directory, even when base_dir points at a shared
            # location such as the current working directory.
            continue
        shutil.rmtree(ds_dir)
        total -= entry["size_bytes"]
        evicted.append(entry["dataset_id"])

    return evicted


def lock_timeout() -> float:
    """Cache-lock acquisition timeout (seconds), from ``DAGNAM_CACHE_LOCK_TIMEOUT``.

    An absent or unparseable value falls back to :data:`DEFAULT_LOCK_TIMEOUT`.
    """
    raw = os.environ.get("DAGNAM_CACHE_LOCK_TIMEOUT")
    if raw is None:
        return DEFAULT_LOCK_TIMEOUT
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_LOCK_TIMEOUT


def dataset_lock(
    dataset_id: str, *, base_dir: Path | None = None, timeout: float | None = None
) -> filelock.FileLock:
    """Exclusive cross-process lock for one dataset's cache dir.

    The lock file is a **sibling** of the dataset directory
    (``<base>/<cache_dir_name>.lock``), never inside it, so eviction can
    ``rmtree`` the dataset dir without deleting a lock another process may be
    holding.
    """
    base = base_dir if base_dir is not None else DEFAULT_CACHE_DIR
    base.mkdir(parents=True, exist_ok=True)
    lock_path = base / f"{cache_dir_name(dataset_id)}.lock"
    return filelock.FileLock(str(lock_path), timeout=lock_timeout() if timeout is None else timeout)


def eviction_lock(
    *, base_dir: Path | None = None, timeout: float | None = None
) -> filelock.FileLock:
    """Global cross-process lock serializing cache eviction (``<base>/.eviction.lock``)."""
    base = base_dir if base_dir is not None else DEFAULT_CACHE_DIR
    base.mkdir(parents=True, exist_ok=True)
    return filelock.FileLock(
        str(base / ".eviction.lock"), timeout=lock_timeout() if timeout is None else timeout
    )


def evict_lru_locked(max_size_bytes: int | None = None, base_dir: Path | None = None) -> list[str]:
    """Evict LRU cache entries under the global eviction lock; best-effort.

    A busy lock (another process/thread already evicting) is **not** an error:
    this logs a warning and returns ``[]`` instead of blocking or raising, since
    eviction is disk housekeeping and must never fail a caller that just
    finished a successful download.
    """
    try:
        with eviction_lock(base_dir=base_dir):
            return evict_lru(max_size_bytes=max_size_bytes, base_dir=base_dir)
    except filelock.Timeout:
        _CACHE_LOG.warning("eviction lock busy; skipping eviction this round")
        return []


def save_metadata(
    dataset_id: str,
    meta: JsonObject,
    base_dir: Path | None = None,
    *,
    data_file: Path | None = None,
) -> None:
    """Write meta.json to cache directory.

    When ``data_file`` is given, also records its ``size`` + ``mtime`` under a
    namespaced ``_cache`` key so a later :func:`verify_cached` can cheaply check
    a cache hit without a full re-hash. **Contract:** callers MUST pass
    ``data_file`` at its FINAL on-disk path (after any staging/rename is
    complete) — recording a pre-rename staging path's mtime would make every
    subsequent cheap check fail spuriously.
    """
    cache_dir = get_cache_dir(dataset_id, base_dir)
    payload = dict(meta)
    if data_file is not None and data_file.exists():
        st = data_file.stat()
        payload["_cache"] = {"file": data_file.name, "size": st.st_size, "mtime": st.st_mtime}
    meta_file = cache_dir / "meta.json"
    meta_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def verify_cached(
    dataset_id: str, server_checksum: str, *, base_dir: Path | None = None, full: bool = False
) -> bool:
    """Verify a cache hit: cheap size+mtime by default, full sha256 on demand.

    On a ``True`` result this calls :func:`touch_cache` first — mirroring
    :func:`is_cached`'s touch-on-hit behavior so a hit resolved via
    ``verify_cached`` still counts as a recent access for LRU eviction. Returns
    ``False`` when the recorded ``_cache`` block is missing, the data file is
    gone, size/mtime differ, or (when ``full``) the sha256 mismatches.
    """
    cache_dir = get_cache_dir(dataset_id, base_dir)
    meta = load_metadata(dataset_id, base_dir)
    info = meta.get("_cache")
    if not isinstance(info, dict):
        return False
    data_file = cache_dir / str(info.get("file", ""))
    if not data_file.exists():
        return False
    st = data_file.stat()
    if st.st_size != info.get("size") or st.st_mtime != info.get("mtime"):
        return False
    if full and compute_file_checksum(data_file) != server_checksum.removeprefix("sha256:"):
        return False
    touch_cache(dataset_id, base_dir)
    return True


def load_metadata(dataset_id: str, base_dir: Path | None = None) -> JsonObject:
    """Read meta.json from cache directory. Returns empty dict if file doesn't exist."""
    cache_dir = get_cache_dir(dataset_id, base_dir)
    meta_file = cache_dir / "meta.json"
    if not meta_file.exists():
        return {}
    return ensure_json_object(json.loads(meta_file.read_text(encoding="utf-8")))


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
