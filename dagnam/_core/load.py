"""Dataset loading entry points.

Server mode contract:

``DAGNAM_INTERNAL``
    When set, bypass HTTP and load from sidecar metadata written by the
    backend training task.
``DAGNAM_META_DIR``
    Directory containing ``{dataset_id}.meta.json`` sidecar files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re

from dagnam._core.auth import get_api_key, get_api_url
from dagnam._core.client import DagnamClient
from dagnam._core.config import get_config_value
from dagnam._core.exceptions import ChecksumError
from dagnam.data.cache import (
    compute_file_checksum,
    evict_lru,
    get_cache_dir,
    is_cached,
    load_metadata,
    save_checksum,
    save_metadata,
    touch_cache,
)
from dagnam.data.dataset import DagnamDataset

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _is_uuid(s: str) -> bool:
    """Return True if *s* looks like a standard UUID (8-4-4-4-12 hex)."""
    return bool(_UUID_RE.match(s))


def load_dataset(
    dataset_id: str,
    api_url: str | None = None,
    api_key: str | None = None,
    cache_dir: str | None = None,
    version: str | None = None,
    presigned_url: str | None = None,
    download_url: str | None = None,
    resume: bool = True,
) -> DagnamDataset:
    """Load a dataset by ID. Auto-downloads and caches if needed.

    In server mode (DAGNAM_INTERNAL=true), reads sidecar metadata from
    DAGNAM_META_DIR and loads directly from the filesystem. In client mode,
    resolves auth, checks cache, downloads if needed, and returns a dataset.
    """
    if os.environ.get("DAGNAM_INTERNAL"):
        return _load_internal(dataset_id)

    resolved_key = get_api_key(override=api_key)
    resolved_url = get_api_url(override=api_url)
    client = DagnamClient(resolved_url, resolved_key)
    is_system = not _is_uuid(dataset_id)

    if is_system:
        meta = client.get_system_dataset_meta(dataset_id, version=version)
    else:
        meta = client.get_dataset_meta(dataset_id, version=version)

    source_type = meta.get("source_type", "")
    if source_type == "system" or is_system:
        try:
            from dagnam.data.loaders.system import resolve_system_dataset

            return resolve_system_dataset(meta)
        except (ImportError, Exception):
            pass

    cache_dir_path: Path | None = Path(cache_dir) if cache_dir is not None else None
    cache_key = f"{dataset_id}@{version}" if version else dataset_id
    effective_download_url = presigned_url or download_url or meta.get("download_url")

    if is_cached(cache_key, meta["checksum"], base_dir=cache_dir_path):
        cached_meta = load_metadata(cache_key, base_dir=cache_dir_path)
        ds_cache_dir = get_cache_dir(cache_key, base_dir=cache_dir_path)
        return DagnamDataset(cached_meta, ds_cache_dir)

    ds_cache_dir = get_cache_dir(cache_key, base_dir=cache_dir_path)
    if is_system:
        downloaded_file = client.download_system_dataset(dataset_id, ds_cache_dir)
    else:
        downloaded_file = client.download_dataset(
            dataset_id,
            ds_cache_dir,
            download_url=effective_download_url,
            filename=meta.get("filename"),
            version=version,
            resume=resume,
        )

    local_checksum = compute_file_checksum(downloaded_file)
    expected_checksum = meta["checksum"].removeprefix("sha256:")
    if local_checksum != expected_checksum:
        raise ChecksumError(
            f"Checksum mismatch for dataset '{dataset_id}': "
            f"expected {meta['checksum']}, got {local_checksum}"
        )

    save_metadata(cache_key, meta, base_dir=cache_dir_path)
    save_checksum(cache_key, meta["checksum"], base_dir=cache_dir_path)
    touch_cache(cache_key, base_dir=cache_dir_path)
    max_cache = get_config_value("max_cache_size", None)
    evict_lru(max_size_bytes=max_cache, base_dir=cache_dir_path)

    return DagnamDataset(meta, ds_cache_dir)


def _load_internal(dataset_id: str) -> DagnamDataset:
    """Load dataset from sidecar metadata for server-side training."""
    meta_dir_env = os.environ.get("DAGNAM_META_DIR", ".dagnam_meta")
    meta_dir = Path(meta_dir_env)
    meta_path = meta_dir / f"{dataset_id}.meta.json"

    if not meta_path.exists():
        storage_path = os.environ.get("DAGNAM_STORAGE_PATH", "/data/uploads/datasets")
        dataset_dir = Path(storage_path) / dataset_id
        legacy_meta = dataset_dir / "meta.json"
        if legacy_meta.exists():
            meta = json.loads(legacy_meta.read_text(encoding="utf-8"))
            return DagnamDataset(meta, dataset_dir)
        raise FileNotFoundError(
            f"Sidecar metadata not found at {meta_path} and no legacy meta.json at {legacy_meta}"
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if meta.get("source_type") == "system":
        try:
            from dagnam.data.loaders.system import resolve_system_dataset

            return resolve_system_dataset(meta)
        except (ImportError, Exception):
            pass

    file_path_str = meta.get("file_path")
    if file_path_str:
        file_path = Path(file_path_str)
        if file_path.exists():
            return DagnamDataset(meta, file_path.parent)

    raise FileNotFoundError(f"Dataset file not found for '{dataset_id}': file_path={file_path_str}")


__all__ = ["load_dataset"]
