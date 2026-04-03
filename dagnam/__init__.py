"""dagnam — Python client library for Dagnam.AI datasets."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dagnam.auth import configure, get_api_key, get_api_url
from dagnam.cache import (
    compute_file_checksum,
    evict_lru,
    get_cache_dir,
    is_cached,
    load_metadata,
    save_checksum,
    save_metadata,
    touch_cache,
)
from dagnam.config import get_config_value
from dagnam.client import DagnamClient
from dagnam.dataset import DagnamDataset
from dagnam.exceptions import ChecksumError

__all__ = ["load_dataset", "configure"]

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
) -> DagnamDataset:
    """Load a dataset by ID. Auto-downloads and caches if needed.

    In server mode (DAGNAM_INTERNAL=true), reads directly from filesystem.
    In client mode, resolves auth → checks cache → downloads if needed → returns DagnamDataset.
    """
    # --- Server mode: bypass HTTP entirely ---
    if os.environ.get("DAGNAM_INTERNAL"):
        storage_path = os.environ.get("DAGNAM_STORAGE_PATH", "/data/uploads/datasets")
        dataset_dir = Path(storage_path) / dataset_id
        meta_file = dataset_dir / "meta.json"
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        return DagnamDataset(meta, dataset_dir)

    # --- Client mode ---
    # 1. Resolve auth
    resolved_key = get_api_key(override=api_key)
    resolved_url = get_api_url(override=api_url)

    # 2. Create HTTP client
    client = DagnamClient(resolved_url, resolved_key)

    # 3. Determine whether this is a system dataset (friendly name) or user dataset (UUID)
    is_system = not _is_uuid(dataset_id)

    # 4. Get metadata from server
    if is_system:
        meta = client.get_system_dataset_meta(dataset_id)
    else:
        meta = client.get_dataset_meta(dataset_id)

    # 5. Resolve cache base dir
    cache_dir_path: Path | None = Path(cache_dir) if cache_dir is not None else None

    # 6. Check cache
    if is_cached(dataset_id, meta["checksum"], base_dir=cache_dir_path):
        cached_meta = load_metadata(dataset_id, base_dir=cache_dir_path)
        ds_cache_dir = get_cache_dir(dataset_id, base_dir=cache_dir_path)
        return DagnamDataset(cached_meta, ds_cache_dir)

    # 7. Download
    ds_cache_dir = get_cache_dir(dataset_id, base_dir=cache_dir_path)
    if is_system:
        downloaded_file = client.download_system_dataset(dataset_id, ds_cache_dir)
    else:
        downloaded_file = client.download_dataset(dataset_id, ds_cache_dir)

    # 8. Verify checksum
    local_checksum = compute_file_checksum(downloaded_file)
    if local_checksum != meta["checksum"]:
        raise ChecksumError(
            f"Checksum mismatch for dataset '{dataset_id}': "
            f"expected {meta['checksum']}, got {local_checksum}"
        )

    # 9. Persist metadata and checksum
    save_metadata(dataset_id, meta, base_dir=cache_dir_path)
    save_checksum(dataset_id, meta["checksum"], base_dir=cache_dir_path)

    # 10. Update access timestamp and run LRU eviction
    touch_cache(dataset_id, base_dir=cache_dir_path)
    max_cache = get_config_value("max_cache_size", None)
    evict_lru(max_size_bytes=max_cache, base_dir=cache_dir_path)

    return DagnamDataset(meta, ds_cache_dir)
