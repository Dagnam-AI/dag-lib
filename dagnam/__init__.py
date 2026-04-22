"""dagnam — Python client library for Dagnam.AI datasets."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dagnam._core.auth import configure, get_api_key, get_api_url
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
from dagnam._core.config import get_config_value
from dagnam._core.client import DagnamClient
from dagnam.data.dataset import DagnamDataset
from dagnam._core.exceptions import ChecksumError
from dagnam.services.inference import deployment_health, inference, inference_batch
from dagnam.services.checkpoints import download_checkpoint
from dagnam.services.training import TrainingEvent, stream_training
from dagnam.services import codegen, deployments, hub, projects
from dagnam.services import datasets_upload
from dagnam._core.lro import LongRunningOperation

__all__ = [
    "load_dataset",
    "configure",
    "inference",
    "inference_batch",
    "deployment_health",
    "download_checkpoint",
    "stream_training",
    "TrainingEvent",
    "codegen",
    "deployments",
    "hub",
    "projects",
    "datasets_upload",
    "LongRunningOperation",
]

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

    In server mode (DAGNAM_INTERNAL=true), reads sidecar metadata from
    DAGNAM_META_DIR and loads directly from the filesystem.
    In client mode, resolves auth → checks cache → downloads if needed → returns DagnamDataset.
    """
    # --- Server mode: bypass HTTP entirely via sidecar metadata ---
    if os.environ.get("DAGNAM_INTERNAL"):
        return _load_internal(dataset_id)

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

    # 5. Detect system datasets by source_type and route to native loader
    source_type = meta.get("source_type", "")
    if source_type == "system" or is_system:
        try:
            from dagnam.data.loaders.system_loader import resolve_system_dataset
            return resolve_system_dataset(meta)
        except (ImportError, Exception):
            # Fall through to normal download path if native loader fails
            pass

    # 6. Resolve cache base dir
    cache_dir_path: Path | None = Path(cache_dir) if cache_dir is not None else None

    # 7. Check cache
    if is_cached(dataset_id, meta["checksum"], base_dir=cache_dir_path):
        cached_meta = load_metadata(dataset_id, base_dir=cache_dir_path)
        ds_cache_dir = get_cache_dir(dataset_id, base_dir=cache_dir_path)
        return DagnamDataset(cached_meta, ds_cache_dir)

    # 8. Download
    ds_cache_dir = get_cache_dir(dataset_id, base_dir=cache_dir_path)
    if is_system:
        downloaded_file = client.download_system_dataset(dataset_id, ds_cache_dir)
    else:
        downloaded_file = client.download_dataset(dataset_id, ds_cache_dir)

    # 9. Verify checksum
    local_checksum = compute_file_checksum(downloaded_file)
    if local_checksum != meta["checksum"]:
        raise ChecksumError(
            f"Checksum mismatch for dataset '{dataset_id}': "
            f"expected {meta['checksum']}, got {local_checksum}"
        )

    # 10. Persist metadata and checksum
    save_metadata(dataset_id, meta, base_dir=cache_dir_path)
    save_checksum(dataset_id, meta["checksum"], base_dir=cache_dir_path)

    # 10. Update access timestamp and run LRU eviction
    touch_cache(dataset_id, base_dir=cache_dir_path)
    max_cache = get_config_value("max_cache_size", None)
    evict_lru(max_size_bytes=max_cache, base_dir=cache_dir_path)

    return DagnamDataset(meta, ds_cache_dir)


def _load_internal(dataset_id: str) -> DagnamDataset:
    """Load dataset from sidecar metadata (server-side training).

    Reads ``.dagnam_meta/{dataset_id}.meta.json`` written by the Celery
    training task.  Routes system datasets to native loaders with shared
    ``TORCH_HOME`` cache, and user datasets to direct filesystem reads.
    """
    meta_dir_env = os.environ.get("DAGNAM_META_DIR", ".dagnam_meta")
    meta_dir = Path(meta_dir_env)
    meta_path = meta_dir / f"{dataset_id}.meta.json"

    if not meta_path.exists():
        # Fallback: legacy path (flat storage)
        storage_path = os.environ.get("DAGNAM_STORAGE_PATH", "/data/uploads/datasets")
        dataset_dir = Path(storage_path) / dataset_id
        legacy_meta = dataset_dir / "meta.json"
        if legacy_meta.exists():
            meta = json.loads(legacy_meta.read_text(encoding="utf-8"))
            return DagnamDataset(meta, dataset_dir)
        raise FileNotFoundError(
            f"Sidecar metadata not found at {meta_path} and no legacy "
            f"meta.json at {legacy_meta}"
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # System datasets: use native loader with shared TORCH_HOME cache
    if meta.get("source_type") == "system":
        try:
            from dagnam.data.loaders.system_loader import resolve_system_dataset
            return resolve_system_dataset(meta)
        except (ImportError, Exception):
            pass  # Fall through to file-based path

    # User datasets: read directly from file_path on the filesystem
    file_path_str = meta.get("file_path")
    if file_path_str:
        file_path = Path(file_path_str)
        if file_path.exists():
            return DagnamDataset(meta, file_path.parent)

    raise FileNotFoundError(
        f"Dataset file not found for '{dataset_id}': file_path={file_path_str}"
    )
