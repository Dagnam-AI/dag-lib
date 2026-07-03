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
import logging
import os
from pathlib import Path
import re

from dagnam._core.auth import get_api_key, get_api_url
from dagnam._core.client import DagnamClient
from dagnam._core.config import get_config_value
from dagnam._core.exceptions import ChecksumError
from dagnam._types import JsonObject, ensure_json_object
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

logger = logging.getLogger(__name__)

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _is_uuid(s: str) -> bool:
    """Return True if *s* looks like a standard UUID (8-4-4-4-12 hex)."""
    return bool(_UUID_RE.match(s))


def _required_meta_str(meta: JsonObject, key: str) -> str:
    value = meta.get(key)
    if isinstance(value, str):
        return value
    raise ValueError(f"Dataset metadata field {key!r} must be a string")


def _optional_meta_str(meta: JsonObject, key: str) -> str | None:
    value = meta.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ValueError(f"Dataset metadata field {key!r} must be a string when provided")


def _resolve_cache_budget() -> int | None:
    """Resolve the configured LRU cache budget in bytes.

    ``max_cache_size`` comes from a user-editable config file, so a corrupt
    value (a hand-typed string, a bool) must NOT crash a just-completed
    download. A non-int is treated as "unset" (``None``), which lets
    ``evict_lru`` fall back to its default budget.
    """
    configured = get_config_value("max_cache_size", None)
    if isinstance(configured, bool) or not isinstance(configured, int):
        return None
    return configured


def load_dataset(
    dataset_id: str,
    api_url: str | None = None,
    api_key: str | None = None,
    cache_dir: str | None = None,
    version: str | None = None,
    presigned_url: str | None = None,
    download_url: str | None = None,
    resume: bool = True,
    show_progress: bool = True,
    binding: dict[str, object] | None = None,
) -> DagnamDataset:
    """Load a dataset by ID. Auto-downloads and caches if needed.

    In server mode (DAGNAM_INTERNAL=true), reads sidecar metadata from
    DAGNAM_META_DIR and loads directly from the filesystem. In client mode,
    resolves auth, checks cache, downloads if needed, and returns a dataset.
    """
    if os.environ.get("DAGNAM_INTERNAL"):
        return _load_internal(dataset_id, binding=binding)

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
            from dagnam.data.loaders.system import load_system_dataset

            return load_system_dataset(meta, binding=binding)
        except Exception as exc:
            # Native resolution is the preferred path; if it fails (e.g. the
            # framework library or tfds isn't installed) fall back to the
            # server-side download below, but surface the real cause first.
            logger.warning(
                "Native system-dataset load failed for %r; falling back to download: %s",
                dataset_id,
                exc,
            )

    cache_dir_path: Path | None = Path(cache_dir) if cache_dir is not None else None
    cache_key = f"{dataset_id}@{version}" if version else dataset_id
    checksum = _required_meta_str(meta, "checksum")
    effective_download_url = (
        presigned_url or download_url or _optional_meta_str(meta, "download_url")
    )

    if is_cached(cache_key, checksum, base_dir=cache_dir_path):
        cached_meta = load_metadata(cache_key, base_dir=cache_dir_path)
        ds_cache_dir = get_cache_dir(cache_key, base_dir=cache_dir_path)
        return DagnamDataset(cached_meta, ds_cache_dir)

    ds_cache_dir = get_cache_dir(cache_key, base_dir=cache_dir_path)
    if is_system:
        downloaded_file = client.download_system_dataset(
            dataset_id,
            ds_cache_dir,
            show_progress=show_progress,
        )
    else:
        downloaded_file = client.download_dataset(
            dataset_id,
            ds_cache_dir,
            download_url=effective_download_url,
            filename=_optional_meta_str(meta, "filename"),
            version=version,
            resume=resume,
            show_progress=show_progress,
        )

    local_checksum = compute_file_checksum(downloaded_file)
    expected_checksum = checksum.removeprefix("sha256:")
    if local_checksum != expected_checksum:
        raise ChecksumError(
            f"Checksum mismatch for dataset '{dataset_id}': "
            f"expected {checksum}, got {local_checksum}"
        )

    save_metadata(cache_key, meta, base_dir=cache_dir_path)
    save_checksum(cache_key, checksum, base_dir=cache_dir_path)
    touch_cache(cache_key, base_dir=cache_dir_path)
    evict_lru(max_size_bytes=_resolve_cache_budget(), base_dir=cache_dir_path)

    return DagnamDataset(meta, ds_cache_dir)


def _load_internal(
    dataset_id: str,
    *,
    binding: dict[str, object] | None = None,
) -> DagnamDataset:
    """Load dataset from sidecar metadata for server-side training."""
    _validate_internal_dataset_id(dataset_id)
    meta_dir_env = os.environ.get("DAGNAM_META_DIR", ".dagnam_meta")
    meta_dir = Path(meta_dir_env)
    meta_path = meta_dir / f"{dataset_id}.meta.json"

    if not meta_path.exists():
        storage_path = os.environ.get("DAGNAM_STORAGE_PATH", "/data/uploads/datasets")
        dataset_dir = Path(storage_path) / dataset_id
        legacy_meta = dataset_dir / "meta.json"
        if legacy_meta.exists():
            meta = ensure_json_object(json.loads(legacy_meta.read_text(encoding="utf-8")))
            return DagnamDataset(meta, dataset_dir)
        raise FileNotFoundError(
            f"Sidecar metadata not found at {meta_path} and no legacy meta.json at {legacy_meta}"
        )

    meta = ensure_json_object(json.loads(meta_path.read_text(encoding="utf-8")))

    if meta.get("source_type") == "system":
        # A system dataset is always resolved via its framework-native loader;
        # its sidecar carries no real on-disk file (the relative file_path is a
        # codegen placeholder), so let any loader error propagate with its real
        # cause rather than masking it as a FileNotFoundError below.
        from dagnam.data.loaders.system import load_system_dataset

        return load_system_dataset(meta, binding=binding)

    file_path_str = meta.get("file_path")
    if isinstance(file_path_str, str) and file_path_str:
        file_path = Path(file_path_str)
        if file_path.exists():
            return DagnamDataset(meta, file_path.parent)

    raise FileNotFoundError(f"Dataset file not found for '{dataset_id}': file_path={file_path_str}")


def _validate_internal_dataset_id(dataset_id: str) -> None:
    raw = str(dataset_id)
    normalized = raw.replace("\\", "/")
    if raw in {"", ".", ".."} or "/" in normalized or ":" in raw or Path(raw).is_absolute():
        raise ValueError(f"Unsafe dataset_id for internal loading: {dataset_id!r}")


__all__ = ["load_dataset"]
