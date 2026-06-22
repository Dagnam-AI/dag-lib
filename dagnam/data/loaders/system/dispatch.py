"""Descriptor-driven system dataset loading."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
from typing import Any, cast

import requests

from dagnam._types import JsonObject
from dagnam.data.dataset import DagnamDataset
from dagnam.data.loaders.system.bound_dataset import BoundNativeDataset
from dagnam.data.loaders.system.common import SYSTEM_CACHE_ROOT
from dagnam.data.loaders.system.decoders import get_decoder

_DOWNLOAD_TIMEOUT = (30, 60)


def detect_installed_framework() -> str:
    """Return a compatibility value for callers that still probe this helper."""
    return "generic"


def _artifact_filename(meta: JsonObject) -> str | None:
    filename = meta.get("filename")
    if isinstance(filename, str) and filename:
        return filename
    artifact = meta.get("artifact")
    if isinstance(artifact, dict) and isinstance(artifact.get("filename"), str):
        return cast("str", artifact["filename"])
    file_path = meta.get("file_path")
    if isinstance(file_path, str) and file_path:
        return Path(file_path).name
    return None


def _artifact_source(meta: JsonObject) -> tuple[str | None, str | None]:
    artifact = meta.get("artifact")
    if isinstance(artifact, dict):
        url = artifact.get("download_url")
        checksum = artifact.get("checksum")
        return (
            url if isinstance(url, str) else None,
            checksum if isinstance(checksum, str) else None,
        )
    url = meta.get("download_url")
    checksum = meta.get("checksum")
    return (
        url if isinstance(url, str) else None,
        checksum if isinstance(checksum, str) else None,
    )


def _copy_local_artifact(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _download_artifact(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    try:
        with requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT) as response:
            response.raise_for_status()
            with tmp.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        tmp.replace(destination)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_verified_file(url: str, destination: Path, expected_sha256: str) -> None:
    if destination.exists() and _sha256(destination) == expected_sha256:
        return
    destination.unlink(missing_ok=True)
    _download_artifact(url, destination)
    actual = _sha256(destination)
    if actual != expected_sha256:
        destination.unlink(missing_ok=True)
        raise ValueError(
            f"Downloaded system dataset checksum mismatch: expected {expected_sha256}, got {actual}"
        )


def _artifact_dir(meta: JsonObject) -> Path:
    """Resolve/download the local artifact directory for a system descriptor."""
    dataset_id = str(meta.get("id") or meta.get("name") or "system")
    cache = SYSTEM_CACHE_ROOT / dataset_id
    cache.mkdir(parents=True, exist_ok=True)
    filename = _artifact_filename(meta)
    if filename is None:
        return cache

    destination = cache / filename
    file_path = meta.get("file_path")
    if isinstance(file_path, str) and file_path:
        source = Path(file_path)
        if source.exists():
            _copy_local_artifact(source, destination)
            return cache

    url, checksum = _artifact_source(meta)
    if url is None:
        return cache
    if Path(url).exists():
        _copy_local_artifact(Path(url), destination)
        return cache
    if checksum:
        _ensure_verified_file(url, destination, checksum.removeprefix("sha256:"))
    elif not destination.exists():
        _download_artifact(url, destination)
    return cache


def load_system_dataset(
    meta: JsonObject,
    *,
    framework: str | None = None,
    transform: object | None = None,
    binding: dict[str, object] | None = None,
) -> DagnamDataset:
    """Load a system dataset from its served descriptor, independent of framework."""
    del framework, transform
    descriptor_format = str(meta["format"])
    raw_layout = meta.get("layout", {})
    if not isinstance(raw_layout, dict) or not raw_layout:
        raise ValueError(f"System dataset {meta.get('name')!r} has no layout descriptor")
    layout = cast("dict[str, object]", raw_layout)
    raw_columns = meta.get("columns", [])
    descriptor_columns = raw_columns if isinstance(raw_columns, list) else []
    columns = cast("list[dict[str, Any]]", descriptor_columns)
    raw_roles = meta.get("column_roles", {})
    column_roles = cast("dict[str, str]", raw_roles if isinstance(raw_roles, dict) else {})
    decoder = get_decoder(descriptor_format)
    artifact = _artifact_dir(meta)
    train_store = decoder.decode(artifact, layout, "train")
    test_store = decoder.decode(artifact, layout, "test")
    resolved_binding = cast("dict[str, Any]", binding or {})
    return DagnamDataset(
        meta,
        artifact,
        _native_train=BoundNativeDataset(train_store, resolved_binding, columns, column_roles),
        _native_test=BoundNativeDataset(test_store, resolved_binding, columns, column_roles),
    )
