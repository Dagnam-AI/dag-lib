"""Model registry — sync SDK surface: push, resolve, download, lineage.

Uses the same checksum-addressed local cache primitives as
``resources/checkpoints.py`` (``dagnam/data/cache.py``), under a sibling
root so model and dataset/checkpoint cache budgets never collide. Unlike
``checkpoints.py``, ``download()`` also serializes concurrent downloads of
the same artifact with ``dataset_lock`` and promotes the file atomically
(``os.replace`` from a same-directory ``.partial`` temp file) — matching
``data/load.py``'s dataset-cache security rules, since a model weight file
is loaded downstream (e.g. via ``torch.load``) and a reader must never
observe a partially-written or corrupted cache entry.
"""

from __future__ import annotations

from collections.abc import Sequence
import logging
import os
from pathlib import Path

import requests

from dagnam._core.client import DagnamClient
from dagnam._core.client.base import (
    ALLOW_REDIRECTS,
    is_success_response,
    safe_download_basename,
    safe_error_body_from_response,
    scrub_secret_params,
)
from dagnam._core.config import get_config_value
from dagnam._core.exceptions import APIError, ChecksumError, ModelError, ModelNotFoundError
from dagnam._core.resolver import resolve_client
from dagnam._types import JsonObject, JsonValue
from dagnam.data.cache import (
    cache_dir_name,
    compute_file_checksum,
    dataset_lock,
    evict_lru_locked,
    get_cache_size,
    touch_cache,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_CACHE_DIR: Path = Path.home() / ".dagnam" / "models"

_ARTIFACT_TYPE_BY_NAME: dict[str, str] = {
    "adapter_model.safetensors": "adapter",
    "adapter_config.json": "adapter",
    "tokenizer.json": "tokenizer",
    "tokenizer_config.json": "tokenizer",
    "special_tokens_map.json": "tokenizer",
    "config.json": "architecture_config",
    "README.md": "readme",
}


def _infer_artifact_type(path: Path) -> str:
    """Infer an artifact's registry type from its filename.

    Table-driven lookup via ``_ARTIFACT_TYPE_BY_NAME``, falling back to
    ``"weights"`` for anything unmatched — new filename patterns extend the
    table, never an ``if name == X`` branch chain.
    """
    return _ARTIFACT_TYPE_BY_NAME.get(path.name, "weights")


def _artifact_filename(artifact: JsonObject, artifact_id: str) -> str:
    """Local filename for a downloaded artifact.

    Prefers the artifact's real ``filename``, falling back to the last path
    segment of its ``logical_key`` (e.g. ``"weights/pytorch_model.bin"`` ->
    ``"pytorch_model.bin"``), and finally the artifact id -- so a
    HuggingFace-shaped multi-file release (config.json, tokenizer.json,
    adapter_model.safetensors) lands under its real names in the same version
    directory and stays reconstructable, instead of every file colliding on
    one generic ``"<id>.bin"``. Both fields are server-supplied strings
    landing in a local path, so they are routed through
    ``safe_download_basename`` (reduced to a bare basename -- no traversal,
    no absolute path, no Windows-reserved device name).
    """
    default = f"{cache_dir_name(artifact_id)}.bin"
    filename = artifact.get("filename")
    if isinstance(filename, str) and filename:
        return safe_download_basename(filename, default=default)
    logical_key = artifact.get("logical_key")
    if isinstance(logical_key, str) and logical_key:
        return safe_download_basename(logical_key, default=default)
    return default


def _put_to_presigned_url(url: str, file_path: Path, headers: JsonValue | None) -> None:
    """PUT a file straight to a presigned object-storage URL.

    Only reached when ``initiate_model_artifact`` reports ``upload_method
    == "PUT"`` (an ``STORAGE_BACKEND=s3`` backend) — the client-layer
    ``upload_model_artifact_direct`` is POST-only, the local-backend
    fallback route. A raw, unauthenticated, non-redirect-following
    ``requests.put`` mirrors ``BaseDagnamClient._get_stream_no_auth``'s
    treatment of presigned URLs: the signature already rides in the query
    string, and following a redirect could smuggle the upload to an
    attacker-controlled host.
    """
    header_map: dict[str, str] = (
        {str(k): str(v) for k, v in headers.items()} if isinstance(headers, dict) else {}
    )
    try:
        with file_path.open("rb") as handle:
            resp = requests.put(
                url,
                data=handle,
                headers=header_map,
                timeout=(10, 3600),
                allow_redirects=ALLOW_REDIRECTS,
            )
    except requests.ConnectionError as exc:
        raise APIError(0, f"Connection failed: {scrub_secret_params(str(exc))}") from exc
    except requests.Timeout as exc:
        raise APIError(0, f"Request timed out: {scrub_secret_params(str(exc))}") from exc
    if not is_success_response(resp):
        raise ModelError(
            f"presigned upload of {file_path.name} failed: "
            f"{resp.status_code} {safe_error_body_from_response(resp)}"
        )


def push(
    *,
    name: str,
    slug: str,
    description: str,
    files: Sequence[str],
    origin: str = "imported",
    license: str = "mit",
    visibility: str = "private",
    client: DagnamClient | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
) -> JsonObject:
    """Create a registry entry + draft version, then upload and finalize.

    Uploads every file in ``files`` as an artifact — each one's registry
    ``artifact_type`` is inferred from its filename via
    :func:`_infer_artifact_type` — and finalizes the version. Returns the
    finalized version's JSON body (``status == "ready"``).

    Raises:
        FileNotFoundError: a path in ``files`` does not exist. Checked up
            front, before any network call, so a typo never leaves an
            orphaned draft entry/version on the server.
        ModelError: the server rejected the entry/version/artifact/finalize
            request (e.g. a duplicate slug), or a presigned upload failed.
    """
    for file_str in files:
        if not Path(file_str).is_file():
            raise FileNotFoundError(f"artifact file not found: {file_str}")

    resolved = resolve_client(client, api_key, api_url)

    entry = resolved.create_model_entry(
        {"name": name, "slug": slug, "description": description, "visibility": visibility}
    )
    version = resolved.create_model_version(
        str(entry["id"]), {"origin": origin, "license": license}
    )
    version_id = str(version["id"])

    for file_str in files:
        file_path = Path(file_str)
        size_bytes = file_path.stat().st_size
        artifact_type = _infer_artifact_type(file_path)
        initiate = resolved.initiate_model_artifact(
            version_id,
            {
                "artifact_type": artifact_type,
                "logical_key": f"{artifact_type}/{file_path.name}",
                "size_bytes": size_bytes,
            },
        )
        upload_url = str(initiate["upload_url"])

        if initiate["upload_method"] == "PUT":
            _put_to_presigned_url(upload_url, file_path, initiate.get("headers"))
        else:
            resolved.upload_model_artifact_direct(upload_url, file_path)

        digest = compute_file_checksum(file_path)
        resolved.complete_model_artifact(
            version_id,
            str(initiate["artifact_id"]),
            {"sha256": digest, "size_bytes": size_bytes},
        )

    return resolved.finalize_model_version(version_id)


def resolve(
    version_id: str,
    *,
    client: DagnamClient | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
) -> JsonObject:
    """Fetch a model version's metadata (status, signature, runtime_manifest, ...).

    Raises:
        ModelNotFoundError: ``version_id`` does not exist.
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_model_version(version_id)


def get_lineage(
    version_id: str,
    *,
    client: DagnamClient | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
) -> JsonObject:
    """Fetch a model version's lineage graph (parent runs/versions/datasets).

    Raises:
        ModelNotFoundError: ``version_id`` does not exist.
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_model_version_lineage(version_id)


def get_task_contract(
    key: str,
    version: str,
    *,
    client: DagnamClient | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
) -> JsonObject:
    """Fetch a versioned task contract by key.

    Raises:
        ModelError: no contract matches ``key``/``version`` — a bare error,
            since the endpoint threads no id for a typed
            ``ModelNotFoundError``.
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_task_contract(key, version)


def download(
    version_id: str,
    artifact_id: str,
    *,
    client: DagnamClient | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Download (or serve from cache) one artifact of a model version.

    Cached under ``~/.dagnam/models/{version_id}/`` as the artifact's real
    filename (its ``filename``, or ``logical_key``'s last segment, or
    ``{artifact_id}.bin`` when neither is present -- see
    :func:`_artifact_filename`). Verifies the artifact's checksum against the
    server-reported value on every call, including a cache hit — a model
    weight file is loaded downstream (e.g. via ``torch.load``), so a swapped
    or corrupted cache entry is a code-execution risk, not just a
    data-integrity nuisance. Concurrent downloads of the same artifact are
    serialized with ``dataset_lock`` and promoted atomically (``os.replace``
    from a same-directory ``.partial`` temp file), so a reader never observes
    a partially-written file.

    A best-effort LRU eviction sweep runs OUTSIDE the per-artifact lock, once
    a ``max_model_cache_size`` (falling back to the shared ``max_cache_size``)
    budget is explicitly configured; with no budget configured, nothing is
    ever evicted. The entry this call just wrote is never a candidate: the
    eviction budget is padded by that entry's own size, and it is always the
    most-recently-touched entry, so an LRU sweep can only reach it once every
    older entry is already gone -- which cannot happen under a padded budget.

    Raises:
        ModelNotFoundError: ``artifact_id`` has no matching artifact on
            ``version_id``.
        ChecksumError: no checksum is available to verify against, or the
            downloaded bytes do not match it.
    """
    resolved = resolve_client(client, api_key, api_url)
    base_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_MODEL_CACHE_DIR
    version_dir = base_dir / cache_dir_name(version_id)

    artifacts = resolved.list_model_version_artifacts(version_id)
    artifact = next(
        (a for a in artifacts if isinstance(a, dict) and a.get("id") == artifact_id), None
    )
    if artifact is None:
        raise ModelNotFoundError(artifact_id)
    raw_sha = artifact.get("sha256")
    expected_sha = raw_sha if isinstance(raw_sha, str) else None
    dest_path = version_dir / _artifact_filename(artifact, artifact_id)

    def _cached() -> bool:
        return bool(
            expected_sha and dest_path.exists() and compute_file_checksum(dest_path) == expected_sha
        )

    if _cached():
        touch_cache(version_id, base_dir=base_dir)
        return dest_path

    with dataset_lock(f"{version_id}/{artifact_id}", base_dir=base_dir):
        if _cached():  # a peer finished downloading while we waited for the lock
            touch_cache(version_id, base_dir=base_dir)
            return dest_path

        version_dir.mkdir(parents=True, exist_ok=True)
        staging_path = dest_path.with_name(dest_path.name + ".partial")

        written, server_sha = resolved.download_model_artifact_stream(
            version_id, artifact_id, staging_path
        )
        actual_sha = compute_file_checksum(written)
        verify_sha = server_sha or expected_sha
        if verify_sha is None:
            written.unlink(missing_ok=True)
            raise ChecksumError(
                f"Artifact '{artifact_id}' has no server-reported checksum to verify "
                "against; refusing to cache an unverified model artifact."
            )
        if actual_sha != verify_sha:
            written.unlink(missing_ok=True)
            raise ChecksumError(
                f"Artifact '{artifact_id}' checksum mismatch: expected {verify_sha}, "
                f"got {actual_sha}"
            )

        os.replace(written, dest_path)  # atomic promotion; never expose a partial file
        touch_cache(version_id, base_dir=base_dir)

    # Evict OUTSIDE the per-artifact lock (T11), under evict_lru_locked's own
    # separate global eviction lock -- best-effort, and only when a budget is
    # explicitly configured (never falls back to evict_lru's default dataset
    # budget; see C1). Padding the budget by the fresh entry's own size means
    # the sweep can evict every older entry but never this one.
    max_bytes = get_config_value("max_model_cache_size", None) or get_config_value(
        "max_cache_size", None
    )
    if isinstance(max_bytes, int):
        try:
            evict_lru_locked(
                max_size_bytes=max_bytes + get_cache_size(version_dir), base_dir=base_dir
            )
        except Exception as exc:
            logger.warning("Model cache eviction failed: %s", exc)
    return dest_path


__all__ = [
    "download",
    "get_lineage",
    "get_task_contract",
    "push",
    "resolve",
]
