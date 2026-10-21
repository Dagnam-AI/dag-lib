"""Model Hub releases: versions, derivatives, design snapshots and download.

Re-exported from :mod:`dagnam.resources.hub` (``dagnam.hub.download`` and friends).
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Optional
from uuid import UUID

from dagnam._core.client import DagnamClient
from dagnam._core.client.base import safe_download_basename
from dagnam._core.client.hub import safe_dest
from dagnam._core.resolver import resolve_client
from dagnam._types import JsonObject, ensure_json_object
from dagnam.resources.hub_publish import normalize_stage


def derivatives(
    model_id: str | UUID,
    *,
    stage: str | list[str] | None = None,
    page: int = 1,
    limit: int = 20,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Public models made from this one, optionally only of the given stage(s).

    >>> dagnam.hub.derivatives("model_abc", stage="adapter")["items"]
    """
    resolved = resolve_client(client, api_key, api_url)
    stages = [stage] if isinstance(stage, str) else stage
    return resolved.list_hub_model_derivatives(
        str(model_id),
        stage=[normalize_stage(item) for item in stages] if stages else None,
        page=page,
        limit=limit,
    )


def get_architecture(
    model_id: str | UUID,
    version_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """A version's design snapshot (the object Studio opens)."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_hub_model_architecture(str(model_id), version_id)


def finalize_version(
    model_id: str | UUID,
    version_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Publish a draft version; a published version never changes."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.finalize_hub_model_version(str(model_id), version_id)


def delete_version(
    model_id: str | UUID,
    version_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> None:
    """Delete a draft version (a published one cannot be deleted)."""
    resolved = resolve_client(client, api_key, api_url)
    resolved.delete_hub_model_version(str(model_id), version_id)


def download(
    model_id: str | UUID,
    dest_dir: str | Path,
    file_id: Optional[str] = None,
    *,
    version_id: Optional[str] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> list[str]:
    """Save a version's files (one file with ``file_id``) under ``dest_dir``; returns their paths.

    Without ``version_id`` the latest published version is used. Without ``file_id``, a version
    that carries a design also gets it saved as ``<slug>-<version>.dag.json``.

    >>> dagnam.hub.download("model_abc", "./weights")
    ['weights/model.safetensors', 'weights/head.onnx']
    """
    resolved = resolve_client(client, api_key, api_url)
    mid = str(model_id)
    saved = [
        str(path)
        for path in resolved.download_hub_model_files(
            mid, dest_dir, file_id=file_id, version_id=version_id
        )
    ]
    if file_id is not None:
        return saved
    model = resolved.get_hub_model(mid)
    if version_id is None:
        record = model.get("latest_version")
    else:
        record = next(
            (
                item
                for item in resolved.list_hub_model_versions(mid)
                if isinstance(item, dict) and item.get("id") == version_id
            ),
            None,
        )
    version: JsonObject = record if isinstance(record, dict) else {}
    if version.get("has_architecture") is True:
        slug = re.sub(r"[^a-z0-9]+", "-", str(model.get("name", "")).lower()).strip("-") or "design"
        name = safe_download_basename(
            f"{slug}-{version.get('version')}.dag.json", default="design.dag.json"
        )
        target = safe_dest(Path(dest_dir), name)
        target.parent.mkdir(parents=True, exist_ok=True)
        design = resolved.get_hub_model_architecture(mid, str(version["id"]))
        target.write_text(json.dumps(design, indent=2), encoding="utf-8")
        saved.append(str(target))
    return saved


def use_in_studio(
    model_id: str | UUID,
    *,
    visibility: Optional[str] = None,
    version_id: Optional[str] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Open a hub model's design as a Studio project.

    ``visibility`` (``public`` or ``private``) and ``version_id`` (default: the latest
    published version) are optional.
    """
    resolved = resolve_client(client, api_key, api_url)
    legacy_use_in_studio = getattr(resolved, "hub_use_in_studio", None)
    if callable(legacy_use_in_studio):
        return ensure_json_object(legacy_use_in_studio(str(model_id)))
    options: JsonObject = {
        key: value
        for key, value in (("visibility", visibility), ("version_id", version_id))
        if value is not None
    }
    return resolved.use_hub_model_in_studio(str(model_id), options or None)


__all__ = [
    "delete_version",
    "derivatives",
    "download",
    "finalize_version",
    "get_architecture",
    "use_in_studio",
]
