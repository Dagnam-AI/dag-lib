"""Model Hub — sync SDK surface.

Wraps the ``/api/v1/hub/*`` routes on top of
:class:`dagnam.client.DagnamClient` for browsing, publishing, and managing
models in the Dagnam Model Hub.

The module exposes plain functions (``dagnam.hub.search(...)``) to match
the Phase 3 style (``dagnam.inference``, ``dagnam.deployments``).
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from dagnam._core._resolver import resolve_client
from dagnam._core.client import DagnamClient


def _stringify_id(value: Any) -> str:
    if isinstance(value, UUID):
        return str(value)
    return str(value)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def search(
    *,
    search: Optional[str] = None,
    task_type: Optional[str] = None,
    framework: Optional[str] = None,
    license: Optional[str] = None,
    tags: Optional[list[str]] = None,
    is_official: Optional[bool] = None,
    is_verified: Optional[bool] = None,
    sort_by: str = "popular",
    page: int = 1,
    limit: int = 20,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> dict:
    """Search the model hub with optional filters.

    >>> dagnam.hub.search(task_type="text-generation", sort_by="popular")["items"]
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.hub_search(
        search=search,
        task_type=task_type,
        framework=framework,
        license=license,
        tags=tags,
        is_official=is_official,
        is_verified=is_verified,
        sort_by=sort_by,
        page=page,
        limit=limit,
    )


def categories(
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> list:
    """Return available model categories."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.hub_categories()


def featured(
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> list:
    """Return featured models."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.hub_featured()


def trending(
    *,
    days: int = 7,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> list:
    """Return trending models over the given number of days."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.hub_trending(days=days)


# ---------------------------------------------------------------------------
# Model CRUD
# ---------------------------------------------------------------------------


def get(
    model_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> dict:
    """Fetch a single model record."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.hub_get(_stringify_id(model_id))


def create(
    *,
    name: str,
    description: str,
    task_type: str,
    framework: str,
    license: str = "mit",
    visibility: str = "public",
    tags: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> dict:
    """Create a new model in the hub.

    >>> dagnam.hub.create(name="my-model", description="...", task_type="text-generation", framework="pytorch")
    """
    resolved = resolve_client(client, api_key, api_url)
    payload: dict[str, Any] = {
        "name": name,
        "description": description,
        "task_type": task_type,
        "framework": framework,
        "license": license,
        "visibility": visibility,
    }
    if tags is not None:
        payload["tags"] = tags
    if metadata is not None:
        payload["metadata"] = metadata
    return resolved.hub_create(payload)


def update(
    model_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Update mutable model fields."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.hub_update(_stringify_id(model_id), kwargs)


def delete(
    model_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> None:
    """Delete a model from the hub."""
    resolved = resolve_client(client, api_key, api_url)
    resolved.hub_delete(_stringify_id(model_id))


# ---------------------------------------------------------------------------
# Files & versions
# ---------------------------------------------------------------------------


def list_files(
    model_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> dict:
    """List files belonging to a model."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.hub_list_files(_stringify_id(model_id))


def download(
    model_id: str,
    *,
    file_id: Optional[str] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> dict:
    """Download a model or a specific file."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.hub_download(_stringify_id(model_id), file_id=file_id)


def list_versions(
    model_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> list:
    """List all versions of a model."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.hub_list_versions(_stringify_id(model_id))


def create_version(
    model_id: str,
    version: str,
    *,
    changelog: Optional[str] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> dict:
    """Create a new version for a model."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.hub_create_version(
        _stringify_id(model_id), version=version, changelog=changelog,
    )


# ---------------------------------------------------------------------------
# Social / community
# ---------------------------------------------------------------------------


def star(
    model_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> dict:
    """Star a model."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.hub_star(_stringify_id(model_id))


def unstar(
    model_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> dict:
    """Remove a star from a model."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.hub_unstar(_stringify_id(model_id))


def starred(
    *,
    sort_by: str = "date_starred",
    page: int = 1,
    limit: int = 20,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> dict:
    """List models starred by the current user."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.hub_starred(sort_by=sort_by, page=page, limit=limit)


def fork(
    model_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> dict:
    """Fork a model into the current user's namespace."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.hub_fork(_stringify_id(model_id))


def list_reviews(
    model_id: str,
    *,
    page: int = 1,
    limit: int = 20,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> dict:
    """List reviews for a model."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.hub_list_reviews(
        _stringify_id(model_id), page=page, limit=limit,
    )


def add_review(
    model_id: str,
    rating: int,
    *,
    review_text: Optional[str] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> dict:
    """Add a review to a model."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.hub_add_review(
        _stringify_id(model_id), rating=rating, review_text=review_text,
    )


# ---------------------------------------------------------------------------
# Studio integration
# ---------------------------------------------------------------------------


def use_in_studio(
    model_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> dict:
    """Import a hub model into Studio for fine-tuning or inference."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.hub_use_in_studio(_stringify_id(model_id))


__all__ = [
    "search",
    "get",
    "create",
    "update",
    "delete",
    "list_files",
    "download",
    "list_versions",
    "create_version",
    "star",
    "unstar",
    "fork",
    "list_reviews",
    "add_review",
    "use_in_studio",
    "categories",
    "featured",
    "trending",
    "starred",
]
