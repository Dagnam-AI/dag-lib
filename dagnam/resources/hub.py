"""Model Hub — sync SDK surface.

Wraps the ``/api/v1/hub/*`` routes on top of
:class:`dagnam.client.DagnamClient` for browsing, publishing, and managing
models in the Dagnam Model Hub.

The module exposes plain functions (``dagnam.hub.search(...)``) to match
the Phase 3 style (``dagnam.inference``, ``dagnam.deployments``).
"""

from __future__ import annotations

from dagnam._types import (
    JsonArray,
    JsonObject,
    JsonValue,
    ensure_json_array,
    ensure_json_object,
    is_json_value,
)
from typing import Optional
from uuid import UUID

from dagnam._core.client import DagnamClient
from dagnam._core.resolver import resolve_client


def _stringify_id(value: object) -> str:
    if isinstance(value, UUID):
        return str(value)
    return str(value)


def _json_payload_value(name: str, value: object) -> JsonValue:
    if is_json_value(value):
        return value
    raise TypeError(f"Hub field {name!r} must be JSON-compatible")


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
) -> JsonObject:
    """Search the model hub with optional filters.

    >>> dagnam.hub.search(task_type="text-generation", sort_by="popular")["items"]
    """
    resolved = resolve_client(client, api_key, api_url)
    legacy_search = getattr(resolved, "hub_search", None)
    if callable(legacy_search):
        return ensure_json_object(
            legacy_search(
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
        )
    return resolved.list_hub_models(
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
) -> JsonArray | str | None:
    """Return available model categories."""
    resolved = resolve_client(client, api_key, api_url)
    legacy_categories = getattr(resolved, "hub_categories", None)
    if callable(legacy_categories):
        return ensure_json_array(legacy_categories())
    return resolved.list_hub_categories()


def featured(
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonArray:
    """Return featured models."""
    resolved = resolve_client(client, api_key, api_url)
    legacy_featured = getattr(resolved, "hub_featured", None)
    if callable(legacy_featured):
        return ensure_json_array(legacy_featured())
    return resolved.get_hub_featured()


def trending(
    *,
    days: int = 7,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonArray:
    """Return trending models over the given number of days."""
    resolved = resolve_client(client, api_key, api_url)
    legacy_trending = getattr(resolved, "hub_trending", None)
    if callable(legacy_trending):
        return ensure_json_array(legacy_trending(days=days))
    return resolved.get_hub_trending(days=days)


# ---------------------------------------------------------------------------
# Model CRUD
# ---------------------------------------------------------------------------


def get(
    model_id: str | UUID,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Fetch a single model record."""
    resolved = resolve_client(client, api_key, api_url)
    legacy_get = getattr(resolved, "hub_get", None)
    if callable(legacy_get):
        return ensure_json_object(legacy_get(_stringify_id(model_id)))
    return resolved.get_hub_model(_stringify_id(model_id))


def create(
    *,
    name: str,
    description: str,
    task_type: str,
    framework: str,
    license: str = "mit",
    visibility: str = "public",
    tags: Optional[list[str]] = None,
    metadata: Optional[JsonObject] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Create a new model in the hub.

    >>> dagnam.hub.create(
    ...     name="my-model", description="...", task_type="text-generation", framework="pytorch"
    ... )
    """
    resolved = resolve_client(client, api_key, api_url)
    payload: JsonObject = {
        "name": name,
        "description": description,
        "task_type": task_type,
        "framework": framework,
        "license": license,
        "visibility": visibility,
    }
    if tags is not None:
        payload["tags"] = [str(tag) for tag in tags]
    if metadata is not None:
        payload["metadata"] = metadata
    legacy_create = getattr(resolved, "hub_create", None)
    if callable(legacy_create):
        return ensure_json_object(legacy_create(payload))
    return resolved.create_hub_model(payload)


def update(
    model_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    **kwargs: object,
) -> JsonObject:
    """Update mutable model fields."""
    resolved = resolve_client(client, api_key, api_url)
    payload: JsonObject = {key: _json_payload_value(key, value) for key, value in kwargs.items()}
    legacy_update = getattr(resolved, "hub_update", None)
    if callable(legacy_update):
        return ensure_json_object(legacy_update(_stringify_id(model_id), payload))
    return resolved.update_hub_model(_stringify_id(model_id), payload)


def delete(
    model_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> None:
    """Delete a model from the hub."""
    resolved = resolve_client(client, api_key, api_url)
    legacy_delete = getattr(resolved, "hub_delete", None)
    if callable(legacy_delete):
        legacy_delete(_stringify_id(model_id))
        return
    resolved.delete_hub_model(_stringify_id(model_id))


# ---------------------------------------------------------------------------
# Files & versions
# ---------------------------------------------------------------------------


def list_files(
    model_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """List files belonging to a model."""
    resolved = resolve_client(client, api_key, api_url)
    legacy_list_files = getattr(resolved, "hub_list_files", None)
    if callable(legacy_list_files):
        return ensure_json_object(legacy_list_files(_stringify_id(model_id)))
    return resolved.list_hub_model_files(_stringify_id(model_id))


def download(
    model_id: str,
    *,
    file_id: Optional[str] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Download a model or a specific file."""
    resolved = resolve_client(client, api_key, api_url)
    legacy_download = getattr(resolved, "hub_download", None)
    if callable(legacy_download):
        return ensure_json_object(legacy_download(_stringify_id(model_id), file_id=file_id))
    return resolved.download_hub_model(_stringify_id(model_id), file_id=file_id)


def list_versions(
    model_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonArray:
    """List all versions of a model."""
    resolved = resolve_client(client, api_key, api_url)
    legacy_list_versions = getattr(resolved, "hub_list_versions", None)
    if callable(legacy_list_versions):
        return ensure_json_array(legacy_list_versions(_stringify_id(model_id)))
    return resolved.list_hub_model_versions(_stringify_id(model_id))


def create_version(
    model_id: str,
    version: str,
    *,
    changelog: Optional[str] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Create a new version for a model."""
    resolved = resolve_client(client, api_key, api_url)
    payload: JsonObject = {"version": version}
    if changelog is not None:
        payload["changelog"] = changelog
    legacy_create_version = getattr(resolved, "hub_create_version", None)
    if callable(legacy_create_version):
        return ensure_json_object(
            legacy_create_version(_stringify_id(model_id), version=version, changelog=changelog)
        )
    return resolved.create_hub_model_version(_stringify_id(model_id), payload)


# ---------------------------------------------------------------------------
# Social / community
# ---------------------------------------------------------------------------


def star(
    model_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Star a model."""
    resolved = resolve_client(client, api_key, api_url)
    legacy_star = getattr(resolved, "hub_star", None)
    if callable(legacy_star):
        return ensure_json_object(legacy_star(_stringify_id(model_id)))
    return resolved.star_hub_model(_stringify_id(model_id))


def unstar(
    model_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Remove a star from a model."""
    resolved = resolve_client(client, api_key, api_url)
    legacy_unstar = getattr(resolved, "hub_unstar", None)
    if callable(legacy_unstar):
        return ensure_json_object(legacy_unstar(_stringify_id(model_id)))
    return resolved.unstar_hub_model(_stringify_id(model_id))


def starred(
    *,
    sort_by: str = "date_starred",
    page: int = 1,
    limit: int = 20,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """List models starred by the current user."""
    resolved = resolve_client(client, api_key, api_url)
    legacy_starred = getattr(resolved, "hub_starred", None)
    if callable(legacy_starred):
        return ensure_json_object(legacy_starred(sort_by=sort_by, page=page, limit=limit))
    return resolved.list_hub_starred(sort_by=sort_by, page=page, limit=limit)


def fork(
    model_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Fork a model into the current user's namespace."""
    resolved = resolve_client(client, api_key, api_url)
    legacy_fork = getattr(resolved, "hub_fork", None)
    if callable(legacy_fork):
        return ensure_json_object(legacy_fork(_stringify_id(model_id)))
    return resolved.fork_hub_model(_stringify_id(model_id))


def list_reviews(
    model_id: str,
    *,
    page: int = 1,
    limit: int = 20,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """List reviews for a model."""
    resolved = resolve_client(client, api_key, api_url)
    legacy_list_reviews = getattr(resolved, "hub_list_reviews", None)
    if callable(legacy_list_reviews):
        return ensure_json_object(
            legacy_list_reviews(_stringify_id(model_id), page=page, limit=limit)
        )
    return resolved.list_hub_model_reviews(
        _stringify_id(model_id),
        page=page,
        limit=limit,
    )


def add_review(
    model_id: str,
    rating: int,
    *,
    review_text: Optional[str] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Add a review to a model."""
    resolved = resolve_client(client, api_key, api_url)
    payload: JsonObject = {"rating": rating}
    if review_text is not None:
        payload["review_text"] = review_text
    legacy_add_review = getattr(resolved, "hub_add_review", None)
    if callable(legacy_add_review):
        return ensure_json_object(
            legacy_add_review(_stringify_id(model_id), rating=rating, review_text=review_text)
        )
    return resolved.add_hub_model_review(_stringify_id(model_id), payload)


# ---------------------------------------------------------------------------
# Studio integration
# ---------------------------------------------------------------------------


def use_in_studio(
    model_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Import a hub model into Studio for fine-tuning or inference."""
    resolved = resolve_client(client, api_key, api_url)
    legacy_use_in_studio = getattr(resolved, "hub_use_in_studio", None)
    if callable(legacy_use_in_studio):
        return ensure_json_object(legacy_use_in_studio(_stringify_id(model_id)))
    return resolved.use_hub_model_in_studio(_stringify_id(model_id))


__all__ = [
    "add_review",
    "categories",
    "create",
    "create_version",
    "delete",
    "download",
    "featured",
    "fork",
    "get",
    "list_files",
    "list_reviews",
    "list_versions",
    "search",
    "star",
    "starred",
    "trending",
    "unstar",
    "update",
    "use_in_studio",
]
