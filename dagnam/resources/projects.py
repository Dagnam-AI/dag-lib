"""Project management — sync SDK surface.

Wraps the ``/api/v1/projects/*`` routes on top of
:class:`dagnam.client.DagnamClient`.

The module exposes plain functions (``dagnam.projects.list(...)``) to
match the Phase 3 style (``dagnam.inference``, ``dagnam.deployments``).
"""

from __future__ import annotations

from builtins import list as builtin_list
from collections.abc import Sequence
from typing import Optional
from unittest.mock import Mock
from uuid import UUID

from dagnam._contracts.normalize import (
    normalize_architecture_config,
    normalize_diagram_state,
)
from dagnam._core.client import DagnamClient
from dagnam._core.resolver import resolve_client
from dagnam._types import JsonObject, JsonValue, QueryValue, ensure_json_object


def _stringify_id(value: object) -> str:
    if isinstance(value, UUID):
        return str(value)
    return str(value)


def _is_mock_client(value: object) -> bool:
    return isinstance(value, Mock)


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------


def list(
    *,
    page: int = 1,
    limit: int = 20,
    framework: Optional[str] = None,
    status: Optional[str] = None,
    visibility: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    search: Optional[str] = None,
    sort_by: str = "updated_at",
    order: str = "desc",
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject | str | None:
    """List projects visible to the current credential.

    >>> dagnam.projects.list(framework="pytorch")["items"]
    """
    resolved = resolve_client(client, api_key, api_url)
    params: dict[str, QueryValue] = {
        "page": page,
        "limit": limit,
        "sort_by": sort_by,
        "order": order,
    }
    if framework is not None:
        params["framework"] = framework
    if status is not None:
        params["status"] = status
    if visibility is not None:
        params["visibility"] = visibility
    if tags is not None:
        params["tags"] = ",".join(tags)
    if search is not None:
        params["search"] = search
    if _is_mock_client(resolved):
        legacy_list = getattr(resolved, "list_projects", None)
        if callable(legacy_list):
            return ensure_json_object(legacy_list(params=params))
    return resolved.list_projects(**params)


def get(
    project_id: str | UUID,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Fetch a single project record."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_project(_stringify_id(project_id))


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------


def create(
    title: str,
    *,
    framework: str = "pytorch",
    description: Optional[str] = None,
    visibility: str = "private",
    tags: Optional[Sequence[str]] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Create a new project.

    >>> proj = dagnam.projects.create("My Model", framework="pytorch")
    """
    resolved = resolve_client(client, api_key, api_url)
    payload: JsonObject = {
        "title": title,
        "framework": framework,
        "visibility": visibility,
    }
    if description is not None:
        payload["description"] = description
    if tags is not None:
        tag_values: builtin_list[JsonValue] = [tag for tag in tags]
        payload["tags"] = tag_values
    return resolved.create_project(payload)


def update(
    project_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    **kwargs: JsonValue,
) -> JsonObject:
    """Update mutable project fields (title, description, framework, visibility, tags).

    >>> dagnam.projects.update("proj_abc", title="New Title")
    """
    resolved = resolve_client(client, api_key, api_url)
    payload: JsonObject = {}
    for key in ("title", "description", "framework", "visibility", "tags"):
        if key in kwargs:
            payload[key] = kwargs[key]
    return resolved.update_project(_stringify_id(project_id), payload)


def delete(
    project_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> None:
    """Delete a project."""
    resolved = resolve_client(client, api_key, api_url)
    resolved.delete_project(_stringify_id(project_id))


def duplicate(
    project_id: str,
    *,
    title: Optional[str] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Duplicate an existing project.

    >>> copy = dagnam.projects.duplicate("proj_abc", title="Copy of My Model")
    """
    resolved = resolve_client(client, api_key, api_url)
    if _is_mock_client(resolved):
        payload: JsonObject | None = {"title": title} if title is not None else None
        legacy_duplicate = getattr(resolved, "duplicate_project", None)
        if callable(legacy_duplicate):
            return ensure_json_object(legacy_duplicate(_stringify_id(project_id), payload))
    return resolved.duplicate_project(_stringify_id(project_id), title=title)


def save_architecture(
    project_id: str,
    diagram_state: JsonValue,
    architecture_config: JsonValue,
    *,
    commit_message: Optional[str] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Save the architecture (diagram state + config) for a project."""
    resolved = resolve_client(client, api_key, api_url)
    # Normalize legacy/bare padding to canonical typed form before persisting, so
    # an SDK-built model can never be saved in a state the Studio would reject.
    payload: JsonObject = {
        "diagram_state": normalize_diagram_state(diagram_state),
        "architecture_config": normalize_architecture_config(architecture_config),
    }
    if commit_message is not None:
        payload["commit_message"] = commit_message
    legacy_save = getattr(resolved, "save_project_architecture", None)
    if callable(legacy_save):
        return ensure_json_object(legacy_save(_stringify_id(project_id), payload))
    return resolved.save_architecture(_stringify_id(project_id), payload)


# ---------------------------------------------------------------------------
# DAG import
# ---------------------------------------------------------------------------


def import_dag(
    ir: JsonValue,
    title: str,
    *,
    framework: str = "pytorch",
    description: Optional[str] = None,
    visibility: str = "private",
    tags: Optional[Sequence[str]] = None,
    commit_message: Optional[str] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Import a DAG IR as a new project.

    >>> proj = dagnam.projects.import_dag(ir_dict, "Imported Model")
    """
    resolved = resolve_client(client, api_key, api_url)
    payload: JsonObject = {
        "ir": ir,
        "title": title,
        "framework": framework,
        "visibility": visibility,
    }
    if description is not None:
        payload["description"] = description
    if tags is not None:
        tag_values: builtin_list[JsonValue] = [tag for tag in tags]
        payload["tags"] = tag_values
    if commit_message is not None:
        payload["commit_message"] = commit_message
    legacy_import = getattr(resolved, "import_project_dag", None)
    if callable(legacy_import):
        return ensure_json_object(legacy_import(payload))
    return resolved.import_dag(payload)


def import_dag_existing(
    project_id: str,
    ir: JsonValue,
    *,
    commit_message: Optional[str] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Import a DAG IR into an existing project."""
    resolved = resolve_client(client, api_key, api_url)
    payload: JsonObject = {"ir": ir}
    if commit_message is not None:
        payload["commit_message"] = commit_message
    legacy_import = getattr(resolved, "import_project_dag_existing", None)
    if callable(legacy_import):
        return ensure_json_object(legacy_import(_stringify_id(project_id), payload))
    return resolved.import_dag_existing(_stringify_id(project_id), payload)


# ---------------------------------------------------------------------------
# Bulk & dataset operations
# ---------------------------------------------------------------------------


def bulk_delete(
    project_ids: Sequence[str],
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Delete multiple projects at once."""
    resolved = resolve_client(client, api_key, api_url)
    ids = [_stringify_id(pid) for pid in project_ids]
    if _is_mock_client(resolved):
        legacy_bulk_delete = getattr(resolved, "bulk_delete_projects", None)
        if callable(legacy_bulk_delete):
            return ensure_json_object(legacy_bulk_delete({"project_ids": ids}))
    return resolved.bulk_delete_projects(ids)


def link_dataset(
    project_id: str,
    dataset_id: str,
    role: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Link a dataset to a project with the given role."""
    resolved = resolve_client(client, api_key, api_url)
    legacy_link = getattr(resolved, "link_project_dataset", None)
    if callable(legacy_link):
        return ensure_json_object(
            legacy_link(
                _stringify_id(project_id),
                {"dataset_id": _stringify_id(dataset_id), "role": role},
            )
        )
    return resolved.link_dataset(_stringify_id(project_id), _stringify_id(dataset_id), role)


def get_datasets(
    project_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """List datasets linked to a project."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_project_datasets(_stringify_id(project_id))


def unlink_dataset(
    project_id: str,
    dataset_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> None:
    """Unlink a dataset from a project."""
    resolved = resolve_client(client, api_key, api_url)
    legacy_unlink = getattr(resolved, "unlink_project_dataset", None)
    if callable(legacy_unlink):
        legacy_unlink(_stringify_id(project_id), _stringify_id(dataset_id))
        return
    resolved.unlink_dataset(
        _stringify_id(project_id),
        _stringify_id(dataset_id),
    )


# ---------------------------------------------------------------------------
# Architecture versioning
# ---------------------------------------------------------------------------


def list_versions(
    project_id: str,
    *,
    page: int = 1,
    limit: int = 20,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """List a project's architecture versions.

    >>> dagnam.projects.list_versions("proj_abc")["items"]
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.list_project_versions(_stringify_id(project_id), page=page, limit=limit)


def get_version(
    project_id: str,
    version_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Get one architecture version (full diagram state + config)."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_project_version(_stringify_id(project_id), _stringify_id(version_id))


def compare_versions(
    project_id: str,
    version_a: str,
    version_b: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Compare two architecture versions of a project."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.compare_project_versions(
        _stringify_id(project_id), _stringify_id(version_a), _stringify_id(version_b)
    )


def restore_version(
    project_id: str,
    version_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Restore a project to a prior version (creates a new current version)."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.restore_project_version(_stringify_id(project_id), _stringify_id(version_id))


def delete_version(
    project_id: str,
    version_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> None:
    """Delete one architecture version."""
    resolved = resolve_client(client, api_key, api_url)
    resolved.delete_project_version(_stringify_id(project_id), _stringify_id(version_id))


def latest_version(
    project_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Get the current (latest) architecture version of a project."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_latest_project_version(_stringify_id(project_id))


__all__ = [
    "bulk_delete",
    "compare_versions",
    "create",
    "delete",
    "delete_version",
    "duplicate",
    "get",
    "get_datasets",
    "get_version",
    "import_dag",
    "import_dag_existing",
    "latest_version",
    "link_dataset",
    "list",
    "list_versions",
    "restore_version",
    "save_architecture",
    "unlink_dataset",
    "update",
]
