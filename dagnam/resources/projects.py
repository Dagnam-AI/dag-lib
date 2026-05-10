"""Project management — sync SDK surface.

Wraps the ``/api/v1/projects/*`` routes on top of
:class:`dagnam.client.DagnamClient`.

The module exposes plain functions (``dagnam.projects.list(...)``) to
match the Phase 3 style (``dagnam.inference``, ``dagnam.deployments``).
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from dagnam._core.client import DagnamClient
from dagnam._core.resolver import resolve_client


def _stringify_id(value: Any) -> str:
    if isinstance(value, UUID):
        return str(value)
    return str(value)


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
    tags: Optional[list[str]] = None,
    search: Optional[str] = None,
    sort_by: str = "updated_at",
    order: str = "desc",
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> dict:
    """List projects visible to the current credential.

    >>> dagnam.projects.list(framework="pytorch")["items"]
    """
    resolved = resolve_client(client, api_key, api_url)
    params: dict[str, Any] = {
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
    return resolved.list_projects(params=params)


def get(
    project_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> dict:
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
    tags: Optional[list[str]] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> dict:
    """Create a new project.

    >>> proj = dagnam.projects.create("My Model", framework="pytorch")
    """
    resolved = resolve_client(client, api_key, api_url)
    payload: dict[str, Any] = {
        "title": title,
        "framework": framework,
        "visibility": visibility,
    }
    if description is not None:
        payload["description"] = description
    if tags is not None:
        payload["tags"] = tags
    return resolved.create_project(payload)


def update(
    project_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Update mutable project fields (title, description, framework, visibility, tags).

    >>> dagnam.projects.update("proj_abc", title="New Title")
    """
    resolved = resolve_client(client, api_key, api_url)
    payload: dict[str, Any] = {}
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
) -> dict:
    """Duplicate an existing project.

    >>> copy = dagnam.projects.duplicate("proj_abc", title="Copy of My Model")
    """
    resolved = resolve_client(client, api_key, api_url)
    payload = {"title": title} if title is not None else None
    return resolved.duplicate_project(_stringify_id(project_id), payload)


def save_architecture(
    project_id: str,
    diagram_state: Any,
    architecture_config: Any,
    *,
    commit_message: Optional[str] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> dict:
    """Save the architecture (diagram state + config) for a project."""
    resolved = resolve_client(client, api_key, api_url)
    payload: dict[str, Any] = {
        "diagram_state": diagram_state,
        "architecture_config": architecture_config,
    }
    if commit_message is not None:
        payload["commit_message"] = commit_message
    return resolved.save_project_architecture(_stringify_id(project_id), payload)


# ---------------------------------------------------------------------------
# DAG import
# ---------------------------------------------------------------------------


def import_dag(
    ir: Any,
    title: str,
    *,
    framework: str = "pytorch",
    description: Optional[str] = None,
    visibility: str = "private",
    tags: Optional[list[str]] = None,
    commit_message: Optional[str] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> dict:
    """Import a DAG IR as a new project.

    >>> proj = dagnam.projects.import_dag(ir_dict, "Imported Model")
    """
    resolved = resolve_client(client, api_key, api_url)
    payload: dict[str, Any] = {
        "ir": ir,
        "title": title,
        "framework": framework,
        "visibility": visibility,
    }
    if description is not None:
        payload["description"] = description
    if tags is not None:
        payload["tags"] = tags
    if commit_message is not None:
        payload["commit_message"] = commit_message
    return resolved.import_project_dag(payload)


def import_dag_existing(
    project_id: str,
    ir: Any,
    *,
    commit_message: Optional[str] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> dict:
    """Import a DAG IR into an existing project."""
    resolved = resolve_client(client, api_key, api_url)
    payload: dict[str, Any] = {"ir": ir}
    if commit_message is not None:
        payload["commit_message"] = commit_message
    return resolved.import_project_dag_existing(_stringify_id(project_id), payload)


# ---------------------------------------------------------------------------
# Bulk & dataset operations
# ---------------------------------------------------------------------------


def bulk_delete(
    project_ids: list[str],
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> dict:
    """Delete multiple projects at once."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.bulk_delete_projects(
        {"project_ids": [_stringify_id(pid) for pid in project_ids]}
    )


def link_dataset(
    project_id: str,
    dataset_id: str,
    role: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> dict:
    """Link a dataset to a project with the given role."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.link_project_dataset(
        _stringify_id(project_id),
        {"dataset_id": _stringify_id(dataset_id), "role": role},
    )


def get_datasets(
    project_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> dict:
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
    resolved.unlink_project_dataset(
        _stringify_id(project_id),
        _stringify_id(dataset_id),
    )


__all__ = [
    "bulk_delete",
    "create",
    "delete",
    "duplicate",
    "get",
    "get_datasets",
    "import_dag",
    "import_dag_existing",
    "link_dataset",
    "list",
    "save_architecture",
    "unlink_dataset",
    "update",
]
