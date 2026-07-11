"""Dataset upload — sync SDK surface.

Provides file and URL-based dataset upload via
:class:`dagnam.client.DagnamClient`.  URL uploads return a
:class:`~dagnam.lro.LongRunningOperation` that polls until the
server-side ingestion completes.
"""

from __future__ import annotations

from typing import Callable, Optional

from dagnam._core.client import DagnamClient
from dagnam._core.lro import LongRunningOperation
from dagnam._core.resolver import resolve_client
from dagnam._types import JsonObject

_SUCCESS_STATES = frozenset({"completed", "ready"})
_FAILURE_STATES = frozenset({"failed"})


def upload(
    path: str,
    name: str,
    dataset_type: str,
    format: str,
    *,
    description: Optional[str] = None,
    visibility: str = "private",
    license: Optional[str] = None,
    progress_cb: Optional[Callable[[int, int], object]] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Upload a local dataset file.

    >>> result = dagnam.datasets.upload(
    ...     "data.csv",
    ...     name="my-dataset",
    ...     dataset_type="tabular",
    ...     format="csv",
    ... )
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.upload_dataset(
        file_path=path,
        name=name,
        dataset_type=dataset_type,
        format=format,
        description=description,
        visibility=visibility,
        license=license,
        progress_cb=progress_cb,
    )


def upload_from_url(
    url: str,
    name: str,
    dataset_type: str,
    format: str,
    *,
    description: Optional[str] = None,
    visibility: str = "private",
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> LongRunningOperation:
    """Start a server-side dataset import from a URL and return an LRO.

    >>> op = dagnam.datasets.upload_from_url(
    ...     "https://example.com/data.parquet",
    ...     name="remote-ds",
    ...     dataset_type="tabular",
    ...     format="parquet",
    ... )
    >>> ds = op.wait(timeout=600).result()
    """
    resolved = resolve_client(client, api_key, api_url)
    initial = resolved.upload_dataset_from_url(
        url=url,
        name=name,
        dataset_type=dataset_type,
        format=format,
        description=description,
        visibility=visibility,
    )
    task_id_value = initial.get("task_id")
    if not isinstance(task_id_value, str):
        raise ValueError("Dataset upload response did not include a string task_id")
    return LongRunningOperation(
        poll=lambda: resolved.get_dataset_task_status(task_id_value),
        success_states=_SUCCESS_STATES,
        failure_states=_FAILURE_STATES,
        state_key="status",
        error_key="error_message",
        name=f"datasets.upload_from_url({task_id_value})",
        initial=initial,
    )


def preview_dataset(
    dataset_id: str,
    rows: int = 10,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Preview a dataset's samples and statistics.

    >>> dagnam.preview_dataset("ds-1", rows=5)["samples"]
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.preview_dataset(dataset_id, rows=rows)


def update_dataset(
    dataset_id: str,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    visibility: Optional[str] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Update a dataset's name, description, and/or visibility.

    At least one of ``name``/``description``/``visibility`` must be provided.

    >>> dagnam.update_dataset("ds-1", name="renamed")
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.update_dataset(
        dataset_id, name=name, description=description, visibility=visibility
    )


def delete_dataset(
    dataset_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> None:
    """Delete a dataset permanently."""
    resolved = resolve_client(client, api_key, api_url)
    resolved.delete_dataset(dataset_id)


def update_dataset_roles(
    dataset_id: str,
    column_roles: dict[str, str],
    task_type_hint: Optional[str] = None,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Set a dataset's per-column roles (and an optional task-type hint).

    >>> dagnam.update_dataset_roles("ds-1", {"species": "target"})
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.update_dataset_roles(dataset_id, column_roles, task_type_hint=task_type_hint)


__all__ = [
    "delete_dataset",
    "preview_dataset",
    "update_dataset",
    "update_dataset_roles",
    "upload",
    "upload_from_url",
]
