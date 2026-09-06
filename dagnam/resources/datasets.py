"""Dataset upload — sync SDK surface.

Provides file and URL-based dataset upload via
:class:`dagnam.client.DagnamClient`.  URL uploads return a
:class:`~dagnam.lro.LongRunningOperation` that polls until the
server-side ingestion completes.
"""

from __future__ import annotations

from builtins import list as builtin_list
from collections.abc import Mapping, Sequence
from typing import Callable, Optional

from dagnam._core.client import DagnamClient
from dagnam._core.lro import LongRunningOperation
from dagnam._core.resolver import resolve_client
from dagnam._types import JsonObject

# ``GET /api/v1/datasets/tasks/{task_id}`` reports the raw task-queue status
# ("SUCCESS"/"FAILURE"/"REVOKED"/...) under ``status`` and a lower-cased
# rendering ("completed"/"failed"/"cancelled"/...) under ``state``. Both
# spellings are accepted so the poll terminates on either shape.
_SUCCESS_STATES = frozenset({"completed", "ready", "success", "SUCCESS"})
_FAILURE_STATES = frozenset({"failed", "failure", "FAILURE", "cancelled", "revoked", "REVOKED"})
# The failure detail is ``error``; ``error_message`` is kept as a fallback.
_ERROR_KEYS = ("error", "error_message")


def _task_lro(client: DagnamClient, initial: JsonObject, name: str) -> LongRunningOperation:
    """Wrap a 202 ``{"task_id": ...}`` response as an LRO over the task status route."""
    task_id_value = initial.get("task_id")
    if not isinstance(task_id_value, str):
        raise ValueError(f"datasets.{name} response did not include a string task_id")
    return LongRunningOperation(
        poll=lambda: client.get_dataset_task_status(task_id_value),
        success_states=_SUCCESS_STATES,
        failure_states=_FAILURE_STATES,
        state_key="status",
        error_key=_ERROR_KEYS,
        name=f"datasets.{name}({task_id_value})",
        initial=initial,
    )


def list(
    *,
    type: str = "all",
    search: Optional[str] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> builtin_list[JsonObject]:
    """List datasets available to the current credential.

    Mirrors the frontend dataset browser: delegates to the client
    ``list_datasets`` (``GET /api/v1/datasets/browse``). ``type`` filters by
    dataset kind (image/text/audio/video/tabular/custom/all); ``search`` filters
    by a name/description substring.

    >>> dagnam.datasets.list(type="tabular")
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.list_datasets(type=type, search=search)


def list_versions(
    dataset_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> builtin_list[JsonObject]:
    """List a dataset's versions (``GET /api/v1/datasets/{id}/versions``).

    >>> dagnam.datasets.list_versions("ds-1")
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.list_dataset_versions(dataset_id)


def list_system(
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> builtin_list[JsonObject]:
    """List the built-in system datasets (``GET /api/v1/datasets/system``).

    >>> dagnam.datasets.list_system()
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.list_system_datasets()


def get(
    dataset_id: str,
    *,
    version: Optional[str] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Fetch one dataset's metadata by id (``GET /api/v1/datasets/{id}/meta``).

    Mirrors the frontend picker reading a single dataset; delegates to the client
    ``get_dataset_meta``. Pass ``version`` to pin a specific dataset version.

    >>> dagnam.datasets.get("ds-1")["name"]
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_dataset_meta(dataset_id, version=version)


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
    return _task_lro(resolved, initial, "upload_from_url")


def create_explicit_splits(
    dataset_id: str,
    version_id: str,
    memberships: Mapping[str, Sequence[int]],
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> LongRunningOperation:
    """Split a dataset version by explicit row membership and return an LRO.

    ``memberships`` maps a split name to the row indices it owns, e.g.
    ``{"train": [0, 1, 2], "test": [3]}``. The server preflights the indices
    against the version's row count (a 422 surfaces immediately) and then
    materializes the splits in a task the LRO polls.
    """
    resolved = resolve_client(client, api_key, api_url)
    initial = resolved.create_explicit_splits(dataset_id, version_id, memberships)
    return _task_lro(resolved, initial, "create_explicit_splits")


def scan_pii(
    dataset_id: str,
    version_id: str,
    policy: Optional[Mapping[str, str]] = None,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> LongRunningOperation:
    """Scan a dataset version for PII and return an LRO over the scan task.

    With no ``policy`` (the default) the scan only reports. A policy maps a
    finding class to ``redact``, ``drop`` or ``ignore``; classes left out are
    ignored. The task result carries the findings plus ``pass_list`` — the
    classes that were looked for — so a clean report is never read as more
    than it is.
    """
    resolved = resolve_client(client, api_key, api_url)
    initial = resolved.scan_pii(dataset_id, version_id, policy)
    return _task_lro(resolved, initial, "scan_pii")


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
    "get",
    "list",
    "list_system",
    "list_versions",
    "preview_dataset",
    "update_dataset",
    "update_dataset_roles",
    "upload",
    "upload_from_url",
]
