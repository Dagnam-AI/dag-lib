"""Dataset upload — sync SDK surface.

Provides file and URL-based dataset upload via
:class:`dagnam.client.DagnamClient`.  URL uploads return a
:class:`~dagnam.lro.LongRunningOperation` that polls until the
server-side ingestion completes.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from dagnam._core.client import DagnamClient
from dagnam._core.lro import LongRunningOperation
from dagnam._core.resolver import resolve_client

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
    progress_cb: Optional[Callable[[int, int], Any]] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> dict:
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
    task_id = initial["task_id"]
    return LongRunningOperation(
        poll=lambda: resolved.get_dataset_task_status(task_id),
        success_states=_SUCCESS_STATES,
        failure_states=_FAILURE_STATES,
        state_key="status",
        error_key="error_message",
        name=f"datasets.upload_from_url({task_id})",
        initial=initial,
    )


__all__ = [
    "upload",
    "upload_from_url",
]
