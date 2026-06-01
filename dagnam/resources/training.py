"""SSE client for live training job events.

This module is a thin adapter over the shared reconnect engine so Python
callers can iterate training events just like the frontend EventSource.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Optional
from uuid import UUID

from dagnam._core.client import DagnamClient
from dagnam._core.resolver import resolve_client
from dagnam._core.sse import (
    DEFAULT_MAX_RECONNECTS,
    TERMINAL_TRAINING_EVENTS,
    SSEEvent,
    iter_with_reconnect,
)
from dagnam._types import JsonObject, QueryValue

TrainingEvent = SSEEvent


def stream_training(
    job_id: str,
    *,
    last_event_id: Optional[str] = None,
    include_heartbeats: bool = False,
    max_reconnects: int = DEFAULT_MAX_RECONNECTS,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> Iterator[TrainingEvent]:
    """Yield training events for ``job_id`` until the job terminates."""
    resolved = resolve_client(client, api_key, api_url)

    def _open(cursor: Optional[str]):
        return resolved.open_training_stream(job_id, last_event_id=cursor)

    yield from iter_with_reconnect(
        _open,
        terminal_events=TERMINAL_TRAINING_EVENTS,
        include_heartbeats=include_heartbeats,
        max_reconnects=max_reconnects,
        resource_label=f"Training stream for job {job_id}",
        last_event_id=last_event_id,
    )


# ---------------------------------------------------------------------------
# Training-job lifecycle
# ---------------------------------------------------------------------------


def _stringify_id(value: object) -> str:
    return str(value)


def create_training_job(
    project_id: str | UUID,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    optimizer: str,
    loss_function: str,
    training_dataset_id: str | UUID,
    framework: str = "pytorch",
    validation_dataset_id: Optional[str | UUID] = None,
    test_dataset_id: Optional[str | UUID] = None,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    config_overrides: Optional[JsonObject] = None,
    max_duration_seconds: Optional[int] = None,
    confirm_resource_warning: bool = False,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Create a platform training job and return the created job record.

    The required hyperparameters and dataset split are assembled into the
    backend ``TrainingConfig``. Pass ``config_overrides`` to set advanced
    ``TrainingConfig`` fields (``lr_scheduler``, ``regularization``,
    ``compute_config``, ``logging_config``, ``resume_*``, etc.); they are merged on
    top of the built config. For total control over the request body, call
    :meth:`DagnamClient.create_training_job` with a raw payload instead.

    >>> job = dagnam.create_training_job(
    ...     "proj_abc",
    ...     epochs=2, batch_size=32, learning_rate=1e-3,
    ...     optimizer="adam", loss_function="cross_entropy",
    ...     training_dataset_id="ds_123",
    ... )
    """
    resolved = resolve_client(client, api_key, api_url)

    dataset_config: JsonObject = {
        "training_dataset_id": _stringify_id(training_dataset_id),
        "train_split": train_split,
        "val_split": val_split,
        "test_split": test_split,
    }
    if validation_dataset_id is not None:
        dataset_config["validation_dataset_id"] = _stringify_id(validation_dataset_id)
    if test_dataset_id is not None:
        dataset_config["test_dataset_id"] = _stringify_id(test_dataset_id)

    config: JsonObject = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "loss_function": loss_function,
        "dataset_config": dataset_config,
    }
    if config_overrides:
        config.update(config_overrides)

    payload: JsonObject = {
        "project_id": _stringify_id(project_id),
        "framework": framework,
        "config": config,
        "confirm_resource_warning": confirm_resource_warning,
    }
    if max_duration_seconds is not None:
        payload["max_duration_seconds"] = max_duration_seconds

    return resolved.create_training_job(payload)


def get_training_job(
    job_id: str | UUID,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Fetch a single training job record."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_training_job(_stringify_id(job_id))


def list_training_jobs(
    *,
    page: int = 1,
    limit: int = 20,
    status: Optional[str | Sequence[str]] = None,
    project_id: Optional[str | UUID] = None,
    sort_by: str = "created_at",
    order: str = "desc",
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """List training jobs visible to the current credential.

    ``status`` accepts a single status or a sequence; it is sent to the backend
    as the comma-separated ``status_filter`` query parameter.

    >>> dagnam.list_training_jobs(status=["running", "completed"])["items"]
    """
    resolved = resolve_client(client, api_key, api_url)
    params: dict[str, QueryValue] = {
        "page": page,
        "limit": limit,
        "sort_by": sort_by,
        "order": order,
    }
    if status is not None:
        params["status_filter"] = status if isinstance(status, str) else ",".join(status)
    if project_id is not None:
        params["project_id"] = _stringify_id(project_id)
    return resolved.list_training_jobs(**params)


def cancel_training_job(
    job_id: str | UUID,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Cancel a non-terminal training job."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.cancel_training_job(_stringify_id(job_id))


def delete_training_jobs(
    job_ids: Sequence[str | UUID],
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Delete one or more training jobs (1-100) in a single request."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.bulk_delete_training_jobs([_stringify_id(jid) for jid in job_ids])


def training_logs(
    job_id: str | UUID,
    *,
    log_level: Optional[str] = None,
    source: Optional[str] = None,
    page: int = 1,
    limit: int = 100,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Fetch historical logs for one training job."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_training_logs(
        _stringify_id(job_id),
        log_level=log_level,
        source=source,
        page=page,
        limit=limit,
    )


def training_metrics(
    job_id: str | UUID,
    *,
    metric_type: Optional[str] = None,
    epoch_start: Optional[int] = None,
    epoch_end: Optional[int] = None,
    epoch_summary: bool = False,
    page: int = 1,
    limit: int = 100,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Fetch historical metrics for one training job."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_training_metrics(
        _stringify_id(job_id),
        metric_type=metric_type,
        epoch_start=epoch_start,
        epoch_end=epoch_end,
        epoch_summary=epoch_summary,
        page=page,
        limit=limit,
    )


def training_metrics_summary(
    job_id: str | UUID,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Fetch aggregate historical metrics for one training job."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_training_metrics_summary(_stringify_id(job_id))


__all__ = [
    "TrainingEvent",
    "cancel_training_job",
    "create_training_job",
    "delete_training_jobs",
    "get_training_job",
    "list_training_jobs",
    "stream_training",
    "training_logs",
    "training_metrics",
    "training_metrics_summary",
]
