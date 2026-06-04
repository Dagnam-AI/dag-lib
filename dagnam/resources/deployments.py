"""Deployment management — sync SDK surface.

Wraps the ``/api/v1/deployments/*`` routes on top of
:class:`dagnam.client.DagnamClient` and returns
:class:`~dagnam.lro.LongRunningOperation` for lifecycle actions whose
effects land asynchronously on the cluster.

The module exposes plain functions (``dagnam.deployments.create(...)``) to
match the Phase 3 style (``dagnam.inference``, ``dagnam.stream_training``).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Optional
from uuid import UUID

from dagnam._core.client import DagnamClient
from dagnam._core.lro import LongRunningOperation
from dagnam._core.resolver import resolve_client
from dagnam._core.sse import TERMINAL_DEPLOYMENT_EVENTS, SSEEvent, iter_with_reconnect
from dagnam._types import JsonMapping, JsonObject

# Terminal status values returned by the deployment status enum.
_ACTIVE_STATES = frozenset({"running"})
_PAUSED_STATES = frozenset({"paused"})
_FAILED_STATES = frozenset({"failed"})


def _stringify_id(value: object) -> str:
    if isinstance(value, UUID):
        return str(value)
    return str(value)


def _json_object_from_mapping(value: JsonMapping) -> JsonObject:
    return {str(key): item for key, item in value.items()}


def _lifecycle_lro(
    client: DagnamClient,
    deployment_id: str,
    initial: JsonMapping,
    *,
    success_states: frozenset[str],
    name: str,
) -> LongRunningOperation:
    """Build an LRO that polls ``GET /deployments/{id}`` until terminal."""
    return LongRunningOperation(
        poll=lambda: client.get_deployment(deployment_id),
        success_states=success_states,
        failure_states=_FAILED_STATES,
        state_key="status",
        error_key="error_message",
        name=f"{name}({deployment_id})",
        initial=initial,
    )


# ---------------------------------------------------------------------------
# Read operations — no LRO needed
# ---------------------------------------------------------------------------


def list(
    *,
    page: int = 1,
    limit: int = 20,
    status: Optional[str] = None,
    platform: Optional[str] = None,
    project_id: Optional[str] = None,
    search: Optional[str] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject | str | None:
    """List deployments visible to the current credential.

    >>> dagnam.deployments.list(status="running")["items"]
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.list_deployments(
        page=page,
        limit=limit,
        status_filter=status,
        platform=platform,
        project_id=project_id,
        search=search,
    )


def get(
    deployment_id: str | UUID,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Fetch a single deployment record."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_deployment(_stringify_id(deployment_id))


def health(
    deployment_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Return the platform-side health row for a deployment."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_deployment_health_full(_stringify_id(deployment_id))


def metrics(
    deployment_id: str,
    *,
    time_range: str = "24h",
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Fetch aggregated deployment metrics for the given time range."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_deployment_metrics(_stringify_id(deployment_id), time_range=time_range)


def logs(
    deployment_id: str,
    *,
    level: Optional[str] = None,
    search: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    page: int = 1,
    limit: int = 100,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Fetch paginated deployment logs."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_deployment_logs(
        _stringify_id(deployment_id),
        level=level,
        search=search,
        start_time=start_time,
        end_time=end_time,
        page=page,
        limit=limit,
    )


# ---------------------------------------------------------------------------
# Write operations — LRO on lifecycle transitions
# ---------------------------------------------------------------------------


def create(
    *,
    name: str,
    project_id: str,
    checkpoint_path: str,
    platform: str,
    deployment_type: str,
    instance_type: str,
    num_instances: int = 1,
    training_job_id: Optional[str] = None,
    checkpoint_id: Optional[str] = None,
    auto_scaling_enabled: bool = False,
    min_instances: Optional[int] = None,
    max_instances: Optional[int] = None,
    region: Optional[str] = None,
    config: Optional[JsonMapping] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> LongRunningOperation:
    """Queue a new deployment and return an LRO.

    The backend responds 202 Accepted immediately with ``status=deploying``;
    the returned LRO polls until the deployment reaches ``running`` or
    ``failed``.  Fire-and-forget callers can inspect ``op.initial()``
    without ever calling ``wait()``.

    >>> op = dagnam.deployments.create(
    ...     name="my-dep",
    ...     project_id="...",
    ...     checkpoint_path="/ckpt.pt",
    ...     platform="fastapi",
    ...     deployment_type="text",
    ...     instance_type="t3.medium",
    ... )
    >>> dep = op.wait(timeout=300).result()
    """
    resolved = resolve_client(client, api_key, api_url)
    payload: JsonObject = {
        "name": name,
        "project_id": _stringify_id(project_id),
        "checkpoint_path": checkpoint_path,
        "platform": platform,
        "deployment_type": deployment_type,
        "instance_type": instance_type,
        "num_instances": num_instances,
        "auto_scaling_enabled": auto_scaling_enabled,
    }
    if training_job_id is not None:
        payload["training_job_id"] = _stringify_id(training_job_id)
    if checkpoint_id is not None:
        payload["checkpoint_id"] = _stringify_id(checkpoint_id)
    if min_instances is not None:
        payload["min_instances"] = min_instances
    if max_instances is not None:
        payload["max_instances"] = max_instances
    if region is not None:
        payload["region"] = region
    if config is not None:
        payload["config"] = _json_object_from_mapping(config)

    initial = resolved.create_deployment(payload)
    deployment_id = _stringify_id(initial["id"])
    return _lifecycle_lro(
        resolved,
        deployment_id,
        initial,
        success_states=_ACTIVE_STATES,
        name="deployments.create",
    )


def update(
    deployment_id: str,
    *,
    name: Optional[str] = None,
    instance_type: Optional[str] = None,
    num_instances: Optional[int] = None,
    auto_scaling_enabled: Optional[bool] = None,
    min_instances: Optional[int] = None,
    max_instances: Optional[int] = None,
    config: Optional[JsonMapping] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Update mutable deployment fields (non-lifecycle)."""
    resolved = resolve_client(client, api_key, api_url)
    payload: JsonObject = {}
    for key, value in (
        ("name", name),
        ("instance_type", instance_type),
        ("num_instances", num_instances),
        ("auto_scaling_enabled", auto_scaling_enabled),
        ("min_instances", min_instances),
        ("max_instances", max_instances),
    ):
        if value is not None:
            payload[key] = value
    if config is not None:
        payload["config"] = _json_object_from_mapping(config)
    return resolved.update_deployment(_stringify_id(deployment_id), payload)


def delete(
    deployment_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> Optional[JsonObject]:
    """Soft-delete a deployment."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.delete_deployment(_stringify_id(deployment_id))


def pause(
    deployment_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> LongRunningOperation:
    """Pause a running deployment (returns an LRO resolving to ``paused``)."""
    resolved = resolve_client(client, api_key, api_url)
    dep_id = _stringify_id(deployment_id)
    initial = resolved.pause_deployment(dep_id)
    return _lifecycle_lro(
        resolved,
        dep_id,
        initial,
        success_states=_PAUSED_STATES,
        name="deployments.pause",
    )


def resume(
    deployment_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> LongRunningOperation:
    """Resume a paused deployment (returns an LRO resolving to ``running``)."""
    resolved = resolve_client(client, api_key, api_url)
    dep_id = _stringify_id(deployment_id)
    initial = resolved.resume_deployment(dep_id)
    return _lifecycle_lro(
        resolved,
        dep_id,
        initial,
        success_states=_ACTIVE_STATES,
        name="deployments.resume",
    )


def scale(
    deployment_id: str,
    num_instances: int,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> LongRunningOperation:
    """Scale a deployment's instance count (returns an LRO)."""
    resolved = resolve_client(client, api_key, api_url)
    dep_id = _stringify_id(deployment_id)
    initial = resolved.scale_deployment(dep_id, num_instances=num_instances)
    return _lifecycle_lro(
        resolved,
        dep_id,
        initial,
        success_states=_ACTIVE_STATES,
        name="deployments.scale",
    )


def rollback(
    deployment_id: str,
    checkpoint_path: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> LongRunningOperation:
    """Roll the deployment back to ``checkpoint_path`` (returns an LRO)."""
    resolved = resolve_client(client, api_key, api_url)
    dep_id = _stringify_id(deployment_id)
    initial = resolved.rollback_deployment(dep_id, checkpoint_path=checkpoint_path)
    return _lifecycle_lro(
        resolved,
        dep_id,
        initial,
        success_states=_ACTIVE_STATES,
        name="deployments.rollback",
    )


# ---------------------------------------------------------------------------
# SSE stream
# ---------------------------------------------------------------------------


def stream_events(
    deployment_id: str,
    *,
    last_event_id: Optional[str] = None,
    include_heartbeats: bool = False,
    max_reconnects: int = 5,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> Iterator[SSEEvent]:
    """Yield deployment lifecycle events until a terminal state.

    Auto-reconnects on transport errors using ``Last-Event-ID`` (up to
    ``max_reconnects`` with exponential backoff), matching the Phase 3
    training-stream behaviour.

    >>> for ev in dagnam.deployments.stream_events("dep_abc"):
    ...     print(ev.event, ev.data)
    """
    resolved = resolve_client(client, api_key, api_url)
    dep_id = _stringify_id(deployment_id)

    def _open(cursor: Optional[str]):
        return resolved.open_deployment_stream(dep_id, last_event_id=cursor)

    return iter_with_reconnect(
        _open,
        terminal_events=TERMINAL_DEPLOYMENT_EVENTS,
        include_heartbeats=include_heartbeats,
        max_reconnects=max_reconnects,
        resource_label=f"Deployment stream for {dep_id}",
        last_event_id=last_event_id,
    )


__all__ = [
    "create",
    "delete",
    "get",
    "health",
    "list",
    "logs",
    "metrics",
    "pause",
    "resume",
    "rollback",
    "scale",
    "stream_events",
    "update",
]
