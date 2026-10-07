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
from dagnam._types import JsonArray, JsonMapping, JsonObject
from dagnam.resources.inference import inference_stream

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


def revisions(
    deployment_id: str,
    *,
    page: int = 1,
    limit: int = 50,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonArray:
    """Fetch a deployment's revision history, newest first."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_deployment_revisions(_stringify_id(deployment_id), page=page, limit=limit)


def create_revision(
    deployment_id: str,
    *,
    model_version_id: str,
    capacity_mode: str = "serverless",
    capacity_policy: Optional[JsonMapping] = None,
    region: Optional[str] = None,
    engine_override: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Roll a deployment to a new model version by creating a revision.

    Returns the revision object as created (``id``, ``revision_number``,
    ``status``, ``is_active``, ...) — a plain object, not an LRO. The platform
    activates the revision asynchronously, so poll :func:`revisions` until the
    new one reports ``is_active``. ``capacity_policy``, ``region`` and
    ``engine_override`` are only sent when given: without a ``capacity_policy``
    the platform keeps the deployment's current one (including a
    :func:`set_warm` pin), or applies its own default when nothing is live yet. An ``idempotency_key`` makes
    a retried call replay the same revision; one is minted when omitted.
    """
    payload: JsonObject = {"model_version_id": model_version_id, "capacity_mode": capacity_mode}
    if capacity_policy is not None:
        payload["capacity_policy"] = _json_object_from_mapping(capacity_policy)
    if region is not None:
        payload["region"] = region
    if engine_override is not None:
        payload["engine_override"] = engine_override
    resolved = resolve_client(client, api_key, api_url)
    return resolved.create_deployment_revision(
        _stringify_id(deployment_id), payload, idempotency_key=idempotency_key
    )


def set_warm(
    deployment_id: str,
    warm: bool,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Keep a deployment warm (``True``) or let it scale to zero when idle (``False``).

    Warm pins one GPU container continuously, so it costs for as long as it is
    on. It is only available on plans that include remote GPU. A deployment
    that is not serving yet answers 409 (:class:`DeploymentStateError`), and one
    that is not yours answers 404 (:class:`DeploymentNotFoundError`). Returns
    the updated capacity.

    >>> dagnam.set_warm("dep_abc123", True)
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.set_deployment_warm(_stringify_id(deployment_id), warm)


# ---------------------------------------------------------------------------
# Write operations — LRO on lifecycle transitions
# ---------------------------------------------------------------------------


def deploy_model_version(
    model_version_id: str,
    *,
    name: Optional[str] = None,
    project_id: Optional[str] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Deploy a model version from the registry and return the new deployment.

    One call creates the deployment and its first serving revision. The result
    carries the deployment's ``api_key`` exactly once, so store it now. Serving
    starts asynchronously: poll :func:`revisions` until the newest revision
    reports ``is_active``. ``name`` defaults to the model name and version, and
    ``project_id`` to the model entry's project.

    >>> dep = dagnam.deployments.deploy_model_version("mv_123", name="support-bot")
    >>> key = dep["api_key"]
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.deploy_model_version(model_version_id, name=name, project_id=project_id)


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

    The deployment serves no requests until it has a revision: create one with
    :func:`create_revision`, or use :func:`deploy_model_version`, which creates
    the deployment and its first revision in one call.

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
    name: str,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Rename a deployment. Serving capacity is managed by the platform."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.update_deployment(_stringify_id(deployment_id), {"name": name})


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


# ---------------------------------------------------------------------------
# Planning — validate config, list platforms
# ---------------------------------------------------------------------------


def validate(
    *,
    name: str,
    project_id: str,
    checkpoint_path: str,
    platform: str,
    deployment_type: str,
    instance_type: str,
    num_instances: int = 1,
    auto_scaling_enabled: bool = False,
    min_instances: Optional[int] = None,
    max_instances: Optional[int] = None,
    region: Optional[str] = None,
    config: Optional[JsonMapping] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Validate a deployment configuration without creating it."""
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
    if min_instances is not None:
        payload["min_instances"] = min_instances
    if max_instances is not None:
        payload["max_instances"] = max_instances
    if region is not None:
        payload["region"] = region
    if config is not None:
        payload["config"] = _json_object_from_mapping(config)
    return resolved.validate_deployment(payload)


def platforms(
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonArray:
    """List available serving platforms and their capabilities."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.list_deployment_platforms()


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

    # open_deployment_stream mints a fresh stream token per call, so iter_with_reconnect re-mints on every reconnect.
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


def predict_stream(
    deployment_id: str,
    inputs: JsonObject,
    *,
    include_heartbeats: bool = False,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> Iterator[SSEEvent]:
    """Stream a prediction from this deployment (alias of ``dagnam.inference_stream``)."""
    return inference_stream(
        deployment_id,
        inputs,
        include_heartbeats=include_heartbeats,
        client=client,
        api_key=api_key,
        api_url=api_url,
    )


__all__ = [
    "create",
    "create_revision",
    "delete",
    "deploy_model_version",
    "get",
    "health",
    "list",
    "logs",
    "metrics",
    "pause",
    "platforms",
    "predict_stream",
    "resume",
    "revisions",
    "set_warm",
    "stream_events",
    "update",
    "validate",
]
