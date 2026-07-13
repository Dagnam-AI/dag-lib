"""Wire-level coverage for the sync deployments client mixin."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import requests

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import (
    APIError,
    AuthError,
    DeploymentNotFoundError,
    DeploymentStateError,
)

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch, RequestsMocker

API = "https://api.test"


# ---------------------------------------------------------------- deployments client


def test_list_deployments_all_params(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/deployments", json={"items": []})
    client.list_deployments(
        page=2,
        limit=10,
        status_filter="active",
        platform="aws",
        project_id="p1",
        search="foo",
    )
    qs = rmock.last_request.qs
    assert qs["page"] == ["2"]
    assert qs["status"] == ["active"]
    assert qs["platform"] == ["aws"]
    assert qs["project_id"] == ["p1"]
    assert qs["search"] == ["foo"]


def test_list_deployments_minimal(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/deployments", json={"items": []})
    client.list_deployments()
    # Only page+limit
    assert set(rmock.last_request.qs.keys()) == {"page", "limit"}


def test_get_deployment(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/deployments/dep1", json={"id": "dep1"})
    assert client.get_deployment("dep1") == {"id": "dep1"}


def test_get_deployment_404(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/deployments/missing", status_code=404)
    with pytest.raises(DeploymentNotFoundError):
        client.get_deployment("missing")


def test_mint_deployment_stream_token_posts_with_bearer_header(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{API}/api/v1/deployments/dep1/stream-access-token", json={"token": "stream-t"})
    assert client.mint_deployment_stream_token("dep1") == "stream-t"
    assert rmock.last_request.headers["Authorization"] == "Bearer k"


def test_mint_deployment_stream_token_401_maps_auth_error(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{API}/api/v1/deployments/dep1/stream-access-token", status_code=401)
    with pytest.raises(AuthError):
        client.mint_deployment_stream_token("dep1")


def test_mint_deployment_stream_token_404_maps_not_found(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{API}/api/v1/deployments/missing/stream-access-token", status_code=404)
    with pytest.raises(DeploymentNotFoundError):
        client.mint_deployment_stream_token("missing")


def test_create_deployment(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/deployments", json={"id": "dep1"})
    assert client.create_deployment({"name": "x"}) == {"id": "dep1"}


def test_update_deployment(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.put(f"{API}/api/v1/deployments/dep1", json={"id": "dep1"})
    assert client.update_deployment("dep1", {}) == {"id": "dep1"}


def test_delete_deployment_empty_body(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.delete(f"{API}/api/v1/deployments/dep1", status_code=204, text="")
    assert client.delete_deployment("dep1") is None


def test_delete_deployment_returns_object(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.delete(f"{API}/api/v1/deployments/dep1", json={"deleted": True})
    assert client.delete_deployment("dep1") == {"deleted": True}


def test_get_deployment_logs_minimal(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/deployments/dep1/logs", json={"items": []})
    client.get_deployment_logs("dep1")
    assert set(rmock.last_request.qs.keys()) == {"page", "limit"}


def test_pause_resume(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/deployments/dep1/pause", json={"paused": True})
    rmock.post(f"{API}/api/v1/deployments/dep1/resume", json={"paused": False})
    assert client.pause_deployment("dep1") == {"paused": True}
    assert client.resume_deployment("dep1") == {"paused": False}


def test_scale_deployment(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.put(f"{API}/api/v1/deployments/dep1/scale", json={"instances": 5})
    client.scale_deployment("dep1", 5)
    assert rmock.last_request.qs == {"num_instances": ["5"]}


def test_rollback_deployment(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/deployments/dep1/rollback", json={"ok": True})
    client.rollback_deployment("dep1", "ckpt/path")
    assert rmock.last_request.qs == {"checkpoint_path": ["ckpt/path"]}


def test_get_deployment_metrics(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/deployments/dep1/metrics", json={"qps": 1})
    client.get_deployment_metrics("dep1", time_range="1h")
    assert rmock.last_request.qs == {"time_range": ["1h"]}


def test_get_deployment_logs_all_params(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/deployments/dep1/logs", json={"items": []})
    client.get_deployment_logs(
        "dep1",
        level="ERROR",
        search="oom",
        start_time="2025-01-01",
        end_time="2025-01-02",
        page=2,
        limit=50,
    )
    qs = rmock.last_request.qs
    for key in ("level", "search", "start_time", "end_time", "page", "limit"):
        assert key in qs


def test_get_deployment_health_full(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/deployments/dep1/health", json={"status": "healthy"})
    assert client.get_deployment_health_full("dep1") == {"status": "healthy"}


def test_open_deployment_stream_sets_headers(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/deployments/dep1/stream-access-token", json={"token": "stream-t"})
    rmock.get(f"{API}/api/v1/deployments/dep1/stream", text="data: x\n\n")
    resp = client.open_deployment_stream("dep1", last_event_id="cursor")
    assert resp.status_code == 200
    req = rmock.last_request
    assert req.headers["Accept"] == "text/event-stream"
    assert req.headers["Last-Event-ID"] == "cursor"
    assert req.qs == {"token": ["stream-t"]}
    assert "api_key" not in req.qs


def test_open_deployment_stream_without_cursor(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/deployments/dep1/stream-access-token", json={"token": "stream-t"})
    rmock.get(f"{API}/api/v1/deployments/dep1/stream", text="")
    resp = client.open_deployment_stream("dep1")
    assert "Last-Event-ID" not in resp.request.headers


def test_open_deployment_stream_connectionerror(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch
) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(client, "mint_deployment_stream_token", lambda _deployment_id: "stream-t")
    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.open_deployment_stream("dep1")


def test_open_deployment_stream_timeout(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch
) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(client, "mint_deployment_stream_token", lambda _deployment_id: "stream-t")
    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.open_deployment_stream("dep1")


def test_deployments_connectionerror_wrapped(client: DagnamClient, rmock: RequestsMocker) -> None:
    client._sleep = lambda _s: None
    rmock.get(f"{API}/api/v1/deployments", exc=requests.ConnectionError("nope"))
    with pytest.raises(APIError, match="Request failed"):
        client.list_deployments()


def test_deployments_timeout_wrapped(client: DagnamClient, rmock: RequestsMocker) -> None:
    client._sleep = lambda _s: None
    rmock.get(f"{API}/api/v1/deployments", exc=requests.Timeout("slow"))
    with pytest.raises(APIError, match="Request failed"):
        client.list_deployments()


def test_deployments_get_retries_transient(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(
        f"{API}/api/v1/deployments/dep1",
        [{"status_code": 503}, {"status_code": 200, "json": {"id": "dep1"}}],
    )
    client._sleep = lambda _s: None
    client._rng = lambda: 1.0
    assert client.get_deployment("dep1") == {"id": "dep1"}
    assert rmock.call_count == 2


def test_deployments_404_not_retried(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/deployments/dep1", status_code=404)
    client._sleep = lambda _s: None
    with pytest.raises(DeploymentNotFoundError):
        client.get_deployment("dep1")
    assert rmock.call_count == 1


def test_deployments_text_response(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/deployments", text="plain", headers={"Content-Type": "text/plain"})
    assert client.list_deployments() == "plain"


# ---------------------------------------------------------------- deployment planning


def test_estimate_cost(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(
        f"{API}/api/v1/deployments/estimate-cost",
        json={"hourly_cost": 0.5, "daily_cost": 12.0, "monthly_cost": 360.0},
    )
    out = client.estimate_cost({"platform": "fastapi", "instance_type": "cpu.small"})
    assert out["monthly_cost"] == 360.0
    assert rmock.last_request.json()["platform"] == "fastapi"


def test_validate_deployment(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/deployments/validate", json={"valid": True, "errors": []})
    assert client.validate_deployment({"name": "x"})["valid"] is True


def test_list_deployment_platforms(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(
        f"{API}/api/v1/deployments-platforms",
        json=[{"platform": "fastapi", "name": "FastAPI"}],
    )
    out = client.list_deployment_platforms()
    first = out[0]
    assert isinstance(first, dict)
    assert first["platform"] == "fastapi"


def test_list_deployment_platforms_rejects_non_array(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.get(f"{API}/api/v1/deployments-platforms", json={"not": "an array"})
    with pytest.raises(TypeError, match="Expected JSON array"):
        client.list_deployment_platforms()


def test_retry_deployment(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/deployments/d1/retry", json={"id": "d1", "status": "deploying"})
    assert client.retry_deployment("d1")["status"] == "deploying"


def test_retry_deployment_404(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/deployments/d1/retry", status_code=404)
    with pytest.raises(DeploymentNotFoundError):
        client.retry_deployment("d1")


def test_collect_deployment_metrics(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(
        f"{API}/api/v1/deployments/dep1/metrics/collect",
        json={"deployment_id": "dep1", "points_created": 60, "backfilled": True},
    )
    result = client.collect_deployment_metrics("dep1", backfill_minutes=120)
    assert result["points_created"] == 60
    assert rmock.last_request.qs["backfill_minutes"] == ["120"]


def test_collect_deployment_metrics_409_maps_state_error(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    from dagnam._core.exceptions import DeploymentStateError

    rmock.post(
        f"{API}/api/v1/deployments/dep1/metrics/collect", status_code=409, text="not running"
    )
    with pytest.raises(DeploymentStateError):
        client.collect_deployment_metrics("dep1")


def test_create_deployment_sends_idempotency_key(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{API}/api/v1/deployments", json={"id": "dep1"})
    client.create_deployment({"project_id": "p1"})
    assert rmock.last_request.headers.get("Idempotency-Key")


def test_create_deployment_retries_with_same_key(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(
        f"{API}/api/v1/deployments",
        [{"status_code": 503}, {"status_code": 201, "json": {"id": "dep1"}}],
    )
    client._sleep = lambda _s: None
    client._rng = lambda: 1.0
    client.create_deployment({"project_id": "p1"})
    keys = {req.headers.get("Idempotency-Key") for req in rmock.request_history}
    assert len(keys) == 1
    assert next(iter(keys))
    assert rmock.call_count == 2


def test_create_deployment_domain_409_raises_immediately(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    """A deployment 409 is a DOMAIN state conflict, mapped by
    ``raise_for_deployment`` to :class:`DeploymentStateError` before the retry
    driver sees a generic ``APIError(409)`` — so it raises immediately and is
    NOT swept into the scoped conflict-retry, even though ``idempotent=True``
    stamps a key. (The training create, whose generic mapper yields
    ``APIError(409)``, is where the client-side conflict-retry engages.)"""
    rmock.post(f"{API}/api/v1/deployments", status_code=409)
    client._sleep = lambda _s: None
    client._rng = lambda: 1.0
    with pytest.raises(DeploymentStateError):
        client.create_deployment({"project_id": "p1"})
    assert rmock.call_count == 1  # domain 409 is terminal, no retry
