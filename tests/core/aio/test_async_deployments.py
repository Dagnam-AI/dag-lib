"""Async deployments client mixin."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.exceptions import (
    APIError,
    DeploymentNotFoundError,
)

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch, RespxMockRouter

API = "https://api.test"

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------- deployments


async def test_async_deployments_full_surface(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/deployments").mock(return_value=httpx.Response(200, json={"items": []}))
    mock.get("/api/v1/deployments/dep1").mock(return_value=httpx.Response(200, json={"id": "dep1"}))
    mock.post("/api/v1/deployments").mock(return_value=httpx.Response(200, json={}))
    mock.put("/api/v1/deployments/dep1").mock(return_value=httpx.Response(200, json={}))
    mock.delete("/api/v1/deployments/dep1").mock(return_value=httpx.Response(204))
    mock.post("/api/v1/deployments/dep1/pause").mock(return_value=httpx.Response(200, json={}))
    mock.post("/api/v1/deployments/dep1/resume").mock(return_value=httpx.Response(200, json={}))
    mock.put("/api/v1/deployments/dep1/scale").mock(return_value=httpx.Response(200, json={}))
    mock.post("/api/v1/deployments/dep1/rollback").mock(return_value=httpx.Response(200, json={}))
    mock.get("/api/v1/deployments/dep1/metrics").mock(return_value=httpx.Response(200, json={}))
    mock.get("/api/v1/deployments/dep1/logs").mock(return_value=httpx.Response(200, json={}))
    mock.get("/api/v1/deployments/dep1/health").mock(return_value=httpx.Response(200, json={}))

    await client.list_deployments(
        status_filter="active",
        platform="aws",
        project_id="p1",
        search="q",
    )
    await client.list_deployments()  # minimal
    await client.get_deployment("dep1")
    await client.create_deployment({})
    await client.update_deployment("dep1", {})
    await client.delete_deployment("dep1")
    await client.pause_deployment("dep1")
    await client.resume_deployment("dep1")
    await client.scale_deployment("dep1", 5)
    await client.rollback_deployment("dep1", "ck-1")
    await client.get_deployment_metrics("dep1")
    await client.get_deployment_logs(
        "dep1",
        level="ERROR",
        search="oom",
        start_time="2025-01-01",
        end_time="2025-01-02",
    )
    await client.get_deployment_health_full("dep1")


async def test_async_get_deployment_404(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/deployments/missing").mock(return_value=httpx.Response(404))
    with pytest.raises(DeploymentNotFoundError):
        await client.get_deployment("missing")


# ---------------------------------------------------------------- deployment SSE stream

_DEP_STREAM_URL = "/api/v1/streaming/deployments/dep1/stream"


async def test_async_mint_deployment_stream_token(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post("/api/v1/deployments/dep1/stream-access-token").mock(
        return_value=httpx.Response(200, json={"token": "dep-stream-t"})
    )
    assert await client.mint_deployment_stream_token("dep1") == "dep-stream-t"
    assert route.calls[0].request.headers["Authorization"] == "Bearer k"


async def test_async_stream_deployment_events(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/deployments/dep1/stream-access-token").mock(
        return_value=httpx.Response(200, json={"token": "stream-t"})
    )
    route = mock.get(_DEP_STREAM_URL).mock(
        return_value=httpx.Response(
            200,
            text='event: deployment_status\ndata: {"status":"running"}\n\nevent: deployment_ready\ndata: ok\n\n',
            headers={"Content-Type": "text/event-stream"},
        )
    )
    events = [e async for e in client.stream_deployment_events("dep1")]
    assert [e.event for e in events] == ["deployment_status", "deployment_ready"]
    assert route.calls[0].request.url.params["token"] == "stream-t"
    assert "api_key" not in route.calls[0].request.url.params


async def test_async_stream_deployment_reconnects_without_terminal(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    # A stream that ends without a terminal event must reconnect (re-mint token,
    # forward the cursor), not stop as if the deployment finished.
    mock.post("/api/v1/deployments/dep1/stream-access-token").mock(
        side_effect=[
            httpx.Response(200, json={"token": "tok-1"}),
            httpx.Response(200, json={"token": "tok-2"}),
        ]
    )
    route = mock.get(_DEP_STREAM_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                text="event: log\ndata: line\nid: 4\n\n",
                headers={"Content-Type": "text/event-stream"},
            ),
            httpx.Response(
                200,
                text="event: deployment_failed\ndata: boom\n\n",
                headers={"Content-Type": "text/event-stream"},
            ),
        ]
    )
    events = [e async for e in client.stream_deployment_events("dep1")]
    assert [e.event for e in events] == ["log", "deployment_failed"]
    assert len(route.calls) == 2
    assert route.calls[1].request.headers["Last-Event-ID"] == "4"
    assert route.calls[1].request.url.params["token"] == "tok-2"


async def test_async_stream_deployment_404(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/deployments/missing/stream-access-token").mock(
        return_value=httpx.Response(200, json={"token": "t"})
    )
    mock.get("/api/v1/streaming/deployments/missing/stream").mock(return_value=httpx.Response(404))
    with pytest.raises(DeploymentNotFoundError):
        _ = [e async for e in client.stream_deployment_events("missing")]


async def test_async_stream_deployment_connect_error(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/deployments/dep1/stream-access-token").mock(
        return_value=httpx.Response(200, json={"token": "t"})
    )
    mock.get(_DEP_STREAM_URL).mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(APIError, match="Connection failed"):
        _ = [e async for e in client.stream_deployment_events("dep1")]


async def test_async_stream_deployment_timeout(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/deployments/dep1/stream-access-token").mock(
        return_value=httpx.Response(200, json={"token": "t"})
    )
    mock.get(_DEP_STREAM_URL).mock(side_effect=httpx.ConnectTimeout("slow"))
    with pytest.raises(APIError, match="Request timed out"):
        _ = [e async for e in client.stream_deployment_events("dep1")]


async def test_async_deployments_text_response(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/deployments").mock(
        return_value=httpx.Response(200, text="plain", headers={"Content-Type": "text/plain"})
    )
    with pytest.raises(TypeError):
        await client.list_deployments()


async def test_async_get_deployment_logs_minimal(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.get("/api/v1/deployments/dep1/logs").mock(
        return_value=httpx.Response(200, json={"items": []})
    )
    await client.get_deployment_logs("dep1")
    url = str(route.calls[0].request.url)
    assert "level" not in url
    assert "search" not in url
    assert "start_time" not in url
    assert "end_time" not in url


async def test_async_delete_deployment_returns_object(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.delete("/api/v1/deployments/dep1").mock(
        return_value=httpx.Response(200, json={"deleted": True})
    )
    assert await client.delete_deployment("dep1") == {"deleted": True}


# ---------------------------------------------------------------- deployment planning


async def test_async_deployment_planning(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.post("/api/v1/deployments/estimate-cost").mock(
        return_value=httpx.Response(200, json={"monthly_cost": 12.0})
    )
    mock.post("/api/v1/deployments/validate").mock(
        return_value=httpx.Response(200, json={"valid": True, "errors": []})
    )
    mock.get("/api/v1/deployments-platforms").mock(
        return_value=httpx.Response(200, json=[{"platform": "fastapi"}])
    )
    mock.post("/api/v1/deployments/d1/retry").mock(
        return_value=httpx.Response(200, json={"id": "d1", "status": "deploying"})
    )

    assert (await client.estimate_cost({"platform": "fastapi"}))["monthly_cost"] == 12.0
    assert (await client.validate_deployment({"name": "x"}))["valid"] is True
    platforms = await client.list_deployment_platforms()
    first = platforms[0]
    assert isinstance(first, dict)
    assert first["platform"] == "fastapi"
    assert (await client.retry_deployment("d1"))["status"] == "deploying"


async def test_async_retry_deployment_404(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.post("/api/v1/deployments/missing/retry").mock(return_value=httpx.Response(404))
    with pytest.raises(DeploymentNotFoundError):
        await client.retry_deployment("missing")


async def test_async_collect_deployment_metrics(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post("/api/v1/deployments/dep1/metrics/collect").mock(
        return_value=httpx.Response(
            200, json={"deployment_id": "dep1", "points_created": 60, "backfilled": True}
        )
    )
    result = await client.collect_deployment_metrics("dep1", backfill_minutes=120)
    assert result["points_created"] == 60
    assert "backfill_minutes=120" in str(route.calls[0].request.url)


async def test_async_collect_deployment_metrics_409(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    from dagnam._core.exceptions import DeploymentStateError

    mock.post("/api/v1/deployments/dep1/metrics/collect").mock(
        return_value=httpx.Response(409, text="no")
    )
    with pytest.raises(DeploymentStateError):
        await client.collect_deployment_metrics("dep1")


# --------------------------------------------------------------------------- transient retry


async def test_async_get_deployment_retries_transient(
    client: AsyncDagnamClient, mock: RespxMockRouter, monkeypatch: PytestMonkeyPatch
) -> None:
    async def _no_sleep(_d: float) -> None: ...

    monkeypatch.setattr(client, "_async_sleep", _no_sleep)
    monkeypatch.setattr(client, "_rng", lambda: 1.0)
    mock.get("/api/v1/deployments/d1").mock(
        side_effect=[
            httpx.Response(503, json={}),
            httpx.Response(200, json={"id": "d1"}),
        ]
    )
    dep = await client.get_deployment("d1")
    assert dep["id"] == "d1"


async def test_async_get_deployment_404_not_retried(
    client: AsyncDagnamClient, mock: RespxMockRouter, monkeypatch: PytestMonkeyPatch
) -> None:
    async def _no_sleep(_d: float) -> None: ...

    monkeypatch.setattr(client, "_async_sleep", _no_sleep)
    route = mock.get("/api/v1/deployments/missing").mock(return_value=httpx.Response(404, json={}))
    with pytest.raises(DeploymentNotFoundError):
        await client.get_deployment("missing")
    assert route.call_count == 1


async def test_async_create_deployment_sends_idempotency_key(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post("/api/v1/deployments").mock(
        return_value=httpx.Response(201, json={"id": "dep1"})
    )
    await client.create_deployment({"project_id": "p1"})
    assert route.calls[-1].request.headers.get("Idempotency-Key")


async def test_async_create_deployment_retries_transient_with_same_key(
    client: AsyncDagnamClient, mock: RespxMockRouter, monkeypatch: PytestMonkeyPatch
) -> None:
    async def _no_sleep(_d: float) -> None: ...

    monkeypatch.setattr(client, "_async_sleep", _no_sleep)
    client._rng = lambda: 1.0
    route = mock.post("/api/v1/deployments").mock(
        side_effect=[
            httpx.Response(503, json={}),
            httpx.Response(201, json={"id": "dep1"}),
        ]
    )
    await client.create_deployment({"project_id": "p1"})
    assert route.call_count == 2
    keys = {c.request.headers.get("Idempotency-Key") for c in route.calls}
    assert len(keys) == 1
    assert next(iter(keys))
