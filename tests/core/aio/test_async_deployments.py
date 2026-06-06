"""Async deployments client mixin."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.exceptions import (
    DeploymentNotFoundError,
)

if TYPE_CHECKING:
    from tests.typing_helpers import RespxMockRouter

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
    await client.rollback_deployment("dep1", "ck")
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
