"""Async deployments client mixin."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.exceptions import (
    APIError,
    DeploymentNotFoundError,
    DeploymentStateError,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from tests.typing_helpers import JsonObject, PytestMonkeyPatch, RespxMockRouter

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

_DEP_STREAM_URL = "/api/v1/deployments/dep1/stream"


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
    mock.get("/api/v1/deployments/missing/stream").mock(return_value=httpx.Response(404))
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
    mock.post("/api/v1/deployments/validate").mock(
        return_value=httpx.Response(200, json={"valid": True, "errors": []})
    )
    mock.get("/api/v1/deployments-platforms").mock(
        return_value=httpx.Response(200, json=[{"platform": "fastapi"}])
    )

    assert (await client.validate_deployment({"name": "x"}))["valid"] is True
    platforms = await client.list_deployment_platforms()
    first = platforms[0]
    assert isinstance(first, dict)
    assert first["platform"] == "fastapi"


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


async def test_async_create_deployment_revision(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post("/api/v1/deployments/dep1/revisions").mock(
        return_value=httpx.Response(201, json={"id": "rev2", "is_active": False})
    )
    payload: JsonObject = {"model_version_id": "mv1", "capacity_mode": "serverless"}
    result = await client.create_deployment_revision("dep1", payload, idempotency_key="key-1")
    assert result == {"id": "rev2", "is_active": False}
    request = route.calls[0].request
    assert json.loads(request.content) == payload
    assert request.headers["Idempotency-Key"] == "key-1"


async def test_async_create_deployment_revision_mints_key_and_maps_404(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    ok = mock.post("/api/v1/deployments/dep1/revisions").mock(
        return_value=httpx.Response(200, json={"id": "rev2"})
    )
    await client.create_deployment_revision("dep1", {"model_version_id": "mv1"})
    assert ok.calls[0].request.headers["Idempotency-Key"]
    mock.post("/api/v1/deployments/missing/revisions").mock(return_value=httpx.Response(404))
    with pytest.raises(DeploymentNotFoundError):
        await client.create_deployment_revision("missing", {"model_version_id": "mv1"})


async def test_async_get_deployment_revisions(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.get("/api/v1/deployments/dep1/revisions").mock(
        return_value=httpx.Response(200, json=[{"id": "rev1", "is_active": True}])
    )
    result = await client.get_deployment_revisions("dep1")
    assert result == [{"id": "rev1", "is_active": True}]
    url = str(route.calls[0].request.url)
    assert "page=1" in url
    assert "limit=50" in url


# --------------------------------------------------------------------------- deploy a model version


async def test_async_deploy_model_version_sends_only_the_version_by_default(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post("/api/v1/deployments/from-model-version").mock(
        return_value=httpx.Response(201, json={"id": "dep1", "api_key": "dep-key"})
    )
    result = await client.deploy_model_version("mv1")
    assert result["api_key"] == "dep-key"
    request = route.calls[0].request
    assert json.loads(request.content) == {"model_version_id": "mv1"}
    assert request.headers["Idempotency-Key"]


async def test_async_deploy_model_version_sends_name_project_and_explicit_key(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post("/api/v1/deployments/from-model-version").mock(
        return_value=httpx.Response(201, json={"id": "dep1"})
    )
    await client.deploy_model_version("mv1", name="bot", project_id="p1", idempotency_key="key-7")
    request = route.calls[0].request
    assert json.loads(request.content) == {
        "model_version_id": "mv1",
        "name": "bot",
        "project_id": "p1",
    }
    assert request.headers["Idempotency-Key"] == "key-7"


# --------------------------------------------------------------------------- replayed creates

_ROTATED = {"key_prefix": "sk_new12", "api_key": "fresh-key"}


async def test_async_rotate_deployment_key(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/deployments/dep1/rotate-key").mock(
        return_value=httpx.Response(200, json=_ROTATED)
    )
    assert await client.rotate_deployment_key("dep1") == _ROTATED


@pytest.mark.parametrize(
    ("path", "call"),
    [
        ("/api/v1/deployments/from-model-version", lambda c: c.deploy_model_version("mv1")),
        ("/api/v1/deployments", lambda c: c.create_deployment({"name": "x"})),
    ],
    ids=["deploy_model_version", "create_deployment"],
)
async def test_async_replayed_create_rotates_to_restore_the_key(
    client: AsyncDagnamClient,
    mock: RespxMockRouter,
    path: str,
    call: Callable[[AsyncDagnamClient], Awaitable[JsonObject]],
) -> None:
    mock.post(path).mock(
        return_value=httpx.Response(
            201,
            json={"id": "dep1", "key_prefix": "sk_old12", "api_key": None},
            headers={"Idempotency-Replayed": "true"},
        )
    )
    rotate = mock.post("/api/v1/deployments/dep1/rotate-key").mock(
        return_value=httpx.Response(200, json=_ROTATED)
    )
    result = await call(client)
    assert result == {"id": "dep1", "key_prefix": "sk_new12", "api_key": "fresh-key"}
    assert len(rotate.calls) == 1


async def test_async_first_create_with_its_key_does_not_rotate(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/deployments/from-model-version").mock(
        return_value=httpx.Response(201, json={"id": "dep1", "api_key": "dep-key"})
    )
    rotate = mock.post("/api/v1/deployments/dep1/rotate-key").mock(
        return_value=httpx.Response(200, json=_ROTATED)
    )
    assert (await client.deploy_model_version("mv1"))["api_key"] == "dep-key"
    assert len(rotate.calls) == 0


@pytest.mark.parametrize(
    "name",
    [
        "collect_deployment_metrics",
        "estimate_cost",
        "retry_deployment",
        "rollback_deployment",
        "scale_deployment",
    ],
)
async def test_async_removed_client_methods_are_gone(name: str) -> None:
    assert not hasattr(AsyncDagnamClient, name)


async def test_async_set_deployment_warm_patches_the_capacity_route(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.patch("/api/v1/deployments/dep1/capacity").mock(
        return_value=httpx.Response(200, json={"warm": True})
    )
    assert await client.set_deployment_warm("dep1", True) == {"warm": True}
    assert json.loads(route.calls[0].request.content) == {"warm": True}


async def test_async_set_deployment_warm_off_sends_false_and_quotes_the_id(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.patch("/api/v1/deployments/a%2Fb/capacity").mock(
        return_value=httpx.Response(200, json={"warm": False})
    )
    await client.set_deployment_warm("a/b", False)
    assert json.loads(route.calls[0].request.content) == {"warm": False}


async def test_async_set_deployment_warm_maps_409_and_404(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.patch("/api/v1/deployments/dep1/capacity").mock(return_value=httpx.Response(409))
    with pytest.raises(DeploymentStateError):
        await client.set_deployment_warm("dep1", True)
    mock.patch("/api/v1/deployments/missing/capacity").mock(return_value=httpx.Response(404))
    with pytest.raises(DeploymentNotFoundError):
        await client.set_deployment_warm("missing", True)
