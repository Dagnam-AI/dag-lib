"""Async inference client mixin."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.exceptions import (
    APIError,
    DeploymentNotFoundError,
    StreamError,
)

if TYPE_CHECKING:
    from tests.typing_helpers import RespxMockRouter

API = "https://api.test"

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------- inference


async def test_async_predict(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.post("/api/v1/inference/dep1/predict").mock(
        return_value=httpx.Response(200, json={"y": 1})
    )
    assert await client.predict("dep1", {"x": 1}) == {"y": 1}


async def test_async_predict_404(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.post("/api/v1/inference/missing/predict").mock(return_value=httpx.Response(404))
    with pytest.raises(DeploymentNotFoundError):
        await client.predict("missing", {})


async def test_async_predict_batch(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.post("/api/v1/inference/dep1/predict/batch").mock(
        return_value=httpx.Response(200, json=[{"y": 1}, {"y": 2}])
    )
    assert await client.predict_batch("dep1", [{"x": 1}, {"x": 2}]) == [{"y": 1}, {"y": 2}]


async def test_async_deployment_health(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/inference/dep1/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )
    assert await client.deployment_health("dep1") == {"status": "healthy"}


async def test_async_inference_schema(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/inference/d1/schema").mock(
        return_value=httpx.Response(
            200, json={"input_schema": {"type": "object"}, "output_schema": {"type": "array"}}
        )
    )
    out = await client.schema("d1")
    assert out["input_schema"] == {"type": "object"}
    assert out["output_schema"] == {"type": "array"}


async def test_async_inference_schema_404(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/inference/missing/schema").mock(return_value=httpx.Response(404))
    with pytest.raises(DeploymentNotFoundError):
        await client.schema("missing")


# ---------------------------------------------------------------- streaming


async def test_async_stream_predict_yields_tokens_until_complete(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/inference/dep1/stream-access-token").mock(
        return_value=httpx.Response(200, json={"token": "t1"})
    )
    body = (
        'event: token\ndata: {"token": "he", "index": 1}\n\n'
        'event: token\ndata: {"token": "llo", "index": 2}\n\n'
        'event: complete\ndata: {"done": true, "total_tokens": 2}\n\n'
    )
    route = mock.get("/api/v1/inference/dep1/predict/stream").mock(
        return_value=httpx.Response(200, text=body, headers={"Content-Type": "text/event-stream"})
    )
    events = [ev async for ev in client.stream_predict("dep1", {"text": "hi"})]
    assert [ev.event for ev in events] == ["token", "token", "complete"]
    assert events[0].data == {"token": "he", "index": 1}
    sent = route.calls[0].request.url
    assert "token=t1" in str(sent)
    assert "input=" in str(sent)


async def test_async_stream_predict_error_event_is_terminal(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/inference/dep1/stream-access-token").mock(
        return_value=httpx.Response(200, json={"token": "t1"})
    )
    body = 'event: error\ndata: {"message": "model blew up"}\n\n'
    mock.get("/api/v1/inference/dep1/predict/stream").mock(
        return_value=httpx.Response(200, text=body, headers={"Content-Type": "text/event-stream"})
    )
    events = [ev async for ev in client.stream_predict("dep1", {"text": "hi"})]
    assert [ev.event for ev in events] == ["error"]
    assert events[0].data == {"message": "model blew up"}


async def test_async_stream_predict_ends_without_terminal_raises(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/inference/dep1/stream-access-token").mock(
        return_value=httpx.Response(200, json={"token": "t1"})
    )
    mock.get("/api/v1/inference/dep1/predict/stream").mock(
        return_value=httpx.Response(
            200,
            text='event: token\ndata: {"token": "a"}\n\n',
            headers={"Content-Type": "text/event-stream"},
        )
    )
    with pytest.raises(StreamError):
        _ = [ev async for ev in client.stream_predict("dep1", {"text": "hi"})]


async def test_async_stream_predict_404_maps_not_found(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/inference/missing/stream-access-token").mock(
        return_value=httpx.Response(200, json={"token": "t1"})
    )
    mock.get("/api/v1/inference/missing/predict/stream").mock(return_value=httpx.Response(404))
    with pytest.raises(DeploymentNotFoundError):
        _ = [ev async for ev in client.stream_predict("missing", {"x": 1})]


async def test_async_stream_predict_connect_error_maps_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/inference/dep1/stream-access-token").mock(
        return_value=httpx.Response(200, json={"token": "t1"})
    )
    mock.get("/api/v1/inference/dep1/predict/stream").mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(APIError):
        _ = [ev async for ev in client.stream_predict("dep1", {"x": 1})]


async def test_async_stream_predict_connect_timeout_maps_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/inference/dep1/stream-access-token").mock(
        return_value=httpx.Response(200, json={"token": "t1"})
    )
    mock.get("/api/v1/inference/dep1/predict/stream").mock(side_effect=httpx.ConnectTimeout("slow"))
    with pytest.raises(APIError):
        _ = [ev async for ev in client.stream_predict("dep1", {"x": 1})]
