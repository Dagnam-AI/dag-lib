"""Async inference client mixin."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.exceptions import (
    AccountSuspendedError,
    APIError,
    AuthError,
    DeploymentNotFoundError,
    StreamError,
)

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch, RespxMockRouter

API = "https://api.test"

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------- inference


async def test_async_predict(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    route = mock.post("/api/v1/inference/dep1/predict").mock(
        return_value=httpx.Response(200, json={"y": 1})
    )
    assert await client.predict("dep1", {"x": 1}) == {"y": 1}
    assert json.loads(route.calls[0].request.content) == {"input": {"x": 1}}


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


async def test_async_mint_inference_stream_token(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    # No longer used by stream_predict (the session returns its own token); still public.
    route = mock.post("/api/v1/inference/dep1/stream-access-token").mock(
        return_value=httpx.Response(200, json={"token": "t1"})
    )
    assert await client.mint_inference_stream_token("dep1") == "t1"
    assert route.calls[0].request.headers["Authorization"] == "Bearer k"


_SESSION = "/api/v1/inference/dep1/predict/stream/session"
_STREAM = "/api/v1/inference/dep1/predict/stream/s1"


def _session_ok(mock: RespxMockRouter) -> None:
    mock.post(_SESSION).mock(
        return_value=httpx.Response(
            200, json={"session_id": "s1", "token": "t1", "expires_in": 300}
        )
    )


async def test_async_stream_predict_yields_tokens_until_complete(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    session = mock.post(_SESSION).mock(
        return_value=httpx.Response(
            200, json={"session_id": "s1", "token": "t1", "expires_in": 300}
        )
    )
    body = (
        'event: token\ndata: {"token": "he", "index": 1}\n\n'
        'event: token\ndata: {"token": "llo", "index": 2}\n\n'
        'event: complete\ndata: {"done": true, "total_tokens": 2}\n\n'
    )
    route = mock.get(_STREAM).mock(
        return_value=httpx.Response(200, text=body, headers={"Content-Type": "text/event-stream"})
    )
    events = [ev async for ev in client.stream_predict("dep1", {"text": "hi"})]
    assert [ev.event for ev in events] == ["token", "token", "complete"]
    assert events[0].data == {"token": "he", "index": 1}
    assert json.loads(session.calls[0].request.content) == {"input": {"text": "hi"}}
    # Only the session's short-lived token rides in the stream URL, never the input.
    assert dict(route.calls[0].request.url.params) == {"token": "t1"}


async def test_async_stream_predict_error_event_is_terminal(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    _session_ok(mock)
    body = 'event: error\ndata: {"message": "model blew up"}\n\n'
    mock.get(_STREAM).mock(
        return_value=httpx.Response(200, text=body, headers={"Content-Type": "text/event-stream"})
    )
    events = [ev async for ev in client.stream_predict("dep1", {"text": "hi"})]
    assert [ev.event for ev in events] == ["error"]
    assert events[0].data == {"message": "model blew up"}


async def test_async_stream_predict_ends_without_terminal_raises(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    _session_ok(mock)
    mock.get(_STREAM).mock(
        return_value=httpx.Response(
            200,
            text='event: token\ndata: {"token": "a"}\n\n',
            headers={"Content-Type": "text/event-stream"},
        )
    )
    with pytest.raises(StreamError):
        _ = [ev async for ev in client.stream_predict("dep1", {"text": "hi"})]


async def test_async_stream_predict_session_404_maps_not_found(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/inference/missing/predict/stream/session").mock(
        return_value=httpx.Response(404)
    )
    with pytest.raises(DeploymentNotFoundError):
        _ = [ev async for ev in client.stream_predict("missing", {"x": 1})]


async def test_async_stream_predict_expired_session_404_maps_not_found(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    _session_ok(mock)
    mock.get(_STREAM).mock(return_value=httpx.Response(404))
    with pytest.raises(DeploymentNotFoundError):
        _ = [ev async for ev in client.stream_predict("dep1", {"x": 1})]


async def test_async_stream_predict_connect_error_maps_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    _session_ok(mock)
    mock.get(_STREAM).mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(APIError):
        _ = [ev async for ev in client.stream_predict("dep1", {"x": 1})]


async def test_async_stream_predict_connect_timeout_maps_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    _session_ok(mock)
    mock.get(_STREAM).mock(side_effect=httpx.ConnectTimeout("slow"))
    with pytest.raises(APIError):
        _ = [ev async for ev in client.stream_predict("dep1", {"x": 1})]


# --------------------------------------------------------------------------- transient retry


async def test_async_schema_retries_transient(
    client: AsyncDagnamClient, mock: RespxMockRouter, monkeypatch: PytestMonkeyPatch
) -> None:
    async def _no_sleep(_d: float) -> None: ...

    monkeypatch.setattr(client, "_async_sleep", _no_sleep)
    monkeypatch.setattr(client, "_rng", lambda: 1.0)
    mock.get("/api/v1/inference/d1/schema").mock(
        side_effect=[
            httpx.Response(503, json={}),
            httpx.Response(200, json={"inputs": []}),
        ]
    )
    out = await client.schema("d1")
    assert out == {"inputs": []}


async def test_async_schema_404_not_retried(
    client: AsyncDagnamClient, mock: RespxMockRouter, monkeypatch: PytestMonkeyPatch
) -> None:
    async def _no_sleep(_d: float) -> None: ...

    monkeypatch.setattr(client, "_async_sleep", _no_sleep)
    route = mock.get("/api/v1/inference/missing/schema").mock(
        return_value=httpx.Response(404, json={})
    )
    with pytest.raises(DeploymentNotFoundError):
        await client.schema("missing")
    assert route.call_count == 1


async def test_async_predict_post_not_retried(
    client: AsyncDagnamClient, mock: RespxMockRouter, monkeypatch: PytestMonkeyPatch
) -> None:
    """A predict POST is non-idempotent: a transient status is not retried."""

    async def _no_sleep(_d: float) -> None: ...

    monkeypatch.setattr(client, "_async_sleep", _no_sleep)
    route = mock.post("/api/v1/inference/d1/predict").mock(
        return_value=httpx.Response(503, json={})
    )
    with pytest.raises(APIError):
        await client.predict("d1", {"x": 1})
    assert route.call_count == 1


# ------------------------------------------- shared account-status 403 mapping
# Twins of the sync assertions in tests/core/client/test_sync_inference.py: both
# transports run the same mapper, so both must raise the same typed error.


async def test_async_predict_suspended_403_raises_account_suspended(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/inference/dep1/predict").mock(
        return_value=httpx.Response(
            403, json={"detail": {"error": "account_suspended", "message": "Account suspended."}}
        )
    )
    with pytest.raises(AccountSuspendedError, match=r"Account suspended\."):
        await client.predict("dep1", {"x": 1})


async def test_async_predict_blocked_ip_403_raises_auth_error(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/inference/dep1/predict").mock(
        return_value=httpx.Response(
            403, json={"detail": {"error": "blocked_ip", "message": "IP not permitted."}}
        )
    )
    with pytest.raises(AuthError, match=r"IP not permitted\."):
        await client.predict("dep1", {"x": 1})
