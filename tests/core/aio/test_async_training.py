"""Wire-level coverage for the async training client mixin.

Async mirror of ``tests/core/client/test_sync_training.py``: exercises the
shared ``_training_req`` transport helper, the eleven request methods, the
metrics-event upload path, and the single-connection async SSE event stream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.exceptions import APIError, AuthError, TrainingJobNotFoundError

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch, RespxMockRouter

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------- request methods


async def test_async_training_full_surface(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/training/jobs").mock(return_value=httpx.Response(200, json={"id": "j1"}))
    mock.post("/api/v1/training/jobs/j1/stream-token").mock(
        return_value=httpx.Response(200, json={"token": "t"})
    )
    mock.post("/api/v1/training/jobs/j1/stream-access-token").mock(
        return_value=httpx.Response(200, json={"token": "stream-t"})
    )
    mock.get("/api/v1/training/jobs/j1").mock(return_value=httpx.Response(200, json={"id": "j1"}))
    mock.get("/api/v1/training/jobs").mock(return_value=httpx.Response(200, json={"items": []}))
    mock.post("/api/v1/training/jobs/j1/cancel").mock(
        return_value=httpx.Response(200, json={"message": "cancelled"})
    )
    mock.post("/api/v1/training/jobs/bulk-delete").mock(
        return_value=httpx.Response(200, json={"succeeded": 1})
    )
    mock.get("/api/v1/training/jobs/j1/logs").mock(
        return_value=httpx.Response(200, json={"items": []})
    )
    mock.get("/api/v1/training/jobs/j1/metrics").mock(
        return_value=httpx.Response(200, json={"items": []})
    )
    mock.get("/api/v1/training/jobs/j1/metrics/summary").mock(
        return_value=httpx.Response(200, json={"epochs": 1})
    )
    mock.post("/api/v1/training/jobs/j1/metrics/events").mock(
        return_value=httpx.Response(200, json={"accepted": 2, "duplicates": 0})
    )

    assert (await client.create_training_job({"project_id": "p1"}))["id"] == "j1"
    assert (await client.register_local_run(project_id="p1", framework="pytorch", config={}))[
        "id"
    ] == "j1"
    assert (
        await client.register_local_run(
            project_id="p1", framework="pytorch", config={}, max_duration_seconds=60
        )
    )["id"] == "j1"
    assert (await client.mint_run_token("j1"))["token"] == "t"
    assert await client.mint_training_stream_token("j1") == "stream-t"
    assert (await client.get_training_job("j1"))["id"] == "j1"
    assert "items" in await client.list_training_jobs(status="running")
    assert (await client.cancel_training_job("j1"))["message"] == "cancelled"
    assert (await client.bulk_delete_training_jobs(["j1"]))["succeeded"] == 1
    assert "items" in await client.get_training_logs("j1", page=1)
    assert "items" in await client.get_training_metrics("j1")
    assert (await client.get_training_metrics_summary("j1"))["epochs"] == 1
    assert (await client.upload_training_events("j1", [{"e": 1}]))["accepted"] == 2


async def test_async_register_local_run_payload(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post("/api/v1/training/jobs").mock(
        return_value=httpx.Response(200, json={"id": "j1"})
    )
    await client.register_local_run(
        project_id="p1", framework="pytorch", config={"epochs": 3}, max_duration_seconds=120
    )
    import json as _json

    body = _json.loads(route.calls[0].request.content)
    assert body == {
        "project_id": "p1",
        "framework": "pytorch",
        "execution_mode": "local",
        "config": {"epochs": 3},
        "max_duration_seconds": 120,
    }


async def test_async_create_collection_404_is_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    # No job_id supplied -> 404 maps to APIError, not TrainingJobNotFoundError.
    mock.post("/api/v1/training/jobs").mock(return_value=httpx.Response(404, text="missing"))
    with pytest.raises(APIError):
        await client.create_training_job({"project_id": "p1"})


async def test_async_get_training_job_404(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/training/jobs/missing").mock(return_value=httpx.Response(404))
    with pytest.raises(TrainingJobNotFoundError):
        await client.get_training_job("missing")


async def test_async_mint_training_stream_token_401_maps_auth_error(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/training/jobs/j1/stream-access-token").mock(return_value=httpx.Response(401))
    with pytest.raises(AuthError):
        await client.mint_training_stream_token("j1")


async def test_async_mint_training_stream_token_404_maps_job_not_found(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/training/jobs/missing/stream-access-token").mock(
        return_value=httpx.Response(404)
    )
    with pytest.raises(TrainingJobNotFoundError):
        await client.mint_training_stream_token("missing")


async def test_async_training_empty_body_raises_type_error(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    # _training_req returns None on an empty body; ensure_json_object rejects it.
    mock.get("/api/v1/training/jobs").mock(return_value=httpx.Response(204))
    with pytest.raises(TypeError, match="Expected JSON object"):
        await client.list_training_jobs()


async def test_async_training_non_json_body_raises_type_error(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    # Non-JSON body falls back to resp.text (a str); ensure_json_object rejects it.
    mock.get("/api/v1/training/jobs").mock(
        return_value=httpx.Response(200, text="plain", headers={"Content-Type": "text/plain"})
    )
    with pytest.raises(TypeError, match="Expected JSON object"):
        await client.list_training_jobs()


async def test_async_upload_events_empty_short_circuits(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    # No HTTP call is made when events is empty.
    route = mock.post("/api/v1/training/jobs/j1/metrics/events").mock(
        return_value=httpx.Response(200, json={"accepted": 99})
    )
    assert await client.upload_training_events("j1", []) == {"accepted": 0, "duplicates": 0}
    assert len(route.calls) == 0


async def test_async_upload_events_default_source(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post("/api/v1/training/jobs/j1/metrics/events").mock(
        return_value=httpx.Response(200, json={"accepted": 1, "duplicates": 0})
    )
    await client.upload_training_events("j1", [{"type": "metric"}])
    import json as _json

    source = _json.loads(route.calls[0].request.content)["source"]
    assert source["kind"] == "local_attach"
    assert isinstance(source["sdk_version"], str)


async def test_async_upload_events_custom_source(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post("/api/v1/training/jobs/j1/metrics/events").mock(
        return_value=httpx.Response(200, json={"accepted": 1, "duplicates": 0})
    )
    await client.upload_training_events("j1", [{"type": "metric"}], source={"kind": "local_stream"})
    import json as _json

    assert _json.loads(route.calls[0].request.content)["source"] == {"kind": "local_stream"}


async def test_async_upload_events_unknown_sdk_version(
    client: AsyncDagnamClient, mock: RespxMockRouter, monkeypatch: PytestMonkeyPatch
) -> None:
    from importlib import metadata

    def _missing(_name: str) -> str:
        raise metadata.PackageNotFoundError("dagnam")

    monkeypatch.setattr(metadata, "version", _missing)
    route = mock.post("/api/v1/training/jobs/j1/metrics/events").mock(
        return_value=httpx.Response(200, json={"accepted": 1, "duplicates": 0})
    )
    await client.upload_training_events("j1", [{"type": "metric"}])
    import json as _json

    source = _json.loads(route.calls[0].request.content)["source"]
    assert source["sdk_version"] == "0+unknown"


async def test_async_upload_events_404_raises_job_not_found(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/training/jobs/j1/metrics/events").mock(return_value=httpx.Response(404))
    with pytest.raises(TrainingJobNotFoundError):
        await client.upload_training_events("j1", [{"type": "metric"}])


# ---------------------------------------------------------------- stream_training_events


_STREAM_URL = "/api/v1/streaming/training-jobs/j1/stream"


async def test_async_stream_training_events(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/training/jobs/j1/stream-access-token").mock(
        return_value=httpx.Response(200, json={"token": "stream-t"})
    )
    body = (
        ": heartbeat\n"
        "event: progress\n"
        'data: {"epoch": 1}\n'
        "id: 7\n"
        "retry: 1000\n"
        "\n"
        "event: complete\n"
        "data: done\n"
        "\n"
    )
    route = mock.get(_STREAM_URL).mock(
        return_value=httpx.Response(200, text=body, headers={"Content-Type": "text/event-stream"})
    )
    events = [e async for e in client.stream_training_events("j1", last_event_id="3")]
    assert events[0].event == "progress"
    assert events[0].data == {"epoch": 1}
    assert events[0].id == "7"
    assert events[0].retry == 1000
    assert events[1].event == "complete"
    assert events[1].data == "done"
    # last_event_id is forwarded as a Last-Event-ID request header.
    assert route.calls[0].request.headers["Last-Event-ID"] == "3"
    assert route.calls[0].request.url.params["token"] == "stream-t"
    assert "api_key" not in route.calls[0].request.url.params


async def test_async_stream_training_events_no_cursor(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/training/jobs/j1/stream-access-token").mock(
        return_value=httpx.Response(200, json={"token": "stream-t"})
    )
    route = mock.get(_STREAM_URL).mock(
        return_value=httpx.Response(
            200,
            text="data: hi\n\nevent: stream_end\ndata: bye\n\n",
            headers={"Content-Type": "text/event-stream"},
        )
    )
    events = [e async for e in client.stream_training_events("j1")]
    assert events[0].data == "hi"
    assert "Last-Event-ID" not in route.calls[0].request.headers


async def test_async_stream_training_reconnects_when_stream_ends_without_terminal(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    # A stream that ends WITHOUT a terminal event is a silent drop, not
    # completion: the client must reconnect (re-minting the token, forwarding
    # the last event id) rather than return as if the job finished.
    mock.post("/api/v1/training/jobs/j1/stream-access-token").mock(
        side_effect=[
            httpx.Response(200, json={"token": "tok-1"}),
            httpx.Response(200, json={"token": "tok-2"}),
        ]
    )
    route = mock.get(_STREAM_URL).mock(
        side_effect=[
            # First connection drops after one event, no terminal marker.
            httpx.Response(
                200,
                text="event: progress\ndata: {}\nid: 9\n\n",
                headers={"Content-Type": "text/event-stream"},
            ),
            # Reconnect resumes and reaches a terminal event.
            httpx.Response(
                200,
                text="event: complete\ndata: done\n\n",
                headers={"Content-Type": "text/event-stream"},
            ),
        ]
    )
    events = [e async for e in client.stream_training_events("j1")]
    assert [e.event for e in events] == ["progress", "complete"]
    assert len(route.calls) == 2  # reconnected
    assert route.calls[1].request.headers["Last-Event-ID"] == "9"  # cursor preserved
    assert route.calls[1].request.url.params["token"] == "tok-2"  # fresh token


async def test_async_stream_training_404(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.post("/api/v1/training/jobs/missing/stream-access-token").mock(
        return_value=httpx.Response(200, json={"token": "stream-t"})
    )
    mock.get("/api/v1/streaming/training-jobs/missing/stream").mock(
        return_value=httpx.Response(404)
    )
    with pytest.raises(TrainingJobNotFoundError):
        _ = [e async for e in client.stream_training_events("missing")]


async def test_async_stream_training_auth_error(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/training/jobs/j1/stream-access-token").mock(
        return_value=httpx.Response(200, json={"token": "stream-t"})
    )
    mock.get(_STREAM_URL).mock(return_value=httpx.Response(401))
    with pytest.raises(AuthError):
        _ = [e async for e in client.stream_training_events("j1")]


async def test_async_stream_training_server_error(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/training/jobs/j1/stream-access-token").mock(
        return_value=httpx.Response(200, json={"token": "stream-t"})
    )
    mock.get(_STREAM_URL).mock(return_value=httpx.Response(500, text="boom"))
    with pytest.raises(APIError):
        _ = [e async for e in client.stream_training_events("j1")]


async def test_async_stream_training_connect_error(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/training/jobs/j1/stream-access-token").mock(
        return_value=httpx.Response(200, json={"token": "stream-t"})
    )
    mock.get(_STREAM_URL).mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(APIError, match="Connection failed"):
        _ = [e async for e in client.stream_training_events("j1")]


async def test_async_stream_training_timeout(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/training/jobs/j1/stream-access-token").mock(
        return_value=httpx.Response(200, json={"token": "stream-t"})
    )
    mock.get(_STREAM_URL).mock(side_effect=httpx.ConnectTimeout("slow"))
    with pytest.raises(APIError, match="Request timed out"):
        _ = [e async for e in client.stream_training_events("j1")]
