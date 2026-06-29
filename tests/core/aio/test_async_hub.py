"""Async hub client mixin."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.exceptions import (
    HubModelNotFoundError,
)

if TYPE_CHECKING:
    from tests.typing_helpers import RespxMockRouter

API = "https://api.test"

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------- hub


async def test_async_list_hub_models(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/hub/models").mock(return_value=httpx.Response(200, json={"items": []}))
    assert await client.list_hub_models() == {"items": []}


async def test_async_get_hub_model_404(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/hub/models/missing").mock(return_value=httpx.Response(404))
    with pytest.raises(HubModelNotFoundError):
        await client.get_hub_model("missing")


async def test_async_create_hub_model(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.post("/api/v1/hub/models").mock(return_value=httpx.Response(200, json={"id": "m1"}))
    assert await client.create_hub_model({}) == {"id": "m1"}


async def test_async_update_hub_model(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.put("/api/v1/hub/models/m1").mock(return_value=httpx.Response(200, json={"id": "m1"}))
    assert await client.update_hub_model("m1", {}) == {"id": "m1"}


async def test_async_delete_hub_model(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.delete("/api/v1/hub/models/m1").mock(return_value=httpx.Response(204))
    assert await client.delete_hub_model("m1") is None


async def test_async_hub_misc(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/hub/models/m1/files").mock(return_value=httpx.Response(200, json={}))
    mock.get("/api/v1/hub/models/m1/download").mock(return_value=httpx.Response(200, json={}))
    mock.get("/api/v1/hub/models/m1/versions").mock(return_value=httpx.Response(200, json=[]))
    mock.post("/api/v1/hub/models/m1/versions").mock(return_value=httpx.Response(200, json={}))
    mock.post("/api/v1/hub/models/m1/star").mock(return_value=httpx.Response(200, json={}))
    mock.delete("/api/v1/hub/models/m1/star").mock(return_value=httpx.Response(200, json={}))
    mock.post("/api/v1/hub/models/m1/fork").mock(return_value=httpx.Response(200, json={}))
    mock.get("/api/v1/hub/models/m1/reviews").mock(return_value=httpx.Response(200, json={}))
    mock.post("/api/v1/hub/models/m1/reviews").mock(return_value=httpx.Response(200, json={}))
    mock.post("/api/v1/hub/models/m1/use-in-studio").mock(return_value=httpx.Response(200, json={}))
    mock.get("/api/v1/hub/categories").mock(return_value=httpx.Response(200, json=[]))
    mock.get("/api/v1/hub/featured").mock(return_value=httpx.Response(200, json=[]))
    mock.get("/api/v1/hub/trending").mock(return_value=httpx.Response(200, json=[]))
    mock.get("/api/v1/hub/starred").mock(return_value=httpx.Response(200, json={}))

    await client.list_hub_model_files("m1")
    await client.download_hub_model("m1", file_id="f1")
    await client.download_hub_model("m1")
    await client.list_hub_model_versions("m1")
    await client.create_hub_model_version("m1", {})
    await client.star_hub_model("m1")
    await client.unstar_hub_model("m1")
    await client.fork_hub_model("m1")
    await client.list_hub_model_reviews("m1")
    await client.add_hub_model_review("m1", {})
    await client.use_hub_model_in_studio("m1")
    await client.list_hub_categories()
    await client.get_hub_featured()
    await client.get_hub_trending()
    await client.list_hub_starred()


async def test_async_hub_text_response(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/hub/categories").mock(
        return_value=httpx.Response(200, text="plain", headers={"Content-Type": "text/plain"})
    )
    assert await client.list_hub_categories() == "plain"


async def test_async_hub_empty_response(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/hub/categories").mock(return_value=httpx.Response(204))
    assert await client.list_hub_categories() is None


# ---------------------------------------------------------------- file upload


async def test_async_hub_upload_model_file(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    f = tmp_path / "weights.bin"
    f.write_bytes(b"\x00\x01\x02")
    route = mock.post("/api/v1/hub/models/m1/files").mock(
        return_value=httpx.Response(
            201, json={"id": "f1", "model_id": "m1", "file_name": "weights.bin"}
        )
    )
    out = await client.upload_model_file("m1", str(f))
    assert out["id"] == "f1"
    assert b'name="file"' in route.calls[0].request.content


async def test_async_hub_upload_model_file_404(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    f = tmp_path / "weights.bin"
    f.write_bytes(b"\x00")
    mock.post("/api/v1/hub/models/missing/files").mock(return_value=httpx.Response(404))
    with pytest.raises(HubModelNotFoundError):
        await client.upload_model_file("missing", str(f))
