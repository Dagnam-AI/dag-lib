"""Async codegen client mixin."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient

if TYPE_CHECKING:
    from tests.typing_helpers import RespxMockRouter

API = "https://api.test"

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------- codegen


async def test_async_generate_code_default(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post("/api/v1/projects/p1/generate-code").mock(
        return_value=httpx.Response(200, json={"task_id": "t1"})
    )
    await client.generate_code(
        "p1", framework="tf", version_id="v2", options={"s": 1}, async_mode=True
    )
    body = route.calls[0].request.read()
    assert b"tf" in body
    assert b"v2" in body
    assert "async_mode=true" in str(route.calls[0].request.url)


async def test_async_generate_code_default_minimal(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post("/api/v1/projects/p1/generate-code").mock(
        return_value=httpx.Response(200, json={})
    )
    await client.generate_code("p1")
    body = route.calls[0].request.read()
    assert b"version_id" not in body
    assert b"options" not in body


async def test_async_generate_code_explicit_payload(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/projects/p1/generate-code").mock(return_value=httpx.Response(200, json={}))
    await client.generate_code("p1", payload={"custom": True})


async def test_async_preview_code(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/projects/p1/code-preview").mock(return_value=httpx.Response(200, json={}))
    await client.preview_code("p1", "pytorch", version_id="v1")
    await client.preview_code("p1", "pytorch")


async def test_async_validate_code(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.post("/api/v1/projects/p1/validate").mock(return_value=httpx.Response(200, json={}))
    await client.validate_code("p1", version_id="v1")
    await client.validate_code("p1")
    await client.validate_architecture("p1")


async def test_async_download_code_returns_bytes(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/projects/p1/download-code").mock(
        return_value=httpx.Response(200, content=b"<code>")
    )
    out = await client.download_code("p1", framework="pytorch", version_id="v1")
    assert out == b"<code>"


async def test_async_download_code_to_file(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.get("/api/v1/projects/p1/download-code").mock(
        return_value=httpx.Response(200, content=b"<code>")
    )
    dest = tmp_path / "out.zip"
    out = await client.download_code("p1", dest_path=dest)
    assert out == dest
    assert dest.read_bytes() == b"<code>"


async def test_async_download_code_zip_alias(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/projects/p1/download-code").mock(
        return_value=httpx.Response(200, content=b"x")
    )
    assert await client.download_code_zip("p1", "pytorch") == b"x"


async def test_async_codegen_text_response(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/projects/p1/code-preview").mock(
        return_value=httpx.Response(200, text="plain", headers={"Content-Type": "text/plain"})
    )
    assert await client.preview_code("p1", "pytorch") == "plain"


async def test_async_codegen_empty_response(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/projects/p1/code-preview").mock(return_value=httpx.Response(204))
    assert await client.preview_code("p1", "pytorch") is None


async def test_async_get_code_status(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/projects/p1/code-status/t1").mock(
        return_value=httpx.Response(200, json={"status": "done"})
    )
    assert await client.get_code_status("p1", "t1") == {"status": "done"}
