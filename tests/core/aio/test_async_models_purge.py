"""The async mirror of the per-version purge route."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

pytestmark = pytest.mark.anyio

if TYPE_CHECKING:
    from respx import MockRouter as RespxMockRouter

    from dagnam._core.aio import AsyncDagnamClient


async def test_async_purge_model_version(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    route = mock.delete("/api/v1/model-versions/v1").mock(return_value=httpx.Response(204))
    assert await client.purge_model_version("v1") is None
    assert route.called


async def test_async_purge_model_version_returns_the_platforms_own_row(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    row = {"kind": "model_version", "id": "v1", "status": "deleted", "code": "deleted"}
    mock.delete("/api/v1/model-versions/v1").mock(return_value=httpx.Response(200, json=row))
    assert await client.purge_model_version("v1") == row
