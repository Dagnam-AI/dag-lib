"""Async inference client mixin."""

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
