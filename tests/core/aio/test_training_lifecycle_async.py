"""Wire-level coverage for the async training lifecycle client methods.

Async mirror of ``tests/core/client/test_training_lifecycle.py``: restart,
restore-from-checkpoint, resource-estimate, and allowed-strategies.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.exceptions import (
    APIError,
    AuthError,
    QuotaExceededError,
    TrainingJobNotFoundError,
)

if TYPE_CHECKING:
    from tests.typing_helpers import RespxMockRouter

pytestmark = pytest.mark.anyio


# ----------------------------------------------------------------- restart


async def test_restart_returns_new_job(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.post("/api/v1/training/jobs/j1/restart").mock(
        return_value=httpx.Response(201, json={"id": "j2"})
    )
    assert (await client.restart_training_job("j1"))["id"] == "j2"


async def test_restart_402_raises_quota(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.post("/api/v1/training/jobs/j1/restart").mock(
        return_value=httpx.Response(402, json={"detail": "plan limit"})
    )
    with pytest.raises(QuotaExceededError):
        await client.restart_training_job("j1")


async def test_restart_404_raises_not_found(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/training/jobs/j1/restart").mock(return_value=httpx.Response(404))
    with pytest.raises(TrainingJobNotFoundError):
        await client.restart_training_job("j1")


# ----------------------------------------------------------------- restore


async def test_restore_returns_new_job(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.post("/api/v1/training/jobs/j1/checkpoints/c1/restore").mock(
        return_value=httpx.Response(201, json={"id": "j2"})
    )
    assert (await client.restore_from_checkpoint("j1", "c1"))["id"] == "j2"


async def test_restore_404_raises_not_found(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/training/jobs/j1/checkpoints/c1/restore").mock(
        return_value=httpx.Response(404)
    )
    with pytest.raises(TrainingJobNotFoundError):
        await client.restore_from_checkpoint("j1", "c1")


async def test_restore_422_raises_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/training/jobs/j1/checkpoints/c1/restore").mock(
        return_value=httpx.Response(422, json={"detail": "checkpoint too old"})
    )
    with pytest.raises(APIError) as exc_info:
        await client.restore_from_checkpoint("j1", "c1")
    assert exc_info.value.status_code == 422


# ---------------------------------------------------------------- estimate


async def test_estimate_posts_config(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    payload = {
        "estimated_memory_mb": 512,
        "estimated_training_time_seconds": 60,
        "estimated_disk_space_mb": 128,
        "estimated_cost_usd": None,
        "warnings": [],
        "recommendations": [],
    }
    route = mock.post("/api/v1/training/estimate-resources").mock(
        return_value=httpx.Response(200, json=payload)
    )
    assert await client.estimate_training_resources({"epochs": 2}) == payload
    assert json.loads(route.calls[-1].request.content) == {"epochs": 2}


async def test_estimate_422_raises_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post("/api/v1/training/estimate-resources").mock(
        return_value=httpx.Response(422, json={"detail": "invalid config"})
    )
    with pytest.raises(APIError) as exc_info:
        await client.estimate_training_resources({})
    assert exc_info.value.status_code == 422


# ------------------------------------------------------------ allowed-strategies


async def test_allowed_strategies_returns_flat_map(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    body = {"cpu": True, "single_gpu": True, "multi_gpu_ddp": False}
    mock.get("/api/v1/training/allowed-strategies").mock(
        return_value=httpx.Response(200, json=body)
    )
    assert await client.get_allowed_strategies() == body


async def test_allowed_strategies_401_raises_autherror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/training/allowed-strategies").mock(return_value=httpx.Response(401))
    with pytest.raises(AuthError):
        await client.get_allowed_strategies()
