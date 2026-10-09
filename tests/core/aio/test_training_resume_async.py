"""Async mirror of ``tests/core/client/test_training_resume.py``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest
from tests.core.client.test_common_credits import TRAINING_BODY

from dagnam._core.exceptions import (
    APIError,
    InsufficientCreditsError,
    TrainingJobNotFoundError,
    TrainingStateError,
)

if TYPE_CHECKING:
    from tests.typing_helpers import RespxMockRouter

    from dagnam._core.aio import AsyncDagnamClient

pytestmark = pytest.mark.anyio

RESUME = "/api/v1/training/jobs/j1/resume"


async def test_resume_returns_the_requeued_job(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.post(RESUME).mock(
        return_value=httpx.Response(200, json={"id": "j1", "status": "pending"})
    )
    assert await client.resume_training_job("j1") == {"id": "j1", "status": "pending"}
    assert route.calls[-1].request.headers["Authorization"] == "Bearer k"


async def test_resume_402_raises_the_typed_credits_refusal(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post(RESUME).mock(return_value=httpx.Response(402, json=TRAINING_BODY))
    with pytest.raises(InsufficientCreditsError) as caught:
        await client.resume_training_job("j1")
    assert caught.value.required_credits == 120


async def test_resume_409_carries_the_platforms_marker(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post(RESUME).mock(
        return_value=httpx.Response(
            409, json={"detail": {"error": "not_paused", "message": "It is running."}}
        )
    )
    with pytest.raises(TrainingStateError) as caught:
        await client.resume_training_job("j1")
    assert (caught.value.reason, caught.value.message) == ("not_paused", "It is running.")


async def test_resume_409_without_a_marker_is_still_typed(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post(RESUME).mock(return_value=httpx.Response(409, json={"detail": "still stopping"}))
    with pytest.raises(TrainingStateError) as caught:
        await client.resume_training_job("j1")
    assert caught.value.reason is None


async def test_resume_404_raises_not_found(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post(RESUME).mock(return_value=httpx.Response(404))
    with pytest.raises(TrainingJobNotFoundError):
        await client.resume_training_job("j1")


async def test_resume_5xx_raises_apierror(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.post(RESUME).mock(return_value=httpx.Response(500, text="boom"))
    with pytest.raises(APIError) as caught:
        await client.resume_training_job("j1")
    assert not isinstance(caught.value, TrainingStateError)
