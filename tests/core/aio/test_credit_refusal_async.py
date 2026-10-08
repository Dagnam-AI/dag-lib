"""The async client types the platform's insufficient-credits 402."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest
from tests.core.client.test_common_credits import INFERENCE_BODY, TRAINING_BODY

from dagnam._core.exceptions import InsufficientCreditsError

if TYPE_CHECKING:
    import respx

    from dagnam._core.aio import AsyncDagnamClient

pytestmark = pytest.mark.anyio

RESTART = "/api/v1/training/jobs/j1/restart"


async def test_the_async_client_raises_it(
    client: AsyncDagnamClient, mock: respx.MockRouter
) -> None:
    mock.post(RESTART).mock(return_value=httpx.Response(402, json=TRAINING_BODY))
    with pytest.raises(InsufficientCreditsError) as caught:
        await client.restart_training_job("j1")
    assert (caught.value.required_credits, caught.value.available_credits) == (120, 30)
    assert caught.value.next_steps == ("request_credits", "reduce_job_size")


async def test_a_refusal_to_a_third_party_carries_no_balance(
    client: AsyncDagnamClient, mock: respx.MockRouter
) -> None:
    mock.post(RESTART).mock(return_value=httpx.Response(402, json=INFERENCE_BODY))
    with pytest.raises(InsufficientCreditsError) as caught:
        await client.restart_training_job("j1")
    assert caught.value.available_credits is None
