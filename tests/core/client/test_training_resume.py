"""Wire-level coverage for ``TrainingClientMixin.resume_training_job``.

``POST /api/v1/training/jobs/{id}/resume`` puts a paused job back in the queue under the same
id. Each refusal the platform documents for it has its own typed error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from tests.core.client.test_common_credits import TRAINING_BODY

from dagnam._core.exceptions import (
    APIError,
    AuthError,
    InsufficientCreditsError,
    TrainingJobNotFoundError,
    TrainingStateError,
)

if TYPE_CHECKING:
    from tests.typing_helpers import RequestsMocker

    from dagnam._core.client import DagnamClient

RESUME = "https://api.test/api/v1/training/jobs/j1/resume"


def test_resume_returns_the_requeued_job(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(RESUME, status_code=200, json={"id": "j1", "status": "pending"})
    assert client.resume_training_job("j1") == {"id": "j1", "status": "pending"}
    assert rmock.last_request.method == "POST"
    assert rmock.last_request.headers["Authorization"] == "Bearer k"
    assert rmock.last_request.text is None


def test_resume_quotes_the_job_id(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(
        "https://api.test/api/v1/training/jobs/a%2Fb/resume",
        json={"id": "a/b", "status": "pending"},
    )
    assert client.resume_training_job("a/b")["id"] == "a/b"


def test_resume_402_raises_the_typed_credits_refusal(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(RESUME, status_code=402, json=TRAINING_BODY)
    with pytest.raises(InsufficientCreditsError) as caught:
        client.resume_training_job("j1")
    assert (caught.value.required_credits, caught.value.available_credits) == (120, 30)


@pytest.mark.parametrize("reason", ["not_paused", "checkpoint_unavailable"])
def test_resume_409_carries_the_platforms_marker(
    client: DagnamClient, rmock: RequestsMocker, reason: str
) -> None:
    rmock.post(
        RESUME,
        status_code=409,
        json={"detail": {"error": reason, "message": f"The job says {reason}."}},
    )
    with pytest.raises(TrainingStateError) as caught:
        client.resume_training_job("j1")
    assert caught.value.reason == reason
    assert caught.value.message == f"The job says {reason}."
    assert caught.value.status_code == 409
    assert isinstance(caught.value, APIError)


def test_resume_409_without_a_marker_is_still_typed(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(
        RESUME,
        status_code=409,
        json={"detail": "the previous run is still stopping; try again in a few minutes"},
    )
    with pytest.raises(TrainingStateError) as caught:
        client.resume_training_job("j1")
    assert caught.value.reason is None
    assert "still stopping" in caught.value.message


def test_resume_404_raises_not_found(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(RESUME, status_code=404, json={"detail": "Training job not found"})
    with pytest.raises(TrainingJobNotFoundError):
        client.resume_training_job("j1")


def test_resume_401_raises_auth(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(RESUME, status_code=401, text="no")
    with pytest.raises(AuthError):
        client.resume_training_job("j1")


def test_resume_5xx_raises_apierror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(RESUME, status_code=500, text="boom")
    with pytest.raises(APIError) as caught:
        client.resume_training_job("j1")
    assert not isinstance(caught.value, TrainingStateError)
    assert caught.value.status_code == 500
