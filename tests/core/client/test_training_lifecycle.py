"""Wire-level coverage for the sync training lifecycle client methods.

Covers ``TrainingClientMixin.restart_training_job``,
``restore_from_checkpoint``, ``estimate_training_resources``, and
``get_allowed_strategies`` - the job restart/restore, resource-estimate, and
allowed-strategies routes added in the training-lifecycle wave.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import (
    APIError,
    AuthError,
    QuotaExceededError,
    TrainingJobNotFoundError,
)

if TYPE_CHECKING:
    from tests.typing_helpers import RequestsMocker

API = "https://api.test"
RESTART = f"{API}/api/v1/training/jobs/j1/restart"
RESTORE = f"{API}/api/v1/training/jobs/j1/checkpoints/c1/restore"
ESTIMATE = f"{API}/api/v1/training/estimate-resources"
STRATEGIES = f"{API}/api/v1/training/allowed-strategies"


# ----------------------------------------------------------------- restart


def test_restart_returns_new_job(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(RESTART, status_code=201, json={"id": "j2", "status": "pending"})
    assert client.restart_training_job("j1") == {"id": "j2", "status": "pending"}
    assert rmock.last_request.method == "POST"


def test_restart_402_raises_quota(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(RESTART, status_code=402, json={"detail": "plan limit"})
    with pytest.raises(QuotaExceededError):
        client.restart_training_job("j1")


def test_restart_404_raises_not_found(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(RESTART, status_code=404, text="missing")
    with pytest.raises(TrainingJobNotFoundError):
        client.restart_training_job("j1")


# ----------------------------------------------------------------- restore


def test_restore_returns_new_job(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(RESTORE, status_code=201, json={"id": "j2"})
    assert client.restore_from_checkpoint("j1", "c1") == {"id": "j2"}


def test_restore_404_raises_not_found(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(RESTORE, status_code=404, text="missing")
    with pytest.raises(TrainingJobNotFoundError):
        client.restore_from_checkpoint("j1", "c1")


def test_restore_422_raises_apierror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(RESTORE, status_code=422, json={"detail": "checkpoint too old"})
    with pytest.raises(APIError) as exc_info:
        client.restore_from_checkpoint("j1", "c1")
    assert exc_info.value.status_code == 422


# ---------------------------------------------------------------- estimate


def test_estimate_posts_config(client: DagnamClient, rmock: RequestsMocker) -> None:
    payload = {
        "estimated_memory_mb": 512,
        "estimated_training_time_seconds": 60,
        "estimated_disk_space_mb": 128,
        "estimated_cost_usd": None,
        "warnings": [],
        "recommendations": [],
    }
    rmock.post(ESTIMATE, json=payload)
    assert client.estimate_training_resources({"epochs": 2}) == payload
    assert rmock.last_request.json() == {"epochs": 2}


def test_estimate_422_raises_apierror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(ESTIMATE, status_code=422, json={"detail": "invalid config"})
    with pytest.raises(APIError) as exc_info:
        client.estimate_training_resources({})
    assert exc_info.value.status_code == 422


# ------------------------------------------------------------ allowed-strategies


def test_allowed_strategies_returns_flat_map(client: DagnamClient, rmock: RequestsMocker) -> None:
    body = {"cpu": True, "single_gpu": True, "multi_gpu_ddp": False}
    rmock.get(STRATEGIES, json=body)
    assert client.get_allowed_strategies() == body
    assert rmock.last_request.method == "GET"


def test_allowed_strategies_401_raises_autherror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.get(STRATEGIES, status_code=401, text="nope")
    with pytest.raises(AuthError):
        client.get_allowed_strategies()
