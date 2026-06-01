from __future__ import annotations

import requests_mock

from dagnam._core.client import DagnamClient

API = "https://api.test"


def test_register_local_run_posts_local_mode():
    client = DagnamClient(API, "key")
    with requests_mock.Mocker() as mocker:
        mocker.post(
            f"{API}/api/v1/training/jobs",
            json={"id": "run_1", "execution_mode": "local", "status": "pending"},
            status_code=201,
        )

        result = client.register_local_run(
            project_id="proj_1",
            framework="pytorch",
            config={"epochs": 1},
        )

    assert result["id"] == "run_1"
    assert mocker.last_request.json()["execution_mode"] == "local"


def test_mint_run_token():
    client = DagnamClient(API, "key")
    with requests_mock.Mocker() as mocker:
        mocker.post(
            f"{API}/api/v1/training/jobs/run_1/stream-token",
            json={"token": "rt", "expires_in": 1800},
        )
        result = client.mint_run_token("run_1")

    assert result == {"token": "rt", "expires_in": 1800}
