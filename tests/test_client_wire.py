"""Wire-level HTTP tests using requests-mock.

Asserts DagnamClient hits the right URLs with the right headers, bodies,
and query params — without running a real server.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import requests_mock as rm_module

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import (
    APIError,
    AuthError,
    CheckpointNotFoundError,
    DeploymentNotFoundError,
)


API = "https://api.test"


@pytest.fixture
def client() -> DagnamClient:
    return DagnamClient(API, "k")


@pytest.fixture
def rmock():
    with rm_module.Mocker() as m:
        yield m


def test_predict_sends_bearer_and_api_key(client, rmock):
    url = f"{API}/api/v1/inference/dep_1/predict"
    rmock.post(url, json={"label": "ok"})

    result = client.predict("dep_1", {"x": 1})

    assert result == {"label": "ok"}
    req = rmock.last_request
    assert req.headers["Authorization"] == "Bearer k"
    assert req.headers["X-API-Key"] == "k"
    assert req.json() == {"x": 1}


def test_predict_batch_wraps_inputs(client, rmock):
    url = f"{API}/api/v1/inference/dep_1/predict/batch"
    rmock.post(url, json=[{"y": 1}, {"y": 2}])

    result = client.predict_batch("dep_1", [{"x": 1}, {"x": 2}])

    assert result == [{"y": 1}, {"y": 2}]
    assert rmock.last_request.json() == {"inputs": [{"x": 1}, {"x": 2}]}


def test_deployment_health_get(client, rmock):
    url = f"{API}/api/v1/inference/dep_1/health"
    rmock.get(url, json={"status": "healthy"})

    assert client.deployment_health("dep_1") == {"status": "healthy"}


def test_checkpoint_download_streams_and_returns_checksum(
    client, rmock, tmp_path
):
    body = b"weights-bytes"
    sha = hashlib.sha256(body).hexdigest()
    url = f"{API}/api/v1/training/jobs/job_1/checkpoints/ck_1/download"
    rmock.get(
        url,
        content=body,
        headers={
            "Content-Disposition": 'attachment; filename="job_1_ck_1.pt"',
            "Content-Length": str(len(body)),
            "X-Checksum-SHA256": sha,
        },
    )
    dest = tmp_path / "ck_1.pt"

    written, expected_sha = client.download_checkpoint_stream(
        "job_1", "ck_1", dest
    )

    assert written == dest
    assert expected_sha == sha
    assert dest.read_bytes() == body
    assert rmock.last_request.headers["Authorization"] == "Bearer k"


def test_sse_uses_api_key_query_param_not_header(client, rmock):
    """Contract: SSE auth goes in ?api_key=..., NOT in a custom header."""
    url = f"{API}/api/v1/streaming/training-jobs/job_1/stream"
    rmock.get(url, text="")

    resp = client.open_training_stream("job_1", last_event_id="42")
    resp.close()

    req = rmock.last_request
    assert req.qs.get("api_key") == ["k"]
    assert req.headers["Accept"] == "text/event-stream"
    assert req.headers["Last-Event-ID"] == "42"
    assert "X-API-Key" not in req.headers


def test_inference_404_maps_to_deployment_error(client, rmock):
    url = f"{API}/api/v1/inference/dep_404/predict"
    rmock.post(url, status_code=404, text="not found")

    with pytest.raises(DeploymentNotFoundError):
        client.predict("dep_404", {"x": 1})


def test_checkpoint_401_maps_to_auth_error(client, rmock, tmp_path):
    url = f"{API}/api/v1/training/jobs/job_1/checkpoints/ck_1/download"
    rmock.get(url, status_code=401, text="unauthorized")

    with pytest.raises(AuthError):
        client.download_checkpoint_stream("job_1", "ck_1", tmp_path / "x.pt")


def test_checkpoint_404_maps_to_checkpoint_not_found(client, rmock, tmp_path):
    url = f"{API}/api/v1/training/jobs/job_1/checkpoints/ck_bad/download"
    rmock.get(url, status_code=404, text="not found")

    with pytest.raises(CheckpointNotFoundError):
        client.download_checkpoint_stream(
            "job_1", "ck_bad", tmp_path / "x.pt"
        )
