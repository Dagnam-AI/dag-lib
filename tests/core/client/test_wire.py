"""Wire-level HTTP tests using requests-mock.

Asserts DagnamClient hits the right URLs with the right headers, bodies,
and query params — without running a real server.
"""

from __future__ import annotations

import hashlib

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


def test_path_identifiers_are_percent_encoded(client, rmock):
    """Untrusted identifiers must not inject extra URL path/query syntax."""
    url = f"{API}/api/v1/datasets/tenant%2Fdata%3Fversion%3Dlatest/meta"
    rmock.get(url, json={"id": "tenant/data?version=latest"})

    assert client.get_dataset_meta("tenant/data?version=latest") == {
        "id": "tenant/data?version=latest"
    }
    assert rmock.last_request.path.lower() == (
        "/api/v1/datasets/tenant%2Fdata%3Fversion%3Dlatest/meta".lower()
    )


def test_checkpoint_download_streams_and_returns_checksum(client, rmock, tmp_path):
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

    written, expected_sha = client.download_checkpoint_stream("job_1", "ck_1", dest)

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
        client.download_checkpoint_stream("job_1", "ck_bad", tmp_path / "x.pt")


def test_checkpoint_redirect_is_rejected_not_written(client, rmock, tmp_path):
    url = f"{API}/api/v1/training/jobs/job_1/checkpoints/ck_1/download"
    rmock.get(url, status_code=302, headers={"Location": "https://evil.test/ck.pt"})
    dest = tmp_path / "x.pt"

    with pytest.raises(APIError):
        client.download_checkpoint_stream("job_1", "ck_1", dest)

    assert not dest.exists()


def test_sse_redirect_is_rejected(client, rmock):
    url = f"{API}/api/v1/streaming/training-jobs/job_1/stream"
    rmock.get(url, status_code=302, headers={"Location": "https://evil.test/stream"})

    with pytest.raises(APIError):
        client.open_training_stream("job_1")


def test_codegen_generate_posts_framework_and_version(client, rmock):
    url = f"{API}/api/v1/projects/p1/generate-code"
    rmock.post(url, json={"files": []})

    result = client.generate_code("p1", framework="tensorflow", version_id="v2")

    assert result == {"files": []}
    req = rmock.last_request
    assert req.json() == {"framework": "tensorflow", "version_id": "v2"}
    assert req.qs == {}


def test_codegen_generate_async_sets_query_param(client, rmock):
    url = f"{API}/api/v1/projects/p1/generate-code"
    rmock.post(url, json={"task_id": "t1", "status": "pending"})

    result = client.generate_code("p1", framework="pytorch", async_mode=True)

    assert result["task_id"] == "t1"
    assert rmock.last_request.qs["async_mode"] == ["true"]
    assert rmock.last_request.json() == {"framework": "pytorch"}


def test_codegen_preview_uses_project_preview_route(client, rmock):
    url = f"{API}/api/v1/projects/p1/code-preview"
    rmock.get(url, json={"files": []})

    result = client.preview_code("p1", framework="pytorch", version_id="v4")

    assert result == {"files": []}
    assert rmock.last_request.qs["framework"] == ["pytorch"]
    assert rmock.last_request.qs["version_id"] == ["v4"]


def test_codegen_validate_posts_to_project_route(client, rmock):
    url = f"{API}/api/v1/projects/p1/validate"
    rmock.post(url, json={"is_valid": True})

    assert client.validate_code("p1", version_id="v1") == {"is_valid": True}
    assert rmock.last_request.qs["version_id"] == ["v1"]


def test_codegen_status_uses_project_status_route(client, rmock):
    url = f"{API}/api/v1/projects/p1/code-status/t1"
    rmock.get(url, json={"status": "completed"})

    assert client.get_code_status("p1", "t1") == {"status": "completed"}


def test_codegen_download_uses_project_download_route(client, rmock, tmp_path):
    body = b"zip-bytes"
    url = f"{API}/api/v1/projects/p1/download-code"
    rmock.get(url, content=body)

    assert client.download_code("p1", framework="flax", version_id="v3") == body
    assert rmock.last_request.qs["framework"] == ["flax"]
    assert rmock.last_request.qs["version_id"] == ["v3"]

    dest = tmp_path / "code.zip"
    rmock.get(url, content=body)
    assert client.download_code("p1", dest_path=dest) == dest
    assert dest.read_bytes() == body
