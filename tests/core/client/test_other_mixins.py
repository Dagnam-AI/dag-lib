"""Coverage for inference, training, codegen, checkpoints sync mixins."""

from __future__ import annotations

import pytest
import requests
import requests_mock as rm_module

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import (
    APIError,
    AuthError,
    CheckpointNotFoundError,
    CodegenError,
    CodegenValidationError,
    DeploymentNotFoundError,
    TrainingJobNotFoundError,
)

API = "https://api.test"


@pytest.fixture
def client() -> DagnamClient:
    return DagnamClient(API, "k")


@pytest.fixture
def rmock():
    with rm_module.Mocker() as m:
        yield m


# ---------------------------------------------------------------- inference


def test_predict_404(client, rmock):
    rmock.post(f"{API}/api/v1/inference/dep1/predict", status_code=404)
    with pytest.raises(DeploymentNotFoundError):
        client.predict("dep1", {"x": 1})


def test_predict_connection_error(client, monkeypatch):
    def _boom(*_a, **_kw):
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "post", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.predict("dep1", {"x": 1})


def test_predict_timeout(client, monkeypatch):
    def _boom(*_a, **_kw):
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "post", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.predict("dep1", {"x": 1})


def test_predict_batch_connection_error(client, monkeypatch):
    def _boom(*_a, **_kw):
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "post", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.predict_batch("dep1", [{"x": 1}])


def test_predict_batch_timeout(client, monkeypatch):
    def _boom(*_a, **_kw):
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "post", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.predict_batch("dep1", [{"x": 1}])


def test_deployment_health_connection_error(client, monkeypatch):
    def _boom(*_a, **_kw):
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.deployment_health("dep1")


def test_deployment_health_timeout(client, monkeypatch):
    def _boom(*_a, **_kw):
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.deployment_health("dep1")


# ---------------------------------------------------------------- training stream


def test_open_training_stream_with_cursor(client, rmock):
    rmock.get(f"{API}/api/v1/streaming/training-jobs/job1/stream", text="")
    resp = client.open_training_stream("job1", last_event_id="c1")
    assert resp.request.headers["Last-Event-ID"] == "c1"
    assert resp.request.headers["Accept"] == "text/event-stream"
    assert rmock.last_request.qs == {"api_key": ["k"]}


def test_open_training_stream_without_cursor(client, rmock):
    rmock.get(f"{API}/api/v1/streaming/training-jobs/job1/stream", text="")
    resp = client.open_training_stream("job1")
    assert "Last-Event-ID" not in resp.request.headers


def test_open_training_stream_401(client, rmock):
    rmock.get(f"{API}/api/v1/streaming/training-jobs/job1/stream", status_code=401)
    with pytest.raises(AuthError):
        client.open_training_stream("job1")


def test_open_training_stream_404(client, rmock):
    rmock.get(f"{API}/api/v1/streaming/training-jobs/job1/stream", status_code=404)
    with pytest.raises(TrainingJobNotFoundError):
        client.open_training_stream("job1")


def test_open_training_stream_500(client, rmock):
    rmock.get(f"{API}/api/v1/streaming/training-jobs/job1/stream", status_code=500, text="boom")
    with pytest.raises(APIError):
        client.open_training_stream("job1")


def test_open_training_stream_connection_error(client, monkeypatch):
    def _boom(*_a, **_kw):
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.open_training_stream("job1")


def test_open_training_stream_timeout(client, monkeypatch):
    def _boom(*_a, **_kw):
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.open_training_stream("job1")


# ---------------------------------------------------------------- codegen


def test_generate_code_default_payload(client, rmock):
    rmock.post(f"{API}/api/v1/projects/p1/generate-code", json={"task_id": "t1"})
    client.generate_code(
        "p1",
        framework="tensorflow",
        version_id="v2",
        options={"strict": True},
        async_mode=True,
    )
    body = rmock.last_request.json()
    assert body == {
        "framework": "tensorflow",
        "version_id": "v2",
        "options": {"strict": True},
    }
    assert rmock.last_request.qs == {"async_mode": ["true"]}


def test_generate_code_explicit_payload(client, rmock):
    rmock.post(f"{API}/api/v1/projects/p1/generate-code", json={})
    client.generate_code("p1", payload={"custom": True})
    assert rmock.last_request.json() == {"custom": True}


def test_preview_code(client, rmock):
    rmock.get(f"{API}/api/v1/projects/p1/code-preview", json={"code": "..."})
    client.preview_code("p1", "pytorch", version_id="v2")
    assert rmock.last_request.qs == {"framework": ["pytorch"], "version_id": ["v2"]}


def test_preview_code_no_version(client, rmock):
    rmock.get(f"{API}/api/v1/projects/p1/code-preview", json={"code": "..."})
    client.preview_code("p1", "pytorch")
    assert rmock.last_request.qs == {"framework": ["pytorch"]}


def test_validate_code_with_and_without_version(client, rmock):
    rmock.post(f"{API}/api/v1/projects/p1/validate", json={"valid": True})
    client.validate_code("p1", version_id="v1")
    assert rmock.last_request.qs == {"version_id": ["v1"]}
    client.validate_code("p1")
    assert rmock.last_request.qs == {}


def test_validate_architecture_alias(client, rmock):
    rmock.post(f"{API}/api/v1/projects/p1/validate", json={"valid": True})
    client.validate_architecture("p1", version_id="v1")


def test_download_code_returns_bytes(client, rmock):
    rmock.get(f"{API}/api/v1/projects/p1/download-code", content=b"<code>")
    result = client.download_code("p1", framework="pytorch", version_id="v1")
    assert result == b"<code>"


def test_download_code_writes_to_file(client, rmock, tmp_path):
    rmock.get(f"{API}/api/v1/projects/p1/download-code", content=b"<code>")
    dest = tmp_path / "out.zip"
    out = client.download_code("p1", framework="pytorch", dest_path=dest)
    assert out == dest
    assert dest.read_bytes() == b"<code>"


def test_download_code_zip_alias(client, rmock):
    rmock.get(f"{API}/api/v1/projects/p1/download-code", content=b"x")
    assert client.download_code_zip("p1", "pytorch") == b"x"


def test_download_code_connection_error(client, monkeypatch):
    def _boom(*_a, **_kw):
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.download_code("p1")


def test_download_code_timeout(client, monkeypatch):
    def _boom(*_a, **_kw):
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.download_code("p1")


def test_codegen_500_raises(client, rmock):
    rmock.post(f"{API}/api/v1/projects/p1/validate", status_code=500, text="boom")
    with pytest.raises(CodegenError):
        client.validate_code("p1")


def test_codegen_400_raises_validation(client, rmock):
    rmock.post(f"{API}/api/v1/projects/p1/validate", status_code=400, text="bad")
    with pytest.raises(CodegenValidationError):
        client.validate_code("p1")


def test_codegen_text_response(client, rmock):
    rmock.get(
        f"{API}/api/v1/projects/p1/code-preview",
        text="plain",
        headers={"Content-Type": "text/plain"},
    )
    assert client.preview_code("p1", "pytorch") == "plain"


def test_codegen_empty_response(client, rmock):
    rmock.get(f"{API}/api/v1/projects/p1/code-preview", text="", status_code=204)
    assert client.preview_code("p1", "pytorch") is None


def test_codegen_connection_error(client, monkeypatch):
    def _boom(*_a, **_kw):
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "request", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.preview_code("p1", "pytorch")


def test_codegen_timeout(client, monkeypatch):
    def _boom(*_a, **_kw):
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "request", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.preview_code("p1", "pytorch")


def test_get_code_status(client, rmock):
    rmock.get(f"{API}/api/v1/projects/p1/code-status/t1", json={"status": "done"})
    assert client.get_code_status("p1", "t1") == {"status": "done"}


# ---------------------------------------------------------------- checkpoints


def test_list_checkpoints(client, rmock):
    rmock.get(f"{API}/api/v1/training/jobs/job1/checkpoints", json=[{"id": "c1"}])
    assert client.list_checkpoints("job1") == [{"id": "c1"}]


def test_list_checkpoints_404(client, rmock):
    rmock.get(f"{API}/api/v1/training/jobs/job1/checkpoints", status_code=404)
    with pytest.raises(TrainingJobNotFoundError):
        client.list_checkpoints("job1")


def test_list_checkpoints_401(client, rmock):
    rmock.get(f"{API}/api/v1/training/jobs/job1/checkpoints", status_code=401)
    with pytest.raises(AuthError):
        client.list_checkpoints("job1")


def test_list_checkpoints_500(client, rmock):
    rmock.get(f"{API}/api/v1/training/jobs/job1/checkpoints", status_code=500, text="boom")
    with pytest.raises(APIError):
        client.list_checkpoints("job1")


def test_list_checkpoints_connection_error(client, monkeypatch):
    def _boom(*_a, **_kw):
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.list_checkpoints("job1")


def test_list_checkpoints_timeout(client, monkeypatch):
    def _boom(*_a, **_kw):
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.list_checkpoints("job1")


def test_download_checkpoint_stream(client, rmock, tmp_path):
    url = f"{API}/api/v1/training/jobs/job1/checkpoints/ckpt1/download"
    rmock.get(url, content=b"weights", headers={"X-Checksum-SHA256": "abc"})
    dest = tmp_path / "ckpt.bin"
    written, checksum = client.download_checkpoint_stream("job1", "ckpt1", dest)
    assert written == dest
    assert checksum == "abc"
    assert dest.read_bytes() == b"weights"


def test_download_checkpoint_stream_401(client, rmock, tmp_path):
    url = f"{API}/api/v1/training/jobs/job1/checkpoints/ckpt1/download"
    rmock.get(url, status_code=401)
    with pytest.raises(AuthError):
        client.download_checkpoint_stream("job1", "ckpt1", tmp_path / "x")


def test_download_checkpoint_stream_404(client, rmock, tmp_path):
    url = f"{API}/api/v1/training/jobs/job1/checkpoints/ckpt1/download"
    rmock.get(url, status_code=404)
    with pytest.raises(CheckpointNotFoundError):
        client.download_checkpoint_stream("job1", "ckpt1", tmp_path / "x")


def test_download_checkpoint_stream_500(client, rmock, tmp_path):
    url = f"{API}/api/v1/training/jobs/job1/checkpoints/ckpt1/download"
    rmock.get(url, status_code=500, text="boom")
    with pytest.raises(APIError):
        client.download_checkpoint_stream("job1", "ckpt1", tmp_path / "x")
