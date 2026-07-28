"""Wire-level HTTP tests using requests-mock.

Asserts DagnamClient hits the right URLs with the right headers, bodies,
and query params - without running a real server.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import requests_mock as rm_module

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import (
    APIError,
    AuthError,
    CheckpointNotFoundError,
    DeploymentNotFoundError,
    QuotaExceededError,
    TrainingJobNotFoundError,
)

if TYPE_CHECKING:
    from tests.typing_helpers import JsonObject, RequestsMocker

API = "https://api.test"


@pytest.fixture
def client() -> DagnamClient:
    return DagnamClient(API, "k")


@pytest.fixture
def rmock():
    with rm_module.Mocker() as m:
        yield m


def test_predict_sends_bearer_only(client: DagnamClient, rmock: RequestsMocker) -> None:
    url = f"{API}/api/v1/inference/dep_1/predict"
    rmock.post(url, json={"label": "ok"})

    result = client.predict("dep_1", {"x": 1})

    assert result == {"label": "ok"}
    req = rmock.last_request
    assert req.headers["Authorization"] == "Bearer k"
    assert "X-API-Key" not in req.headers
    assert req.json() == {"x": 1}


def test_predict_batch_wraps_inputs(client: DagnamClient, rmock: RequestsMocker) -> None:
    url = f"{API}/api/v1/inference/dep_1/predict/batch"
    rmock.post(url, json=[{"y": 1}, {"y": 2}])

    result = client.predict_batch("dep_1", [{"x": 1}, {"x": 2}])

    assert result == [{"y": 1}, {"y": 2}]
    assert rmock.last_request.json() == {"inputs": [{"x": 1}, {"x": 2}]}


def test_deployment_health_get(client: DagnamClient, rmock: RequestsMocker) -> None:
    url = f"{API}/api/v1/inference/dep_1/health"
    rmock.get(url, json={"status": "healthy"})

    assert client.deployment_health("dep_1") == {"status": "healthy"}


def test_path_identifiers_are_percent_encoded(client: DagnamClient, rmock: RequestsMocker) -> None:
    """Untrusted identifiers must not inject extra URL path/query syntax."""
    url = f"{API}/api/v1/datasets/tenant%2Fdata%3Fversion%3Dlatest/meta"
    rmock.get(url, json={"id": "tenant/data?version=latest"})

    assert client.get_dataset_meta("tenant/data?version=latest") == {
        "id": "tenant/data?version=latest"
    }
    assert rmock.last_request.path.lower() == (
        "/api/v1/datasets/tenant%2Fdata%3Fversion%3Dlatest/meta".lower()
    )


def test_checkpoint_download_streams_and_returns_checksum(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
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


def test_sse_mints_stream_token_query_param_not_api_key(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    """Contract: SSE uses a minted short-lived ?token=, never ?api_key=."""
    rmock.post(
        f"{API}/api/v1/training/jobs/job_1/stream-access-token",
        json={"token": "stream-t"},
    )
    url = f"{API}/api/v1/streaming/training-jobs/job_1/stream"
    rmock.get(url, text="")

    resp = client.open_training_stream("job_1", last_event_id="42")
    resp.close()

    req = rmock.last_request
    assert req.qs.get("token") == ["stream-t"]
    assert "api_key" not in req.qs
    assert req.headers["Accept"] == "text/event-stream"
    assert req.headers["Last-Event-ID"] == "42"
    assert "X-API-Key" not in req.headers


def test_inference_404_maps_to_deploymenterror(client: DagnamClient, rmock: RequestsMocker) -> None:
    url = f"{API}/api/v1/inference/dep_404/predict"
    rmock.post(url, status_code=404, text="not found")

    with pytest.raises(DeploymentNotFoundError):
        client.predict("dep_404", {"x": 1})


def test_checkpoint_401_maps_to_autherror(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    url = f"{API}/api/v1/training/jobs/job_1/checkpoints/ck_1/download"
    rmock.get(url, status_code=401, text="unauthorized")

    with pytest.raises(AuthError):
        client.download_checkpoint_stream("job_1", "ck_1", tmp_path / "x.pt")


def test_checkpoint_404_maps_to_checkpoint_not_found(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    url = f"{API}/api/v1/training/jobs/job_1/checkpoints/ck_bad/download"
    rmock.get(url, status_code=404, text="not found")

    with pytest.raises(CheckpointNotFoundError):
        client.download_checkpoint_stream("job_1", "ck_bad", tmp_path / "x.pt")


def test_checkpoint_redirect_is_followed_to_presigned_url(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    """A redirect Location (presigned object-storage URL) is followed and streamed.

    The redirect follow-up must NOT carry the API key (presigned URLs are
    self-authenticating and reject a forwarded Authorization header).
    """
    body = b"weights-from-s3"
    sha = hashlib.sha256(body).hexdigest()
    url = f"{API}/api/v1/training/jobs/job_1/checkpoints/ck_1/download"
    presigned = "https://bucket.s3.example.com/ck_1?sig=xyz"
    rmock.get(url, status_code=307, headers={"Location": presigned})
    rmock.get(presigned, content=body, headers={"X-Checksum-SHA256": sha})
    dest = tmp_path / "ck_1.pt"

    written, expected_sha = client.download_checkpoint_stream("job_1", "ck_1", dest)

    assert written == dest
    assert dest.read_bytes() == body
    assert expected_sha == sha
    # The presigned follow-up (the last request) carries no Authorization header.
    assert "Authorization" not in rmock.last_request.headers


def test_sse_redirect_is_rejected(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(
        f"{API}/api/v1/training/jobs/job_1/stream-access-token",
        json={"token": "stream-t"},
    )
    url = f"{API}/api/v1/streaming/training-jobs/job_1/stream"
    rmock.get(url, status_code=302, headers={"Location": "https://evil.test/stream"})

    with pytest.raises(APIError):
        client.open_training_stream("job_1")


def test_codegen_generate_posts_framework_and_version(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    url = f"{API}/api/v1/projects/p1/generate-code"
    rmock.post(url, json={"files": []})

    result = client.generate_code("p1", framework="tensorflow", version_id="v2")

    assert result == {"files": []}
    req = rmock.last_request
    assert req.json() == {"framework": "tensorflow", "version_id": "v2"}
    assert req.qs == {}


def test_codegen_generate_async_sets_query_param(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    url = f"{API}/api/v1/projects/p1/generate-code"
    rmock.post(url, json={"task_id": "t1", "status": "pending"})

    result = client.generate_code("p1", framework="pytorch", async_mode=True)

    assert result["task_id"] == "t1"
    assert rmock.last_request.qs["async_mode"] == ["true"]
    assert rmock.last_request.json() == {"framework": "pytorch"}


def test_codegen_preview_uses_project_preview_route(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    url = f"{API}/api/v1/projects/p1/code-preview"
    rmock.get(url, json={"files": []})

    result = client.preview_code("p1", framework="pytorch", version_id="v4")

    assert result == {"files": []}
    assert rmock.last_request.qs["framework"] == ["pytorch"]
    assert rmock.last_request.qs["version_id"] == ["v4"]


def test_codegen_validate_posts_to_project_route(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    url = f"{API}/api/v1/projects/p1/validate"
    rmock.post(url, json={"is_valid": True})

    assert client.validate_code("p1", version_id="v1") == {"is_valid": True}
    assert rmock.last_request.qs["version_id"] == ["v1"]


def test_codegen_status_uses_project_status_route(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    url = f"{API}/api/v1/projects/p1/code-status/t1"
    rmock.get(url, json={"status": "completed"})

    assert client.get_code_status("p1", "t1") == {"status": "completed"}


def test_codegen_download_uses_project_download_route(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
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


# --------------------------------------------------------------- training jobs


def test_create_training_job_posts_payload(client: DagnamClient, rmock: RequestsMocker) -> None:
    url = f"{API}/api/v1/training/jobs"
    rmock.post(url, status_code=201, json={"id": "j1", "status": "pending"})

    payload: JsonObject = {"project_id": "p1", "framework": "pytorch", "config": {"epochs": 1}}
    result = client.create_training_job(payload)

    assert result == {"id": "j1", "status": "pending"}
    req = rmock.last_request
    assert req.json() == payload
    assert req.headers["Authorization"] == "Bearer k"


def test_create_training_job_over_limit_maps_to_quota_exceeded(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    """A 402 plan-limit rejection surfaces as QuotaExceededError."""
    url = f"{API}/api/v1/training/jobs"
    rmock.post(url, status_code=402, json={"detail": "limit_exceeded"})

    with pytest.raises(QuotaExceededError):
        client.create_training_job({"project_id": "p1"})


def test_get_training_job(client: DagnamClient, rmock: RequestsMocker) -> None:
    url = f"{API}/api/v1/training/jobs/j1"
    rmock.get(url, json={"id": "j1", "status": "running"})

    assert client.get_training_job("j1") == {"id": "j1", "status": "running"}


def test_get_training_job_404_maps_to_not_found(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    url = f"{API}/api/v1/training/jobs/missing"
    rmock.get(url, status_code=404, text="not found")

    with pytest.raises(TrainingJobNotFoundError):
        client.get_training_job("missing")


def test_list_training_jobs_passes_query(client: DagnamClient, rmock: RequestsMocker) -> None:
    url = f"{API}/api/v1/training/jobs"
    rmock.get(url, json={"items": [], "total": 0})

    client.list_training_jobs(page=2, limit=5, status_filter="running,completed", project_id="p1")
    qs = rmock.last_request.qs
    assert qs["page"] == ["2"]
    assert qs["status_filter"] == ["running,completed"]
    assert qs["project_id"] == ["p1"]


def test_cancel_training_job_posts(client: DagnamClient, rmock: RequestsMocker) -> None:
    url = f"{API}/api/v1/training/jobs/j1/cancel"
    rmock.post(url, json={"message": "Training job cancelled successfully"})

    assert client.cancel_training_job("j1") == {"message": "Training job cancelled successfully"}


def test_cancel_terminal_job_maps_to_apierror(client: DagnamClient, rmock: RequestsMocker) -> None:
    url = f"{API}/api/v1/training/jobs/j1/cancel"
    rmock.post(url, status_code=400, json={"detail": "Cannot cancel job with status completed"})

    with pytest.raises(APIError) as excinfo:
        client.cancel_training_job("j1")
    assert excinfo.value.status_code == 400


def test_bulk_delete_training_jobs_posts_ids(client: DagnamClient, rmock: RequestsMocker) -> None:
    url = f"{API}/api/v1/training/jobs/bulk-delete"
    rmock.post(url, json={"deleted": 2})

    assert client.bulk_delete_training_jobs(["j1", "j2"]) == {"deleted": 2}
    assert rmock.last_request.json() == {"job_ids": ["j1", "j2"]}


def test_training_job_path_is_percent_encoded(client: DagnamClient, rmock: RequestsMocker) -> None:
    url = f"{API}/api/v1/training/jobs/a%2Fb/cancel"
    rmock.post(url, json={"message": "ok"})

    client.cancel_training_job("a/b")
    assert rmock.last_request.path.lower() == "/api/v1/training/jobs/a%2fb/cancel"


def test_training_logs_pass_query(client: DagnamClient, rmock: RequestsMocker) -> None:
    url = f"{API}/api/v1/training/jobs/job-1/logs"
    rmock.get(url, json={"items": []})

    assert client.get_training_logs("job-1", log_level="error", page=2, limit=5) == {"items": []}
    assert rmock.last_request.qs == {
        "log_level": ["error"],
        "page": ["2"],
        "limit": ["5"],
    }


def test_training_metrics_pass_query(client: DagnamClient, rmock: RequestsMocker) -> None:
    url = f"{API}/api/v1/training/jobs/job-1/metrics"
    rmock.get(url, json={"items": []})

    assert client.get_training_metrics("job-1", metric_type="train_loss", epoch_summary=True) == {
        "items": []
    }
    assert rmock.last_request.qs == {
        "metric_type": ["train_loss"],
        "epoch_summary": ["true"],
    }


def test_training_metrics_summary_wire(client: DagnamClient, rmock: RequestsMocker) -> None:
    url = f"{API}/api/v1/training/jobs/job-1/metrics/summary"
    rmock.get(url, json={"best_epoch": 2})

    assert client.get_training_metrics_summary("job-1") == {"best_epoch": 2}


# ----------------------------------------------------------------- account/usage


def test_get_entitlements(client: DagnamClient, rmock: RequestsMocker) -> None:
    url = f"{API}/api/v1/users/me/entitlements"
    rmock.get(url, json={"plan": {"code": "pro"}, "limits": []})

    assert client.get_entitlements() == {"plan": {"code": "pro"}, "limits": []}
    assert rmock.last_request.headers["Authorization"] == "Bearer k"


def test_get_storage_quota(client: DagnamClient, rmock: RequestsMocker) -> None:
    url = f"{API}/api/v1/datasets/storage/quota"
    rmock.get(url, json={"used_bytes": 1, "limit_bytes": 100})

    assert client.get_storage_quota() == {"used_bytes": 1, "limit_bytes": 100}


def test_get_api_key_usage(client: DagnamClient, rmock: RequestsMocker) -> None:
    url = f"{API}/api/v1/users/me/api-keys/key_1/usage"
    rmock.get(url, json={"usage_count": 7})

    assert client.get_api_key_usage("key_1") == {"usage_count": 7}


def test_entitlements_401_maps_to_autherror(client: DagnamClient, rmock: RequestsMocker) -> None:
    url = f"{API}/api/v1/users/me/entitlements"
    rmock.get(url, status_code=401, text="unauthorized")

    with pytest.raises(AuthError):
        client.get_entitlements()
