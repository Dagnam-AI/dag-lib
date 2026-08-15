"""Wire-level coverage for the sync run-scoped artifact-push client methods.

The four routes a training run calls to hand its own output to the model
registry with nothing but a run token: initiate, upload, complete, finalize.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import requests

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import APIError, AuthError, TrainingJobNotFoundError

if TYPE_CHECKING:
    from pathlib import Path

    from tests.typing_helpers import PytestMonkeyPatch, RequestsMocker

API = "https://api.test"


# ----------------------------------------------------------------- initiate


def test_initiate_run_artifacts_posts_the_declared_file_set(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(
        f"{API}/api/v1/training/jobs/j1/artifacts",
        json={"version_id": "v1", "status": "draft", "artifacts": []},
    )
    result = client.initiate_run_artifacts(
        "j1", {"files": [{"filename": "model.safetensors", "size_bytes": 7}]}
    )
    assert result["version_id"] == "v1"
    assert rmock.last_request.json() == {
        "files": [{"filename": "model.safetensors", "size_bytes": 7}]
    }
    assert rmock.last_request.headers["Authorization"] == "Bearer k"


def test_initiate_run_artifacts_quotes_the_job_id(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{API}/api/v1/training/jobs/a%2Fb/artifacts", json={"version_id": "v1"})
    assert client.initiate_run_artifacts("a/b", {"files": []})["version_id"] == "v1"


def test_initiate_run_artifacts_maps_404_to_job_not_found(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{API}/api/v1/training/jobs/j1/artifacts", status_code=404)
    with pytest.raises(TrainingJobNotFoundError):
        client.initiate_run_artifacts("j1", {"files": []})


def test_initiate_run_artifacts_surfaces_a_conflict(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(
        f"{API}/api/v1/training/jobs/j1/artifacts",
        status_code=409,
        json={"detail": "a concurrent push for this run is already in progress; retry"},
    )
    with pytest.raises(APIError, match="concurrent push") as excinfo:
        client.initiate_run_artifacts("j1", {"files": []})
    assert excinfo.value.status_code == 409


def test_initiate_run_artifacts_maps_401_to_auth_error(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{API}/api/v1/training/jobs/j1/artifacts", status_code=401)
    with pytest.raises(AuthError):
        client.initiate_run_artifacts("j1", {"files": []})


# ------------------------------------------------------------------- upload


def test_upload_run_artifact_posts_multipart(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"\x00\x01\x02")
    rmock.post(f"{API}/api/v1/training/jobs/j1/artifacts/a1/upload", status_code=204)

    assert client.upload_run_artifact("j1", "a1", weights) is True
    body = rmock.last_request.text
    assert body is not None
    assert 'name="file"' in body
    assert rmock.last_request.headers["Authorization"] == "Bearer k"


def test_upload_run_artifact_returns_false_when_already_committed(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"\x00")
    rmock.post(
        f"{API}/api/v1/training/jobs/j1/artifacts/a1/upload",
        status_code=409,
        json={"detail": "artifact a1 is already verified; its bytes cannot be replaced"},
    )
    assert client.upload_run_artifact("j1", "a1", weights) is False


def test_upload_run_artifact_maps_404_to_job_not_found(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"\x00")
    rmock.post(f"{API}/api/v1/training/jobs/j1/artifacts/nope/upload", status_code=404)
    with pytest.raises(TrainingJobNotFoundError):
        client.upload_run_artifact("j1", "nope", weights)


def test_upload_run_artifact_surfaces_a_rejected_body(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"\x00")
    rmock.post(
        f"{API}/api/v1/training/jobs/j1/artifacts/a1/upload",
        status_code=422,
        json={"detail": "upload exceeds the declared size"},
    )
    with pytest.raises(APIError, match="exceeds the declared size"):
        client.upload_run_artifact("j1", "a1", weights)


def test_upload_run_artifact_connectionerror(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"\x00")

    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "post", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.upload_run_artifact("j1", "a1", weights)


def test_upload_run_artifact_timeout(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"\x00")

    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "post", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.upload_run_artifact("j1", "a1", weights)


# --------------------------------------------------------- complete/finalize


def test_complete_run_artifact_posts_the_digest(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(
        f"{API}/api/v1/training/jobs/j1/artifacts/a1/complete",
        json={"id": "a1", "verification_status": "verified"},
    )
    result = client.complete_run_artifact("j1", "a1", {"sha256": "abc", "size_bytes": 3})
    assert result["verification_status"] == "verified"
    assert rmock.last_request.json() == {"sha256": "abc", "size_bytes": 3}


def test_complete_run_artifact_maps_404_to_job_not_found(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{API}/api/v1/training/jobs/j1/artifacts/a1/complete", status_code=404)
    with pytest.raises(TrainingJobNotFoundError):
        client.complete_run_artifact("j1", "a1", {"sha256": "abc", "size_bytes": 3})


def test_finalize_run_artifacts_commits_the_version(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(
        f"{API}/api/v1/training/jobs/j1/artifacts:finalize",
        json={"version_id": "v1", "status": "ready", "artifacts": []},
    )
    assert client.finalize_run_artifacts("j1") == {
        "version_id": "v1",
        "status": "ready",
        "artifacts": [],
    }


def test_finalize_run_artifacts_surfaces_an_incomplete_version(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(
        f"{API}/api/v1/training/jobs/j1/artifacts:finalize",
        status_code=422,
        json={"detail": "cannot finalize a version with an artifact that is not verified"},
    )
    with pytest.raises(APIError, match="not verified"):
        client.finalize_run_artifacts("j1")
