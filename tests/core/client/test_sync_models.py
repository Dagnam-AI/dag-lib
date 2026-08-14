"""Wire-level coverage for the sync ModelsClientMixin (model registry)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import requests

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import (
    APIError,
    AuthError,
    ModelError,
    ModelNotFoundError,
    ResponseError,
)

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch, RequestsMocker

API = "https://api.test"


# ------------------------------------------------------------------- entries


def test_create_model_entry(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/models", json={"id": "m1", "slug": "tiny-chat"}, status_code=201)
    result = client.create_model_entry({"name": "tiny-chat", "slug": "tiny-chat"})
    assert result == {"id": "m1", "slug": "tiny-chat"}


def test_create_model_entry_sends_idempotency_key(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{API}/api/v1/models", json={"id": "m1"}, status_code=201)
    client.create_model_entry({"name": "x", "slug": "x"})
    assert rmock.last_request.headers.get("Idempotency-Key")


def test_create_model_entry_409_duplicate_slug(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/models", status_code=409, text="slug already exists")
    with pytest.raises(ModelError):
        client.create_model_entry({"name": "x", "slug": "dup"})


def test_create_model_entry_404_project_not_found(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    """No model exists yet at create time, so a 404 maps to bare ModelError."""
    rmock.post(f"{API}/api/v1/models", status_code=404, text="Project not found")
    with pytest.raises(ModelError) as exc_info:
        client.create_model_entry({"name": "x", "slug": "y", "project_id": "missing"})
    assert not isinstance(exc_info.value, ModelNotFoundError)


def test_get_model_entry(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/models/m1", json={"id": "m1", "name": "tiny-chat"})
    assert client.get_model_entry("m1") == {"id": "m1", "name": "tiny-chat"}


def test_get_model_entry_404(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/models/missing", status_code=404, text="Model not found")
    with pytest.raises(ModelNotFoundError):
        client.get_model_entry("missing")


def test_get_model_entry_401(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/models/m1", status_code=401, text="unauthorized")
    with pytest.raises(AuthError):
        client.get_model_entry("m1")


def test_get_model_entry_500_raises_apierror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/models/m1", status_code=500, text="boom")
    client._sleep = lambda _s: None  # 500 on a GET is transient -> retried; don't sleep
    with pytest.raises(APIError):
        client.get_model_entry("m1")


def test_get_model_entry_non_json_body_raises(client: DagnamClient, rmock: RequestsMocker) -> None:
    """A 200 with a non-JSON body falls back to text, then fails object-narrowing."""
    rmock.get(f"{API}/api/v1/models/m1", text="not json", headers={"Content-Type": "text/plain"})
    with pytest.raises(ResponseError):
        client.get_model_entry("m1")


def test_list_model_entries_returns_array(client: DagnamClient, rmock: RequestsMocker) -> None:
    """GET /api/v1/models returns a bare JSON array, not an {"items": [...]} envelope."""
    rmock.get(f"{API}/api/v1/models", json=[{"id": "m1"}, {"id": "m2"}])
    assert client.list_model_entries() == [{"id": "m1"}, {"id": "m2"}]


def test_list_model_entries_query_params(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/models", json=[])
    client.list_model_entries(search="tiny", page=2, limit=10)
    assert rmock.last_request.qs == {"search": ["tiny"], "page": ["2"], "limit": ["10"]}


def test_update_model_entry(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.patch(f"{API}/api/v1/models/m1", json={"id": "m1", "name": "renamed"})
    assert client.update_model_entry("m1", {"name": "renamed"}) == {"id": "m1", "name": "renamed"}


def test_update_model_entry_404(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.patch(f"{API}/api/v1/models/missing", status_code=404)
    with pytest.raises(ModelNotFoundError):
        client.update_model_entry("missing", {"name": "x"})


def test_delete_model_entry_empty_body(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.delete(f"{API}/api/v1/models/m1", status_code=204, text="")
    assert client.delete_model_entry("m1") is None


def test_delete_model_entry_404(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.delete(f"{API}/api/v1/models/missing", status_code=404)
    with pytest.raises(ModelNotFoundError):
        client.delete_model_entry("missing")


# ------------------------------------------------------------------ versions


def test_create_model_version(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(
        f"{API}/api/v1/models/m1/versions",
        json={"id": "v1", "status": "draft"},
        status_code=201,
    )
    result = client.create_model_version("m1", {"origin": "trained"})
    assert result == {"id": "v1", "status": "draft"}


def test_create_model_version_sends_idempotency_key(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{API}/api/v1/models/m1/versions", json={"id": "v1"}, status_code=201)
    client.create_model_version("m1", {"origin": "trained"})
    assert rmock.last_request.headers.get("Idempotency-Key")


def test_create_model_version_404(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/models/missing/versions", status_code=404)
    with pytest.raises(ModelNotFoundError):
        client.create_model_version("missing", {"origin": "trained"})


def test_list_model_versions_returns_array(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/models/m1/versions", json=[{"id": "v1"}, {"id": "v2"}])
    assert client.list_model_versions("m1") == [{"id": "v1"}, {"id": "v2"}]


def test_get_model_version(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/model-versions/v1", json={"id": "v1", "status": "ready"})
    assert client.get_model_version("v1") == {"id": "v1", "status": "ready"}


def test_get_model_version_404(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/model-versions/missing", status_code=404)
    with pytest.raises(ModelNotFoundError):
        client.get_model_version("missing")


def test_get_model_version_lineage(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(
        f"{API}/api/v1/model-versions/v1/lineage",
        json={"version_id": "v1", "edges": []},
    )
    assert client.get_model_version_lineage("v1") == {"version_id": "v1", "edges": []}


def test_get_model_version_lineage_404(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/model-versions/missing/lineage", status_code=404)
    with pytest.raises(ModelNotFoundError):
        client.get_model_version_lineage("missing")


# ----------------------------------------------------------------- artifacts


def test_initiate_model_artifact(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(
        f"{API}/api/v1/model-versions/v1/artifacts:initiate",
        json={"artifact_id": "a1", "upload_method": "POST", "upload_url": "/x"},
        status_code=201,
    )
    result = client.initiate_model_artifact("v1", {"logical_key": "weights", "size_bytes": 10})
    assert result["artifact_id"] == "a1"


def test_initiate_model_artifact_sends_idempotency_key(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(
        f"{API}/api/v1/model-versions/v1/artifacts:initiate",
        json={"artifact_id": "a1", "upload_method": "POST", "upload_url": "/x"},
        status_code=201,
    )
    client.initiate_model_artifact("v1", {"logical_key": "weights", "size_bytes": 10})
    assert rmock.last_request.headers.get("Idempotency-Key")


def test_initiate_model_artifact_404_not_draft(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(
        f"{API}/api/v1/model-versions/v1/artifacts:initiate",
        status_code=404,
        text="Model version not found or not in draft status",
    )
    with pytest.raises(ModelNotFoundError):
        client.initiate_model_artifact("v1", {"logical_key": "weights", "size_bytes": 10})


def test_complete_model_artifact(client: DagnamClient, rmock: RequestsMocker) -> None:
    sha = hashlib.sha256(b"weights").hexdigest()
    rmock.post(
        f"{API}/api/v1/model-versions/v1/artifacts/a1/complete",
        json={"id": "a1", "verification_status": "verified"},
    )
    result = client.complete_model_artifact("v1", "a1", {"sha256": sha, "size_bytes": 7})
    assert result["verification_status"] == "verified"


def test_complete_model_artifact_404(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/model-versions/v1/artifacts/missing/complete", status_code=404)
    with pytest.raises(ModelNotFoundError):
        client.complete_model_artifact("v1", "missing", {"sha256": "x", "size_bytes": 1})


def test_finalize_model_version(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/model-versions/v1/finalize", json={"id": "v1", "status": "ready"})
    assert client.finalize_model_version("v1")["status"] == "ready"


def test_finalize_model_version_422_invalid_state(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(
        f"{API}/api/v1/model-versions/v1/finalize",
        status_code=422,
        text="cannot finalize a version with zero artifacts",
    )
    with pytest.raises(ModelError):
        client.finalize_model_version("v1")


def test_finalize_model_version_404(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/model-versions/missing/finalize", status_code=404)
    with pytest.raises(ModelNotFoundError):
        client.finalize_model_version("missing")


def test_get_task_contract(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(
        f"{API}/api/v1/task-contracts/chat/versions/v1",
        json={"key": "chat", "version": "v1"},
    )
    assert client.get_task_contract("chat", "v1") == {"key": "chat", "version": "v1"}


def test_get_task_contract_404(client: DagnamClient, rmock: RequestsMocker) -> None:
    """A task-contract lookup has no model/version id, so 404 -> bare ModelError."""
    rmock.get(f"{API}/api/v1/task-contracts/missing/versions/v1", status_code=404)
    with pytest.raises(ModelError) as exc_info:
        client.get_task_contract("missing", "v1")
    assert not isinstance(exc_info.value, ModelNotFoundError)


def test_list_model_version_artifacts_returns_array(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.get(f"{API}/api/v1/model-versions/v1/artifacts", json=[{"id": "a1"}, {"id": "a2"}])
    assert client.list_model_version_artifacts("v1") == [{"id": "a1"}, {"id": "a2"}]


def test_list_model_version_artifacts_404(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/model-versions/missing/artifacts", status_code=404)
    with pytest.raises(ModelNotFoundError):
        client.list_model_version_artifacts("missing")


# --------------------------------------------------------- direct file upload


def test_upload_model_artifact_direct(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    f = tmp_path / "weights.bin"
    f.write_bytes(b"\x00\x01\x02")
    rmock.post(
        f"{API}/api/v1/model-versions/v1/artifacts/a1/upload",
        status_code=204,
    )
    client.upload_model_artifact_direct("/api/v1/model-versions/v1/artifacts/a1/upload", f)
    body = rmock.last_request.text
    assert body is not None
    assert 'name="file"' in body
    assert rmock.last_request.headers["Authorization"] == "Bearer k"


def test_upload_model_artifact_direct_404(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    f = tmp_path / "weights.bin"
    f.write_bytes(b"\x00")
    rmock.post(f"{API}/api/v1/model-versions/v1/artifacts/missing/upload", status_code=404)
    with pytest.raises(ModelError):
        client.upload_model_artifact_direct("/api/v1/model-versions/v1/artifacts/missing/upload", f)


def test_upload_model_artifact_direct_connectionerror(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    f = tmp_path / "weights.bin"
    f.write_bytes(b"\x00")

    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "post", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.upload_model_artifact_direct("/api/v1/model-versions/v1/artifacts/a1/upload", f)


def test_upload_model_artifact_direct_timeout(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    f = tmp_path / "weights.bin"
    f.write_bytes(b"\x00")

    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "post", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.upload_model_artifact_direct("/api/v1/model-versions/v1/artifacts/a1/upload", f)


# ------------------------------------------------------- artifact download


def test_download_model_artifact_stream_direct(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    body = b"weights-bytes"
    sha = hashlib.sha256(body).hexdigest()
    url = f"{API}/api/v1/model-versions/v1/artifacts/a1/download"
    rmock.get(
        url,
        content=body,
        headers={"Content-Length": str(len(body)), "X-Checksum-SHA256": sha},
    )
    dest = tmp_path / "a1.bin"

    written, expected_sha = client.download_model_artifact_stream("v1", "a1", dest)

    assert written == dest
    assert expected_sha == sha
    assert dest.read_bytes() == body
    assert rmock.last_request.headers["Authorization"] == "Bearer k"


def test_download_model_artifact_stream_redirect_to_presigned(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    """A 307 Location (presigned object-storage URL) is followed with no auth header."""
    body = b"weights-from-s3"
    sha = hashlib.sha256(body).hexdigest()
    url = f"{API}/api/v1/model-versions/v1/artifacts/a1/download"
    presigned = "https://bucket.s3.example.com/a1?sig=xyz"
    rmock.get(url, status_code=307, headers={"Location": presigned})
    rmock.get(presigned, content=body, headers={"X-Checksum-SHA256": sha})
    dest = tmp_path / "a1.bin"

    written, expected_sha = client.download_model_artifact_stream("v1", "a1", dest)

    assert written == dest
    assert dest.read_bytes() == body
    assert expected_sha == sha
    assert "Authorization" not in rmock.last_request.headers


def test_download_model_artifact_stream_redirect_preserves_api_checksum(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    """An unauthenticated redirect target cannot replace the API checksum."""
    url = f"{API}/api/v1/model-versions/v1/artifacts/a1/download"
    presigned = "https://bucket.s3.example.com/a1?sig=untrusted"
    rmock.get(
        url,
        status_code=307,
        headers={"Location": presigned, "X-Checksum-SHA256": "authenticated-sha"},
    )
    rmock.get(
        presigned,
        content=b"substituted-weights",
        headers={"X-Checksum-SHA256": "attacker-controlled-sha"},
    )

    _written, expected_sha = client.download_model_artifact_stream("v1", "a1", tmp_path / "a1.bin")

    assert expected_sha == "authenticated-sha"


def test_download_model_artifact_stream_401(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    url = f"{API}/api/v1/model-versions/v1/artifacts/a1/download"
    rmock.get(url, status_code=401, text="unauthorized")
    with pytest.raises(AuthError):
        client.download_model_artifact_stream("v1", "a1", tmp_path / "x.bin")


def test_download_model_artifact_stream_404(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    url = f"{API}/api/v1/model-versions/v1/artifacts/missing/download"
    rmock.get(url, status_code=404, text="not found")
    with pytest.raises(ModelNotFoundError):
        client.download_model_artifact_stream("v1", "missing", tmp_path / "x.bin")


def test_download_model_artifact_stream_other_error(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    url = f"{API}/api/v1/model-versions/v1/artifacts/a1/download"
    rmock.get(url, status_code=500, text="boom")
    with pytest.raises(APIError):
        client.download_model_artifact_stream("v1", "a1", tmp_path / "x.bin")
