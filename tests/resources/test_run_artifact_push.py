"""Tests for ``dagnam.models.push_run_artifacts`` — a run pushing its own output.

Unlike ``push``, the caller names nothing: it declares its filenames and sizes
and the server derives the registry entry, version, artifact type and storage
key from the job the run token is scoped to.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import requests_mock as rm_module

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import APIError, ModelError
from dagnam.resources import models

if TYPE_CHECKING:
    from pathlib import Path

    from tests.typing_helpers import PytestMonkeyPatch


def _target(artifact_id: str, filename: str, **overrides: object) -> dict[str, object]:
    """One ``artifacts`` entry of a push response, local-backend shaped."""
    target: dict[str, object] = {
        "artifact_id": artifact_id,
        "filename": filename,
        "logical_key": f"weights/{filename}",
        "artifact_type": "weights",
        "committed": False,
        "upload_method": "POST",
        "upload_url": f"/api/v1/training/jobs/j1/artifacts/{artifact_id}/upload",
        "headers": {},
    }
    target.update(overrides)
    return target


def _client(*targets: dict[str, object]) -> MagicMock:
    client = MagicMock(spec=DagnamClient)
    client.initiate_run_artifacts.return_value = {
        "version_id": "v1",
        "status": "draft",
        "artifacts": list(targets),
    }
    client.upload_run_artifact.return_value = True
    client.complete_run_artifact.return_value = {"verification_status": "verified"}
    client.finalize_run_artifacts.return_value = {"version_id": "v1", "status": "ready"}
    return client


def test_uploads_every_file_and_finalizes(tmp_path: Path) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    config = tmp_path / "config.json"
    config.write_bytes(b"{}")
    client = _client(_target("a1", "model.safetensors"), _target("a2", "config.json"))

    result = models.push_run_artifacts(
        job_id="j1", files=[str(weights), str(config)], client=client
    )

    assert result["status"] == "ready"
    client.initiate_run_artifacts.assert_called_once_with(
        "j1",
        {
            "files": [
                {"filename": "model.safetensors", "size_bytes": 7},
                {"filename": "config.json", "size_bytes": 2},
            ]
        },
    )
    assert [call.args for call in client.upload_run_artifact.call_args_list] == [
        ("j1", "a1", weights),
        ("j1", "a2", config),
    ]
    assert client.complete_run_artifact.call_args_list[0].args == (
        "j1",
        "a1",
        {"sha256": hashlib.sha256(b"weights").hexdigest(), "size_bytes": 7},
    )
    client.finalize_run_artifacts.assert_called_once_with("j1")


def test_uses_the_presigned_put_when_the_server_returns_one(tmp_path: Path) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    client = _client(
        _target(
            "a1",
            "model.safetensors",
            upload_method="PUT",
            upload_url="https://s3.example.com/bucket/key?X-Amz-Signature=abc",
            headers={"Content-Type": "application/octet-stream"},
        )
    )

    with rm_module.Mocker() as m:
        m.put("https://s3.example.com/bucket/key", status_code=200)
        result = models.push_run_artifacts(job_id="j1", files=[str(weights)], client=client)
        assert m.last_request is not None
        assert m.last_request.headers["Content-Type"] == "application/octet-stream"

    assert result["status"] == "ready"
    client.upload_run_artifact.assert_not_called()
    client.complete_run_artifact.assert_called_once()


def test_missing_file_raises_before_any_network_call(tmp_path: Path) -> None:
    client = _client()
    present = tmp_path / "model.safetensors"
    present.write_bytes(b"weights")

    with pytest.raises(FileNotFoundError, match=r"nope\.safetensors"):
        models.push_run_artifacts(
            job_id="j1",
            files=[str(present), str(tmp_path / "nope.safetensors")],
            client=client,
        )

    client.initiate_run_artifacts.assert_not_called()


def test_a_mid_upload_failure_surfaces(tmp_path: Path) -> None:
    first = tmp_path / "model.safetensors"
    first.write_bytes(b"weights")
    second = tmp_path / "config.json"
    second.write_bytes(b"{}")
    client = _client(_target("a1", "model.safetensors"), _target("a2", "config.json"))
    client.upload_run_artifact.side_effect = APIError(500, "storage unavailable")

    with pytest.raises(APIError, match="storage unavailable"):
        models.push_run_artifacts(job_id="j1", files=[str(first), str(second)], client=client)

    client.complete_run_artifact.assert_not_called()
    client.finalize_run_artifacts.assert_not_called()


def test_a_failing_presigned_put_surfaces(tmp_path: Path) -> None:
    """The object-storage upload leg — the production path — must not be silent.

    It raises ``ModelError``, which is NOT an ``APIError``: a caller that
    catches only ``APIError`` would let this escape. The version must be left
    un-finalized so nothing downstream resolves a half-pushed model.
    """
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    config = tmp_path / "config.json"
    config.write_bytes(b"{}")
    client = _client(
        _target(
            "a1",
            "model.safetensors",
            upload_method="PUT",
            upload_url="https://s3.example.com/bucket/key?X-Amz-Signature=expired",
        ),
        _target("a2", "config.json"),
    )

    with rm_module.Mocker() as m:
        m.put("https://s3.example.com/bucket/key", status_code=403, text="Request has expired")
        with pytest.raises(ModelError, match=r"model\.safetensors"):
            models.push_run_artifacts(job_id="j1", files=[str(weights), str(config)], client=client)

    client.complete_run_artifact.assert_not_called()
    client.upload_run_artifact.assert_not_called()  # the second file never started
    client.finalize_run_artifacts.assert_not_called()


def test_skips_a_file_an_earlier_attempt_already_committed(tmp_path: Path) -> None:
    done = tmp_path / "model.safetensors"
    done.write_bytes(b"weights")
    pending = tmp_path / "config.json"
    pending.write_bytes(b"{}")
    client = _client(_target("a1", "model.safetensors"), _target("a2", "config.json"))
    client.upload_run_artifact.side_effect = [False, True]

    result = models.push_run_artifacts(job_id="j1", files=[str(done), str(pending)], client=client)

    assert result["status"] == "ready"
    assert [call.args[1] for call in client.complete_run_artifact.call_args_list] == ["a2"]
    client.finalize_run_artifacts.assert_called_once_with("j1")


def test_a_committed_entry_is_neither_uploaded_nor_completed(tmp_path: Path) -> None:
    """A resumed push must not re-write bytes the server already verified.

    ``committed`` is the authoritative signal and the entry carries no upload
    capability at all — no route, and on object storage no presigned URL. A
    client that inferred "already there" from a conflict on the upload leg
    would see no conflict here (there is no upload to conflict with) and would
    happily overwrite a verified artifact.
    """
    done = tmp_path / "model.safetensors"
    done.write_bytes(b"weights")
    pending = tmp_path / "config.json"
    pending.write_bytes(b"{}")
    client = _client(
        _target("a1", "model.safetensors", committed=True, upload_method=None, upload_url=None),
        _target(
            "a2",
            "config.json",
            upload_method="PUT",
            upload_url="https://s3.example.com/bucket/key?X-Amz-Signature=abc",
        ),
    )

    with rm_module.Mocker() as m:
        m.put("https://s3.example.com/bucket/key", status_code=200)
        result = models.push_run_artifacts(
            job_id="j1", files=[str(done), str(pending)], client=client
        )
        assert m.call_count == 1  # only the uncommitted entry was written

    assert result["status"] == "ready"
    client.upload_run_artifact.assert_not_called()
    assert [call.args[1] for call in client.complete_run_artifact.call_args_list] == ["a2"]
    client.finalize_run_artifacts.assert_called_once_with("j1")


def test_every_entry_committed_leaves_only_the_finalize(tmp_path: Path) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    client = _client(
        _target("a1", "model.safetensors", committed=True, upload_method=None, upload_url=None)
    )

    assert models.push_run_artifacts(job_id="j1", files=[str(weights)], client=client) == {
        "version_id": "v1",
        "status": "ready",
    }
    client.upload_run_artifact.assert_not_called()
    client.complete_run_artifact.assert_not_called()
    client.finalize_run_artifacts.assert_called_once_with("j1")


def test_a_completed_push_rerun_returns_the_same_version(tmp_path: Path) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    client = _client()
    client.initiate_run_artifacts.return_value = {
        "version_id": "v1",
        "status": "ready",
        "artifacts": [],
    }

    result = models.push_run_artifacts(job_id="j1", files=[str(weights)], client=client)

    assert result == {"version_id": "v1", "status": "ready"}
    client.upload_run_artifact.assert_not_called()
    client.complete_run_artifact.assert_not_called()


def test_the_job_id_defaults_to_the_run_environment(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    client = _client(_target("a1", "model.safetensors"))
    monkeypatch.setenv("DAGNAM_JOB_ID", "j-from-env")

    models.push_run_artifacts(files=[str(weights)], client=client)

    assert client.initiate_run_artifacts.call_args.args[0] == "j-from-env"
    client.finalize_run_artifacts.assert_called_once_with("j-from-env")


def test_without_a_job_id_it_raises_before_any_network_call(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    client = _client()
    monkeypatch.delenv("DAGNAM_JOB_ID", raising=False)

    with pytest.raises(ModelError, match="DAGNAM_JOB_ID"):
        models.push_run_artifacts(files=[str(weights)], client=client)

    client.initiate_run_artifacts.assert_not_called()
