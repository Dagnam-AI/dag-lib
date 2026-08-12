"""Tests for the dagnam.resources.models push/download/cache surface."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests
import requests_mock as rm_module

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import (
    APIError,
    ChecksumError,
    ModelError,
    ModelNotFoundError,
)
from dagnam.resources import models


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fake_download(body: bytes):
    """Return a ``download_model_artifact_stream`` side_effect writing *body*."""
    sha = _sha256(body)

    def _side_effect(version_id: str, artifact_id: str, dest: Path) -> tuple[Path, str]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        return dest, sha

    return _side_effect, sha


@pytest.fixture
def model_cache(tmp_path: Path) -> Path:
    d = tmp_path / "models"
    d.mkdir()
    return d


class TestInferArtifactType:
    def test_adapter_file(self) -> None:
        assert models._infer_artifact_type(Path("adapter_model.safetensors")) == "adapter"

    def test_adapter_config_file(self) -> None:
        assert models._infer_artifact_type(Path("adapter_config.json")) == "adapter"

    def test_tokenizer_file(self) -> None:
        assert models._infer_artifact_type(Path("tokenizer.json")) == "tokenizer"

    def test_tokenizer_config_file(self) -> None:
        assert models._infer_artifact_type(Path("tokenizer_config.json")) == "tokenizer"

    def test_special_tokens_map_file(self) -> None:
        assert models._infer_artifact_type(Path("special_tokens_map.json")) == "tokenizer"

    def test_config_file(self) -> None:
        assert models._infer_artifact_type(Path("config.json")) == "architecture_config"

    def test_readme_file(self) -> None:
        assert models._infer_artifact_type(Path("README.md")) == "readme"

    def test_default_is_weights(self) -> None:
        assert models._infer_artifact_type(Path("pytorch_model.bin")) == "weights"


class TestPush:
    def test_pushes_files_and_finalizes(self, tmp_path: Path) -> None:
        weight_file = tmp_path / "adapter_model.safetensors"
        weight_file.write_bytes(b"weights")

        client = MagicMock(spec=DagnamClient)
        client.create_model_entry.return_value = {"id": "entry-1", "slug": "tiny-chat"}
        client.create_model_version.return_value = {"id": "version-1", "status": "draft"}
        client.initiate_model_artifact.return_value = {
            "artifact_id": "artifact-1",
            "upload_method": "POST",
            "upload_url": "/api/v1/model-versions/version-1/artifacts/artifact-1/upload",
            "headers": {},
        }
        client.complete_model_artifact.return_value = {
            "id": "artifact-1",
            "verification_status": "verified",
        }
        client.finalize_model_version.return_value = {"id": "version-1", "status": "ready"}

        result = models.push(
            name="tiny-chat",
            slug="tiny-chat",
            description="A tiny chat model",
            files=[str(weight_file)],
            client=client,
        )

        assert result["status"] == "ready"
        client.create_model_entry.assert_called_once()
        client.upload_model_artifact_direct.assert_called_once_with(
            "/api/v1/model-versions/version-1/artifacts/artifact-1/upload", weight_file
        )
        client.finalize_model_version.assert_called_once_with("version-1")

    def test_pushes_via_presigned_put(self, tmp_path: Path) -> None:
        weight_file = tmp_path / "pytorch_model.bin"
        weight_file.write_bytes(b"weights")

        client = MagicMock(spec=DagnamClient)
        client.create_model_entry.return_value = {"id": "entry-1"}
        client.create_model_version.return_value = {"id": "version-1"}
        client.initiate_model_artifact.return_value = {
            "artifact_id": "artifact-1",
            "upload_method": "PUT",
            "upload_url": "https://s3.example.com/bucket/key?X-Amz-Signature=abc",
            "headers": {"Content-Type": "application/octet-stream"},
        }
        client.complete_model_artifact.return_value = {"id": "artifact-1"}
        client.finalize_model_version.return_value = {"id": "version-1", "status": "ready"}

        with rm_module.Mocker() as m:
            m.put("https://s3.example.com/bucket/key", status_code=200)
            result = models.push(
                name="m", slug="m", description="d", files=[str(weight_file)], client=client
            )
            assert m.last_request is not None
            assert m.last_request.headers["Content-Type"] == "application/octet-stream"

        assert result["status"] == "ready"
        client.upload_model_artifact_direct.assert_not_called()

    def test_missing_file_raises_before_any_network_call(self, tmp_path: Path) -> None:
        client = MagicMock(spec=DagnamClient)
        missing = tmp_path / "nope.safetensors"

        with pytest.raises(FileNotFoundError):
            models.push(name="m", slug="m", description="d", files=[str(missing)], client=client)

        client.create_model_entry.assert_not_called()

    def test_missing_second_file_raises_before_any_network_call(self, tmp_path: Path) -> None:
        # The upfront validation loop must check every file before the FIRST
        # network call -- a typo on file #2 must not leave an orphaned draft
        # entry/version on the server from file #1 having already been valid.
        present = tmp_path / "config.json"
        present.write_text("{}")
        missing = tmp_path / "nope.bin"
        client = MagicMock(spec=DagnamClient)

        with pytest.raises(FileNotFoundError):
            models.push(
                name="m",
                slug="m",
                description="d",
                files=[str(present), str(missing)],
                client=client,
            )

        client.create_model_entry.assert_not_called()
        client.create_model_version.assert_not_called()

    def test_pushes_multiple_files_with_inferred_types(self, tmp_path: Path) -> None:
        weight_file = tmp_path / "pytorch_model.bin"
        weight_file.write_bytes(b"w")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")

        client = MagicMock(spec=DagnamClient)
        client.create_model_entry.return_value = {"id": "entry-1"}
        client.create_model_version.return_value = {"id": "version-1"}
        client.initiate_model_artifact.return_value = {
            "artifact_id": "artifact-1",
            "upload_method": "POST",
            "upload_url": "/upload",
            "headers": {},
        }
        client.complete_model_artifact.return_value = {}
        client.finalize_model_version.return_value = {"id": "version-1", "status": "ready"}

        models.push(
            name="m",
            slug="m",
            description="d",
            files=[str(weight_file), str(config_file)],
            client=client,
        )

        assert client.initiate_model_artifact.call_count == 2
        artifact_types = [
            call.args[1]["artifact_type"] for call in client.initiate_model_artifact.call_args_list
        ]
        assert artifact_types == ["weights", "architecture_config"]
        assert client.complete_model_artifact.call_count == 2


class TestPutToPresignedUrl:
    def test_success_sends_no_auth_header(self, tmp_path: Path) -> None:
        f = tmp_path / "weights.bin"
        f.write_bytes(b"\x00\x01\x02")

        with rm_module.Mocker() as m:
            m.put("https://s3.example.com/upload", status_code=200)
            models._put_to_presigned_url(
                "https://s3.example.com/upload", f, {"Content-Type": "application/octet-stream"}
            )
            assert m.last_request is not None
            assert m.last_request.headers["Content-Type"] == "application/octet-stream"
            assert "Authorization" not in m.last_request.headers

    def test_non_dict_headers_default_to_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "weights.bin"
        f.write_bytes(b"\x00")

        with rm_module.Mocker() as m:
            m.put("https://s3.example.com/upload", status_code=204)
            models._put_to_presigned_url("https://s3.example.com/upload", f, None)

    def test_non_success_status_raises_model_error(self, tmp_path: Path) -> None:
        f = tmp_path / "weights.bin"
        f.write_bytes(b"\x00")

        with rm_module.Mocker() as m:
            m.put("https://s3.example.com/upload", status_code=403, text="Forbidden")
            with pytest.raises(ModelError, match="403"):
                models._put_to_presigned_url("https://s3.example.com/upload", f, None)

    def test_connection_error_maps_to_api_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f = tmp_path / "weights.bin"
        f.write_bytes(b"\x00")

        def _boom(*_a: object, **_kw: object) -> None:
            raise requests.ConnectionError("nope")

        monkeypatch.setattr(requests, "put", _boom)
        with pytest.raises(APIError, match="Connection failed"):
            models._put_to_presigned_url("https://s3.example.com/upload", f, None)

    def test_timeout_maps_to_api_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f = tmp_path / "weights.bin"
        f.write_bytes(b"\x00")

        def _boom(*_a: object, **_kw: object) -> None:
            raise requests.Timeout("slow")

        monkeypatch.setattr(requests, "put", _boom)
        with pytest.raises(APIError, match="Request timed out"):
            models._put_to_presigned_url("https://s3.example.com/upload", f, None)


class TestResolve:
    def test_delegates_to_client(self) -> None:
        client = MagicMock(spec=DagnamClient)
        client.get_model_version.return_value = {"id": "v1", "status": "ready"}

        result = models.resolve("v1", client=client)

        assert result == {"id": "v1", "status": "ready"}
        client.get_model_version.assert_called_once_with("v1")


class TestGetLineage:
    def test_delegates_to_client(self) -> None:
        client = MagicMock(spec=DagnamClient)
        client.get_model_version_lineage.return_value = {"parents": []}

        result = models.get_lineage("v1", client=client)

        assert result == {"parents": []}
        client.get_model_version_lineage.assert_called_once_with("v1")


class TestGetTaskContract:
    def test_delegates_to_client(self) -> None:
        client = MagicMock(spec=DagnamClient)
        client.get_task_contract.return_value = {"key": "chat", "version": "1.0"}

        result = models.get_task_contract("chat", "1.0", client=client)

        assert result == {"key": "chat", "version": "1.0"}
        client.get_task_contract.assert_called_once_with("chat", "1.0")


class TestDownload:
    def test_fresh_download_writes_verifies_and_caches(self, model_cache: Path) -> None:
        body = b"weights-bytes"
        side, sha = _fake_download(body)
        client = MagicMock(spec=DagnamClient)
        client.list_model_version_artifacts.return_value = [{"id": "a1", "sha256": sha}]
        client.download_model_artifact_stream.side_effect = side

        path = models.download("v1", "a1", client=client, cache_dir=model_cache)

        assert path == model_cache / "v1" / "a1.bin"
        assert path.read_bytes() == body
        assert not (model_cache / "v1" / "a1.bin.partial").exists()

    def test_cache_hit_skips_stream_download(self, model_cache: Path) -> None:
        body = b"already-cached"
        sha = _sha256(body)
        cached = model_cache / "v1" / "a1.bin"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(body)
        client = MagicMock(spec=DagnamClient)
        client.list_model_version_artifacts.return_value = [{"id": "a1", "sha256": sha}]

        path = models.download("v1", "a1", client=client, cache_dir=model_cache)

        assert path == cached
        client.download_model_artifact_stream.assert_not_called()

    def test_stale_local_file_triggers_redownload(self, model_cache: Path) -> None:
        stale = model_cache / "v1" / "a1.bin"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"stale-wrong-bytes")
        body = b"fresh-correct-bytes"
        side, sha = _fake_download(body)
        client = MagicMock(spec=DagnamClient)
        client.list_model_version_artifacts.return_value = [{"id": "a1", "sha256": sha}]
        client.download_model_artifact_stream.side_effect = side

        path = models.download("v1", "a1", client=client, cache_dir=model_cache)

        assert path.read_bytes() == body
        client.download_model_artifact_stream.assert_called_once()

    def test_peer_finishes_while_waiting_for_lock(
        self, model_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = b"peer-downloaded-this"
        sha = _sha256(body)
        dest = model_cache / "v1" / "a1.bin"

        class _FakeLock:
            def __enter__(self) -> _FakeLock:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(body)
                return self

            def __exit__(self, *exc_info: object) -> bool:
                return False

        monkeypatch.setattr(models, "dataset_lock", lambda *a, **k: _FakeLock())
        client = MagicMock(spec=DagnamClient)
        client.list_model_version_artifacts.return_value = [{"id": "a1", "sha256": sha}]

        path = models.download("v1", "a1", client=client, cache_dir=model_cache)

        assert path == dest
        client.download_model_artifact_stream.assert_not_called()

    def test_artifact_not_found_raises(self, model_cache: Path) -> None:
        client = MagicMock(spec=DagnamClient)
        client.list_model_version_artifacts.return_value = [{"id": "other", "sha256": "x"}]

        with pytest.raises(ModelNotFoundError):
            models.download("v1", "missing", client=client, cache_dir=model_cache)

    def test_non_string_sha256_falls_back_to_server_checksum(self, model_cache: Path) -> None:
        body = b"weights"
        side, _ = _fake_download(body)
        client = MagicMock(spec=DagnamClient)
        client.list_model_version_artifacts.return_value = [{"id": "a1", "sha256": 12345}]
        client.download_model_artifact_stream.side_effect = side

        path = models.download("v1", "a1", client=client, cache_dir=model_cache)

        assert path.read_bytes() == body

    def test_missing_checksum_everywhere_raises_and_cleans_up(self, model_cache: Path) -> None:
        def side_effect(version_id: str, artifact_id: str, dest: Path) -> tuple[Path, None]:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"unverifiable")
            return dest, None

        client = MagicMock(spec=DagnamClient)
        client.list_model_version_artifacts.return_value = [{"id": "a1"}]
        client.download_model_artifact_stream.side_effect = side_effect

        with pytest.raises(ChecksumError, match="no server-reported checksum"):
            models.download("v1", "a1", client=client, cache_dir=model_cache)

        assert not (model_cache / "v1" / "a1.bin").exists()
        assert not (model_cache / "v1" / "a1.bin.partial").exists()

    def test_checksum_mismatch_raises_and_cleans_up(self, model_cache: Path) -> None:
        def side_effect(version_id: str, artifact_id: str, dest: Path) -> tuple[Path, str]:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"actual-bytes")
            return dest, "wrong" * 12

        client = MagicMock(spec=DagnamClient)
        client.list_model_version_artifacts.return_value = [{"id": "a1", "sha256": "irrelevant"}]
        client.download_model_artifact_stream.side_effect = side_effect

        with pytest.raises(ChecksumError, match="checksum mismatch"):
            models.download("v1", "a1", client=client, cache_dir=model_cache)

        assert not (model_cache / "v1" / "a1.bin").exists()
        assert not (model_cache / "v1" / "a1.bin.partial").exists()

    def test_eviction_failure_is_logged_not_raised(
        self,
        model_cache: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        body = b"weights"
        side, sha = _fake_download(body)
        client = MagicMock(spec=DagnamClient)
        client.list_model_version_artifacts.return_value = [{"id": "a1", "sha256": sha}]
        client.download_model_artifact_stream.side_effect = side

        def _boom(*_a: object, **_kw: object) -> list[str]:
            raise OSError("disk full")

        monkeypatch.setattr(models, "evict_lru_locked", _boom)

        with caplog.at_level(logging.WARNING):
            path = models.download("v1", "a1", client=client, cache_dir=model_cache)

        assert path.read_bytes() == body
        assert any("eviction" in record.message.lower() for record in caplog.records)

    def test_uses_default_cache_dir_when_omitted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        default_root = tmp_path / "models"
        monkeypatch.setattr(models, "DEFAULT_MODEL_CACHE_DIR", default_root)
        body = b"weights"
        side, sha = _fake_download(body)
        client = MagicMock(spec=DagnamClient)
        client.list_model_version_artifacts.return_value = [{"id": "a1", "sha256": sha}]
        client.download_model_artifact_stream.side_effect = side

        path = models.download("v1", "a1", client=client)

        assert path == default_root / "v1" / "a1.bin"
