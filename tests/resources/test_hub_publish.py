"""``dagnam.hub.publish`` and ``finalize`` (``dagnam/resources/hub_publish.py``): retries,
resume, duplicate names, and the finalize contract."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dagnam import hub
from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import APIError, HubError, PayloadTooLargeError, UploadError
from dagnam._types import JsonObject

Progress = Callable[[str, int, int, str], object]


def _client(tmp_path: Path) -> tuple[MagicMock, list[str]]:
    """A client whose every hub call succeeds, and two local files to publish."""
    client = MagicMock(spec=DagnamClient)
    client.create_hub_model.return_value = {"id": "m1", "name": "n"}
    client.upload_model_file.return_value = {"id": "f1"}
    client.create_hub_model_version.return_value = {"id": "v1", "version": "1.0.0"}
    client.finalize_hub_model.return_value = {"id": "m1", "status": "published"}
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"x")
    head = tmp_path / "head.onnx"
    head.write_bytes(b"y")
    return client, [str(weights), str(head)]


def _publish(
    client: MagicMock,
    files: list[str],
    *,
    max_retries_per_file: int = 2,
    on_file_progress: Progress | None = None,
) -> JsonObject:
    return hub.publish(
        name="n",
        description="d",
        task_type="classification",
        framework="pytorch",
        files=files,
        max_retries_per_file=max_retries_per_file,
        on_file_progress=on_file_progress,
        client=client,
    )


class TestRetries:
    def test_a_rejected_file_is_not_retried(self, tmp_path: Path) -> None:
        """A 400/422 (HubError) cannot succeed on a repeat: one attempt, then the halt."""
        client, files = _client(tmp_path)
        client.upload_model_file.side_effect = HubError("Unsupported file extension")
        states: list[str] = []
        with pytest.raises(UploadError, match=r"weights\.safetensors"):
            _publish(client, files, on_file_progress=lambda _p, _i, _t, s: states.append(s))
        assert client.upload_model_file.call_count == 1
        assert states == ["uploading", "failed"]
        client.finalize_hub_model.assert_not_called()

    @pytest.mark.parametrize("status", [403, 404, 409, 410, 422])
    def test_a_4xx_api_error_is_not_retried(self, tmp_path: Path, status: int) -> None:
        client, files = _client(tmp_path)
        client.upload_model_file.side_effect = APIError(status, "no")
        with pytest.raises(UploadError):
            _publish(client, files)
        assert client.upload_model_file.call_count == 1

    @pytest.mark.parametrize("status", [0, 408, 429, 500, 503])
    def test_a_transient_failure_is_retried(self, tmp_path: Path, status: int) -> None:
        client, files = _client(tmp_path)
        client.upload_model_file.side_effect = [
            APIError(status, "blip"),
            {"id": "f1"},
            {"id": "f2"},
        ]
        result = _publish(client, files)
        assert client.upload_model_file.call_count == 3
        assert result["finalized"] is True

    def test_an_upload_error_is_not_retried(self, tmp_path: Path) -> None:
        client, files = _client(tmp_path)
        client.upload_model_file.side_effect = UploadError("refused")
        with pytest.raises(UploadError, match="after 1 attempt"):
            _publish(client, files, max_retries_per_file=5)
        assert client.upload_model_file.call_count == 1

    def test_a_file_too_large_is_not_retried_and_names_the_draft(self, tmp_path: Path) -> None:
        """A 413 is the platform's request ceiling; a repeat sends the same bytes."""
        client, files = _client(tmp_path)
        client.upload_model_file.side_effect = PayloadTooLargeError("Request body too large")
        with pytest.raises(UploadError, match=r"model m1 halted.*too large"):
            _publish(client, files, max_retries_per_file=5)
        assert client.upload_model_file.call_count == 1

    def test_a_file_that_changed_mid_send_is_not_retried_and_names_the_draft(
        self, tmp_path: Path
    ) -> None:
        """The client raises the file's own OSError; publish halts with the resume hint."""
        client, files = _client(tmp_path)
        client.upload_model_file.side_effect = OSError("a file shrank while it was being sent")
        with pytest.raises(UploadError, match=r"model m1 halted.*shrank"):
            _publish(client, files, max_retries_per_file=5)
        assert client.upload_model_file.call_count == 1
        client.finalize_hub_model.assert_not_called()


class TestVersion:
    def test_version_without_changelog_sends_the_version_only(self, tmp_path: Path) -> None:
        client, files = _client(tmp_path)
        result = hub.publish(
            name="n",
            description="d",
            task_type="classification",
            framework="pytorch",
            files=files,
            version="1.0.0",
            client=client,
        )
        client.create_hub_model_version.assert_called_once_with("m1", {"version": "1.0.0"})
        assert result["version"] == {"id": "v1", "version": "1.0.0"}

    def test_version_with_changelog_sends_both(self, tmp_path: Path) -> None:
        client, files = _client(tmp_path)
        hub.publish(
            name="n",
            description="d",
            task_type="classification",
            framework="pytorch",
            files=files,
            version="1.0.0",
            changelog="first",
            client=client,
        )
        client.create_hub_model_version.assert_called_once_with(
            "m1", {"version": "1.0.0", "changelog": "first"}
        )


class TestMove:
    def test_hub_publish_lives_in_hub_publish(self) -> None:
        assert hub.publish.__module__ == "dagnam.resources.hub_publish"

    def test_create_uses_the_shared_payload(self) -> None:
        client = MagicMock(spec=DagnamClient)
        client.create_hub_model.return_value = {"id": "m1"}
        hub.create(
            name="n",
            description="d",
            task_type="classification",
            framework="pytorch",
            tags=["a"],
            metadata={"k": "v"},
            client=client,
        )
        payload = client.create_hub_model.call_args.args[0]
        assert payload == {
            "name": "n",
            "description": "d",
            "task_type": "classification",
            "framework": "pytorch",
            "license": "mit",
            "visibility": "public",
            "tags": ["a"],
            "metadata": {"k": "v"},
        }
