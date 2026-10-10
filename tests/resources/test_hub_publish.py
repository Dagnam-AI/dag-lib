"""``dagnam.hub.publish`` and ``finalize`` (``dagnam/resources/hub_publish.py``): retries,
resume, duplicate names, and the finalize contract."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dagnam import hub
from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import (
    APIError,
    HubError,
    HubModelNotFoundError,
    PayloadTooLargeError,
    UploadError,
)
from dagnam._types import JsonObject
from dagnam.resources.hub_publish import stored_name

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
    model_id: str | None = None,
    max_retries_per_file: int = 2,
    on_file_progress: Progress | None = None,
) -> JsonObject:
    return hub.publish(
        name="n",
        description="d",
        task_type="classification",
        framework="pytorch",
        files=files,
        model_id=model_id,
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


class TestFinalize:
    def test_finalize_delegates_to_the_client(self) -> None:
        client = MagicMock(spec=DagnamClient)
        client.finalize_hub_model.return_value = {"id": "m1", "status": "published"}
        assert hub.finalize("m1", client=client)["status"] == "published"
        client.finalize_hub_model.assert_called_once_with("m1")

    def test_finalize_404_propagates(self) -> None:
        client = MagicMock(spec=DagnamClient)
        client.finalize_hub_model.side_effect = HubModelNotFoundError("m1")
        with pytest.raises(HubModelNotFoundError):
            hub.finalize("m1", client=client)

    def test_finalize_is_exported(self) -> None:
        assert "finalize" in hub.__all__

    def test_publish_reports_the_finalized_model(self, tmp_path: Path) -> None:
        client, files = _client(tmp_path)
        result = _publish(client, files)
        assert result["finalized"] is True
        assert result["model"] == {"id": "m1", "status": "published"}


class TestResume:
    def test_model_id_skips_create_and_uploads_into_the_draft(self, tmp_path: Path) -> None:
        client, files = _client(tmp_path)
        result = _publish(client, files, model_id="draft-1")
        client.create_hub_model.assert_not_called()
        targets = [call.args[0] for call in client.upload_model_file.call_args_list]
        assert targets == ["draft-1", "draft-1"]
        client.finalize_hub_model.assert_called_once_with("draft-1")
        assert result["finalized"] is True
        assert result["model"] == {"id": "m1", "status": "published"}

    def test_resume_skips_a_file_the_draft_already_holds(self, tmp_path: Path) -> None:
        """The platform answers 409 to a name the model already has: skipped, not a halt."""
        client, files = _client(tmp_path)
        client.upload_model_file.side_effect = [APIError(409, "already on the model"), {"id": "f2"}]
        states: list[str] = []
        result = _publish(
            client,
            files,
            model_id="draft-1",
            on_file_progress=lambda _p, _i, _t, s: states.append(s),
        )
        assert states == ["uploading", "skipped", "uploading", "uploaded"]
        assert result["files"] == [{"id": "f2"}]
        client.finalize_hub_model.assert_called_once_with("draft-1")

    def test_resume_skip_needs_no_progress_callback(self, tmp_path: Path) -> None:
        client, files = _client(tmp_path)
        client.upload_model_file.side_effect = [APIError(409, "dup"), {"id": "f2"}]
        assert _publish(client, files, model_id="draft-1")["files"] == [{"id": "f2"}]

    def test_409_on_a_fresh_publish_halts_with_the_resume_hint(self, tmp_path: Path) -> None:
        client, files = _client(tmp_path)
        client.upload_model_file.side_effect = APIError(409, "dup")
        with pytest.raises(UploadError, match=r"hub\.publish\(\.\.\., model_id='m1'\)"):
            _publish(client, files)
        assert client.upload_model_file.call_count == 1
        client.finalize_hub_model.assert_not_called()

    def test_409_after_a_retry_of_the_same_file_counts_as_uploaded(self, tmp_path: Path) -> None:
        """The first attempt stored the file but its answer was lost; the retry's 409 is success."""
        client, files = _client(tmp_path)
        client.upload_model_file.side_effect = [
            APIError(503, "blip"),
            APIError(409, "already on the model"),
            {"id": "f2"},
        ]
        states: list[str] = []
        result = _publish(client, files, on_file_progress=lambda _p, _i, _t, s: states.append(s))
        assert states == ["uploading", "retrying", "skipped", "uploading", "uploaded"]
        assert result["files"] == [{"id": "f2"}]
        client.create_hub_model.assert_called_once()
        client.finalize_hub_model.assert_called_once_with("m1")

    def test_duplicate_base_names_are_refused_before_any_network_call(self, tmp_path: Path) -> None:
        client, files = _client(tmp_path)
        (tmp_path / "b").mkdir()
        twin = tmp_path / "b" / "weights.safetensors"
        twin.write_bytes(b"z")
        with pytest.raises(ValueError, match=r"weights\.safetensors"):
            _publish(client, [*files, str(twin)])
        client.create_hub_model.assert_not_called()
        client.upload_model_file.assert_not_called()

    @pytest.mark.parametrize(
        ("first", "second"),
        [("a.pt", "a.safetensors"), ("a.pt", "a.pth"), ("A.PT", "a.safetensors")],
    )
    def test_names_that_collide_after_conversion_are_refused(
        self, tmp_path: Path, first: str, second: str
    ) -> None:
        """A PyTorch file is stored as <stem>.safetensors, so these pairs are one name on the hub."""
        client, _files = _client(tmp_path)
        pair = [tmp_path / first, tmp_path / second]
        for path in pair:
            path.write_bytes(b"z")
        with pytest.raises(ValueError, match=r"a\.safetensors"):
            _publish(client, [str(path) for path in pair])
        client.create_hub_model.assert_not_called()


@pytest.mark.parametrize(
    ("local", "stored"),
    [
        ("weights.pt", "weights.safetensors"),
        ("weights.pth", "weights.safetensors"),
        ("v2.final.PT", "v2.final.safetensors"),
        ("weights.safetensors", "weights.safetensors"),
        ("head.onnx", "head.onnx"),
        ("noext", "noext"),
    ],
)
def test_stored_name_maps_pytorch_files_to_safetensors(local: str, stored: str) -> None:
    assert stored_name(f"/tmp/some/dir/{local}") == stored
