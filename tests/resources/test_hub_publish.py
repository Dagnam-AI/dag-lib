"""``dagnam.hub.publish`` and ``finalize`` (``dagnam/resources/hub_publish.py``): retries,
resume, duplicate names, and the finalize contract."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
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
from dagnam.resources.hub_publish import (
    base_fields,
    normalize_stage,
    split_architecture,
    stored_name,
)

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
    def test_the_first_version_is_named_in_the_create_payload(self, tmp_path: Path) -> None:
        client, files = _client(tmp_path)
        result = _publish(client, files)
        assert client.create_hub_model.call_args.args[0]["version"] == "1.0.0"
        assert "changelog" not in client.create_hub_model.call_args.args[0]
        client.create_hub_model_version.assert_not_called()
        assert result["version"] is None

    def test_version_and_changelog_are_sent_in_the_create_payload(self, tmp_path: Path) -> None:
        client, files = _client(tmp_path)
        hub.publish(
            name="n",
            description="d",
            task_type="classification",
            framework="pytorch",
            files=files,
            version="2.0.0",
            changelog="first",
            client=client,
        )
        payload = client.create_hub_model.call_args.args[0]
        assert (payload["version"], payload["changelog"]) == ("2.0.0", "first")

    def test_the_result_version_is_the_finalized_models_latest_version(
        self, tmp_path: Path
    ) -> None:
        client, files = _client(tmp_path)
        latest = {"id": "v1", "version": "1.0.0"}
        client.finalize_hub_model.return_value = {"id": "m1", "latest_version": latest}
        assert _publish(client, files)["version"] == latest


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


def test_docstring_examples_use_task_types_the_platform_accepts() -> None:
    assert 'task_type="generation"' in (hub.search.__doc__ or "")
    assert 'task_type="classification"' in (hub.create.__doc__ or "")
    for fn in (hub.search, hub.create):
        assert "text-generation" not in (fn.__doc__ or "")
    assert "500 MB" in (hub.upload_file.__doc__ or "")
    assert "500 MB" in (hub.publish.__doc__ or "")


DESIGN: JsonObject = {"diagram_state": {"nodes": []}, "architecture_config": {"layers": []}}
IR: JsonObject = {"layers": [], "connections": []}
BASE_ID = "0d6c7c64-8f1a-4b2b-9a7e-3f2f5a8e1c11"


def _design(client: MagicMock, **kwargs: Any) -> JsonObject:
    return hub.publish(
        name="n",
        description="d",
        task_type="classification",
        framework="pytorch",
        client=client,
        **kwargs,
    )


def _sent(client: MagicMock) -> JsonObject:
    payload: JsonObject = client.create_hub_model.call_args.args[0]
    return payload


class TestDesign:
    def test_a_design_publishes_with_no_files_and_no_stage(self, tmp_path: Path) -> None:
        client, _files = _client(tmp_path)
        _design(client, architecture=DESIGN)
        payload = _sent(client)
        assert payload["architecture"] == DESIGN
        assert payload["version"] == "1.0.0"
        assert "stage" not in payload
        client.upload_model_file.assert_not_called()
        client.create_hub_model_version.assert_not_called()
        client.finalize_hub_model.assert_called_once_with("m1")

    def test_a_codegen_ir_is_sent_as_architecture_ir(self, tmp_path: Path) -> None:
        client, _files = _client(tmp_path)
        _design(client, architecture=IR)
        payload = _sent(client)
        assert payload["architecture_ir"] == IR
        assert "architecture" not in payload

    def test_a_project_design_sends_the_project_and_its_version(self, tmp_path: Path) -> None:
        client, _files = _client(tmp_path)
        _design(client, project_id="p1", architecture_version_id="av1")
        payload = _sent(client)
        assert (payload["project_id"], payload["architecture_version_id"]) == ("p1", "av1")

    def test_two_design_sources_are_refused_before_any_call(self, tmp_path: Path) -> None:
        client, _files = _client(tmp_path)
        with pytest.raises(ValueError, match="not both"):
            _design(client, architecture=DESIGN, project_id="p1")
        client.create_hub_model.assert_not_called()

    def test_a_project_version_needs_the_project(self, tmp_path: Path) -> None:
        client, _files = _client(tmp_path)
        with pytest.raises(ValueError, match="needs project_id"):
            _design(client, architecture_version_id="av1")
        client.create_hub_model.assert_not_called()

    def test_split_architecture_tells_studio_state_from_an_ir(self) -> None:
        assert split_architecture(DESIGN) == ("architecture", DESIGN)
        assert split_architecture(IR) == ("architecture_ir", IR)


class TestStageAndBase:
    @pytest.mark.parametrize(
        ("given", "sent"),
        [
            ("pre-trained", "pretrained"),
            ("Fine-Tuned", "finetuned"),
            ("adapter", "adapter"),
            (" design ", "design"),
        ],
    )
    def test_a_stage_is_normalized(self, tmp_path: Path, given: str, sent: str) -> None:
        client, files = _client(tmp_path)
        _design(client, files=[] if sent == "design" else files, stage=given, base="B")
        assert _sent(client)["stage"] == sent
        assert normalize_stage(given) == sent

    def test_post_trained_alone_names_the_five_stages(self) -> None:
        with pytest.raises(ValueError, match="finetuned, adapter, quantized, merged, distilled"):
            normalize_stage("post-trained")

    def test_an_unknown_stage_is_refused_before_any_call(self, tmp_path: Path) -> None:
        client, files = _client(tmp_path)
        with pytest.raises(ValueError, match="Unknown stage 'bogus'"):
            _design(client, files=files, stage="bogus")
        client.create_hub_model.assert_not_called()

    def test_a_post_trained_stage_needs_a_base_before_any_call(self, tmp_path: Path) -> None:
        client, files = _client(tmp_path)
        with pytest.raises(ValueError, match="needs base"):
            _design(client, files=files, stage="adapter")
        client.create_hub_model.assert_not_called()

    def test_resuming_a_draft_needs_no_base(self, tmp_path: Path) -> None:
        client, files = _client(tmp_path)
        _design(client, files=files, stage="adapter", model_id="m1")
        client.finalize_hub_model.assert_called_once_with("m1")

    def test_a_design_with_files_is_refused_before_any_call(self, tmp_path: Path) -> None:
        client, files = _client(tmp_path)
        with pytest.raises(ValueError, match="design takes no files"):
            _design(client, files=files, stage="design")
        client.create_hub_model.assert_not_called()

    def test_a_plain_name_is_a_base_model_name(self, tmp_path: Path) -> None:
        client, files = _client(tmp_path)
        _design(client, files=files, base="Qwen/Qwen2.5-0.5B-Instruct")
        payload = _sent(client)
        assert payload["base_model_name"] == "Qwen/Qwen2.5-0.5B-Instruct"
        assert "base_model_id" not in payload

    def test_a_uuid_is_a_base_model_id(self, tmp_path: Path) -> None:
        client, files = _client(tmp_path)
        _design(client, files=files, base=BASE_ID)
        payload = _sent(client)
        assert payload["base_model_id"] == BASE_ID
        assert "base_model_name" not in payload
        assert base_fields("x") == {"base_model_name": "x"}


class TestRegistryCard:
    def test_a_registry_version_is_sent_alone(self, tmp_path: Path) -> None:
        client, _files = _client(tmp_path)
        _design(client, registry_version_id="r1")
        payload = _sent(client)
        assert payload["registry_version_id"] == "r1"
        assert "stage" not in payload
        client.upload_model_file.assert_not_called()

    @pytest.mark.parametrize("extra", ["stage", "architecture", "files"])
    def test_a_registry_card_takes_no_stage_design_or_files(
        self, tmp_path: Path, extra: str
    ) -> None:
        client, files = _client(tmp_path)
        with pytest.raises(ValueError, match="registry card"):
            _design(
                client,
                registry_version_id="r1",
                stage="design" if extra == "stage" else None,
                architecture=DESIGN if extra == "architecture" else None,
                files=files if extra == "files" else (),
            )
        client.create_hub_model.assert_not_called()


class TestLicence:
    def test_a_listed_licence_is_sent(self, tmp_path: Path) -> None:
        client, files = _client(tmp_path)
        _design(client, files=files, license="gemma", license_name="Gemma terms")
        payload = _sent(client)
        assert (payload["license"], payload["license_name"]) == ("gemma", "Gemma terms")

    def test_an_unknown_licence_is_refused_and_lists_the_known_ones(self, tmp_path: Path) -> None:
        client, files = _client(tmp_path)
        with pytest.raises(ValueError, match=r"cc-by-2\.0.*mit, apache2"):
            _design(client, files=files, license="cc-by-2.0")
        client.create_hub_model.assert_not_called()

    @pytest.mark.parametrize("visibility", ["public", "unlisted"])
    def test_a_listed_custom_licence_needs_a_url(self, tmp_path: Path, visibility: str) -> None:
        client, files = _client(tmp_path)
        with pytest.raises(ValueError, match="license_url"):
            _design(client, files=files, license="custom", visibility=visibility)
        client.create_hub_model.assert_not_called()

    def test_a_custom_licence_with_a_url_or_private_passes(self, tmp_path: Path) -> None:
        client, files = _client(tmp_path)
        _design(client, files=files, license="custom", license_url="https://example.com/l")
        assert _sent(client)["license_url"] == "https://example.com/l"
        _design(client, files=files, license="custom", visibility="private")


class TestFinalizeRefusal:
    @pytest.mark.parametrize("status", [409, 500])
    def test_publish_names_the_draft_when_finalize_is_refused(
        self, tmp_path: Path, status: int
    ) -> None:
        client, files = _client(tmp_path)
        client.finalize_hub_model.side_effect = APIError(status, "base_required")
        with pytest.raises(HubError) as raised:
            _publish(client, files)
        message = str(raised.value)
        assert "m1" in message
        assert "base_required" in message
        assert "hub.finalize('m1')" in message
        assert isinstance(raised.value.__cause__, APIError)

    def test_publish_version_names_the_draft_when_finalize_is_refused(self, tmp_path: Path) -> None:
        client, files = _client(tmp_path)
        client.finalize_hub_model_version.side_effect = APIError(409, "design_has_files")
        with pytest.raises(HubError) as raised:
            hub.publish_version("m1", "1.1.0", files=files, client=client)
        message = str(raised.value)
        assert "Version v1 of hub model m1" in message
        assert "design_has_files" in message
        assert "hub.finalize_version('m1', 'v1')" in message
        assert "hub.delete_version('m1', 'v1')" in message


class TestPublishVersion:
    def test_it_creates_a_draft_uploads_into_it_and_finalizes(self, tmp_path: Path) -> None:
        client, files = _client(tmp_path)
        client.finalize_hub_model_version.return_value = {"id": "v1", "status": "published"}
        out = hub.publish_version("m1", "1.1.0", files=files[:1], changelog="c", client=client)
        client.create_hub_model_version.assert_called_once_with(
            "m1", {"version": "1.1.0", "changelog": "c"}
        )
        client.upload_model_file.assert_called_once_with("m1", files[0], "v1")
        client.finalize_hub_model_version.assert_called_once_with("m1", "v1")
        assert out == {"id": "v1", "status": "published"}

    def test_a_design_version_sends_the_source_and_uploads_nothing(self, tmp_path: Path) -> None:
        client, _files = _client(tmp_path)
        hub.publish_version("m1", "1.1.0", architecture=DESIGN, client=client)
        client.create_hub_model_version.assert_called_once_with(
            "m1", {"version": "1.1.0", "architecture": DESIGN}
        )
        client.upload_model_file.assert_not_called()

    def test_a_failed_upload_says_how_to_finish_or_remove_the_draft(self, tmp_path: Path) -> None:
        client, files = _client(tmp_path)
        client.upload_model_file.side_effect = HubError("Unsupported file extension")
        with pytest.raises(UploadError) as raised:
            hub.publish_version("m1", "1.1.0", files=files, client=client)
        message = str(raised.value)
        assert "version v1 of hub model m1" in message
        assert "hub.delete_version('m1', 'v1')" in message
        assert "hub.finalize_version('m1', 'v1')" in message
        client.finalize_hub_model_version.assert_not_called()

    def test_a_missing_file_is_refused_before_any_call(self, tmp_path: Path) -> None:
        client, _files = _client(tmp_path)
        with pytest.raises(FileNotFoundError):
            hub.publish_version("m1", "1.1.0", files=[str(tmp_path / "gone.onnx")], client=client)
        client.create_hub_model_version.assert_not_called()


class TestPassThrough:
    def test_upload_file_passes_the_version(self) -> None:
        client = MagicMock(spec=DagnamClient)
        hub.upload_file("m1", "w.onnx", version_id="v1", client=client)
        client.upload_model_file.assert_called_once_with("m1", "w.onnx", "v1")

    def test_use_in_studio_sends_the_options_it_was_given(self) -> None:
        client = MagicMock(spec=DagnamClient)
        hub.use_in_studio("m1", visibility="private", client=client)
        client.use_hub_model_in_studio.assert_called_once_with("m1", {"visibility": "private"})
        client.reset_mock()
        hub.use_in_studio("m1", version_id="v1", client=client)
        client.use_hub_model_in_studio.assert_called_once_with("m1", {"version_id": "v1"})
        client.reset_mock()
        hub.use_in_studio("m1", client=client)
        client.use_hub_model_in_studio.assert_called_once_with("m1", None)

    def test_the_new_names_are_exported_from_hub(self) -> None:
        for name in (
            "publish_version",
            "derivatives",
            "get_architecture",
            "finalize_version",
            "delete_version",
        ):
            assert name in hub.__all__
            assert callable(getattr(hub, name))
