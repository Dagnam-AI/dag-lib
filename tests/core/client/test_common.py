"""Coverage for ``dagnam._core.client.common`` URL/header/error mapping helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest

from dagnam._core.client import common
from dagnam._core.exceptions import (
    APIError,
    ArchitectureVersionNotFoundError,
    AuthError,
    CheckpointNotFoundError,
    CodegenError,
    CodegenValidationError,
    DatasetNotFoundError,
    DeploymentNotFoundError,
    DeploymentStateError,
    DeploymentValidationError,
    HubError,
    HubModelNotFoundError,
    ProjectNotFoundError,
    QuotaExceededError,
    TaskNotFoundError,
    TrainingJobNotFoundError,
    UploadError,
)


def _resp(status: int, *, text: str = "", content_type: str | None = None, content: bytes = b""):
    headers: dict[str, str] = {}
    if content_type is not None:
        headers["Content-Type"] = content_type
    return SimpleNamespace(
        status_code=status,
        ok=200 <= status < 300,
        text=text,
        content=content or text.encode(),
        headers=headers,
    )


# build_url --------------------------------------------------------------


def test_build_url_strips_trailing_slash() -> None:
    assert common.build_url("https://api.test/", "/api/v1/x") == "https://api.test/api/v1/x"


def test_build_url_adds_leading_slash() -> None:
    assert common.build_url("https://api.test", "api/v1/x") == "https://api.test/api/v1/x"


def test_quote_path_segment_encodes_slashes() -> None:
    assert common.quote_path_segment("a/b?c") == "a%2Fb%3Fc"


# Header builders --------------------------------------------------------


def test_bearer_headers_with_extras() -> None:
    h = common.bearer_headers("KEY", extra={"X-Trace": "abc"})
    assert h == {"Authorization": "Bearer KEY", "X-Trace": "abc"}


def test_bearer_headers_without_extras() -> None:
    assert common.bearer_headers("KEY") == {"Authorization": "Bearer KEY"}


def test_inference_headers_sets_both_keys() -> None:
    assert common.inference_headers("KEY") == {
        "Authorization": "Bearer KEY",
        "X-API-Key": "KEY",
    }


# safe_response_text -----------------------------------------------------


def test_safe_response_text_streaming_with_type() -> None:
    r = SimpleNamespace(
        _content=False, headers={"Content-Type": "application/json"}, content=b"", text=""
    )
    assert common.safe_response_text(r) == "<streaming application/json body omitted>"


def test_safe_response_text_streaming_without_type() -> None:
    r = SimpleNamespace(_content=False, headers={}, content=b"", text="")
    assert common.safe_response_text(r) == "<streaming response body omitted>"


def test_safe_response_text_binary_body_reports_bytes() -> None:
    r = _resp(400, content_type="application/octet-stream", content=b"\x00\x01\x02")
    assert common.safe_response_text(r) == "<3 bytes of application/octet-stream>"


def test_safe_response_text_truncates_long_bodies() -> None:
    body = "x" * 3000
    r = _resp(400, text=body, content_type="text/plain")
    result = common.safe_response_text(r)
    assert result.startswith("x" * 2048)
    assert "[truncated, 3000 chars total]" in result


def test_safe_response_text_handles_decode_failure() -> None:
    class _BadText:
        headers: ClassVar[dict[str, str]] = {"Content-Type": "text/plain"}
        content = b"abcd"
        _content = b"abcd"

        @property
        def text(self):
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "boom")

    assert common.safe_response_text(_BadText()) == "<4 bytes; failed to decode body>"


def test_safe_response_text_tolerates_missing_headers_attribute() -> None:
    class _NoHeaders:
        content = b""
        text = "ok"

        @property
        def headers(self):
            raise AttributeError("no headers")

    # getattr default kicks in; text path returns "ok"
    assert common.safe_response_text(_NoHeaders()) == "ok"


def test_safe_response_text_coerces_non_string_text() -> None:
    r = SimpleNamespace(headers={"Content-Type": "text/plain"}, content=b"123", text=123)
    assert common.safe_response_text(r) == "123"


# raise_for_generic ------------------------------------------------------


def test_raise_for_generic_no_op_on_2xx() -> None:
    common.raise_for_generic(_resp(204))


def test_raise_for_generic_uses_ok_attribute_when_no_status_code() -> None:
    r = SimpleNamespace(ok=True, status_code=None, text="", headers={}, content=b"")
    # _ok returns True via ok attribute (status_code is not int)
    # But raise_for_generic also reads status_code with int(...); short-circuit on _ok
    common.raise_for_generic(r)


def test_raise_for_generic_401_raises_auth_error() -> None:
    with pytest.raises(AuthError):
        common.raise_for_generic(_resp(401))


def test_raise_for_generic_404_with_arg_uses_not_found() -> None:
    with pytest.raises(DatasetNotFoundError):
        common.raise_for_generic(_resp(404), DatasetNotFoundError, "ds-1")


def test_raise_for_generic_404_without_arg_uses_default_message() -> None:
    with pytest.raises(DatasetNotFoundError):
        common.raise_for_generic(_resp(404), DatasetNotFoundError)


def test_raise_for_generic_404_with_no_exc_class_falls_through_to_apierror() -> None:
    with pytest.raises(APIError):
        common.raise_for_generic(_resp(404))


def test_raise_for_generic_413_raises_quota() -> None:
    with pytest.raises(QuotaExceededError):
        common.raise_for_generic(_resp(413, text="big"))


def test_raise_for_generic_413_uses_default_message_on_empty_body() -> None:
    with pytest.raises(QuotaExceededError, match="Storage quota exceeded"):
        common.raise_for_generic(_resp(413))


def test_raise_for_generic_other_codes_become_apierror() -> None:
    with pytest.raises(APIError):
        common.raise_for_generic(_resp(500, text="boom"))


# raise_for_dataset / training_job / checkpoint / task -------------------


def test_raise_for_dataset_404() -> None:
    with pytest.raises(DatasetNotFoundError):
        common.raise_for_dataset(_resp(404), "ds-1")


def test_raise_for_training_job_404() -> None:
    with pytest.raises(TrainingJobNotFoundError):
        common.raise_for_training_job(_resp(404), "job-1")


def test_raise_for_checkpoint_404() -> None:
    with pytest.raises(CheckpointNotFoundError):
        common.raise_for_checkpoint(_resp(404), "ckpt-1")


def test_raise_for_task_404() -> None:
    with pytest.raises(TaskNotFoundError):
        common.raise_for_task(_resp(404), "task-1")


# raise_for_deployment ---------------------------------------------------


def test_raise_for_deployment_ok() -> None:
    common.raise_for_deployment(_resp(200), "dep-1")


def test_raise_for_deployment_409_state() -> None:
    with pytest.raises(DeploymentStateError):
        common.raise_for_deployment(_resp(409, text="bad state"), "dep-1")


@pytest.mark.parametrize("code", [400, 422])
def test_raise_for_deployment_validation(code: int) -> None:
    with pytest.raises(DeploymentValidationError):
        common.raise_for_deployment(_resp(code, text="bad input"), "dep-1")


def test_raise_for_deployment_404() -> None:
    with pytest.raises(DeploymentNotFoundError):
        common.raise_for_deployment(_resp(404), "dep-1")


# raise_for_hub ----------------------------------------------------------


def test_raise_for_hub_ok() -> None:
    common.raise_for_hub(_resp(200))


def test_raise_for_hub_401() -> None:
    with pytest.raises(AuthError):
        common.raise_for_hub(_resp(401))


def test_raise_for_hub_404_with_model_id() -> None:
    with pytest.raises(HubModelNotFoundError):
        common.raise_for_hub(_resp(404), model_id="model-1")


def test_raise_for_hub_404_without_model_id() -> None:
    with pytest.raises(HubError):
        common.raise_for_hub(_resp(404))


@pytest.mark.parametrize("code", [400, 422])
def test_raise_for_hub_validation(code: int) -> None:
    with pytest.raises(HubError):
        common.raise_for_hub(_resp(code, text="bad"))


def test_raise_for_hub_other_code() -> None:
    with pytest.raises(APIError):
        common.raise_for_hub(_resp(500))


# raise_for_project ------------------------------------------------------


def test_raise_for_project_ok() -> None:
    common.raise_for_project(_resp(200))


def test_raise_for_project_401() -> None:
    with pytest.raises(AuthError):
        common.raise_for_project(_resp(401))


def test_raise_for_project_404_with_id() -> None:
    with pytest.raises(ProjectNotFoundError):
        common.raise_for_project(_resp(404), project_id="proj-1")


def test_raise_for_project_404_without_id_maps_to_arch_version() -> None:
    with pytest.raises(ArchitectureVersionNotFoundError):
        common.raise_for_project(_resp(404, text="v3"))


def test_raise_for_project_404_without_id_default_message() -> None:
    with pytest.raises(ArchitectureVersionNotFoundError):
        common.raise_for_project(_resp(404))


def test_raise_for_project_other_code() -> None:
    with pytest.raises(APIError):
        common.raise_for_project(_resp(500))


# raise_for_codegen ------------------------------------------------------


def test_raise_for_codegen_ok() -> None:
    common.raise_for_codegen(_resp(200))


def test_raise_for_codegen_401() -> None:
    with pytest.raises(AuthError):
        common.raise_for_codegen(_resp(401))


@pytest.mark.parametrize("code", [400, 422])
def test_raise_for_codegen_validation(code: int) -> None:
    with pytest.raises(CodegenValidationError):
        common.raise_for_codegen(_resp(code))


def test_raise_for_codegen_500() -> None:
    with pytest.raises(CodegenError):
        common.raise_for_codegen(_resp(500, text="boom"))


def test_raise_for_codegen_other_code() -> None:
    with pytest.raises(APIError):
        common.raise_for_codegen(_resp(503))


# raise_for_upload -------------------------------------------------------


def test_raise_for_upload_ok() -> None:
    common.raise_for_upload(_resp(200))


def test_raise_for_upload_401() -> None:
    with pytest.raises(AuthError):
        common.raise_for_upload(_resp(401))


def test_raise_for_upload_413() -> None:
    with pytest.raises(QuotaExceededError):
        common.raise_for_upload(_resp(413, text="too big"))


def test_raise_for_upload_413_default_message() -> None:
    with pytest.raises(QuotaExceededError, match="Storage quota exceeded"):
        common.raise_for_upload(_resp(413))


@pytest.mark.parametrize("code", [400, 422])
def test_raise_for_upload_validation(code: int) -> None:
    with pytest.raises(UploadError):
        common.raise_for_upload(_resp(code))


def test_raise_for_upload_other_code() -> None:
    with pytest.raises(APIError):
        common.raise_for_upload(_resp(500))


# _ok branches -----------------------------------------------------------


def test_ok_returns_false_for_unknown_response_shape() -> None:
    r = SimpleNamespace(status_code=None, ok=None)
    assert common._ok(r) is False
