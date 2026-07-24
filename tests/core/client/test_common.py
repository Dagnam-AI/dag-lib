"""Coverage for ``dagnam._core.client.common`` URL/header/error mapping helpers."""

from __future__ import annotations

import json
from typing import ClassVar

import pytest

from dagnam._core.client import common
from dagnam._core.exceptions import (
    AccountLockedError,
    AccountSuspendedError,
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
    EmailNotVerifiedError,
    HubError,
    HubModelNotFoundError,
    InvalidURLError,
    PayloadTooLargeError,
    ProjectNotFoundError,
    QuotaExceededError,
    ResponseError,
    TaskNotFoundError,
    TrainingJobNotFoundError,
    UploadError,
)


class _Resp:
    def __init__(
        self,
        status: int,
        *,
        text: object = "",
        content_type: str | None = None,
        content: bytes = b"",
        content_marker: object = b"",
    ) -> None:
        self.status_code = status
        self.ok = 200 <= status < 300
        self.text = text
        self.content = content or str(text).encode()
        self.headers: dict[str, str] = {}
        self._content = content_marker
        if content_type is not None:
            self.headers["Content-Type"] = content_type


class _StatuslessResp:
    def __init__(self, *, ok: object = True) -> None:
        self.ok = ok
        self.text = ""
        self.content = b""
        self.headers: dict[str, str] = {}

    @property
    def status_code(self) -> object:
        return None


def _resp(
    status: int, *, text: object = "", content_type: str | None = None, content: bytes = b""
) -> _Resp:
    return _Resp(status, text=text, content_type=content_type, content=content)


class _JsonResp:
    def __init__(self, payload: object) -> None:
        self._payload = payload
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self.text = ""
        self.content = b""

    def json(self) -> object:
        return self._payload


# response_json_* helpers ------------------------------------------------


def test_response_json_value_returns_scalar() -> None:
    assert common.response_json_value(_JsonResp(42)) == 42


def test_response_json_object_returns_object() -> None:
    assert common.response_json_object(_JsonResp({"a": 1})) == {"a": 1}


def test_response_json_array_returns_array() -> None:
    assert common.response_json_array(_JsonResp([1, 2, 3])) == [1, 2, 3]


# requests_query_params --------------------------------------------------


def test_requests_query_params_none_returns_none() -> None:
    assert common.requests_query_params(None) is None


def test_requests_query_params_skips_none_value() -> None:
    assert common.requests_query_params({"a": None, "b": "1"}) == [("b", "1")]


def test_requests_query_params_scalar_types() -> None:
    result = common.requests_query_params({"i": 1, "f": 1.5, "b": True, "s": "x"})
    assert result == [("i", "1"), ("f", "1.5"), ("b", "True"), ("s", "x")]


def test_requests_query_params_iterable_drops_none_items() -> None:
    result = common.requests_query_params({"tags": ["a", None, "b"]})
    assert result == [("tags", "a"), ("tags", "b")]


def test_query_scalar_to_string_none_returns_empty() -> None:
    assert common._query_scalar_to_string(None) == ""


def test_query_scalar_to_string_value() -> None:
    assert common._query_scalar_to_string(7) == "7"


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


def test_stream_query_params_uses_short_lived_token_query() -> None:
    assert common.stream_query_params("abc") == {"token": "abc"}


def test_inference_headers_removed() -> None:
    assert not hasattr(common, "inference_headers")


# safe_response_text -----------------------------------------------------


def test_safe_response_text_streaming_with_type() -> None:
    r = _Resp(200, content_type="application/json", content_marker=False)
    assert common.safe_response_text(r) == "<streaming application/json body omitted>"


def test_safe_response_text_streaming_without_type() -> None:
    r = _Resp(200, content_marker=False)
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
        def status_code(self) -> int:
            return 400

        @property
        def text(self) -> str:
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "boom")

    # Hand-rolled response fake (class-level attrs) does not statically satisfy
    # the ResponseLike protocol, but exercises the decode-failure path at runtime.
    assert (
        common.safe_response_text(_BadText())  # pyright: ignore[reportArgumentType]
        == "<4 bytes; failed to decode body>"
    )


def test_safe_response_text_tolerates_missing_headers_attribute() -> None:
    class _NoHeaders:
        content = b""
        text = "ok"

        @property
        def headers(self) -> dict[str, str]:
            raise AttributeError("no headers")

        @property
        def status_code(self) -> int:
            return 200

    # getattr default kicks in; text path returns "ok"
    assert common.safe_response_text(_NoHeaders()) == "ok"


def test_safe_response_text_coerces_non_string_text() -> None:
    r = _Resp(400, content_type="text/plain", content=b"123", text=123)
    assert common.safe_response_text(r) == "123"


def test_safe_response_text_unwraps_fastapi_string_detail() -> None:
    r = _resp(422, text='{"detail":"dataset is required"}', content_type="application/json")
    assert common.safe_response_text(r) == "dataset is required"


def test_safe_response_text_formats_fastapi_validation_detail() -> None:
    r = _resp(
        422,
        text=(
            '{"detail":[{"type":"missing","loc":["body","project_id"],'
            '"msg":"Field required","input":{}}]}'
        ),
        content_type="application/json",
    )
    assert common.safe_response_text(r) == "body.project_id: Field required"


# _format_fastapi_detail edge cases --------------------------------------


def test_fastapi_detail_list_skips_non_dict_items() -> None:
    # A list detail with a non-dict item exercises the ``continue`` arm; with
    # no usable messages it falls through to json.dumps of the detail.
    r = _resp(422, text='{"detail":["just a string"]}', content_type="application/json")
    assert common.safe_response_text(r) == '["just a string"]'


def test_fastapi_detail_list_item_without_string_msg_is_skipped() -> None:
    # item is a dict but ``msg`` is not a str -> the loop continues without
    # appending, and with no usable messages falls through to json.dumps.
    r = _resp(
        422,
        text='{"detail":[{"loc":["body"],"msg":123}]}',
        content_type="application/json",
    )
    assert common.safe_response_text(r) == '[{"loc": ["body"], "msg": 123}]'


def test_fastapi_detail_non_str_non_list_falls_through_to_dumps() -> None:
    # detail is a dict with no ``message`` key -> json.dumps(detail)
    r = _resp(422, text='{"detail":{"code":1}}', content_type="application/json")
    assert common.safe_response_text(r) == '{"code": 1}'


def test_fastapi_detail_dict_with_message_returns_clean_message() -> None:
    # A structured codegen error detail surfaces its human-readable ``message``
    # (clean field triage), not the JSON-stringified dict.
    r = _resp(
        422,
        text=(
            '{"detail":{"error":"code_generation_failed",'
            '"message":"Unknown legacy loss: DiceBCELoss"}}'
        ),
        content_type="application/json",
    )
    assert common.safe_response_text(r) == "Unknown legacy loss: DiceBCELoss"


def test_safe_response_text_streaming_text_decode_failure_returns_empty() -> None:
    class _StreamingBadText:
        _content = False
        content = b""

        def __init__(self) -> None:
            self.headers = {"Content-Type": "text/plain"}

        @property
        def status_code(self) -> int:
            return 500

        @property
        def text(self) -> str:
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "boom")

    # text raises -> text_value = "" -> empty -> falls to "<streaming ...>"
    assert (
        common.safe_response_text(_StreamingBadText())  # pyright: ignore[reportArgumentType]
        == "<streaming text/plain body omitted>"
    )


def test_safe_response_text_binary_body_len_typeerror_reports_zero() -> None:
    class _NoLenContent:
        def __bool__(self) -> bool:
            return True

        def __len__(self) -> int:
            raise TypeError("no len")

    class _BinaryNoLen:
        _content = b""

        def __init__(self) -> None:
            self.headers = {"Content-Type": "application/octet-stream"}
            self.content = _NoLenContent()

        @property
        def status_code(self) -> int:
            return 400

    assert (
        common.safe_response_text(_BinaryNoLen())  # pyright: ignore[reportArgumentType]
        == "<0 bytes of application/octet-stream>"
    )


def test_safe_response_text_decode_failure_len_typeerror_reports_zero() -> None:
    class _NoLenContent:
        def __bool__(self) -> bool:
            return True

        def __len__(self) -> int:
            raise TypeError("no len")

    class _TextDecodeNoLen:
        _content = b""

        def __init__(self) -> None:
            self.headers = {"Content-Type": "text/plain"}
            self.content = _NoLenContent()

        @property
        def status_code(self) -> int:
            return 400

        @property
        def text(self) -> str:
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "boom")

    assert (
        common.safe_response_text(_TextDecodeNoLen())  # pyright: ignore[reportArgumentType]
        == "<0 bytes; failed to decode body>"
    )


# raise_for_generic ------------------------------------------------------


def test_raise_for_generic_no_op_on_2xx() -> None:
    common.raise_for_generic(_resp(204))


def test_raise_for_generic_uses_ok_attribute_when_no_status_code() -> None:
    r = _StatuslessResp()
    # _ok returns True via ok attribute (status_code is not int)
    # But raise_for_generic also reads status_code with int(...); short-circuit on _ok
    common.raise_for_generic(r)


def test_raise_for_generic_401_raises_autherror() -> None:
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


def test_entitlement_402_raises_quota_with_actionable_message() -> None:
    """A backend plan-limit rejection (402) maps to QuotaExceededError carrying the
    rejection message + remediation hints, across every raise_for_* entry point."""
    import json

    body = json.dumps(
        {
            "error": "limit_exceeded",
            "limit_key": "projects.count",
            "message": "You have reached the Pro plan limit for projects.count.",
            "remediation_hints": ["Upgrade your plan", "Delete unused projects"],
        }
    )
    for raiser in (
        common.raise_for_generic,
        common.raise_for_project,
        common.raise_for_codegen,
        common.raise_for_hub,
        common.raise_for_upload,
    ):
        with pytest.raises(QuotaExceededError) as excinfo:
            raiser(_resp(402, text=body, content_type="application/json"))
        message = str(excinfo.value)
        assert "Pro plan limit for projects.count" in message
        assert "Upgrade your plan" in message


def test_entitlement_402_falls_back_to_default_on_non_json_body() -> None:
    with pytest.raises(QuotaExceededError) as excinfo:
        common.raise_for_generic(_resp(402, text=""))
    assert "Plan limit reached" in str(excinfo.value)


def test_entitlement_402_message_without_hints() -> None:
    import json

    body = json.dumps({"message": "Limit reached", "remediation_hints": []})
    with pytest.raises(QuotaExceededError) as excinfo:
        common.raise_for_generic(_resp(402, text=body, content_type="application/json"))
    assert str(excinfo.value) == "Limit reached"


def test_entitlement_402_non_dict_json_falls_back() -> None:
    with pytest.raises(QuotaExceededError) as excinfo:
        common.raise_for_generic(_resp(402, text="[1, 2, 3]", content_type="application/json"))
    assert "Plan limit reached" in str(excinfo.value)


def test_raise_for_generic_413_uses_default_message_on_empty_body() -> None:
    with pytest.raises(PayloadTooLargeError, match="Upload exceeds the maximum allowed size"):
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


def test_raise_for_codegen_structured_422_detail_surfaces_message() -> None:
    body = (
        '{"detail": {"error": "code_generation_failed", '
        '"message": "Unknown legacy loss: DiceBCELoss"}}'
    )

    with pytest.raises(CodegenValidationError) as exc:
        common.raise_for_codegen(_resp(422, text=body, content_type="application/json"))

    assert "DiceBCELoss" in str(exc.value)


def test_raise_for_codegen_500() -> None:
    with pytest.raises(CodegenError):
        common.raise_for_codegen(_resp(500, text="boom"))


def test_raise_for_codegen_other_code() -> None:
    with pytest.raises(APIError):
        common.raise_for_codegen(_resp(503))


def test_safe_response_text_reads_streaming_json_error_body() -> None:
    class StreamingJsonResponse:
        status_code = 500
        _content = False
        content = b""
        text = '{"detail":"Failed to create ZIP file"}'

        def __init__(self) -> None:
            self.headers = {"Content-Type": "application/json"}

    assert common.safe_response_text(StreamingJsonResponse()) == "Failed to create ZIP file"


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
    with pytest.raises(PayloadTooLargeError, match="Upload exceeds the maximum allowed size"):
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
    r = _StatuslessResp(ok=None)
    assert common._ok(r) is False


# ResponseError mapping (Task 6 — decode/shape failures -> ResponseError) ------


class _RaisingResp:
    """Response whose .json() decode fails or yields a wrong-shape payload."""

    def __init__(
        self, payload: object = None, exc: Exception | None = None, status_code: int = 200
    ) -> None:
        self._payload = payload
        self._exc = exc
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.text = ""
        self.content = b""

    def json(self) -> object:
        if self._exc is not None:
            raise self._exc
        return self._payload


def test_response_json_object_maps_decode_error() -> None:
    resp = _RaisingResp(exc=ValueError("Expecting value"), status_code=502)
    with pytest.raises(ResponseError) as ei:
        common.response_json_object(resp)
    assert ei.value.status_code == 502


def test_response_json_object_maps_wrong_shape() -> None:
    with pytest.raises(ResponseError):
        common.response_json_object(_RaisingResp(payload=[1, 2, 3]))  # list, not object


def test_response_json_array_maps_wrong_shape() -> None:
    with pytest.raises(ResponseError):
        common.response_json_array(_RaisingResp(payload={"a": 1}))


def test_response_json_value_maps_decode_error() -> None:
    with pytest.raises(ResponseError):
        common.response_json_value(_RaisingResp(exc=ValueError("boom")))


def test_response_status_defaults_to_zero_when_non_int() -> None:
    resp = _RaisingResp(exc=ValueError("boom"), status_code=0)
    resp.status_code = "oops"  # type: ignore[assignment]
    with pytest.raises(ResponseError) as ei:
        common.response_json_value(resp)
    assert ei.value.status_code == 0


# email-not-verified marker mapping ----------------------------------------


def _email_body(url: str | None = "https://app.dagnam.ai/verify") -> str:
    detail: dict[str, str] = {"error": "email_not_verified", "message": "Email not verified."}
    if url is not None:
        detail["verification_url"] = url
    return json.dumps({"detail": detail})


def test_email_not_verified_on_training_dispatch_via_generic():
    with pytest.raises(EmailNotVerifiedError) as ei:
        common.raise_for_generic(_resp(403, text=_email_body(), content_type="application/json"))
    assert ei.value.verification_url == "https://app.dagnam.ai/verify"
    assert "app.dagnam.ai/verify" in str(ei.value)


def test_email_not_verified_on_deployment_dispatch():
    with pytest.raises(EmailNotVerifiedError):
        common.raise_for_deployment(
            _resp(403, text=_email_body(), content_type="application/json"), "dep-1"
        )


def test_email_not_verified_on_upload_dispatch():
    with pytest.raises(EmailNotVerifiedError):
        common.raise_for_upload(_resp(403, text=_email_body(), content_type="application/json"))


def test_email_not_verified_top_level_error_key_without_detail_wrapper():
    body = json.dumps({"error": "email_not_verified", "message": "nope"})
    with pytest.raises(EmailNotVerifiedError) as ei:
        common.raise_for_generic(_resp(403, text=body, content_type="application/json"))
    assert ei.value.verification_url is None  # no url in body


def test_email_not_verified_without_url_message_only():
    with pytest.raises(EmailNotVerifiedError) as ei:
        common.raise_for_generic(
            _resp(403, text=_email_body(url=None), content_type="application/json")
        )
    assert ei.value.verification_url is None
    assert "Email not verified." in str(ei.value)


def test_plain_403_without_marker_is_not_email_error():
    # A 403 that is not the email-verification gate falls through to APIError
    # (raise_for_generic has no dedicated 403 branch).
    with pytest.raises(APIError) as ei:
        common.raise_for_generic(_resp(403, text="Forbidden"))
    assert not isinstance(ei.value, EmailNotVerifiedError)
    assert ei.value.status_code == 403


def test_403_with_non_email_marker_is_not_email_error():
    body = json.dumps({"detail": {"error": "something_else"}})
    with pytest.raises(APIError):
        common.raise_for_generic(_resp(403, text=body, content_type="application/json"))


# account-status + blocked-IP marker mapping ------------------------------------


def _account_body(code: str, message: str = "…") -> str:
    return json.dumps({"detail": {"error": code, "message": message}})


def test_account_suspended_on_training_dispatch_via_generic():
    with pytest.raises(AccountSuspendedError) as ei:
        common.raise_for_generic(
            _resp(
                403,
                text=_account_body("account_suspended", "Account suspended."),
                content_type="application/json",
            )
        )
    assert "Account suspended." in str(ei.value)


def test_account_suspended_on_deployment_dispatch():
    with pytest.raises(AccountSuspendedError):
        common.raise_for_deployment(
            _resp(403, text=_account_body("account_suspended"), content_type="application/json"),
            "dep-1",
        )


def test_account_suspended_on_upload_dispatch():
    with pytest.raises(AccountSuspendedError):
        common.raise_for_upload(
            _resp(403, text=_account_body("account_suspended"), content_type="application/json")
        )


def test_account_locked_on_training_dispatch_via_generic():
    with pytest.raises(AccountLockedError) as ei:
        common.raise_for_generic(
            _resp(
                423,
                text=_account_body("account_locked", "Too many failed attempts."),
                content_type="application/json",
            )
        )
    assert "Too many failed attempts." in str(ei.value)


def test_account_locked_status_is_distinct_from_suspended():
    # A 403 carrying the "account_locked" marker (wrong status for that marker)
    # must NOT raise AccountLockedError — the two are keyed on status AND marker.
    with pytest.raises(APIError) as ei:
        common.raise_for_generic(
            _resp(403, text=_account_body("account_locked"), content_type="application/json")
        )
    assert not isinstance(ei.value, (AccountLockedError, AccountSuspendedError))


def test_blocked_ip_raises_existing_auth_error_not_a_new_type():
    # Deliberate: no dedicated exception class for a blocked IP (see Task 1).
    with pytest.raises(AuthError) as ei:
        common.raise_for_upload(
            _resp(
                403,
                text=_account_body("blocked_ip", "Request blocked."),
                content_type="application/json",
            )
        )
    assert not isinstance(ei.value, (AccountSuspendedError, AccountLockedError))
    assert "Request blocked." in str(ei.value)


def test_plain_403_without_account_marker_is_unaffected():
    with pytest.raises(APIError) as ei:
        common.raise_for_generic(_resp(403, text="Forbidden"))
    assert not isinstance(ei.value, (AccountSuspendedError, AccountLockedError))


def test_check_account_status_no_op_on_ok_status():
    common._check_account_status(
        _resp(200, text=_account_body("account_suspended"), content_type="application/json")
    )


# marker-helper branch coverage ------------------------------------------------


def test_error_code_reads_top_level_error():
    assert (
        common._error_code(
            _resp(400, text='{"error":"invalid_url"}', content_type="application/json")
        )
        == "invalid_url"
    )


def test_error_code_reads_detail_wrapped_error():
    assert (
        common._error_code(
            _resp(400, text='{"detail":{"error":"invalid_url"}}', content_type="application/json")
        )
        == "invalid_url"
    )


def test_error_code_none_when_error_not_string():
    assert (
        common._error_code(_resp(400, text='{"error":123}', content_type="application/json"))
        is None
    )


def test_error_code_none_on_unparseable_body():
    assert common._error_code(_resp(400, text="not json", content_type="text/plain")) is None


def test_response_payload_none_on_non_dict_json():
    assert (
        common._response_payload(_resp(400, text="[1,2,3]", content_type="application/json"))
        is None
    )


def test_response_payload_none_when_text_raises():
    class _NoText:
        status_code = 400

        @property
        def text(self) -> str:
            raise RuntimeError("boom")

    assert common._response_payload(_NoText()) is None  # pyright: ignore[reportArgumentType]


def test_check_email_verification_no_op_on_non_403():
    # non-403 returns without raising even if the marker is present
    common._check_email_verification(
        _resp(200, text=_email_body(), content_type="application/json")
    )


# 413 payload-too-large mapping --------------------------------------------


def test_413_maps_to_payload_too_large_via_generic():
    with pytest.raises(PayloadTooLargeError) as ei:
        common.raise_for_generic(_resp(413, text="File exceeds the 500 MB upload limit"))
    assert "500 MB" in str(ei.value)
    assert isinstance(ei.value, QuotaExceededError)  # still catchable as quota


def test_413_maps_to_payload_too_large_via_upload():
    with pytest.raises(PayloadTooLargeError):
        common.raise_for_upload(_resp(413, text="too big"))


def test_413_default_message_is_size_focused():
    with pytest.raises(PayloadTooLargeError, match="Upload exceeds the maximum allowed size"):
        common.raise_for_upload(_resp(413))


# dataset-from-URL (SSRF) rejection mapping ---------------------------------


def test_upload_url_ssrf_rejection_maps_to_invalid_url():
    body = json.dumps({"detail": {"error": "invalid_url", "message": "URL host is not allowed"}})
    with pytest.raises(InvalidURLError) as ei:
        common.raise_for_upload(_resp(400, text=body, content_type="application/json"))
    assert isinstance(ei.value, UploadError)  # still catchable as UploadError
    assert "not allowed" in str(ei.value)


def test_upload_url_rejected_marker_also_maps_to_invalid_url():
    body = json.dumps({"error": "url_rejected", "message": "bad scheme"})
    with pytest.raises(InvalidURLError):
        common.raise_for_upload(_resp(422, text=body, content_type="application/json"))


def test_upload_400_without_url_marker_stays_plain_upload_error():
    with pytest.raises(UploadError) as ei:
        common.raise_for_upload(_resp(400, text="name is required"))
    assert not isinstance(ei.value, InvalidURLError)
