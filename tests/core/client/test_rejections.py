"""Coverage for ``dagnam._core.client.rejections``: the error bodies any route can return.

Most cases go through the ``raise_for_*`` mappers, which all run
``check_common`` first, so each rejection is proven typed on every mapper.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from typing import ClassVar

import pytest

from dagnam._core._resume import IN_PROGRESS_TEXT
from dagnam._core.client import common, rejections
from dagnam._core.exceptions import (
    AccountLockedError,
    AccountPendingDeletionError,
    AccountSuspendedError,
    APIError,
    AuthError,
    DeploymentStateError,
    EmailNotVerifiedError,
    InsufficientCreditsError,
    PermissionDeniedError,
    QuotaExceededError,
)


class _Resp:
    def __init__(self, status: int, *, text: str = "", content_type: str | None = None) -> None:
        self.status_code = status
        self.ok = 200 <= status < 300
        self.text = text
        self.content = text.encode()
        self.headers: dict[str, str] = {}
        self._content = b""
        if content_type is not None:
            self.headers["Content-Type"] = content_type


def _resp(status: int, *, text: str = "", content_type: str | None = None) -> _Resp:
    return _Resp(status, text=text, content_type=content_type)


# plan-limit and credit 402 ------------------------------------------------------


def test_entitlement_402_raises_quota_with_actionable_message() -> None:
    """A backend plan-limit rejection (402) maps to QuotaExceededError carrying the
    rejection message + remediation hints, across every raise_for_* entry point."""
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
    body = json.dumps({"message": "Limit reached", "remediation_hints": []})
    with pytest.raises(QuotaExceededError) as excinfo:
        common.raise_for_generic(_resp(402, text=body, content_type="application/json"))
    assert str(excinfo.value) == "Limit reached"


def test_entitlement_402_plain_text_body_is_the_message() -> None:
    with pytest.raises(QuotaExceededError) as excinfo:
        common.raise_for_generic(_resp(402, text="Out of credits for this month.\n"))
    assert str(excinfo.value) == "Out of credits for this month."


def test_entitlement_402_plain_text_body_is_capped() -> None:
    with pytest.raises(QuotaExceededError) as excinfo:
        common.raise_for_generic(_resp(402, text="x" * 5000))
    assert len(str(excinfo.value)) == rejections.MAX_ERROR_BODY


def test_entitlement_402_unread_streaming_body_is_not_drained() -> None:
    class _StreamingResp:
        _content = False
        status_code = 402
        content = b""
        headers: ClassVar[dict[str, str]] = {"Content-Type": "text/plain"}
        text_reads = 0

        @property
        def text(self) -> str:
            type(self).text_reads += 1
            return "Out of credits."

    with pytest.raises(QuotaExceededError, match="Plan limit reached"):
        rejections.check_common(_StreamingResp())  # pyright: ignore[reportArgumentType]
    assert _StreamingResp.text_reads == 0


def test_entitlement_402_unreadable_body_falls_back() -> None:
    class _NoText:
        _content = b"x"
        status_code = 402

        @property
        def text(self) -> str:
            raise RuntimeError("boom")

    with pytest.raises(QuotaExceededError, match="Plan limit reached"):
        rejections.check_common(_NoText())  # pyright: ignore[reportArgumentType]


def test_entitlement_402_non_dict_json_falls_back() -> None:
    with pytest.raises(QuotaExceededError) as excinfo:
        common.raise_for_generic(_resp(402, text="[1, 2, 3]", content_type="application/json"))
    assert "Plan limit reached" in str(excinfo.value)


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


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(document.cookie)",
        "http://phishing.example/verify",
        "https://",
        "https://[unclosed",
        "/verify",
    ],
)
def test_unsafe_verification_url_is_dropped_not_echoed(url: str) -> None:
    """Only an absolute https URL is echoed as a "go here" instruction.

    The API base URL is caller-configurable, so a hostile or misconfigured host
    could otherwise have the SDK present a ``javascript:`` URL or a plaintext
    phishing origin as authoritative.
    """
    with pytest.raises(EmailNotVerifiedError) as ei:
        common.raise_for_generic(
            _resp(403, text=_email_body(url=url), content_type="application/json")
        )
    assert ei.value.verification_url is None
    assert url not in str(ei.value)
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
    # Deliberate: no dedicated exception class for a blocked IP (see `_check_account_status`).
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


_EVERY_MAPPER: list[Callable[[_Resp], None]] = [
    common.raise_for_generic,
    common.raise_for_hub,
    common.raise_for_project,
    common.raise_for_codegen,
    common.raise_for_upload,
]


@pytest.mark.parametrize("raiser", _EVERY_MAPPER)
def test_account_suspended_is_typed_on_every_mapper(raiser: Callable[[_Resp], None]) -> None:
    # The suspended-account 403 comes from the shared auth dependency, so it can
    # arrive on ANY route: every mapper must surface the same typed error.
    with pytest.raises(AccountSuspendedError):
        raiser(_resp(403, text=_account_body("account_suspended"), content_type="application/json"))


@pytest.mark.parametrize("raiser", _EVERY_MAPPER)
def test_account_locked_is_typed_on_every_mapper(raiser: Callable[[_Resp], None]) -> None:
    with pytest.raises(AccountLockedError):
        raiser(_resp(423, text=_account_body("account_locked"), content_type="application/json"))


@pytest.mark.parametrize("raiser", _EVERY_MAPPER)
def test_blocked_ip_is_typed_on_every_mapper(raiser: Callable[[_Resp], None]) -> None:
    # The IP blocklist middleware runs on every route, not just dispatch ones.
    with pytest.raises(AuthError):
        raiser(_resp(403, text=_account_body("blocked_ip"), content_type="application/json"))


@pytest.mark.parametrize("raiser", _EVERY_MAPPER)
def test_email_not_verified_is_typed_on_every_mapper(raiser: Callable[[_Resp], None]) -> None:
    with pytest.raises(EmailNotVerifiedError):
        raiser(_resp(403, text=_email_body(), content_type="application/json"))


def test_account_suspended_on_hub_carries_status_code() -> None:
    with pytest.raises(AccountSuspendedError) as ei:
        common.raise_for_hub(
            _resp(
                403,
                text=_account_body("account_suspended", "Account suspended."),
                content_type="application/json",
            ),
            model_id="model-1",
        )
    assert isinstance(ei.value, APIError)
    assert ei.value.status_code == 403


def test_plain_403_without_account_marker_is_unaffected():
    with pytest.raises(APIError) as ei:
        common.raise_for_generic(_resp(403, text="Forbidden"))
    assert not isinstance(ei.value, (AccountSuspendedError, AccountLockedError))


def test_check_account_status_no_op_on_ok_status():
    resp = _resp(200, text=_account_body("account_suspended"), content_type="application/json")
    rejections._check_account_status(resp, rejections._error_detail_source(resp))


# marker-helper branch coverage ------------------------------------------------


def _marker(resp: _Resp) -> str | None:
    return rejections.error_marker(rejections._error_detail_source(resp))


def test_error_code_reads_top_level_error():
    assert _marker(_resp(400, text='{"error":"invalid_url"}', content_type="application/json")) == (
        "invalid_url"
    )


def test_error_code_reads_detail_wrapped_error():
    assert (
        _marker(
            _resp(400, text='{"detail":{"error":"invalid_url"}}', content_type="application/json")
        )
        == "invalid_url"
    )


def test_error_code_none_when_error_not_string():
    assert _marker(_resp(400, text='{"error":123}', content_type="application/json")) is None


def test_error_code_none_on_unparseable_body():
    assert _marker(_resp(400, text="not json", content_type="text/plain")) is None


def test_response_payload_none_on_non_dict_json():
    assert (
        rejections._response_payload(_resp(400, text="[1,2,3]", content_type="application/json"))
        is None
    )


def test_response_payload_none_when_text_raises():
    class _NoText:
        status_code = 400

        @property
        def text(self) -> str:
            raise RuntimeError("boom")

    assert rejections._response_payload(_NoText()) is None  # pyright: ignore[reportArgumentType]


def test_response_payload_never_drains_an_unread_streaming_body():
    """A response opened with ``stream=True`` must not be materialized here.

    ``requests`` marks an unread streaming body with ``_content is False``;
    touching ``.text`` then drains the WHOLE body synchronously with no size
    cap (a hang or OOM on a large error response), which is exactly what
    ``safe_response_text`` already refuses to do.
    """

    class _StreamingResp:
        _content = False
        status_code = 403
        content = b""

        def __init__(self) -> None:
            self.headers = {"Content-Type": "application/json"}
            self.text_reads = 0

        @property
        def text(self) -> str:
            self.text_reads += 1
            return _account_body("account_suspended")

    resp = _StreamingResp()
    assert rejections._response_payload(resp) is None  # pyright: ignore[reportArgumentType]
    assert resp.text_reads == 0
    # Consequence of the guard: a marker is not recovered from an undrained
    # streaming body, so the mapper falls back to a plain APIError rather than
    # blocking on an unbounded read.
    with pytest.raises(APIError) as ei:
        common.raise_for_deployment(resp, "dep-1")  # pyright: ignore[reportArgumentType]
    assert not isinstance(ei.value, AccountSuspendedError)


def test_response_payload_caps_the_parsed_body_length():
    # A marker never appears past the cap; a huge body is truncated (and then
    # fails to parse) instead of being handed whole to json.loads.
    padding = " " * 4000
    body = "{" + padding + '"error":"email_not_verified"}'
    assert (
        rejections._response_payload(_resp(403, text=body, content_type="application/json")) is None
    )


def test_check_email_verification_no_op_on_non_403():
    # non-403 returns without raising even if the marker is present
    resp = _resp(200, text=_email_body(), content_type="application/json")
    rejections._check_email_verification(resp, rejections._error_detail_source(resp))


# credits, plan gates, scope and account state (the auth status convention) ------


def _json(status: int, payload: object) -> _Resp:
    return _resp(status, text=json.dumps(payload), content_type="application/json")


def test_insufficient_credits_402_reads_user_message_and_carries_the_numbers() -> None:
    # The credit preflight wraps its body in ``detail`` and names the text
    # ``user_message``: both used to be missed, leaving "Plan limit reached".
    # The sentence is the platform's own, so nothing is appended to it.
    body = {
        "detail": {
            "error": "insufficient_credits",
            "user_message": "Your credit balance does not cover this job's estimated cost.",
            "required_credits": 120,
            "available_credits": 40,
        }
    }
    with pytest.raises(InsufficientCreditsError) as ei:
        common.raise_for_training_job(_json(402, body), "job-1")
    assert str(ei.value) == "Your credit balance does not cover this job's estimated cost."
    assert ei.value.required_credits == 120
    assert ei.value.available_credits == 40


def test_402_credit_numbers_must_be_integers() -> None:
    body = {"error": "insufficient_credits", "required_credits": "lots", "available_credits": True}
    with pytest.raises(QuotaExceededError) as ei:
        common.raise_for_generic(_json(402, body))
    assert not isinstance(ei.value, InsufficientCreditsError)
    assert str(ei.value) == "Plan limit reached"


def test_402_string_detail_surfaces_the_text() -> None:
    with pytest.raises(QuotaExceededError) as ei:
        common.raise_for_generic(_json(402, {"detail": "plan limit"}))
    assert str(ei.value) == "plan limit"


_PERMISSION_BODIES = [
    ("insufficient_scope", {"detail": {"error": "insufficient_scope", "message": "Needs write."}}),
    (
        "api_key_not_allowed",
        {"detail": {"error": "api_key_not_allowed", "message": "Needs a web session."}},
    ),
    ("feature_gated", {"error": "feature_gated", "message": "Not on your plan."}),
    (
        "remote_gpu_not_entitled",
        {"detail": {"error": "remote_gpu_not_entitled", "user_message": "No remote GPU."}},
    ),
    (
        "compute_strategy_not_in_tier",
        {
            "detail": {
                "error": "compute_strategy_not_in_tier",
                "user_message": "Strategy needs Pro.",
                "remediation_hints": ["upgrade_to_pro"],
            }
        },
    ),
]


@pytest.mark.parametrize(("reason", "payload"), _PERMISSION_BODIES)
@pytest.mark.parametrize("raiser", _EVERY_MAPPER)
def test_permission_refusals_are_typed_on_every_mapper(
    raiser: Callable[[_Resp], None], reason: str, payload: dict[str, object]
) -> None:
    with pytest.raises(PermissionDeniedError) as ei:
        raiser(_json(403, payload))
    assert ei.value.reason == reason
    assert ei.value.status_code == 403
    assert "{" not in str(ei.value)  # a readable message, never the raw JSON body


def test_insufficient_scope_carries_the_required_scopes() -> None:
    body = {
        "detail": {
            "error": "insufficient_scope",
            "message": "API key does not have permission for this operation",
            "required_scopes": ["write", "write:datasets"],
        }
    }
    with pytest.raises(PermissionDeniedError) as ei:
        common.raise_for_generic(_json(403, body))
    assert ei.value.required_scopes == ["write", "write:datasets"]


@pytest.mark.parametrize(
    ("scopes", "expected"),
    [
        (None, None),
        ("write", None),  # not a list
        (["write", 3, None, "read"], ["write", "read"]),  # non-strings dropped
    ],
)
def test_required_scopes_ignores_malformed_values(scopes: object, expected: object) -> None:
    detail: dict[str, object] = {"error": "insufficient_scope", "message": "No."}
    if scopes is not None:
        detail["required_scopes"] = scopes
    with pytest.raises(PermissionDeniedError) as ei:
        common.raise_for_generic(_json(403, {"detail": detail}))
    assert ei.value.required_scopes == expected


def test_permission_refusal_without_message_names_the_reason() -> None:
    with pytest.raises(PermissionDeniedError) as ei:
        common.raise_for_generic(_json(403, {"detail": {"error": "insufficient_scope"}}))
    assert "insufficient_scope" in str(ei.value)


def test_unknown_403_marker_stays_plain_api_error() -> None:
    with pytest.raises(APIError) as ei:
        common.raise_for_generic(_json(403, {"detail": {"error": "admin_only"}}))
    assert not isinstance(ei.value, PermissionDeniedError)


def test_permission_marker_on_another_status_is_not_misread() -> None:
    with pytest.raises(APIError) as ei:
        common.raise_for_generic(_json(400, {"error": "feature_gated", "message": "x"}))
    assert not isinstance(ei.value, PermissionDeniedError)


@pytest.mark.parametrize("raiser", _EVERY_MAPPER)
def test_account_pending_deletion_is_typed_on_every_mapper(
    raiser: Callable[[_Resp], None],
) -> None:
    with pytest.raises(AccountPendingDeletionError) as ei:
        raiser(
            _resp(
                403,
                text=_account_body("account_pending_deletion", "Scheduled."),
                content_type="application/json",
            )
        )
    assert "Scheduled." in str(ei.value)
    assert isinstance(ei.value, AccountSuspendedError)  # existing handlers keep catching it


def test_account_pending_deletion_default_message() -> None:
    with pytest.raises(AccountPendingDeletionError, match="scheduled for deletion"):
        common.raise_for_generic(_json(403, {"detail": {"error": "account_pending_deletion"}}))


def test_deployment_limit_409_is_a_state_error_with_its_reason() -> None:
    body = {"detail": {"error": "deployment_limit_reached", "message": "Stop one first."}}
    with pytest.raises(DeploymentStateError) as ei:
        common.raise_for_deployment(_json(409, body), "deployment")
    assert ei.value.reason == "deployment_limit_reached"
    assert str(ei.value) == "Stop one first."


# The idempotency conflict: older servers send only the text, newer ones add the
# ``idempotency_in_progress`` marker. Both normalize to the one message the
# retry driver keys on.
_IDEMPOTENCY_BODIES = [
    pytest.param({"detail": IN_PROGRESS_TEXT}, id="text-only"),
    pytest.param(
        {"detail": {"error": "idempotency_in_progress", "message": IN_PROGRESS_TEXT}},
        id="marker-and-text",
    ),
    pytest.param(
        {"detail": {"error": "idempotency_in_progress", "message": "Still running."}},
        id="marker-with-other-text",
    ),
    pytest.param(
        {"detail": IN_PROGRESS_TEXT, "error": "idempotency_in_progress"},
        id="top-level-marker",
    ),
]


@pytest.mark.parametrize("payload", _IDEMPOTENCY_BODIES)
@pytest.mark.parametrize(
    "raiser",
    [
        common.raise_for_generic,
        lambda r: common.raise_for_deployment(r, "deployment"),
        lambda r: common.raise_for_model(r),
    ],
)
def test_idempotency_conflict_normalizes_to_the_retryable_api_error(
    raiser: Callable[[_Resp], None], payload: dict[str, object]
) -> None:
    with pytest.raises(APIError) as ei:
        raiser(_json(409, payload))
    assert type(ei.value) is APIError  # never DeploymentStateError / ModelError
    assert ei.value.status_code == 409
    assert ei.value.message == IN_PROGRESS_TEXT


def test_idempotency_marker_on_another_status_is_not_misread() -> None:
    body = {"detail": {"error": "idempotency_in_progress", "message": "x"}}
    with pytest.raises(APIError) as ei:
        common.raise_for_generic(_json(422, body))
    assert ei.value.message == "x"


def test_check_common_returns_the_decoded_detail_when_nothing_matches() -> None:
    source = rejections.check_common(_json(400, {"detail": {"error": "invalid_url"}}))
    assert source == {"error": "invalid_url"}
