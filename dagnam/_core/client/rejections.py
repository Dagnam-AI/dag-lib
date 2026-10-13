"""Map the error bodies any route can return to typed exceptions.

A plan-limit or credit 402, the email-verification gate, account state
(suspended, pending deletion, locked), a blocked source IP, the scope,
session-only and plan-gate 403s, and the idempotency 409 come from shared
server-side layers, so they can arrive on every route. Every ``raise_for_*``
mapper in :mod:`dagnam._core.client.common` runs :func:`check_common` first;
otherwise the same rejection would surface as a typed error on one endpoint
and a bare ``APIError`` on another.
"""

from __future__ import annotations

import json
from urllib.parse import urlparse

from dagnam._core._resume import IN_PROGRESS_MARKER, IN_PROGRESS_TEXT
from dagnam._core.client.credits import credit_refusal
from dagnam._core.exceptions import (
    AccountLockedError,
    AccountPendingDeletionError,
    AccountSuspendedError,
    APIError,
    AuthError,
    EmailNotVerifiedError,
    PermissionDeniedError,
    QuotaExceededError,
)
from dagnam._types import JsonObject, ResponseLike

MAX_ERROR_BODY = 2048
UNPARSEABLE_BODY = (ValueError, TypeError, RecursionError)
"""What decoding a response body can raise: not JSON, the wrong shape, or nested too deeply.

``json`` raises ``RecursionError`` (not a ``ValueError``) for a body of thousands of nested
brackets, which a hostile or broken server can send in a few kilobytes.
"""


def status_code(resp: ResponseLike) -> int:
    """The response's integer status code, or 0 when it has none."""
    code = resp.status_code
    return code if isinstance(code, int) else 0


def _check_entitlement(resp: ResponseLike, source: JsonObject | None) -> None:
    """Raise QuotaExceededError for a plan-limit or credit rejection (HTTP 402).

    Centralizes the mapping so every `raise_for_*` surfaces plan-limit errors as a
    clear QuotaExceededError rather than a generic APIError. The platform's
    ``insufficient_credits`` body becomes an InsufficientCreditsError. A plan limit sends
    ``{message, remediation_hints, ...}`` at the top level and surfaces its text and
    hints, so the caller sees an actionable error instead of a bare status code.
    """
    if status_code(resp) != 402:
        return
    refusal = credit_refusal(_response_payload(resp))
    if refusal is not None:
        raise refusal
    if source is None:
        raise QuotaExceededError(_plain_text_body(resp) or "Plan limit reached")
    message = _error_message(source) or "Plan limit reached"
    hints = source.get("remediation_hints")
    if isinstance(hints, list) and hints:
        message = f"{message} ({'; '.join(str(h) for h in hints)})"
    raise QuotaExceededError(message)


EMAIL_NOT_VERIFIED_CODE = "email_not_verified"


def _raw_text(resp: ResponseLike) -> str:
    """Up to ``MAX_ERROR_BODY`` characters of the body, or "" when it cannot be read.

    An unread streaming body (``stream=True``) counts as unreadable: touching
    ``.text`` would force an uncapped synchronous drain of the whole body.
    Callers that want its text read the body first.
    """
    if getattr(resp, "_content", None) is False:
        return ""
    try:
        return str(resp.text)[:MAX_ERROR_BODY]
    except Exception:
        return ""


def _plain_text_body(resp: ResponseLike) -> str | None:
    """The capped body when it is plain text rather than JSON, else None."""
    text = _raw_text(resp).strip()
    if not text:
        return None
    try:
        json.loads(text)
    except UNPARSEABLE_BODY:
        return text
    return None


def _response_payload(resp: ResponseLike) -> JsonObject | None:
    """Best-effort decode of a response body to a JSON object, or None.

    Reads the raw body (not ``safe_response_text``, which flattens a FastAPI
    ``detail`` and would drop the machine-readable ``error`` marker). Never
    raises: a body that cannot be read or is not a JSON object yields None.

    It reads through ``_raw_text``, whose two guards keep this cheap on an
    error path that also runs for responses opened with ``stream=True``: an
    unread streaming body is skipped, and the text is capped at
    ``MAX_ERROR_BODY``. A marker never appears past that cap, and a body
    truncated mid-JSON simply fails to parse and yields None.
    """
    try:
        data = json.loads(_raw_text(resp))
    except UNPARSEABLE_BODY:
        return None
    return data if isinstance(data, dict) else None


def _error_detail_source(resp: ResponseLike) -> JsonObject | None:
    """The object that carries the ``error``/``message`` fields.

    Handles both a top-level ``{"error": ...}`` body and a FastAPI
    ``{"detail": {"error": ...}}`` wrapper (from ``HTTPException(detail={...})``).
    """
    data = _response_payload(resp)
    if data is None:
        return None
    detail = data.get("detail")
    if isinstance(detail, dict):
        return detail
    return data


def error_marker(source: JsonObject | None) -> str | None:
    """The machine-readable error marker carried by a decoded error body, or None."""
    if source is None:
        return None
    code = source.get("error")
    return code if isinstance(code, str) else None


def _error_message(source: JsonObject | None) -> str | None:
    """The human-readable message carried by a decoded error body, or None.

    Most bodies name it ``message``; the training preflight names it
    ``user_message``; a plain FastAPI error is a string ``detail``.
    """
    if source is None:
        return None
    for key in ("message", "user_message", "detail"):
        message = source.get(key)
        if isinstance(message, str) and message:
            return message
    return None


def _safe_verification_url(source: JsonObject | None) -> str | None:
    """The server-supplied verification link, but only when it is safe to echo.

    The message built around this URL tells the user to go there, and the API
    base URL is caller-configurable (``DAGNAM_API_URL``), so a hostile or
    misconfigured host could otherwise have the SDK present a ``javascript:``
    URL or a plaintext phishing origin as authoritative. Only an absolute
    ``https`` URL is surfaced; anything else is dropped and the caller keeps
    the bare message.
    """
    url = source.get("verification_url") if source is not None else None
    if not isinstance(url, str) or not url:
        return None
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        return None
    return url


def _check_email_verification(resp: ResponseLike, source: JsonObject | None) -> None:
    """Raise EmailNotVerifiedError for a 403 email-verification-gate rejection.

    Keyed on the ``email_not_verified`` marker so a 403 raised for any other
    reason falls through to the mapper's existing handling.
    """
    if status_code(resp) != 403:
        return
    if error_marker(source) != EMAIL_NOT_VERIFIED_CODE:
        return
    text = _error_message(source) or "Email address is not verified."
    verification_url = _safe_verification_url(source)
    if verification_url is not None:
        text = f"{text} Verify your email at {verification_url}, then retry."
    raise EmailNotVerifiedError(text, verification_url=verification_url)


ACCOUNT_SUSPENDED_CODE = "account_suspended"
ACCOUNT_PENDING_DELETION_CODE = "account_pending_deletion"
ACCOUNT_LOCKED_CODE = "account_locked"
BLOCKED_IP_CODE = "blocked_ip"
# 403s from a caller the server knows: a key without the scope, a key on a
# session-only route, and the plan gates. A new key or a retry will not help.
PERMISSION_DENIED_CODES = frozenset(
    {
        "insufficient_scope",
        "api_key_not_allowed",
        "feature_gated",
        "remote_gpu_not_entitled",
        "compute_strategy_not_in_tier",
    }
)


def _check_account_status(resp: ResponseLike, source: JsonObject | None) -> None:
    """Raise the typed error for an account-state or blocked-IP rejection.

    ``account_suspended`` raises AccountSuspendedError, ``account_pending_deletion``
    AccountPendingDeletionError (both 403), ``account_locked`` AccountLockedError
    (423), and ``blocked_ip`` AuthError (403).

    Keyed on status CODE + marker together (not either alone) so a 403/423
    raised for any other reason falls through to the mapper's existing
    handling, and a marker on the wrong status code is never misread — see
    ``test_account_locked_status_is_distinct_from_suspended``.

    A blocked IP deliberately raises the existing ``AuthError`` rather than a
    new exception type: unlike an unverified email or a lockout window, there
    is no differently-actionable step an SDK caller can take programmatically
    (it cannot retry from a different source IP on its own), so a dedicated
    class would only add a name with no behavioral value.
    """
    code = status_code(resp)
    marker = error_marker(source)
    text = _error_message(source)
    if code == 403 and marker == ACCOUNT_SUSPENDED_CODE:
        raise AccountSuspendedError(text or "This account has been suspended.")
    if code == 403 and marker == ACCOUNT_PENDING_DELETION_CODE:
        raise AccountPendingDeletionError(text or "This account is scheduled for deletion.")
    if code == 423 and marker == ACCOUNT_LOCKED_CODE:
        raise AccountLockedError(text or "This account is temporarily locked. Try again later.")
    if code == 403 and marker == BLOCKED_IP_CODE:
        raise AuthError(text or "Request blocked: this IP address is not permitted.")


def _check_permission(resp: ResponseLike, source: JsonObject | None) -> None:
    """Raise PermissionDeniedError for a 403 carrying a scope, session or plan marker."""
    if source is None or status_code(resp) != 403:
        return
    marker = error_marker(source)
    if marker not in PERMISSION_DENIED_CODES:
        return
    text = _error_message(source) or f"Permission denied ({marker})."
    raise PermissionDeniedError(text, reason=marker, required_scopes=_required_scopes(source))


def _required_scopes(source: JsonObject) -> list[str] | None:
    """The ``required_scopes`` an ``insufficient_scope`` 403 names, or None."""
    scopes = source.get("required_scopes")
    if not isinstance(scopes, list):
        return None
    return [scope for scope in scopes if isinstance(scope, str)]


def _check_idempotency_conflict(resp: ResponseLike, source: JsonObject | None) -> None:
    """Raise the retryable ``APIError(409)`` for the idempotency conflict.

    The server answers it while the first request with the same key is still
    running. Newer servers mark it ``idempotency_in_progress``; older ones send
    only the text. Both become one ``APIError`` with that text, which is what
    the retry driver keys on, and never a domain 409 such as
    ``DeploymentStateError``.
    """
    if status_code(resp) != 409:
        return
    if error_marker(source) == IN_PROGRESS_MARKER or _error_message(source) == IN_PROGRESS_TEXT:
        raise APIError(409, IN_PROGRESS_TEXT)


def check_common(resp: ResponseLike) -> JsonObject | None:
    """Raise the typed error for any cross-cutting rejection in ``resp``.

    Returns the decoded error-detail object so a mapper that needs a further
    marker (e.g. the upload URL-rejection codes) does not decode the body
    again.
    """
    source = _error_detail_source(resp)
    _check_entitlement(resp, source)
    _check_email_verification(resp, source)
    _check_account_status(resp, source)
    _check_permission(resp, source)
    _check_idempotency_conflict(resp, source)
    return source
