"""Shared helpers for sync & async clients.

Centralizes URL building, header construction, and HTTP status → exception
mapping so the sync (``requests``) and async (``httpx``) clients cannot drift.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from urllib.parse import quote, urlparse

from dagnam._core.exceptions import (
    AccountLockedError,
    AccountSuspendedError,
    APIError,
    ArchitectureVersionNotFoundError,
    AuthError,
    CheckpointNotFoundError,
    CodegenError,
    CodegenValidationError,
    DagnamError,
    DatasetNotFoundError,
    DeploymentNotFoundError,
    DeploymentStateError,
    DeploymentValidationError,
    EmailNotVerifiedError,
    HubError,
    HubModelNotFoundError,
    InvalidURLError,
    ModelError,
    ModelNotFoundError,
    PayloadTooLargeError,
    ProjectNotFoundError,
    QuotaExceededError,
    ResponseError,
    TaskNotFoundError,
    TrainingJobNotFoundError,
    UploadError,
)
from dagnam._types import (
    JsonArray,
    JsonObject,
    JsonResponseLike,
    JsonValue,
    QueryParams,
    QueryScalar,
    ResponseLike,
    ensure_json_array,
    ensure_json_object,
    ensure_json_value,
)

API_BASE = "/api/v1"
_MAX_ERROR_BODY = 2048
_TEXT_CONTENT_MARKERS = ("text/", "json", "xml", "javascript", "yaml", "html")


def build_url(api_url: str, path: str) -> str:
    """Join a base API URL with a path that may or may not start with /."""
    return f"{api_url.rstrip('/')}{path if path.startswith('/') else '/' + path}"


def quote_path_segment(value: str) -> str:
    """Percent-encode an untrusted value for use as one URL path segment."""
    return quote(str(value), safe="")


def _response_status(resp: JsonResponseLike) -> int:
    code = getattr(resp, "status_code", 0)
    return code if isinstance(code, int) else 0


def response_json_value(resp: JsonResponseLike) -> JsonValue:
    """Decode a response body and validate that it is JSON-compatible."""
    try:
        return ensure_json_value(resp.json())
    except (ValueError, TypeError) as exc:
        raise ResponseError(_response_status(resp), f"malformed response body: {exc}") from exc


def response_json_object(resp: JsonResponseLike) -> JsonObject:
    """Decode a response body and validate that it is a JSON object."""
    try:
        return ensure_json_object(resp.json())
    except (ValueError, TypeError) as exc:
        raise ResponseError(_response_status(resp), f"malformed response body: {exc}") from exc


def response_json_array(resp: JsonResponseLike) -> JsonArray:
    """Decode a response body and validate that it is a JSON array."""
    try:
        return ensure_json_array(resp.json())
    except (ValueError, TypeError) as exc:
        raise ResponseError(_response_status(resp), f"malformed response body: {exc}") from exc


def bearer_headers(api_key: str, *, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return {Authorization: Bearer <key>} plus object extras."""
    h = {"Authorization": f"Bearer {api_key}"}
    if extra:
        h.update(extra)
    return h


def stream_query_params(token: str) -> dict[str, str]:
    """Query params carrying a short-lived stream-access token for SSE reads.

    Long-lived API keys stay in headers; only short-lived, resource-scoped stream
    tokens appear in URL query strings.
    """
    return {"token": token}


def requests_query_params(params: QueryParams | None) -> list[tuple[str, str]] | None:
    """Convert SDK query params to a requests-compatible repeated-key list."""
    if params is None:
        return None
    converted: list[tuple[str, str]] = []
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, str | int | float | bool):
            converted.append((key, str(value)))
            continue
        for item in value:
            if item is not None:
                converted.append((key, _query_scalar_to_string(item)))
    return converted


def _query_scalar_to_string(value: QueryScalar) -> str:
    if value is None:
        return ""
    return str(value)


# ---------------------------------------------------------------------------
# Generic status mapper
# ---------------------------------------------------------------------------


def _text(resp: ResponseLike) -> str:
    return safe_response_text(resp)


def _short_error_text(text: str) -> str:
    if len(text) > _MAX_ERROR_BODY:
        return text[:_MAX_ERROR_BODY] + f"... [truncated, {len(text)} chars total]"
    return text


def _format_fastapi_detail(text: str) -> str:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return text
    if not isinstance(payload, dict) or "detail" not in payload:
        return text

    detail = payload["detail"]
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        messages = []
        for item in detail:
            if not isinstance(item, dict):
                continue
            location = item.get("loc")
            field = ".".join(str(part) for part in location) if isinstance(location, list) else ""
            message = item.get("msg")
            if isinstance(message, str):
                messages.append(f"{field}: {message}" if field else message)
        if messages:
            return "; ".join(messages)
    if isinstance(detail, dict):
        # Structured error details (e.g. codegen's
        # ``{"error": ..., "message": ...}``) surface their human-readable
        # message rather than the JSON-stringified dict.
        message = detail.get("message")
        if isinstance(message, str):
            return message
    return json.dumps(detail, default=str)


def safe_response_text(resp: ResponseLike) -> str:
    """Return a short, text-safe response body for exception messages."""
    try:
        headers = resp.headers
        content_type = str(headers.get("Content-Type") or headers.get("content-type") or "").lower()
    except AttributeError:
        content_type = ""

    streaming_marker = getattr(resp, "_content", None)
    if streaming_marker is False:
        if content_type and any(marker in content_type for marker in _TEXT_CONTENT_MARKERS):
            try:
                text_value = str(resp.text)
            except Exception:
                text_value = ""
            if text_value:
                return _short_error_text(_format_fastapi_detail(text_value))
        if content_type:
            return f"<streaming {content_type} body omitted>"
        return "<streaming response body omitted>"

    content = resp.content or b""
    if content_type and not any(marker in content_type for marker in _TEXT_CONTENT_MARKERS):
        try:
            body_len = len(content)
        except TypeError:
            body_len = 0
        return f"<{body_len} bytes of {content_type}>"

    try:
        text = resp.text
    except Exception:
        try:
            body_len = len(content)
        except TypeError:
            body_len = 0
        return f"<{body_len} bytes; failed to decode body>"
    text_value = str(text)
    return _short_error_text(_format_fastapi_detail(text_value))


def _ok(resp: ResponseLike) -> bool:
    status_code = resp.status_code
    if isinstance(status_code, int):
        return 200 <= status_code < 300
    return bool(getattr(resp, "ok", False))


def _status_code(resp: ResponseLike) -> int:
    status_code = resp.status_code
    return status_code if isinstance(status_code, int) else 0


def _entitlement_message(resp: ResponseLike) -> str:
    """Build a user-facing message from a backend LimitRejection payload.

    The backend returns `{message, remediation_hints, required_plan, ...}` (see
    entitlements.LimitRejection). We surface the message plus any remediation hints
    so the caller sees an actionable error instead of a bare status code.
    """
    body = _text(resp)
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return body or "Plan limit reached"
    if not isinstance(data, dict):
        return "Plan limit reached"
    message = str(data.get("message") or "Plan limit reached")
    hints = data.get("remediation_hints")
    if isinstance(hints, list) and hints:
        message = f"{message} ({'; '.join(str(h) for h in hints)})"
    return message


def _check_entitlement(resp: ResponseLike) -> None:
    """Raise QuotaExceededError for an entitlement/plan-limit rejection (HTTP 402).

    Centralizes the mapping so every `raise_for_*` surfaces plan-limit errors as a
    clear QuotaExceededError rather than a generic APIError. (The server maps a
    numeric plan-limit hit to 402 Payment Required.)
    """
    if _status_code(resp) == 402:
        raise QuotaExceededError(_entitlement_message(resp))


EMAIL_NOT_VERIFIED_CODE = "email_not_verified"
INVALID_URL_CODES = frozenset({"invalid_url", "url_rejected"})


def _response_payload(resp: ResponseLike) -> JsonObject | None:
    """Best-effort decode of a response body to a JSON object, or None.

    Reads the raw body (not ``safe_response_text``, which flattens a FastAPI
    ``detail`` and would drop the machine-readable ``error`` marker). Never
    raises: a body that cannot be read or is not a JSON object yields None.

    Two guards keep this cheap on an error path that also runs for responses
    opened with ``stream=True``: an unread streaming body is skipped entirely
    (touching ``.text`` would force an uncapped synchronous drain of the whole
    body — the same guard ``safe_response_text`` applies), and the text that IS
    parsed is capped at ``_MAX_ERROR_BODY``. A marker never appears past that
    cap, and a body truncated mid-JSON simply fails to parse and yields None.
    """
    if getattr(resp, "_content", None) is False:
        return None
    try:
        raw = str(resp.text)
    except Exception:
        return None
    try:
        data = json.loads(raw[:_MAX_ERROR_BODY])
    except (ValueError, TypeError):
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


def _error_marker(source: JsonObject | None) -> str | None:
    """The machine-readable error marker carried by a decoded error body, or None."""
    if source is None:
        return None
    code = source.get("error")
    return code if isinstance(code, str) else None


def _error_message(source: JsonObject | None) -> str | None:
    """The human-readable message carried by a decoded error body, or None."""
    if source is None:
        return None
    message = source.get("message")
    return message if isinstance(message, str) else None


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
    if _status_code(resp) != 403:
        return
    if _error_marker(source) != EMAIL_NOT_VERIFIED_CODE:
        return
    text = _error_message(source) or "Email address is not verified."
    verification_url = _safe_verification_url(source)
    if verification_url is not None:
        text = f"{text} Verify your email at {verification_url}, then retry."
    raise EmailNotVerifiedError(text, verification_url=verification_url)


ACCOUNT_SUSPENDED_CODE = "account_suspended"
ACCOUNT_LOCKED_CODE = "account_locked"
BLOCKED_IP_CODE = "blocked_ip"


def _check_account_status(resp: ResponseLike, source: JsonObject | None) -> None:
    """Raise AccountSuspendedError/AccountLockedError/AuthError for an account-moderation or blocked-IP rejection.

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
    code = _status_code(resp)
    marker = _error_marker(source)
    text = _error_message(source)
    if code == 403 and marker == ACCOUNT_SUSPENDED_CODE:
        raise AccountSuspendedError(text or "This account has been suspended.")
    if code == 423 and marker == ACCOUNT_LOCKED_CODE:
        raise AccountLockedError(text or "This account is temporarily locked. Try again later.")
    if code == 403 and marker == BLOCKED_IP_CODE:
        raise AuthError(text or "Request blocked: this IP address is not permitted.")


def _check_common(resp: ResponseLike) -> JsonObject | None:
    """Map the cross-cutting rejections that ANY endpoint can return.

    A plan-limit 402, the email-verification gate, account suspension/lockout
    and a blocked source IP are raised by shared server-side middleware and
    dependencies, so they can arrive on every route — every ``raise_for_*``
    mapper runs this first, otherwise the same rejection would surface as a
    typed error on one endpoint and a bare ``APIError`` on another.

    Returns the decoded error-detail object so a mapper that needs a further
    marker (e.g. the upload URL-rejection codes) does not decode the body
    again.
    """
    _check_entitlement(resp)
    source = _error_detail_source(resp)
    _check_email_verification(resp, source)
    _check_account_status(resp, source)
    return source


def raise_for_generic(
    resp: ResponseLike,
    not_found_exc: type[DagnamError] | None = None,
    not_found_arg: str | None = None,
) -> None:
    """Map a response to (Auth|*NotFound|Quota|API)Error if not OK."""
    if _ok(resp):
        return
    _check_common(resp)
    code = _status_code(resp)
    if code == 401:
        raise AuthError("Authentication failed: invalid or expired API key")
    if code == 404 and not_found_exc is not None:
        if not_found_arg is not None:
            raise not_found_exc(not_found_arg)
        raise not_found_exc("not found")
    if code == 413:
        raise PayloadTooLargeError(_text(resp) or "Upload exceeds the maximum allowed size")
    raise APIError(code, _text(resp))


def raise_for_dataset(resp: ResponseLike, dataset_id: str) -> None:
    raise_for_generic(resp, DatasetNotFoundError, dataset_id)


def raise_for_deployment(resp: ResponseLike, deployment_id: str) -> None:
    if _ok(resp):
        return
    _check_common(resp)
    code = _status_code(resp)
    if code == 409:
        raise DeploymentStateError(_text(resp))
    if code in (400, 422):
        raise DeploymentValidationError(_text(resp))
    raise_for_generic(resp, DeploymentNotFoundError, deployment_id)


def raise_for_training_job(resp: ResponseLike, job_id: str) -> None:
    raise_for_generic(resp, TrainingJobNotFoundError, job_id)


def raise_for_checkpoint(resp: ResponseLike, checkpoint_id: str) -> None:
    raise_for_generic(resp, CheckpointNotFoundError, checkpoint_id)


def raise_for_hub(resp: ResponseLike, model_id: str | None = None) -> None:
    if _ok(resp):
        return
    _check_common(resp)
    code = _status_code(resp)
    if code == 401:
        raise AuthError("Authentication failed: invalid or expired API key")
    if code == 404:
        if model_id:
            raise HubModelNotFoundError(model_id)
        raise HubError(_text(resp))
    if code in (400, 422):
        raise HubError(_text(resp))
    raise APIError(code, _text(resp))


def raise_for_model(resp: ResponseLike, model_id: str | None = None) -> None:
    """Map a registry model-entry/version/artifact response to its typed error.

    Mirrors ``raise_for_hub`` exactly, with one addition: 409 (a duplicate
    ``slug`` on ``create_model_entry``) joins 400/422 as a ``ModelError`` since
    the registry API uses it for that one conflict case.
    """
    if _ok(resp):
        return
    _check_common(resp)
    code = _status_code(resp)
    if code == 401:
        raise AuthError("Authentication failed: invalid or expired API key")
    if code == 404:
        if model_id:
            raise ModelNotFoundError(model_id)
        raise ModelError(_text(resp))
    if code in (400, 409, 422):
        raise ModelError(_text(resp))
    raise APIError(code, _text(resp))


def raise_for_project(resp: ResponseLike, project_id: str | None = None) -> None:
    if _ok(resp):
        return
    _check_common(resp)
    code = _status_code(resp)
    if code == 401:
        raise AuthError("Authentication failed: invalid or expired API key")
    if code == 404 and project_id:
        raise ProjectNotFoundError(project_id)
    if code == 404:
        raise ArchitectureVersionNotFoundError(_text(resp) or "unknown")
    raise APIError(code, _text(resp))


def raise_for_codegen(resp: ResponseLike) -> None:
    if _ok(resp):
        return
    _check_common(resp)
    code = _status_code(resp)
    if code == 401:
        raise AuthError("Authentication failed: invalid or expired API key")
    if code in (400, 422):
        raise CodegenValidationError(_text(resp))
    if code == 500:
        raise CodegenError(_text(resp))
    raise APIError(code, _text(resp))


def raise_for_upload(resp: ResponseLike) -> None:
    if _ok(resp):
        return
    source = _check_common(resp)
    code = _status_code(resp)
    if code == 401:
        raise AuthError("Authentication failed: invalid or expired API key")
    if code == 413:
        raise PayloadTooLargeError(_text(resp) or "Upload exceeds the maximum allowed size")
    if code in (400, 422):
        if _error_marker(source) in INVALID_URL_CODES:
            raise InvalidURLError(_text(resp) or "The dataset source URL was rejected")
        raise UploadError(_text(resp))
    raise APIError(code, _text(resp))


def raise_for_task(resp: ResponseLike, task_id: str) -> None:
    raise_for_generic(resp, TaskNotFoundError, task_id)
