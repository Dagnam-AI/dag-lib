"""Shared helpers for sync & async clients.

Centralizes URL building, header construction, and HTTP status → exception
mapping so the sync (``requests``) and async (``httpx``) clients cannot drift.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from urllib.parse import quote

from dagnam._core.exceptions import (
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
    HubError,
    HubModelNotFoundError,
    ProjectNotFoundError,
    QuotaExceededError,
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


def response_json_value(resp: JsonResponseLike) -> JsonValue:
    """Decode a response body and validate that it is JSON-compatible."""
    return ensure_json_value(resp.json())


def response_json_object(resp: JsonResponseLike) -> JsonObject:
    """Decode a response body and validate that it is a JSON object."""
    return ensure_json_object(resp.json())


def response_json_array(resp: JsonResponseLike) -> JsonArray:
    """Decode a response body and validate that it is a JSON array."""
    return ensure_json_array(resp.json())


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
    clear QuotaExceededError rather than a generic APIError. (Backend maps a numeric
    plan-limit hit to 402 Payment Required; see backend G088 fix.)
    """
    if _status_code(resp) == 402:
        raise QuotaExceededError(_entitlement_message(resp))


def raise_for_generic(
    resp: ResponseLike,
    not_found_exc: type[DagnamError] | None = None,
    not_found_arg: str | None = None,
) -> None:
    """Map a response to (Auth|*NotFound|Quota|API)Error if not OK."""
    if _ok(resp):
        return
    _check_entitlement(resp)
    code = _status_code(resp)
    if code == 401:
        raise AuthError("Authentication failed: invalid or expired API key")
    if code == 404 and not_found_exc is not None:
        if not_found_arg is not None:
            raise not_found_exc(not_found_arg)
        raise not_found_exc("not found")
    if code == 413:
        raise QuotaExceededError(_text(resp) or "Storage quota exceeded")
    raise APIError(code, _text(resp))


def raise_for_dataset(resp: ResponseLike, dataset_id: str) -> None:
    raise_for_generic(resp, DatasetNotFoundError, dataset_id)


def raise_for_deployment(resp: ResponseLike, deployment_id: str) -> None:
    if _ok(resp):
        return
    _check_entitlement(resp)
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
    _check_entitlement(resp)
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


def raise_for_project(resp: ResponseLike, project_id: str | None = None) -> None:
    if _ok(resp):
        return
    _check_entitlement(resp)
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
    _check_entitlement(resp)
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
    _check_entitlement(resp)
    code = _status_code(resp)
    if code == 401:
        raise AuthError("Authentication failed: invalid or expired API key")
    if code == 413:
        raise QuotaExceededError(_text(resp) or "Storage quota exceeded")
    if code in (400, 422):
        raise UploadError(_text(resp))
    raise APIError(code, _text(resp))


def raise_for_task(resp: ResponseLike, task_id: str) -> None:
    raise_for_generic(resp, TaskNotFoundError, task_id)
