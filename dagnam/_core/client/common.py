"""Shared helpers for sync & async clients.

Centralizes URL building, header construction, and HTTP status → exception
mapping so the sync (``requests``) and async (``httpx``) clients cannot drift.
The rejections any route can return are decoded in
:mod:`dagnam._core.client.rejections`; every mapper here runs its
``check_common`` first.
"""

from __future__ import annotations

from collections.abc import Mapping
import io
import json
from urllib.parse import quote, urlparse

import requests

from dagnam._core.client.rejections import (
    MAX_ERROR_BODY as _MAX_ERROR_BODY,
    UNPARSEABLE_BODY,
    check_common,
    error_marker,
    status_code as _status_code,
)
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
    HubRuleError,
    InvalidURLError,
    ModelError,
    ModelNotFoundError,
    PayloadTooLargeError,
    ProjectNotFoundError,
    ResponseError,
    TaskNotFoundError,
    TrainingJobNotFoundError,
    TrainingStateError,
    UploadError,
    VersionKeptError,
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
INVALID_URL_CODES = frozenset({"invalid_url", "url_rejected"})
_TEXT_CONTENT_MARKERS = ("text/", "json", "xml", "javascript", "yaml", "html")


def build_url(api_url: str, path: str) -> str:
    """Join a base API URL with a path that may or may not start with /."""
    return f"{api_url.rstrip('/')}{path if path.startswith('/') else '/' + path}"


def quote_path_segment(value: str) -> str:
    """Percent-encode an untrusted value for use as one URL path segment."""
    return quote(str(value), safe="")


def same_origin(api_url: str, url: str) -> bool:
    """True when ``url`` has the API's scheme and host: the only origin the API key may go to."""
    base, other = urlparse(api_url), urlparse(url)
    return (base.scheme.lower(), base.netloc.lower()) == (
        other.scheme.lower(),
        other.netloc.lower(),
    )


def _response_status(resp: JsonResponseLike) -> int:
    code = getattr(resp, "status_code", 0)
    return code if isinstance(code, int) else 0


def response_json_value(resp: JsonResponseLike) -> JsonValue:
    """Decode a response body and validate that it is JSON-compatible."""
    try:
        return ensure_json_value(resp.json())
    except UNPARSEABLE_BODY as exc:
        raise ResponseError(_response_status(resp), f"malformed response body: {exc}") from exc


def response_json_object(resp: JsonResponseLike) -> JsonObject:
    """Decode a response body and validate that it is a JSON object."""
    try:
        return ensure_json_object(resp.json())
    except UNPARSEABLE_BODY as exc:
        raise ResponseError(_response_status(resp), f"malformed response body: {exc}") from exc


def response_json_array(resp: JsonResponseLike) -> JsonArray:
    """Decode a response body and validate that it is a JSON array."""
    try:
        return ensure_json_array(resp.json())
    except UNPARSEABLE_BODY as exc:
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


def short_error_text(text: str) -> str:
    """``text`` cut to ``_MAX_ERROR_BODY`` characters, with a note of how long it was."""
    if len(text) > _MAX_ERROR_BODY:
        return text[:_MAX_ERROR_BODY] + f"... [truncated, {len(text)} chars total]"
    return text


MAX_ERROR_READ = 64 * 1024
"""Most bytes read from a streamed error body: the mapper needs a marker and a message, not a page."""
ERROR_READ_CHUNK = 8192


def read_error_prefix(resp: requests.Response) -> requests.Response:
    """A copy of a streamed error response holding at most ``MAX_ERROR_READ`` bytes of its body.

    A hostile server or a proxy error page can stream a body without end, so ``resp.content``
    must not be what drains it. ``resp`` is closed; use the copy.
    """
    body = bytearray()
    for chunk in resp.iter_content(chunk_size=ERROR_READ_CHUNK):
        body += chunk
        if len(body) >= MAX_ERROR_READ:
            break
    capped = requests.Response()
    capped.status_code = resp.status_code
    capped.headers = resp.headers
    capped.encoding = resp.encoding
    capped.url = resp.url
    capped.raw = io.BytesIO(bytes(body[:MAX_ERROR_READ]))
    _ = capped.content  # marks the copy as read: the mappers skip an unread stream
    resp.close()
    return capped


def _format_fastapi_detail(text: str) -> str:
    try:
        payload = json.loads(text)
    except UNPARSEABLE_BODY:
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
                return short_error_text(_format_fastapi_detail(text_value))
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
    return short_error_text(_format_fastapi_detail(text_value))


def _ok(resp: ResponseLike) -> bool:
    status_code = resp.status_code
    if isinstance(status_code, int):
        return 200 <= status_code < 300
    return bool(getattr(resp, "ok", False))


def raise_for_generic(
    resp: ResponseLike,
    not_found_exc: type[DagnamError] | None = None,
    not_found_arg: str | None = None,
) -> None:
    """Map a response to (Auth|*NotFound|Quota|API)Error if not OK."""
    if _ok(resp):
        return
    check_common(resp)
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
    source = check_common(resp)
    code = _status_code(resp)
    if code == 409:
        raise DeploymentStateError(_text(resp), reason=error_marker(source))
    if code in (400, 422):
        raise DeploymentValidationError(_text(resp))
    raise_for_generic(resp, DeploymentNotFoundError, deployment_id)


def raise_for_training_job(resp: ResponseLike, job_id: str) -> None:
    raise_for_generic(resp, TrainingJobNotFoundError, job_id)


def raise_for_training_state(resp: ResponseLike, job_id: str) -> None:
    """Like :func:`raise_for_training_job`, but a 409 is a :class:`TrainingStateError`.

    Its ``reason`` is the platform's ``error`` marker when the body carries one.
    """
    if _status_code(resp) == 409:
        source = check_common(resp)
        raise TrainingStateError(_text(resp), reason=error_marker(source))
    raise_for_training_job(resp, job_id)


def raise_for_checkpoint(resp: ResponseLike, checkpoint_id: str) -> None:
    raise_for_generic(resp, CheckpointNotFoundError, checkpoint_id)


def raise_for_hub(resp: ResponseLike, model_id: str | None = None) -> None:
    if _ok(resp):
        return
    source = check_common(resp)
    code = _status_code(resp)
    if code == 401:
        raise AuthError("Authentication failed: invalid or expired API key")
    if code == 404:
        if model_id:
            raise HubModelNotFoundError(model_id)
        raise HubError(_text(resp))
    if code == 413:
        raise PayloadTooLargeError(_text(resp) or "Upload exceeds the maximum allowed size")
    if code in (400, 422):
        raise HubError(_text(resp))
    if code == 409 and (rule := error_marker(source)) is not None:
        raise HubRuleError(409, _text(resp), code=rule)
    raise APIError(code, _text(resp))


def raise_for_model(resp: ResponseLike, model_id: str | None = None) -> None:
    """Map a registry model-entry/version/artifact response to its typed error.

    Mirrors ``raise_for_hub`` exactly, with one addition: 409 (a duplicate
    ``slug`` on ``create_model_entry``) joins 400/422 as a ``ModelError`` since
    the registry API uses it for that one conflict case.
    """
    if _ok(resp):
        return
    check_common(resp)
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


def raise_for_purge(resp: JsonResponseLike, model_id: str | None = None) -> None:
    """``raise_for_model``, except a 409 whose ``detail`` says ``status: kept`` keeps its body.

    The platform's refusal of a version a live endpoint serves is a decision, not a failure: the
    parsed ``detail`` rides on :class:`VersionKeptError` instead of being flattened to its message.
    """
    if _status_code(resp) == 409:
        try:
            body = resp.json()
        except UNPARSEABLE_BODY:
            body = None
        detail = body.get("detail") if isinstance(body, dict) else None
        if isinstance(detail, dict) and detail.get("status") == "kept":
            raise VersionKeptError(detail)
    raise_for_model(resp, model_id)


def raise_for_project(resp: ResponseLike, project_id: str | None = None) -> None:
    if _ok(resp):
        return
    check_common(resp)
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
    check_common(resp)
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
    source = check_common(resp)
    code = _status_code(resp)
    if code == 401:
        raise AuthError("Authentication failed: invalid or expired API key")
    if code == 413:
        raise PayloadTooLargeError(_text(resp) or "Upload exceeds the maximum allowed size")
    if code in (400, 422):
        if error_marker(source) in INVALID_URL_CODES:
            raise InvalidURLError(_text(resp) or "The dataset source URL was rejected")
        raise UploadError(_text(resp))
    raise APIError(code, _text(resp))


def raise_for_task(resp: ResponseLike, task_id: str) -> None:
    raise_for_generic(resp, TaskNotFoundError, task_id)
