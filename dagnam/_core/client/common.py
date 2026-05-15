"""Shared helpers for sync & async clients.

Centralizes URL building, header construction, and HTTP status → exception
mapping so the sync (``requests``) and async (``httpx``) clients cannot drift.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
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

API_BASE = "/api/v1"
_MAX_ERROR_BODY = 2048
_TEXT_CONTENT_MARKERS = ("text/", "json", "xml", "javascript", "yaml", "html")


def build_url(api_url: str, path: str) -> str:
    """Join a base API URL with a path that may or may not start with /."""
    return f"{api_url.rstrip('/')}{path if path.startswith('/') else '/' + path}"


def quote_path_segment(value: str) -> str:
    """Percent-encode an untrusted value for use as one URL path segment."""
    return quote(str(value), safe="")


def bearer_headers(api_key: str, *, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return {Authorization: Bearer <key>} plus any extras."""
    h = {"Authorization": f"Bearer {api_key}"}
    if extra:
        h.update(extra)
    return h


def inference_headers(api_key: str) -> dict[str, str]:
    """Bearer + X-API-Key — the inference endpoints accept either."""
    return {"Authorization": f"Bearer {api_key}", "X-API-Key": api_key}


# ---------------------------------------------------------------------------
# Generic status mapper
# ---------------------------------------------------------------------------


class _Resp:
    """Protocol-like shim so sync/async responses can share status mapping."""

    status_code: int
    ok: bool
    text: str


def _text(resp: Any) -> str:
    return safe_response_text(resp)


def safe_response_text(resp: Any) -> str:
    """Return a short, text-safe response body for exception messages."""
    headers = getattr(resp, "headers", {}) or {}
    try:
        content_type = str(headers.get("Content-Type") or headers.get("content-type") or "").lower()
    except AttributeError:
        content_type = ""

    if getattr(resp, "_content", None) is False:
        if content_type:
            return f"<streaming {content_type} body omitted>"
        return "<streaming response body omitted>"

    content = getattr(resp, "content", b"") or b""
    if content_type and not any(marker in content_type for marker in _TEXT_CONTENT_MARKERS):
        try:
            body_len = len(content)
        except TypeError:
            body_len = 0
        return f"<{body_len} bytes of {content_type}>"

    try:
        text = getattr(resp, "text", "")
    except Exception:
        try:
            body_len = len(content)
        except TypeError:
            body_len = 0
        return f"<{body_len} bytes; failed to decode body>"
    if not isinstance(text, str):
        text = str(text)

    if len(text) > _MAX_ERROR_BODY:
        return text[:_MAX_ERROR_BODY] + f"... [truncated, {len(text)} chars total]"
    return text


def _ok(resp: Any) -> bool:
    ok = getattr(resp, "ok", None)
    code = getattr(resp, "status_code", None)
    if isinstance(code, int):
        return 200 <= code < 300
    if ok is not None:
        return bool(ok)
    return False


def raise_for_generic(
    resp: Any, not_found_exc: type[DagnamError] | None = None, not_found_arg: Any = None
) -> None:
    """Map a response to (Auth|*NotFound|API)Error if not OK."""
    if _ok(resp):
        return
    code = int(resp.status_code)
    if code == 401:
        raise AuthError("Authentication failed: invalid or expired API key")
    if code == 404 and not_found_exc is not None:
        if not_found_arg is not None:
            raise not_found_exc(not_found_arg)
        raise not_found_exc("not found")
    if code == 413:
        raise QuotaExceededError(_text(resp) or "Storage quota exceeded")
    raise APIError(code, _text(resp))


def raise_for_dataset(resp: Any, dataset_id: str) -> None:
    raise_for_generic(resp, DatasetNotFoundError, dataset_id)


def raise_for_deployment(resp: Any, deployment_id: str) -> None:
    if _ok(resp):
        return
    code = int(resp.status_code)
    if code == 409:
        raise DeploymentStateError(_text(resp))
    if code in (400, 422):
        raise DeploymentValidationError(_text(resp))
    raise_for_generic(resp, DeploymentNotFoundError, deployment_id)


def raise_for_training_job(resp: Any, job_id: str) -> None:
    raise_for_generic(resp, TrainingJobNotFoundError, job_id)


def raise_for_checkpoint(resp: Any, checkpoint_id: str) -> None:
    raise_for_generic(resp, CheckpointNotFoundError, checkpoint_id)


def raise_for_hub(resp: Any, model_id: str | None = None) -> None:
    if _ok(resp):
        return
    code = int(resp.status_code)
    if code == 401:
        raise AuthError("Authentication failed: invalid or expired API key")
    if code == 404:
        if model_id:
            raise HubModelNotFoundError(model_id)
        raise HubError(_text(resp))
    if code in (400, 422):
        raise HubError(_text(resp))
    raise APIError(code, _text(resp))


def raise_for_project(resp: Any, project_id: str | None = None) -> None:
    if _ok(resp):
        return
    code = int(resp.status_code)
    if code == 401:
        raise AuthError("Authentication failed: invalid or expired API key")
    if code == 404 and project_id:
        raise ProjectNotFoundError(project_id)
    if code == 404:
        raise ArchitectureVersionNotFoundError(_text(resp) or "unknown")
    raise APIError(code, _text(resp))


def raise_for_codegen(resp: Any) -> None:
    if _ok(resp):
        return
    code = int(resp.status_code)
    if code == 401:
        raise AuthError("Authentication failed: invalid or expired API key")
    if code in (400, 422):
        raise CodegenValidationError(_text(resp))
    if code == 500:
        raise CodegenError(_text(resp))
    raise APIError(code, _text(resp))


def raise_for_upload(resp: Any) -> None:
    if _ok(resp):
        return
    code = int(resp.status_code)
    if code == 401:
        raise AuthError("Authentication failed: invalid or expired API key")
    if code == 413:
        raise QuotaExceededError(_text(resp) or "Storage quota exceeded")
    if code in (400, 422):
        raise UploadError(_text(resp))
    raise APIError(code, _text(resp))


def raise_for_task(resp: Any, task_id: str) -> None:
    raise_for_generic(resp, TaskNotFoundError, task_id)
