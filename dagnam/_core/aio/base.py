"""Base async client transport for Dagnam.AI."""

from __future__ import annotations

try:
    import httpx
except ImportError as _exc:
    raise ImportError(
        "dagnam.aio requires httpx. Install with: pip install 'dagnam[aio]'"
    ) from _exc

import re

from dagnam._types import FormData, JsonValue, QueryParams, UploadFiles
from dagnam._core.client.common import bearer_headers, safe_response_text
from dagnam._core.exceptions import APIError, AuthError, TrainingJobNotFoundError

DEFAULT_TIMEOUT = 30
_WINDOWS_RESERVED_FILENAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


class BaseAsyncDagnamClient:
    """Shared httpx transport helpers for the async Dagnam client."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout, follow_redirects=False)

    async def __aenter__(self) -> BaseAsyncDagnamClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        return bearer_headers(self.api_key)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: QueryParams | None = None,
        json: JsonValue = None,
        data: FormData | None = None,
        files: UploadFiles | None = None,
        headers: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> httpx.Response:
        url = f"{self.api_url}{path}"
        try:
            return await self._client.request(
                method,
                url,
                headers=headers or self._headers(),
                params=params,
                json=json,
                data=data,
                files=files,
                timeout=timeout or self.timeout,
            )
        except httpx.ConnectError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc


def raise_for_job_response(resp: httpx.Response, job_id: str) -> None:
    """Map training-job response errors (mirrors sync client.raise_for_job_response)."""
    if 200 <= resp.status_code < 300:
        return
    code = resp.status_code
    if code == 401:
        raise AuthError("Authentication failed: invalid or expired API key")
    if code == 404:
        raise TrainingJobNotFoundError(job_id)
    raise APIError(code, safe_response_text(resp))


def parse_content_disposition_filename(header: str | None) -> str:
    """Extract filename from Content-Disposition header."""
    if not header:
        return "data"
    m = re.search(r'filename="([^"]*)"', header) or re.search(r"filename=([^\s;]+)", header)
    return _sanitize_filename(m.group(1)) if m else "data"


def _sanitize_filename(filename: str) -> str:
    filename = filename.strip()
    normalized = filename.replace("\\", "/")
    windows_stem = normalized.rstrip(" .").split(".", 1)[0].lower()
    if (
        "/" in normalized
        or ":" in normalized
        or normalized in {"", ".", ".."}
        or windows_stem in _WINDOWS_RESERVED_FILENAMES
    ):
        raise ValueError(f"Unsafe filename in Content-Disposition header: {filename!r}")
    return normalized
