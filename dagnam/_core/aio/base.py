"""Base async client transport for Dagnam.AI."""

from __future__ import annotations

try:
    import httpx
except ImportError as _exc:
    raise ImportError(
        "dagnam.aio requires httpx. Install with: pip install 'dagnam[aio]'"
    ) from _exc

from pathlib import PurePosixPath
import re

from dagnam._core.client.common import bearer_headers, safe_response_text
from dagnam._core.exceptions import APIError, AuthError, TrainingJobNotFoundError
from dagnam._types import FormData, JsonValue, QueryParams, UploadFiles

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

    async def _get_no_auth(self, url: str) -> httpx.Response:
        """GET an absolute URL with NO auth header (e.g. a presigned URL).

        Presigned S3/GCS URLs carry their own signature in the query string and
        reject (or are confused by) a forwarded ``Authorization`` header, so the
        redirect follow-up must not send the API key.
        """
        try:
            return await self._client.get(url, timeout=self.timeout)
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


def _extract_content_disposition_raw(header: str | None) -> str | None:
    """Extract the raw (unsanitized) filename value from a Content-Disposition header.

    Supports both ``filename="name"`` and ``filename=name`` forms. Returns
    ``None`` when the header is absent or carries no filename parameter.
    """
    if not header:
        return None
    m = re.search(r'filename="([^"]*)"', header) or re.search(r"filename=([^\s;]+)", header)
    return m.group(1) if m else None


def parse_content_disposition_filename(header: str | None) -> str:
    """Extract filename from Content-Disposition header.

    Rejects (raises ``ValueError``) any separator, drive letter, ``..``, or
    Windows-reserved name via ``_sanitize_filename`` - see the sync mirror in
    ``dagnam._core.client.base`` for the rationale.
    """
    raw = _extract_content_disposition_raw(header)
    return _sanitize_filename(raw) if raw is not None else "data"


def content_disposition_safe_name(header: str | None, *, default: str) -> str:
    """Extract a Content-Disposition filename that is always safe to join under a directory.

    Mirrors ``dagnam._core.client.base.content_disposition_safe_name``: reduces
    a hostile or malformed filename to its bare basename instead of rejecting
    it. The basename is stripped of every path separator, drive letter, and
    colon prefix, and reserved device stems fall back to ``default``, so the
    returned name provably joins strictly inside ``dest_dir`` on both POSIX
    and Windows - which is why no ``is_relative_to`` runtime assertion is
    needed (it would be an unreachable/uncoverable branch given these
    guarantees).
    """
    raw = _extract_content_disposition_raw(header)
    if raw is None:
        return default
    # Reduce to a bare basename with NO path separator, drive letter, or NTFS
    # alternate-data-stream prefix: PurePosixPath(...).name strips "/" and "\\"
    # components, then rsplit(":", 1)[-1] drops any leading "<drive>:" / "name:stream"
    # prefix (colon is a path-defining char on Windows, a supported platform).
    # The result therefore contains no "/", "\\", or ":" and always joins
    # strictly inside dest_dir on POSIX and Windows alike. A Windows reserved
    # device stem (CON, NUL, COM1, ...) is rejected to "default" so the write
    # can never be redirected to a console/device instead of a file in dest_dir.
    candidate = PurePosixPath(raw.replace("\\", "/")).name.rsplit(":", 1)[-1]
    reserved_stem = candidate.rstrip(" .").split(".", 1)[0].lower()
    if candidate in {"", ".", ".."} or reserved_stem in _WINDOWS_RESERVED_FILENAMES:
        return default
    return candidate


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
