"""Base async client transport for Dagnam.AI."""

from __future__ import annotations

try:
    import httpx
except ImportError as _exc:
    raise ImportError(
        "dagnam.aio requires httpx. Install with: pip install 'dagnam[aio]'"
    ) from _exc

import asyncio
import logging
from pathlib import Path, PurePosixPath
import random
import re
from typing import Awaitable, Callable
import uuid

from dagnam._core._retry import RetryBudget, run_with_retry_async
from dagnam._core.client.base import resolve_max_download_bytes
from dagnam._core.client.common import bearer_headers, safe_response_text
from dagnam._core.exceptions import (
    APIError,
    AuthError,
    DownloadTooLargeError,
    TrainingJobNotFoundError,
)
from dagnam._types import FormData, JsonValue, QueryParams, UploadFiles

DEFAULT_TIMEOUT = 30
# SSE streams are quiet between events (~30s server heartbeat); a read timeout
# comfortably above that avoids spurious ReadTimeouts that would churn the
# reconnect loop and re-mint a stream token. Mirrors the sync SSE_READ_TIMEOUT.
SSE_READ_TIMEOUT = 90
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
        self._retry_budget = RetryBudget()
        self._async_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
        self._rng: Callable[[], float] = random.random

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
        raise_for: Callable[[httpx.Response], None] | None = None,
        idempotent: bool = False,
        idempotency_key: str | None = None,
    ) -> httpx.Response:
        """Issue an async request, retrying transient failures when ``raise_for`` is set.

        With ``raise_for=None`` (default) behavior is unchanged: no status raise
        and no retry — existing async call sites stay byte-compatible until each
        mixin opts in. When ``raise_for`` is supplied, transient 429/5xx (and
        status-0 transport failures) retry per the shared policy; the mapper
        still runs on the final response so domain 404s keep their exact types.
        """
        url = f"{self.api_url}{path}"
        req_headers = dict(headers or self._headers())
        if idempotent and idempotency_key is None:
            idempotency_key = str(uuid.uuid4())
        if idempotency_key is not None:
            req_headers["Idempotency-Key"] = idempotency_key
        method_upper = method.upper()
        retryable = raise_for is not None and (
            method_upper in {"GET", "HEAD", "PUT", "DELETE"} or bool(idempotency_key)
        )

        async def _attempt() -> httpx.Response:
            try:
                resp = await self._client.request(
                    method_upper,
                    url,
                    headers=req_headers,
                    params=params,
                    json=json,
                    data=data,
                    files=files,
                    timeout=timeout or self.timeout,
                )
            except httpx.TransportError as exc:
                # httpx.TransportError is the umbrella base covering
                # ConnectError/TimeoutException *and* ReadError, WriteError,
                # CloseError, ProtocolError, ProxyError — the sync client's
                # blanket requests.RequestException catch has no narrower
                # equivalent, so match that breadth here. Preserve the existing
                # timeout-specific message where it applies.
                if isinstance(exc, httpx.TimeoutException):
                    raise APIError(0, f"Request timed out: {exc}") from exc
                raise APIError(0, f"Connection failed: {exc}") from exc
            if raise_for is not None:
                try:
                    raise_for(resp)
                except APIError as exc:
                    exc.retry_after_header = resp.headers.get("Retry-After")
                    raise
            return resp

        return await run_with_retry_async(
            _attempt,
            retryable=retryable,
            budget=self._retry_budget,
            sleep=self._async_sleep,
            rng=self._rng,
            logger=logging.getLogger("dagnam.http"),
            label=f"{method_upper} {url}",
            idempotency_key=idempotency_key,
        )

    async def _stream_response_to_file(self, resp: httpx.Response, dest: Path) -> None:
        """Stream an open httpx response body to ``dest``, bounded by the cap.

        ``resp`` must come from an open ``self._client.stream(...)`` context so
        the body is never buffered in memory (a large checkpoint would OOM). An
        oversized ``Content-Length`` is rejected up-front and a body that crosses
        ``max_download_bytes`` mid-stream aborts and deletes the partial file.
        """
        max_bytes = resolve_max_download_bytes()
        total = resp.headers.get("content-length")
        total_bytes = int(total) if total is not None else None
        if total_bytes is not None and total_bytes > max_bytes:
            raise DownloadTooLargeError(
                0, f"Download of {total_bytes} bytes exceeds max_download_bytes={max_bytes}"
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        try:
            with open(dest, "wb") as fh:
                async for chunk in resp.aiter_bytes():
                    written += len(chunk)
                    if written > max_bytes:
                        raise DownloadTooLargeError(
                            0, f"Download exceeded max_download_bytes={max_bytes}"
                        )
                    fh.write(chunk)
        except DownloadTooLargeError:
            await asyncio.to_thread(dest.unlink, missing_ok=True)
            raise


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
