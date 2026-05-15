"""Base HTTP client helpers for the Dagnam.AI API."""

from __future__ import annotations

from pathlib import Path
import re

import requests
from tqdm import tqdm

from dagnam._core.client.common import safe_response_text
from dagnam._core.exceptions import (
    APIError,
    AuthError,
    DatasetNotFoundError,
    DeploymentNotFoundError,
    TrainingJobNotFoundError,
)

_CHUNK_SIZE = 8192  # 8KB
_TIMEOUT = 30  # seconds (used for both connect and per-read on non-streaming calls)

# For streaming downloads, requests' single-int timeout only applies to the
# initial connect + header phase. Once headers arrive, a stalled body will hang
# forever. We pass a (connect, read) tuple so the per-chunk read timeout fires
# on dead sockets mid-download and the loop can fail fast.
_STREAM_CONNECT_TIMEOUT = 30  # seconds
_STREAM_READ_TIMEOUT = 60  # seconds — per-chunk read timeout
_ALLOW_REDIRECTS = False
_WINDOWS_RESERVED_FILENAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def _is_success_response(response: requests.Response) -> bool:
    code = getattr(response, "status_code", None)
    if isinstance(code, int):
        return 200 <= code < 300
    return bool(getattr(response, "ok", False))


def _safe_error_body(response: requests.Response) -> str:
    """Extract a short, log-safe error body from a failed HTTP response."""
    return safe_response_text(response)


class BaseDagnamClient:
    """Shared transport helpers for the synchronous Dagnam client."""

    def __init__(self, api_url: str, api_key: str) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _raise_for_status(response: requests.Response, dataset_id: str) -> None:
        """Map HTTP error codes to library exceptions."""
        if _is_success_response(response):
            return
        code = response.status_code
        if code == 401:
            raise AuthError("Authentication failed: invalid or expired API key")
        if code == 404:
            raise DatasetNotFoundError(dataset_id)
        raise APIError(code, _safe_error_body(response))

    @staticmethod
    def _stream_response_to_file(resp: requests.Response, dest: Path) -> Path:
        """Write a streaming response body to ``dest`` with a tqdm progress bar."""
        total = resp.headers.get("Content-Length")
        total_bytes = int(total) if total is not None else None

        dest.parent.mkdir(parents=True, exist_ok=True)
        with (
            open(dest, "wb") as fh,
            tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                disable=total_bytes is None,
            ) as bar,
        ):
            for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                fh.write(chunk)
                bar.update(len(chunk))
        return dest

    def _get_stream(self, url: str) -> requests.Response:
        """Issue a streaming GET with the standard auth header + timeouts.

        Uses a ``(connect, read)`` timeout tuple so that streaming downloads
        which stall mid-body — e.g. when a proxy silently drops the
        connection — fail fast on the next chunk read instead of hanging
        forever.
        """
        try:
            return requests.get(
                url,
                headers=self._headers(),
                timeout=(_STREAM_CONNECT_TIMEOUT, _STREAM_READ_TIMEOUT),
                stream=True,
                allow_redirects=_ALLOW_REDIRECTS,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

    @staticmethod
    def _append_stream_to_file(resp: requests.Response, dest: Path) -> None:
        """Append streaming response body to an existing file."""
        total = resp.headers.get("Content-Length")
        total_bytes = int(total) if total is not None else None

        with (
            open(dest, "ab") as fh,
            tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                disable=total_bytes is None,
            ) as bar,
        ):
            for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                fh.write(chunk)
                bar.update(len(chunk))

    @staticmethod
    def _raise_for_deployment(response: requests.Response, deployment_id: str) -> None:
        if _is_success_response(response):
            return
        code = response.status_code
        if code == 401:
            raise AuthError("Authentication failed: invalid or expired API key")
        if code == 404:
            raise DeploymentNotFoundError(deployment_id)
        raise APIError(code, _safe_error_body(response))

    @staticmethod
    def _raise_for_job(response: requests.Response, job_id: str) -> None:
        if _is_success_response(response):
            return
        code = response.status_code
        if code == 401:
            raise AuthError("Authentication failed: invalid or expired API key")
        if code == 404:
            raise TrainingJobNotFoundError(job_id)
        raise APIError(code, _safe_error_body(response))


def _parse_filename(header: str | None) -> str:
    """Extract filename from a Content-Disposition header value.

    Supports both ``filename="name"`` and ``filename=name`` forms.
    Returns ``"data"`` when the header is absent or contains no filename.
    """
    if not header:
        return "data"
    # Try quoted form first: filename="..."
    match = re.search(r'filename="([^"]*)"', header)
    if match:
        return _sanitize_filename(match.group(1))
    # Try unquoted form: filename=...
    match = re.search(r"filename=([^\s;]+)", header)
    if match:
        return _sanitize_filename(match.group(1))
    return "data"


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
