"""Base HTTP client helpers for the Dagnam.AI API."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path, PurePosixPath
import re
import sys

import requests
from tqdm import tqdm

from dagnam._core.client.common import safe_response_text
from dagnam._core.config import get_config_value
from dagnam._core.exceptions import (
    APIError,
    AuthError,
    DatasetNotFoundError,
    DeploymentNotFoundError,
    DownloadTooLargeError,
    TrainingJobNotFoundError,
)
from dagnam._types import JsonArray, JsonObject, JsonValue, ResponseLike, StatusResponseLike

_CHUNK_SIZE = 8192  # 8KB
DEFAULT_TIMEOUT = 30  # seconds (used for both connect and per-read on non-streaming calls)

# A hostile or compromised server (or redirect target) can stream an unbounded
# body and exhaust the client's disk. Every on-disk download funnels through the
# two stream writers below, so a single cap here bounds them all. 100 GiB is far
# above any real dataset/checkpoint; override via the ``max_download_bytes``
# config key.
DEFAULT_MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024 * 1024  # 100 GiB

# For streaming downloads, requests' single-int timeout only applies to the
# initial connect + header phase. Once headers arrive, a stalled body will hang
# forever. We pass a (connect, read) tuple so the per-chunk read timeout fires
# on dead sockets mid-download and the loop can fail fast.
STREAM_CONNECT_TIMEOUT = 30  # seconds
STREAM_READ_TIMEOUT = 60  # seconds — per-chunk read timeout
# SSE streams are long-lived and quiet between events; the server sends a
# heartbeat every ~30s. A read timeout comfortably above that (3x) tolerates a
# missed heartbeat / a slow-to-start stream without a spurious ReadTimeout, while
# still failing a genuinely dead socket so the reconnect loop can recover.
SSE_READ_TIMEOUT = 90  # seconds
ALLOW_REDIRECTS = False


# Query-parameter names whose VALUES are secrets (SSE stream tokens, presigned
# object-storage signatures). requests/urllib3 embed the full request URL in
# their exception text, so a raw ``{exc}`` in an error message would leak the
# token into logs, tracebacks, and ``--debug`` output. Scrub the values before
# they reach any user-visible message.
_SENSITIVE_QUERY_PARAMS = re.compile(
    r"(?i)([?&]?[\w.-]*(?:token|signature|credential|sig|key)[\w.-]*=)[^&\s'\"]+"
)


def scrub_secret_params(text: str) -> str:
    """Mask sensitive query-parameter values embedded in error/exception text."""
    return _SENSITIVE_QUERY_PARAMS.sub(r"\1***", text)


def resolve_max_download_bytes() -> int:
    """Configured download ceiling in bytes; a non-int/<=0 value uses the default."""
    configured = get_config_value("max_download_bytes", DEFAULT_MAX_DOWNLOAD_BYTES)
    if isinstance(configured, bool) or not isinstance(configured, int) or configured <= 0:
        return DEFAULT_MAX_DOWNLOAD_BYTES
    return configured


def _progress_disabled(total_bytes: int | None, *, show_progress: bool) -> bool:
    r"""Whether the tqdm download bar should be suppressed.

    Disabled without a known total (nothing to measure against), when the caller
    opts out, or in a non-TTY (CI logs, notebooks, a pipe) where a
    carriage-return progress bar is just ``\r`` spam.
    """
    return total_bytes is None or not show_progress or not sys.stderr.isatty()


_WINDOWS_RESERVED_FILENAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def is_success_response(response: StatusResponseLike) -> bool:
    code = getattr(response, "status_code", None)
    if isinstance(code, int):
        return 200 <= code < 300
    return bool(getattr(response, "ok", False))


def is_redirect_response(response: ResponseLike) -> bool:
    """True for a 3xx redirect that carries a ``Location`` to follow."""
    code = response.status_code
    if not isinstance(code, int) or not (300 <= code < 400):
        return False
    return bool(response.headers.get("Location"))


def safe_error_body_from_response(response: ResponseLike) -> str:
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
    def _expect_object(value: JsonValue | str | None) -> JsonObject:
        """Narrow a decoded response body to a JSON object or raise.

        Shared by the resource mixins so every ``GET``/``POST`` that promises an
        object body fails loudly (rather than mis-typing) when the backend
        returns a list, scalar, or empty body.
        """
        if isinstance(value, dict):
            return value
        raise TypeError(f"Expected JSON object, got {type(value).__name__}")

    @staticmethod
    def _expect_array(value: JsonValue | str | None) -> JsonArray:
        """Narrow a decoded response body to a JSON array or raise.

        Mirrors ``_expect_object`` for endpoints that return a bare JSON array
        (e.g. the sessions list) rather than an object.
        """
        if isinstance(value, list):
            return value
        raise TypeError(f"Expected JSON array, got {type(value).__name__}")

    @staticmethod
    def _raise_for_status(response: requests.Response, dataset_id: str) -> None:
        """Map HTTP error codes to library exceptions."""
        if is_success_response(response):
            return
        code = response.status_code
        if code == 401:
            raise AuthError("Authentication failed: invalid or expired API key")
        if code == 404:
            raise DatasetNotFoundError(dataset_id)
        raise APIError(code, safe_error_body_from_response(response))

    @staticmethod
    def _stream_response_to_file(
        resp: requests.Response,
        dest: Path,
        *,
        show_progress: bool = True,
    ) -> Path:
        """Write a streaming response body to ``dest`` with a tqdm progress bar.

        Bounded by ``max_download_bytes``: an oversized ``Content-Length`` is
        rejected up-front, and a body that crosses the ceiling mid-stream aborts
        and deletes the partial file so a hostile server cannot fill the disk.
        """
        max_bytes = resolve_max_download_bytes()
        total = resp.headers.get("Content-Length")
        total_bytes = int(total) if total is not None else None
        if total_bytes is not None and total_bytes > max_bytes:
            resp.close()
            raise DownloadTooLargeError(
                0, f"Download of {total_bytes} bytes exceeds max_download_bytes={max_bytes}"
            )

        dest.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        try:
            with (
                closing(resp),
                open(dest, "wb") as fh,
                tqdm(
                    total=total_bytes,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    disable=_progress_disabled(total_bytes, show_progress=show_progress),
                ) as bar,
            ):
                for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                    written += len(chunk)
                    if written > max_bytes:
                        raise DownloadTooLargeError(
                            0, f"Download exceeded max_download_bytes={max_bytes}"
                        )
                    fh.write(chunk)
                    bar.update(len(chunk))
        except DownloadTooLargeError:
            dest.unlink(missing_ok=True)
            raise
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
                timeout=(STREAM_CONNECT_TIMEOUT, STREAM_READ_TIMEOUT),
                stream=True,
                allow_redirects=ALLOW_REDIRECTS,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {scrub_secret_params(str(exc))}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {scrub_secret_params(str(exc))}") from exc

    @staticmethod
    def _get_stream_no_auth(url: str) -> requests.Response:
        """Stream a URL with NO auth header (e.g. a presigned object-storage URL).

        Presigned S3/GCS URLs carry their own signature in the query string and
        reject (or are confused by) a forwarded ``Authorization`` header, so the
        redirect follow-up must not send the API key.
        """
        try:
            return requests.get(
                url,
                timeout=(STREAM_CONNECT_TIMEOUT, STREAM_READ_TIMEOUT),
                stream=True,
                allow_redirects=ALLOW_REDIRECTS,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {scrub_secret_params(str(exc))}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {scrub_secret_params(str(exc))}") from exc

    @staticmethod
    def _append_stream_to_file(
        resp: requests.Response,
        dest: Path,
        *,
        show_progress: bool = True,
    ) -> None:
        """Append streaming response body to an existing file.

        Bounded by ``max_download_bytes`` over the *resumed total* (already-written
        bytes plus the incoming body), so resuming cannot be used to sidestep the
        cap. A breach aborts and deletes the partial file.
        """
        max_bytes = resolve_max_download_bytes()
        total = resp.headers.get("Content-Length")
        total_bytes = int(total) if total is not None else None
        written = dest.stat().st_size if dest.exists() else 0
        if total_bytes is not None and written + total_bytes > max_bytes:
            resp.close()
            dest.unlink(missing_ok=True)
            raise DownloadTooLargeError(
                0, f"Resumed download exceeds max_download_bytes={max_bytes}"
            )

        try:
            with (
                closing(resp),
                open(dest, "ab") as fh,
                tqdm(
                    total=total_bytes,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    disable=_progress_disabled(total_bytes, show_progress=show_progress),
                ) as bar,
            ):
                for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                    written += len(chunk)
                    if written > max_bytes:
                        raise DownloadTooLargeError(
                            0, f"Download exceeded max_download_bytes={max_bytes}"
                        )
                    fh.write(chunk)
                    bar.update(len(chunk))
        except DownloadTooLargeError:
            dest.unlink(missing_ok=True)
            raise

    @staticmethod
    def _raise_for_deployment(response: requests.Response, deployment_id: str) -> None:
        if is_success_response(response):
            return
        code = response.status_code
        if code == 401:
            raise AuthError("Authentication failed: invalid or expired API key")
        if code == 404:
            raise DeploymentNotFoundError(deployment_id)
        raise APIError(code, safe_error_body_from_response(response))

    @staticmethod
    def raise_for_job_response(response: requests.Response, job_id: str) -> None:
        if is_success_response(response):
            return
        code = response.status_code
        if code == 401:
            raise AuthError("Authentication failed: invalid or expired API key")
        if code == 404:
            raise TrainingJobNotFoundError(job_id)
        raise APIError(code, safe_error_body_from_response(response))


def _extract_content_disposition_raw(header: str | None) -> str | None:
    """Extract the raw (unsanitized) filename value from a Content-Disposition header.

    Supports both ``filename="name"`` and ``filename=name`` forms. Returns
    ``None`` when the header is absent or carries no filename parameter.
    """
    if not header:
        return None
    # Try quoted form first: filename="..."
    match = re.search(r'filename="([^"]*)"', header)
    if match:
        return match.group(1)
    # Try unquoted form: filename=...
    match = re.search(r"filename=([^\s;]+)", header)
    if match:
        return match.group(1)
    return None


def parse_content_disposition_filename(header: str | None) -> str:
    """Extract filename from a Content-Disposition header value.

    Supports both ``filename="name"`` and ``filename=name`` forms.
    Returns ``"data"`` when the header is absent or contains no filename.
    Rejects (raises ``ValueError``) any separator, drive letter, ``..``, or
    Windows-reserved name via ``_sanitize_filename`` - the right contract for
    a dataset download, where an unsafe name should abort rather than silently
    land somewhere unexpected.
    """
    raw = _extract_content_disposition_raw(header)
    if raw is None:
        return "data"
    return _sanitize_filename(raw)


def content_disposition_safe_name(header: str | None, *, default: str) -> str:
    """Extract a Content-Disposition filename that is always safe to join under a directory.

    Unlike :func:`parse_content_disposition_filename` (which *rejects* any
    separator/``..``/reserved name by raising - the right contract for
    dataset downloads), this reduces a hostile or malformed filename to its
    bare basename instead of aborting. The basename is stripped of every path
    separator, drive letter, and colon prefix, and reserved device stems fall
    back to ``default``, so the returned name provably joins strictly inside
    ``dest_dir`` on both POSIX and Windows - which is why no ``is_relative_to``
    runtime assertion is needed (it would be an unreachable/uncoverable
    branch given these guarantees).
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
