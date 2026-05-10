"""Base HTTP client helpers for the Dagnam.AI API."""

from __future__ import annotations

from pathlib import Path
import re

import requests
from tqdm import tqdm

from dagnam._core.exceptions import (
    APIError,
    AuthError,
    DatasetNotFoundError,
    DeploymentNotFoundError,
    TrainingJobNotFoundError,
)

_CHUNK_SIZE = 8192  # 8KB
_TIMEOUT = 30  # seconds


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
        if response.ok:
            return
        code = response.status_code
        if code == 401:
            raise AuthError("Authentication failed: invalid or expired API key")
        if code == 404:
            raise DatasetNotFoundError(dataset_id)
        raise APIError(code, response.text)

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
        """Issue a streaming GET with the standard auth header + timeout."""
        try:
            return requests.get(url, headers=self._headers(), timeout=_TIMEOUT, stream=True)
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
        if response.ok:
            return
        code = response.status_code
        if code == 401:
            raise AuthError("Authentication failed: invalid or expired API key")
        if code == 404:
            raise DeploymentNotFoundError(deployment_id)
        raise APIError(code, response.text)

    @staticmethod
    def _raise_for_job(response: requests.Response, job_id: str) -> None:
        if response.ok:
            return
        code = response.status_code
        if code == 401:
            raise AuthError("Authentication failed: invalid or expired API key")
        if code == 404:
            raise TrainingJobNotFoundError(job_id)
        raise APIError(code, response.text)


def _parse_filename(header: str | None) -> str:
    """Extract filename from a Content-Disposition header value.

    Supports both ``filename="name"`` and ``filename=name`` forms.
    Returns ``"data"`` when the header is absent or contains no filename.
    """
    if not header:
        return "data"
    # Try quoted form first: filename="..."
    match = re.search(r'filename="([^"]+)"', header)
    if match:
        return match.group(1)
    # Try unquoted form: filename=...
    match = re.search(r"filename=([^\s;]+)", header)
    if match:
        return match.group(1)
    return "data"
