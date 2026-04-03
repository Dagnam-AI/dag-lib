"""HTTP client for the Dagnam.AI API."""

from __future__ import annotations

import re
from pathlib import Path

import requests
from tqdm import tqdm

from dagnam.exceptions import APIError, AuthError, DatasetNotFoundError

_CHUNK_SIZE = 8192  # 8KB
_TIMEOUT = 30  # seconds


class DagnamClient:
    """Thin wrapper around the Dagnam.AI REST API."""

    def __init__(self, api_url: str, api_key: str) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_datasets(self, type: str = "all", search: str | None = None) -> list[dict]:
        """GET /api/v1/datasets/browse — List available datasets."""
        url = f"{self.api_url}/api/v1/datasets/browse"
        params: dict[str, str] = {"type": type}
        if search:
            params["search"] = search
        try:
            resp = requests.get(url, headers=self._headers(), params=params, timeout=_TIMEOUT)
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc
        self._raise_for_status(resp, "browse")
        return resp.json()

    def get_dataset_meta(self, dataset_id: str) -> dict:
        """Fetch dataset metadata from the API.

        GET /api/v1/datasets/{dataset_id}/meta
        """
        url = f"{self.api_url}/api/v1/datasets/{dataset_id}/meta"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=_TIMEOUT)
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        self._raise_for_status(resp, dataset_id)
        return resp.json()

    # ------------------------------------------------------------------
    # System dataset endpoints
    # ------------------------------------------------------------------

    def list_system_datasets(self) -> list[dict]:
        """GET /api/v1/datasets/system — List all system datasets."""
        url = f"{self.api_url}/api/v1/datasets/system"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=_TIMEOUT)
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc
        self._raise_for_status(resp, "system")
        return resp.json()

    def get_system_dataset_meta(self, dataset_id: str) -> dict:
        """GET /api/v1/datasets/system/{dataset_id} — Get system dataset metadata."""
        url = f"{self.api_url}/api/v1/datasets/system/{dataset_id}"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=_TIMEOUT)
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc
        self._raise_for_status(resp, dataset_id)
        return resp.json()

    def download_system_dataset(self, dataset_id: str, output_dir: Path) -> Path:
        """Stream-download a system dataset file with a tqdm progress bar.

        GET /api/v1/datasets/system/{dataset_id}/download

        Returns the path to the downloaded file.
        """
        url = f"{self.api_url}/api/v1/datasets/system/{dataset_id}/download"
        try:
            resp = requests.get(
                url,
                headers=self._headers(),
                timeout=_TIMEOUT,
                stream=True,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        self._raise_for_status(resp, dataset_id)

        filename = _parse_filename(resp.headers.get("Content-Disposition"))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dest = output_dir / filename

        total = resp.headers.get("Content-Length")
        total_bytes = int(total) if total is not None else None

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

    # ------------------------------------------------------------------
    # User dataset endpoints
    # ------------------------------------------------------------------

    def download_dataset(self, dataset_id: str, output_dir: Path) -> Path:
        """Stream-download a dataset file with a tqdm progress bar.

        GET /api/v1/datasets/{dataset_id}/download

        Returns the path to the downloaded file.
        """
        url = f"{self.api_url}/api/v1/datasets/{dataset_id}/download"
        try:
            resp = requests.get(
                url,
                headers=self._headers(),
                timeout=_TIMEOUT,
                stream=True,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        self._raise_for_status(resp, dataset_id)

        # Resolve filename from Content-Disposition, fallback to "data"
        filename = _parse_filename(resp.headers.get("Content-Disposition"))

        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dest = output_dir / filename

        total = resp.headers.get("Content-Length")
        total_bytes = int(total) if total is not None else None

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
