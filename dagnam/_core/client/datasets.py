"""Synchronous datasets client methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dagnam._core.client.base import _TIMEOUT, APIError, _parse_filename, requests


class DatasetsClientMixin:
    """Datasets resource methods for DagnamClient."""

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

    def get_dataset_meta(self, dataset_id: str, version: str | None = None) -> dict:
        """Fetch dataset metadata from the API.

        GET /api/v1/datasets/{dataset_id}/meta[?version=...]
        """
        url = f"{self.api_url}/api/v1/datasets/{dataset_id}/meta"
        params = {"version": version} if version else None
        try:
            resp = requests.get(url, headers=self._headers(), params=params, timeout=_TIMEOUT)
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        self._raise_for_status(resp, dataset_id)
        return resp.json()

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

    def get_system_dataset_meta(self, dataset_id: str, version: str | None = None) -> dict:
        """GET /api/v1/datasets/system/{dataset_id} — Get system dataset metadata."""
        url = f"{self.api_url}/api/v1/datasets/system/{dataset_id}"
        params = {"version": version} if version else None
        try:
            resp = requests.get(url, headers=self._headers(), params=params, timeout=_TIMEOUT)
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
        resp = self._get_stream(url)
        self._raise_for_status(resp, dataset_id)
        filename = _parse_filename(resp.headers.get("Content-Disposition"))
        return self._stream_response_to_file(resp, Path(output_dir) / filename)

    def download_dataset(
        self,
        dataset_id: str,
        output_dir: Path,
        *,
        download_url: str | None = None,
        filename: str | None = None,
        version: str | None = None,
        resume: bool = True,
    ) -> Path:
        """Stream-download a dataset file with optional resume support.

        GET /api/v1/datasets/{dataset_id}/download

        When *download_url* is provided (e.g. a presigned URL), the request
        is made without authentication headers.

        When *resume* is True and a ``.part`` file exists, sends a
        ``Range: bytes={offset}-`` header.  If the server responds with
        ``206 Partial Content``, appends to the partial file.  If the
        server responds with ``200 OK``, discards the partial and restarts.

        Args:
            dataset_id: Dataset identifier.
            output_dir: Directory to save the downloaded file.
            download_url: Optional presigned URL (skips auth).
            filename: Optional filename override. If not provided, parsed
                from Content-Disposition header.
            version: Optional dataset version for authenticated downloads.
            resume: Whether to attempt resuming partial downloads.

        Returns:
            Path to the downloaded file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine URL and headers
        if download_url:
            url = download_url
            headers: dict[str, str] = {}
            params = None
        else:
            url = f"{self.api_url}/api/v1/datasets/{dataset_id}/download"
            headers = self._headers()
            params = {"version": version} if version else None

        # Check for partial download to resume
        part_path: Path | None = output_dir / f"{filename}.part" if filename else None
        if filename and resume:
            if part_path.exists() and part_path.stat().st_size > 0:
                offset = part_path.stat().st_size
                headers["Range"] = f"bytes={offset}-"
        elif not resume and filename:
            # Clean up any existing .part file
            if part_path and part_path.exists():
                part_path.unlink()
            part_path = None

        # Make the request
        try:
            resp = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=_TIMEOUT,
                stream=True,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        self._raise_for_status(resp, dataset_id)

        # Determine filename from response if not provided
        if not filename:
            filename = _parse_filename(resp.headers.get("Content-Disposition"))
            part_path = output_dir / f"{filename}.part"

        dest = output_dir / filename
        part_path = part_path or output_dir / f"{filename}.part"

        # Handle resume logic
        if "Range" in headers and resp.status_code == 206:
            # Append to existing partial file
            self._append_stream_to_file(resp, part_path)
            # Rename .part to final filename
            part_path.replace(dest)
        elif "Range" in headers and resp.status_code == 200:
            # Server doesn't support Range — restart full download
            part_path.unlink(missing_ok=True)
            self._stream_response_to_file(resp, part_path)
            part_path.replace(dest)
        else:
            # Normal full download
            if part_path.exists():
                part_path.unlink()
            self._stream_response_to_file(resp, part_path)
            part_path.replace(dest)

        return dest

    def upload_dataset(
        self,
        file_path: str | Path,
        name: str,
        dataset_type: str,
        format: str,
        description: str | None = None,
        visibility: str = "private",
        license: str | None = None,
        progress_cb: Any = None,
    ) -> dict:
        from dagnam._core.client.common import raise_for_upload

        url = f"{self.api_url}/api/v1/datasets/upload"
        fields = {
            "name": name,
            "type": dataset_type,
            "format": format,
            "visibility": visibility,
        }
        if description:
            fields["description"] = description
        if license:
            fields["license"] = license

        file_path = Path(file_path)
        try:
            with open(file_path, "rb") as fh:
                files = {"file": (file_path.name, fh)}
                resp = requests.post(
                    url,
                    headers=self._headers(),
                    data=fields,
                    files=files,
                    timeout=None,
                )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        raise_for_upload(resp)
        return resp.json()

    def upload_dataset_from_url(
        self,
        url: str,
        name: str,
        dataset_type: str,
        format: str,
        description: str | None = None,
        visibility: str = "private",
    ) -> dict:
        from dagnam._core.client.common import raise_for_upload

        endpoint = f"{self.api_url}/api/v1/datasets/upload-url"
        body: dict[str, str] = {
            "url": url,
            "name": name,
            "type": dataset_type,
            "format": format,
            "visibility": visibility,
        }
        if description:
            body["description"] = description
        try:
            resp = requests.post(
                endpoint,
                headers=self._headers(),
                json=body,
                timeout=_TIMEOUT,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        raise_for_upload(resp)
        return resp.json()

    def get_dataset_task_status(self, task_id: str) -> dict:
        from dagnam._core.client.common import raise_for_task

        url = f"{self.api_url}/api/v1/datasets/tasks/{task_id}"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=_TIMEOUT)
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        raise_for_task(resp, task_id)
        return resp.json()
