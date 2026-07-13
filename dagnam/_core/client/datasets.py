"""Synchronous datasets client methods."""

from __future__ import annotations

from pathlib import Path
import re

from dagnam._core.client.base import (
    ALLOW_REDIRECTS,
    DEFAULT_TIMEOUT,
    STREAM_CONNECT_TIMEOUT,
    STREAM_READ_TIMEOUT,
    APIError,
    BaseDagnamClient,
    parse_content_disposition_filename,
    requests,
    safe_download_basename,
)
from dagnam._core.client.common import (
    quote_path_segment,
    raise_for_task,
    raise_for_upload,
    response_json_array,
    response_json_object,
)
from dagnam._types import JsonObject, ensure_json_object


class DatasetsClientMixin(BaseDagnamClient):
    """Datasets resource methods for DagnamClient."""

    def list_datasets(self, type: str = "all", search: str | None = None) -> list[JsonObject]:
        """GET /api/v1/datasets/browse — List available datasets."""
        url = f"{self.api_url}/api/v1/datasets/browse"
        params: dict[str, str] = {"type": type}
        if search:
            params["search"] = search
        resp = self._request(
            "GET",
            url,
            raise_for=lambda r: self._raise_for_status(r, "browse"),
            params=params,
            allow_redirects=ALLOW_REDIRECTS,
        )
        return [item for item in response_json_array(resp) if isinstance(item, dict)]

    def get_dataset_meta(self, dataset_id: str, version: str | None = None) -> JsonObject:
        """Fetch dataset metadata from the API.

        GET /api/v1/datasets/{dataset_id}/meta[?version=...]
        """
        dataset_path = quote_path_segment(dataset_id)
        url = f"{self.api_url}/api/v1/datasets/{dataset_path}/meta"
        params = {"version": version} if version else None
        resp = self._request(
            "GET",
            url,
            raise_for=lambda r: self._raise_for_status(r, dataset_id),
            params=params,
            allow_redirects=ALLOW_REDIRECTS,
        )
        return response_json_object(resp)

    def list_system_datasets(self) -> list[JsonObject]:
        """GET /api/v1/datasets/system — List all system datasets."""
        url = f"{self.api_url}/api/v1/datasets/system"
        resp = self._request(
            "GET",
            url,
            raise_for=lambda r: self._raise_for_status(r, "system"),
            allow_redirects=ALLOW_REDIRECTS,
        )
        return [item for item in response_json_array(resp) if isinstance(item, dict)]

    def get_system_dataset_meta(self, dataset_id: str, version: str | None = None) -> JsonObject:
        """GET /api/v1/datasets/system/{dataset_id} — Get system dataset metadata."""
        dataset_path = quote_path_segment(dataset_id)
        url = f"{self.api_url}/api/v1/datasets/system/{dataset_path}"
        params = {"version": version} if version else None
        resp = self._request(
            "GET",
            url,
            raise_for=lambda r: self._raise_for_status(r, dataset_id),
            params=params,
            allow_redirects=ALLOW_REDIRECTS,
        )
        return response_json_object(resp)

    def download_system_dataset(
        self,
        dataset_id: str,
        output_dir: Path,
        *,
        show_progress: bool = True,
    ) -> Path:
        """Stream-download a system dataset file with a tqdm progress bar.

        GET /api/v1/datasets/system/{dataset_id}/download

        Returns the path to the downloaded file.
        """
        dataset_path = quote_path_segment(dataset_id)
        url = f"{self.api_url}/api/v1/datasets/system/{dataset_path}/download"
        resp = self._get_stream(url)
        self._raise_for_status(resp, dataset_id)
        filename = parse_content_disposition_filename(resp.headers.get("Content-Disposition"))
        return self._stream_response_to_file(
            resp,
            Path(output_dir) / filename,
            show_progress=show_progress,
        )

    def download_dataset(
        self,
        dataset_id: str,
        output_dir: Path,
        *,
        download_url: str | None = None,
        filename: str | None = None,
        version: str | None = None,
        resume: bool = True,
        show_progress: bool = True,
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

        # The explicit filename is often server-controlled (dataset
        # metadata["filename"]). Reduce it to a bare basename so an absolute path
        # or ``..`` traversal cannot escape output_dir and overwrite an arbitrary
        # file (e.g. ~/.bashrc) — a compromised-server arbitrary-write / RCE. The
        # Content-Disposition fallback below is sanitized separately.
        if filename is not None:
            filename = safe_download_basename(filename, default="dataset")

        # Determine URL and headers
        if download_url:
            url = download_url
            headers: dict[str, str] = {}
            params = None
        else:
            dataset_path = quote_path_segment(dataset_id)
            url = f"{self.api_url}/api/v1/datasets/{dataset_path}/download"
            headers = self._headers()
            params = {"version": version} if version else None

        # Check for partial download to resume
        part_path: Path | None = output_dir / f"{filename}.part" if filename else None
        if filename and resume and part_path is not None:
            if part_path.exists() and part_path.stat().st_size > 0:
                offset = part_path.stat().st_size
                headers["Range"] = f"bytes={offset}-"
        elif not resume and filename:
            # Clean up object existing .part file
            if part_path and part_path.exists():
                part_path.unlink()
            part_path = None

        # Make the request
        try:
            resp = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=DEFAULT_TIMEOUT,
                stream=True,
                allow_redirects=bool(download_url),
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        self._raise_for_status(resp, dataset_id)

        # Determine filename from response if not provided
        if not filename:
            filename = parse_content_disposition_filename(resp.headers.get("Content-Disposition"))
            part_path = output_dir / f"{filename}.part"

        dest = output_dir / filename
        part_path = part_path or output_dir / f"{filename}.part"

        # Handle resume logic
        if "Range" in headers and resp.status_code == 206:
            # Validate the server honoured the exact Range we asked for before
            # appending — a 206 whose Content-Range starts at a different offset
            # (server ignored/miscounted the range) would silently corrupt the
            # file. On any mismatch or a missing/garbled header, discard the
            # partial and restart a clean full download.
            requested_start = int(headers["Range"].split("=", 1)[1].split("-", 1)[0])
            content_range = resp.headers.get("Content-Range", "")
            match = re.match(r"bytes\s+(\d+)-", content_range)
            if match is None or int(match.group(1)) != requested_start:
                part_path.unlink(missing_ok=True)
                resp.close()
                return self.download_dataset(
                    dataset_id,
                    output_dir,
                    download_url=download_url,
                    filename=filename,
                    version=version,
                    resume=False,
                    show_progress=show_progress,
                )
            # Append to existing partial file
            self._append_stream_to_file(resp, part_path, show_progress=show_progress)
            # Rename .part to final filename
            part_path.replace(dest)
        elif "Range" in headers and resp.status_code == 200:
            # Server doesn't support Range — restart full download
            part_path.unlink(missing_ok=True)
            self._stream_response_to_file(resp, part_path, show_progress=show_progress)
            part_path.replace(dest)
        else:
            # Normal full download
            if part_path.exists():
                part_path.unlink()
            self._stream_response_to_file(resp, part_path, show_progress=show_progress)
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
        progress_cb: object = None,
    ) -> JsonObject:
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
                    # A (connect, read) tuple rather than None: the per-socket
                    # read timeout resets on each write, so a progressing large
                    # upload is unaffected while a dead socket fails fast instead
                    # of hanging forever.
                    timeout=(STREAM_CONNECT_TIMEOUT, STREAM_READ_TIMEOUT),
                    allow_redirects=ALLOW_REDIRECTS,
                )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        raise_for_upload(resp)
        return response_json_object(resp)

    def upload_dataset_from_url(
        self,
        url: str,
        name: str,
        dataset_type: str,
        format: str,
        description: str | None = None,
        visibility: str = "private",
    ) -> JsonObject:
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
        resp = self._request(
            "POST",
            endpoint,
            raise_for=lambda r: raise_for_upload(r),
            json=body,
            allow_redirects=ALLOW_REDIRECTS,
        )
        return response_json_object(resp)

    def get_dataset_task_status(self, task_id: str) -> JsonObject:
        task_path = quote_path_segment(task_id)
        url = f"{self.api_url}/api/v1/datasets/tasks/{task_path}"
        resp = self._request(
            "GET",
            url,
            raise_for=lambda r: raise_for_task(r, task_id),
            allow_redirects=ALLOW_REDIRECTS,
        )
        return response_json_object(resp)

    def preview_dataset(self, dataset_id: str, rows: int = 10) -> JsonObject:
        """Preview a dataset's samples and statistics.

        ``GET /api/v1/datasets/{dataset_id}/preview?rows=N``. Works for both
        system and user datasets. Returns the raw ``{"samples": [...],
        "statistics": {...}}`` object; image datasets carry base64 sample data.
        """
        dataset_path = quote_path_segment(dataset_id)
        url = f"{self.api_url}/api/v1/datasets/{dataset_path}/preview"
        resp = self._request(
            "GET",
            url,
            raise_for=lambda r: self._raise_for_status(r, dataset_id),
            params={"rows": rows},
            allow_redirects=ALLOW_REDIRECTS,
        )
        return response_json_object(resp)

    def update_dataset(
        self,
        dataset_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        visibility: str | None = None,
    ) -> JsonObject:
        """Update a dataset's mutable fields. ``PUT /api/v1/datasets/{dataset_id}``.

        Sends only the provided fields as multipart form data (``name``,
        ``description``, ``visibility``). At least one field is required; a call
        with all three omitted raises ``ValueError`` before any network request.
        """
        if name is None and description is None and visibility is None:
            raise ValueError(
                "update_dataset requires at least one of name, description, or visibility"
            )
        fields: dict[str, str] = {}
        if name is not None:
            fields["name"] = name
        if description is not None:
            fields["description"] = description
        if visibility is not None:
            fields["visibility"] = visibility

        dataset_path = quote_path_segment(dataset_id)
        url = f"{self.api_url}/api/v1/datasets/{dataset_path}"
        resp = self._request(
            "PUT",
            url,
            raise_for=lambda r: self._raise_for_status(r, dataset_id),
            data=fields,
            allow_redirects=ALLOW_REDIRECTS,
        )
        return response_json_object(resp)

    def delete_dataset(self, dataset_id: str) -> None:
        """Delete a dataset. ``DELETE /api/v1/datasets/{dataset_id}`` (204 No Content)."""
        dataset_path = quote_path_segment(dataset_id)
        url = f"{self.api_url}/api/v1/datasets/{dataset_path}"
        self._request(
            "DELETE",
            url,
            raise_for=lambda r: self._raise_for_status(r, dataset_id),
            allow_redirects=ALLOW_REDIRECTS,
        )

    def update_dataset_roles(
        self,
        dataset_id: str,
        column_roles: dict[str, str],
        task_type_hint: str | None = None,
    ) -> JsonObject:
        """Set a dataset's column roles. ``PATCH /api/v1/datasets/{dataset_id}/roles``.

        Sends a JSON body of ``column_roles`` (required) plus an optional
        ``task_type_hint``. Returns the confirmed roles object.
        """
        body: JsonObject = {
            "column_roles": ensure_json_object(column_roles),
            "task_type_hint": task_type_hint,
        }
        dataset_path = quote_path_segment(dataset_id)
        url = f"{self.api_url}/api/v1/datasets/{dataset_path}/roles"
        resp = self._request(
            "PATCH",
            url,
            raise_for=lambda r: self._raise_for_status(r, dataset_id),
            json=body,
            allow_redirects=ALLOW_REDIRECTS,
        )
        return response_json_object(resp)
