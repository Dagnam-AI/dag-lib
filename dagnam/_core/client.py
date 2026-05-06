"""HTTP client for the Dagnam.AI API."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from dagnam._core.exceptions import (
    APIError,
    AuthError,
    CheckpointNotFoundError,
    DatasetNotFoundError,
    DeploymentNotFoundError,
    TrainingJobNotFoundError,
)

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
            return requests.get(
                url, headers=self._headers(), timeout=_TIMEOUT, stream=True
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

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

    # ------------------------------------------------------------------
    # User dataset endpoints
    # ------------------------------------------------------------------

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


    # ------------------------------------------------------------------
    # Inference endpoints (Phase 3)
    # ------------------------------------------------------------------

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

    def predict(self, deployment_id: str, inputs: dict, timeout: int = _TIMEOUT) -> dict:
        """POST /api/v1/inference/{deployment_id}/predict"""
        url = f"{self.api_url}/api/v1/inference/{deployment_id}/predict"
        headers = {**self._headers(), "X-API-Key": self.api_key}
        try:
            resp = requests.post(url, headers=headers, json=inputs, timeout=timeout)
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc
        self._raise_for_deployment(resp, deployment_id)
        return resp.json()

    def predict_batch(
        self, deployment_id: str, inputs: list, timeout: int = _TIMEOUT
    ) -> list:
        """POST /api/v1/inference/{deployment_id}/predict/batch"""
        url = f"{self.api_url}/api/v1/inference/{deployment_id}/predict/batch"
        headers = {**self._headers(), "X-API-Key": self.api_key}
        try:
            resp = requests.post(
                url, headers=headers, json={"inputs": inputs}, timeout=timeout
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc
        self._raise_for_deployment(resp, deployment_id)
        return resp.json()

    def deployment_health(self, deployment_id: str) -> dict:
        """GET /api/v1/inference/{deployment_id}/health"""
        url = f"{self.api_url}/api/v1/inference/{deployment_id}/health"
        headers = {**self._headers(), "X-API-Key": self.api_key}
        try:
            resp = requests.get(url, headers=headers, timeout=_TIMEOUT)
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc
        self._raise_for_deployment(resp, deployment_id)
        return resp.json()

    # ------------------------------------------------------------------
    # Training checkpoint endpoints (Phase 3)
    # ------------------------------------------------------------------

    def list_checkpoints(self, job_id: str) -> list[dict]:
        """GET /api/v1/training/jobs/{job_id}/checkpoints"""
        url = f"{self.api_url}/api/v1/training/jobs/{job_id}/checkpoints"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=_TIMEOUT)
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc
        self._raise_for_job(resp, job_id)
        return resp.json()

    def download_checkpoint_stream(
        self, job_id: str, checkpoint_id: str, dest_path: Path
    ) -> tuple[Path, str | None]:
        """Stream-download a checkpoint file to dest_path.

        GET /api/v1/training/jobs/{job_id}/checkpoints/{checkpoint_id}/download

        Returns (dest_path, expected_sha256) — the caller must verify.
        """
        url = (
            f"{self.api_url}/api/v1/training/jobs/{job_id}"
            f"/checkpoints/{checkpoint_id}/download"
        )
        resp = self._get_stream(url)

        if not resp.ok:
            code = resp.status_code
            if code == 401:
                raise AuthError("Authentication failed: invalid or expired API key")
            if code == 404:
                raise CheckpointNotFoundError(checkpoint_id)
            raise APIError(code, resp.text)

        expected_checksum = resp.headers.get("X-Checksum-SHA256")
        written = self._stream_response_to_file(resp, Path(dest_path))
        return written, expected_checksum

    # ------------------------------------------------------------------
    # Deployment management endpoints (Phase 4)
    # ------------------------------------------------------------------

    def _deployment_request(
        self,
        method: str,
        path: str,
        *,
        deployment_id: str | None = None,
        params: dict | None = None,
        json_body: Any = None,
        timeout: int = _TIMEOUT,
    ) -> dict | list | None:
        """Issue an authenticated request against a deployment route.

        Maps transport errors to ``APIError(0, …)``, translates status
        codes through :func:`_common.raise_for_deployment`, and decodes
        JSON on success.  Returns ``None`` for empty bodies (e.g. 204).
        """
        from dagnam._core._common import raise_for_deployment

        url = f"{self.api_url}{path}"
        try:
            resp = requests.request(
                method,
                url,
                headers=self._headers(),
                params=params,
                json=json_body,
                timeout=timeout,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        raise_for_deployment(resp, deployment_id or "deployment")

        if not resp.content:
            return None
        try:
            return resp.json()
        except ValueError:
            return resp.text

    def list_deployments(
        self,
        *,
        page: int = 1,
        limit: int = 20,
        status_filter: str | None = None,
        platform: str | None = None,
        project_id: str | None = None,
        search: str | None = None,
    ) -> dict:
        """GET /api/v1/deployments"""
        params: dict[str, Any] = {"page": page, "limit": limit}
        if status_filter is not None:
            params["status"] = status_filter
        if platform is not None:
            params["platform"] = platform
        if project_id is not None:
            params["project_id"] = project_id
        if search is not None:
            params["search"] = search
        return self._deployment_request("GET", "/api/v1/deployments", params=params)

    def get_deployment(self, deployment_id: str) -> dict:
        """GET /api/v1/deployments/{id}"""
        return self._deployment_request(
            "GET",
            f"/api/v1/deployments/{deployment_id}",
            deployment_id=deployment_id,
        )

    def create_deployment(self, payload: dict) -> dict:
        """POST /api/v1/deployments"""
        return self._deployment_request(
            "POST", "/api/v1/deployments", json_body=payload
        )

    def update_deployment(self, deployment_id: str, payload: dict) -> dict:
        """PUT /api/v1/deployments/{id}"""
        return self._deployment_request(
            "PUT",
            f"/api/v1/deployments/{deployment_id}",
            deployment_id=deployment_id,
            json_body=payload,
        )

    def delete_deployment(self, deployment_id: str) -> dict | None:
        """DELETE /api/v1/deployments/{id}"""
        return self._deployment_request(
            "DELETE",
            f"/api/v1/deployments/{deployment_id}",
            deployment_id=deployment_id,
        )

    def pause_deployment(self, deployment_id: str) -> dict:
        return self._deployment_request(
            "POST",
            f"/api/v1/deployments/{deployment_id}/pause",
            deployment_id=deployment_id,
        )

    def resume_deployment(self, deployment_id: str) -> dict:
        return self._deployment_request(
            "POST",
            f"/api/v1/deployments/{deployment_id}/resume",
            deployment_id=deployment_id,
        )

    def scale_deployment(self, deployment_id: str, num_instances: int) -> dict:
        return self._deployment_request(
            "PUT",
            f"/api/v1/deployments/{deployment_id}/scale",
            deployment_id=deployment_id,
            params={"num_instances": num_instances},
        )

    def rollback_deployment(self, deployment_id: str, checkpoint_path: str) -> dict:
        return self._deployment_request(
            "POST",
            f"/api/v1/deployments/{deployment_id}/rollback",
            deployment_id=deployment_id,
            params={"checkpoint_path": checkpoint_path},
        )

    def get_deployment_metrics(self, deployment_id: str, time_range: str = "24h") -> dict:
        return self._deployment_request(
            "GET",
            f"/api/v1/deployments/{deployment_id}/metrics",
            deployment_id=deployment_id,
            params={"time_range": time_range},
        )

    def get_deployment_logs(
        self,
        deployment_id: str,
        *,
        level: str | None = None,
        search: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        page: int = 1,
        limit: int = 100,
    ) -> dict:
        params: dict[str, Any] = {"page": page, "limit": limit}
        if level is not None:
            params["level"] = level
        if search is not None:
            params["search"] = search
        if start_time is not None:
            params["start_time"] = start_time
        if end_time is not None:
            params["end_time"] = end_time
        return self._deployment_request(
            "GET",
            f"/api/v1/deployments/{deployment_id}/logs",
            deployment_id=deployment_id,
            params=params,
        )

    def get_deployment_health_full(self, deployment_id: str) -> dict:
        """GET /api/v1/deployments/{id}/health — platform-side health row.

        Distinct from :meth:`deployment_health` which hits the *inference*
        endpoint.  This returns the deployment's own health_status column.
        """
        return self._deployment_request(
            "GET",
            f"/api/v1/deployments/{deployment_id}/health",
            deployment_id=deployment_id,
        )

    def open_deployment_stream(
        self, deployment_id: str, last_event_id: str | None = None
    ) -> requests.Response:
        """Open an SSE stream for a deployment (``?api_key=`` auth).

        GET /api/v1/deployments/{id}/stream?api_key=...
        """
        from dagnam._core._common import raise_for_deployment

        url = f"{self.api_url}/api/v1/deployments/{deployment_id}/stream"
        params = {"api_key": self.api_key}
        headers = {"Accept": "text/event-stream"}
        if last_event_id:
            headers["Last-Event-ID"] = last_event_id
        try:
            resp = requests.get(
                url, params=params, headers=headers, stream=True, timeout=_TIMEOUT
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        raise_for_deployment(resp, deployment_id)
        return resp


    # ------------------------------------------------------------------
    # Hub endpoints
    # ------------------------------------------------------------------

    def _hub_request(
        self,
        method: str,
        path: str,
        *,
        model_id: str | None = None,
        params: dict | None = None,
        json_body: Any = None,
        timeout: int = _TIMEOUT,
    ) -> dict | list | None:
        from dagnam._core._common import raise_for_hub

        url = f"{self.api_url}{path}"
        try:
            resp = requests.request(
                method, url, headers=self._headers(),
                params=params, json=json_body, timeout=timeout,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        raise_for_hub(resp, model_id)

        if not resp.content:
            return None
        try:
            return resp.json()
        except ValueError:
            return resp.text

    def list_hub_models(self, **filter_params) -> dict:
        return self._hub_request("GET", "/api/v1/hub/models", params=filter_params)

    def get_hub_model(self, model_id: str) -> dict:
        return self._hub_request("GET", f"/api/v1/hub/models/{model_id}", model_id=model_id)

    def create_hub_model(self, payload: dict) -> dict:
        return self._hub_request("POST", "/api/v1/hub/models", json_body=payload)

    def update_hub_model(self, model_id: str, payload: dict) -> dict:
        return self._hub_request("PUT", f"/api/v1/hub/models/{model_id}", model_id=model_id, json_body=payload)

    def delete_hub_model(self, model_id: str) -> None:
        self._hub_request("DELETE", f"/api/v1/hub/models/{model_id}", model_id=model_id)

    def list_hub_model_files(self, model_id: str) -> dict:
        return self._hub_request("GET", f"/api/v1/hub/models/{model_id}/files", model_id=model_id)

    def download_hub_model(self, model_id: str, file_id: str | None = None) -> dict:
        path = f"/api/v1/hub/models/{model_id}/download"
        params = {"file_id": file_id} if file_id else None
        return self._hub_request("GET", path, model_id=model_id, params=params)

    def list_hub_model_versions(self, model_id: str) -> list:
        return self._hub_request("GET", f"/api/v1/hub/models/{model_id}/versions", model_id=model_id)

    def create_hub_model_version(self, model_id: str, payload: dict) -> dict:
        return self._hub_request("POST", f"/api/v1/hub/models/{model_id}/versions", model_id=model_id, json_body=payload)

    def star_hub_model(self, model_id: str) -> dict:
        return self._hub_request("POST", f"/api/v1/hub/models/{model_id}/star", model_id=model_id)

    def unstar_hub_model(self, model_id: str) -> dict:
        return self._hub_request("DELETE", f"/api/v1/hub/models/{model_id}/star", model_id=model_id)

    def fork_hub_model(self, model_id: str) -> dict:
        return self._hub_request("POST", f"/api/v1/hub/models/{model_id}/fork", model_id=model_id)

    def list_hub_model_reviews(self, model_id: str, page: int = 1, limit: int = 20) -> dict:
        return self._hub_request("GET", f"/api/v1/hub/models/{model_id}/reviews", model_id=model_id, params={"page": page, "limit": limit})

    def add_hub_model_review(self, model_id: str, payload: dict) -> dict:
        return self._hub_request("POST", f"/api/v1/hub/models/{model_id}/reviews", model_id=model_id, json_body=payload)

    def use_hub_model_in_studio(self, model_id: str) -> dict:
        return self._hub_request("POST", f"/api/v1/hub/models/{model_id}/use-in-studio", model_id=model_id)

    def list_hub_categories(self) -> list:
        return self._hub_request("GET", "/api/v1/hub/categories")

    def get_hub_featured(self) -> list:
        return self._hub_request("GET", "/api/v1/hub/featured")

    def get_hub_trending(self, days: int = 7) -> list:
        return self._hub_request("GET", "/api/v1/hub/trending", params={"days": days})

    def list_hub_starred(self, sort_by: str = "date_starred", page: int = 1, limit: int = 20) -> dict:
        return self._hub_request("GET", "/api/v1/hub/starred", params={"sort_by": sort_by, "page": page, "limit": limit})

    # ------------------------------------------------------------------
    # Project endpoints
    # ------------------------------------------------------------------

    def _project_request(
        self,
        method: str,
        path: str,
        *,
        project_id: str | None = None,
        params: dict | None = None,
        json_body: Any = None,
        data: Any = None,
        files: Any = None,
        timeout: int = _TIMEOUT,
    ) -> dict | list | None:
        from dagnam._core._common import raise_for_project

        url = f"{self.api_url}{path}"
        try:
            resp = requests.request(
                method, url, headers=self._headers(),
                params=params, json=json_body, data=data, files=files,
                timeout=timeout,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        raise_for_project(resp, project_id)

        if not resp.content:
            return None
        try:
            return resp.json()
        except ValueError:
            return resp.text

    def list_projects(self, **filter_params) -> dict:
        return self._project_request("GET", "/api/v1/projects", params=filter_params)

    def get_project(self, project_id: str) -> dict:
        return self._project_request("GET", f"/api/v1/projects/{project_id}", project_id=project_id)

    def create_project(self, payload: dict) -> dict:
        return self._project_request("POST", "/api/v1/projects", json_body=payload)

    def update_project(self, project_id: str, payload: dict) -> dict:
        return self._project_request("PUT", f"/api/v1/projects/{project_id}", project_id=project_id, json_body=payload)

    def delete_project(self, project_id: str) -> None:
        self._project_request("DELETE", f"/api/v1/projects/{project_id}", project_id=project_id)

    def duplicate_project(self, project_id: str, title: str | None = None) -> dict:
        body = {"title": title} if title else None
        return self._project_request("POST", f"/api/v1/projects/{project_id}/duplicate", project_id=project_id, json_body=body)

    def save_architecture(self, project_id: str, payload: dict) -> dict:
        return self._project_request("POST", f"/api/v1/projects/{project_id}/architecture", project_id=project_id, json_body=payload)

    def import_dag(self, payload: dict) -> dict:
        return self._project_request("POST", "/api/v1/projects/import", json_body=payload)

    def import_dag_existing(self, project_id: str, payload: dict) -> dict:
        return self._project_request("POST", f"/api/v1/projects/{project_id}/import", project_id=project_id, json_body=payload)

    def bulk_delete_projects(self, project_ids: list[str]) -> dict:
        return self._project_request("POST", "/api/v1/projects/bulk-delete", json_body={"project_ids": project_ids})

    def link_dataset(self, project_id: str, dataset_id: str, role: str) -> dict:
        return self._project_request("POST", f"/api/v1/projects/{project_id}/datasets", project_id=project_id, json_body={"dataset_id": dataset_id, "role": role})

    def get_project_datasets(self, project_id: str) -> dict:
        return self._project_request("GET", f"/api/v1/projects/{project_id}/datasets", project_id=project_id)

    def unlink_dataset(self, project_id: str, dataset_id: str) -> None:
        self._project_request("DELETE", f"/api/v1/projects/{project_id}/datasets/{dataset_id}", project_id=project_id)

    # ------------------------------------------------------------------
    # Codegen endpoints
    # ------------------------------------------------------------------

    def _codegen_request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json_body: Any = None,
        timeout: int = _TIMEOUT,
    ) -> dict | list | None:
        from dagnam._core._common import raise_for_codegen

        url = f"{self.api_url}{path}"
        try:
            resp = requests.request(
                method, url, headers=self._headers(),
                params=params, json=json_body, timeout=timeout,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        raise_for_codegen(resp)

        if not resp.content:
            return None
        try:
            return resp.json()
        except ValueError:
            return resp.text

    def generate_code(self, project_id: str, payload: dict, async_mode: bool = False) -> dict:
        params = {"async": "true"} if async_mode else None
        return self._codegen_request("POST", f"/api/v1/codegen/{project_id}/generate", json_body=payload, params=params)

    def preview_code(self, project_id: str, framework: str, version_id: str | None = None) -> dict:
        params: dict[str, str] = {"framework": framework}
        if version_id:
            params["version_id"] = version_id
        return self._codegen_request("GET", f"/api/v1/codegen/{project_id}/preview", params=params)

    def validate_architecture(self, project_id: str, version_id: str | None = None) -> dict:
        params = {"version_id": version_id} if version_id else None
        return self._codegen_request("POST", f"/api/v1/codegen/{project_id}/validate", params=params)

    def download_code_zip(
        self,
        project_id: str,
        framework: str,
        version_id: str | None = None,
        dest_path: Path | str | None = None,
    ) -> Path | bytes:
        from dagnam._core._common import raise_for_codegen

        params: dict[str, str] = {"framework": framework}
        if version_id:
            params["version_id"] = version_id
        url = f"{self.api_url}/api/v1/codegen/{project_id}/download"
        try:
            resp = requests.get(
                url, headers=self._headers(), params=params,
                stream=bool(dest_path), timeout=_TIMEOUT,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        raise_for_codegen(resp)

        if dest_path:
            return self._stream_response_to_file(resp, Path(dest_path))
        return resp.content

    def get_code_status(self, project_id: str, task_id: str) -> dict:
        return self._codegen_request("GET", f"/api/v1/codegen/{project_id}/status/{task_id}")

    # ------------------------------------------------------------------
    # Dataset upload endpoints
    # ------------------------------------------------------------------

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
        from dagnam._core._common import raise_for_upload

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
                    url, headers=self._headers(),
                    data=fields, files=files, timeout=None,
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
        from dagnam._core._common import raise_for_upload

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
                endpoint, headers=self._headers(), json=body, timeout=_TIMEOUT,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        raise_for_upload(resp)
        return resp.json()

    def get_dataset_task_status(self, task_id: str) -> dict:
        from dagnam._core._common import raise_for_task

        url = f"{self.api_url}/api/v1/datasets/tasks/{task_id}"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=_TIMEOUT)
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        raise_for_task(resp, task_id)
        return resp.json()


    # ------------------------------------------------------------------
    # Training SSE stream (Phase 3)
    # ------------------------------------------------------------------

    def open_training_stream(
        self, job_id: str, last_event_id: str | None = None
    ) -> requests.Response:
        """Open an SSE stream for a training job.

        GET /api/v1/streaming/training-jobs/{job_id}/stream?api_key=...

        Returns the raw streaming Response; the caller is responsible for
        wrapping it (e.g. via sseclient-py) and closing it.
        """
        url = (
            f"{self.api_url}/api/v1/streaming/training-jobs/{job_id}/stream"
        )
        params = {"api_key": self.api_key}
        headers = {"Accept": "text/event-stream"}
        if last_event_id:
            headers["Last-Event-ID"] = last_event_id
        try:
            resp = requests.get(
                url, params=params, headers=headers, stream=True, timeout=_TIMEOUT
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        if not resp.ok:
            code = resp.status_code
            if code == 401:
                raise AuthError("Authentication failed: invalid or expired API key")
            if code == 404:
                raise TrainingJobNotFoundError(job_id)
            raise APIError(code, resp.text)
        return resp


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
