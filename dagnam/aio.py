"""Async client for Dagnam.AI — requires ``pip install 'dagnam[aio]'``.

Mirrors the sync :class:`~dagnam.client.DagnamClient` surface on
``httpx.AsyncClient``.

Usage::

    import dagnam.aio as da

    async with da.AsyncDagnamClient(api_url, api_key) as client:
        models = await client.list_hub_models(search="resnet")
"""

from __future__ import annotations

try:
    import httpx
except ImportError as _exc:
    raise ImportError(
        "dagnam.aio requires httpx. Install with: pip install 'dagnam[aio]'"
    ) from _exc

from pathlib import Path
from typing import Any

from dagnam._core._common import (
    bearer_headers,
    inference_headers,
    raise_for_codegen,
    raise_for_dataset,
    raise_for_deployment,
    raise_for_hub,
    raise_for_project,
    raise_for_task,
    raise_for_upload,
)
from dagnam._core.exceptions import (
    APIError,
    AuthError,
    CheckpointNotFoundError,
    TrainingJobNotFoundError,
)

__all__ = ["AsyncDagnamClient"]

_TIMEOUT = 30


class AsyncDagnamClient:
    """Async wrapper around the Dagnam.AI REST API using ``httpx``."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        timeout: int = _TIMEOUT,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def __aenter__(self) -> AsyncDagnamClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return bearer_headers(self.api_key)

    def _inference_headers(self) -> dict[str, str]:
        return inference_headers(self.api_key)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json: Any = None,
        data: dict | None = None,
        files: Any = None,
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

    # Domain-level request helpers ----------------------------------------

    async def _deployment_req(
        self,
        method: str,
        path: str,
        *,
        deployment_id: str | None = None,
        params: dict | None = None,
        json_body: Any = None,
        timeout: int | None = None,
    ) -> dict | list | None:
        resp = await self._request(method, path, params=params, json=json_body, timeout=timeout)
        raise_for_deployment(resp, deployment_id or "deployment")
        if not resp.content:
            return None
        try:
            return resp.json()
        except ValueError:
            return resp.text

    async def _hub_req(
        self,
        method: str,
        path: str,
        *,
        model_id: str | None = None,
        params: dict | None = None,
        json_body: Any = None,
    ) -> dict | list | None:
        resp = await self._request(method, path, params=params, json=json_body)
        raise_for_hub(resp, model_id)
        if not resp.content:
            return None
        try:
            return resp.json()
        except ValueError:
            return resp.text

    async def _project_req(
        self,
        method: str,
        path: str,
        *,
        project_id: str | None = None,
        params: dict | None = None,
        json_body: Any = None,
        data: dict | None = None,
        files: Any = None,
    ) -> dict | list | None:
        resp = await self._request(method, path, params=params, json=json_body, data=data, files=files)
        raise_for_project(resp, project_id)
        if not resp.content:
            return None
        try:
            return resp.json()
        except ValueError:
            return resp.text

    async def _codegen_req(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json_body: Any = None,
    ) -> dict | list | None:
        resp = await self._request(method, path, params=params, json=json_body)
        raise_for_codegen(resp)
        if not resp.content:
            return None
        try:
            return resp.json()
        except ValueError:
            return resp.text

    # ------------------------------------------------------------------
    # Dataset endpoints
    # ------------------------------------------------------------------

    async def list_datasets(self, type: str = "all", search: str | None = None) -> list[dict]:
        params: dict[str, str] = {"type": type}
        if search:
            params["search"] = search
        resp = await self._request("GET", "/api/v1/datasets/browse", params=params)
        raise_for_dataset(resp, "browse")
        return resp.json()

    async def get_dataset_meta(self, dataset_id: str) -> dict:
        resp = await self._request("GET", f"/api/v1/datasets/{dataset_id}/meta")
        raise_for_dataset(resp, dataset_id)
        return resp.json()

    async def list_system_datasets(self) -> list[dict]:
        resp = await self._request("GET", "/api/v1/datasets/system")
        raise_for_dataset(resp, "system")
        return resp.json()

    async def get_system_dataset_meta(self, dataset_id: str) -> dict:
        resp = await self._request("GET", f"/api/v1/datasets/system/{dataset_id}")
        raise_for_dataset(resp, dataset_id)
        return resp.json()

    async def download_dataset(self, dataset_id: str, output_dir: Path) -> Path:
        resp = await self._request("GET", f"/api/v1/datasets/{dataset_id}/download")
        raise_for_dataset(resp, dataset_id)
        filename = _parse_cd(resp.headers.get("content-disposition"))
        dest = Path(output_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return dest

    async def download_system_dataset(self, dataset_id: str, output_dir: Path) -> Path:
        resp = await self._request("GET", f"/api/v1/datasets/system/{dataset_id}/download")
        raise_for_dataset(resp, dataset_id)
        filename = _parse_cd(resp.headers.get("content-disposition"))
        dest = Path(output_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return dest

    # ------------------------------------------------------------------
    # Inference endpoints
    # ------------------------------------------------------------------

    async def predict(self, deployment_id: str, inputs: dict, timeout: int | None = None) -> dict:
        resp = await self._request(
            "POST",
            f"/api/v1/inference/{deployment_id}/predict",
            json=inputs,
            headers=self._inference_headers(),
            timeout=timeout,
        )
        raise_for_deployment(resp, deployment_id)
        return resp.json()

    async def predict_batch(self, deployment_id: str, inputs: list, timeout: int | None = None) -> list:
        resp = await self._request(
            "POST",
            f"/api/v1/inference/{deployment_id}/predict/batch",
            json={"inputs": inputs},
            headers=self._inference_headers(),
            timeout=timeout,
        )
        raise_for_deployment(resp, deployment_id)
        return resp.json()

    async def deployment_health(self, deployment_id: str) -> dict:
        resp = await self._request(
            "GET",
            f"/api/v1/inference/{deployment_id}/health",
            headers=self._inference_headers(),
        )
        raise_for_deployment(resp, deployment_id)
        return resp.json()

    # ------------------------------------------------------------------
    # Training checkpoint endpoints
    # ------------------------------------------------------------------

    async def list_checkpoints(self, job_id: str) -> list[dict]:
        resp = await self._request("GET", f"/api/v1/training/jobs/{job_id}/checkpoints")
        _raise_for_job(resp, job_id)
        return resp.json()

    async def download_checkpoint(
        self, job_id: str, checkpoint_id: str, dest_path: Path
    ) -> tuple[Path, str | None]:
        resp = await self._request(
            "GET",
            f"/api/v1/training/jobs/{job_id}/checkpoints/{checkpoint_id}/download",
        )
        if not resp.is_success:
            code = resp.status_code
            if code == 401:
                raise AuthError("Authentication failed: invalid or expired API key")
            if code == 404:
                raise CheckpointNotFoundError(checkpoint_id)
            raise APIError(code, resp.text)
        expected_checksum = resp.headers.get("x-checksum-sha256")
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return dest, expected_checksum

    # ------------------------------------------------------------------
    # Deployment management endpoints
    # ------------------------------------------------------------------

    async def list_deployments(
        self,
        *,
        page: int = 1,
        limit: int = 20,
        status_filter: str | None = None,
        platform: str | None = None,
        project_id: str | None = None,
        search: str | None = None,
    ) -> dict:
        params: dict[str, Any] = {"page": page, "limit": limit}
        if status_filter is not None:
            params["status"] = status_filter
        if platform is not None:
            params["platform"] = platform
        if project_id is not None:
            params["project_id"] = project_id
        if search is not None:
            params["search"] = search
        return await self._deployment_req("GET", "/api/v1/deployments", params=params)

    async def get_deployment(self, deployment_id: str) -> dict:
        return await self._deployment_req("GET", f"/api/v1/deployments/{deployment_id}", deployment_id=deployment_id)

    async def create_deployment(self, payload: dict) -> dict:
        return await self._deployment_req("POST", "/api/v1/deployments", json_body=payload)

    async def update_deployment(self, deployment_id: str, payload: dict) -> dict:
        return await self._deployment_req("PUT", f"/api/v1/deployments/{deployment_id}", deployment_id=deployment_id, json_body=payload)

    async def delete_deployment(self, deployment_id: str) -> dict | None:
        return await self._deployment_req("DELETE", f"/api/v1/deployments/{deployment_id}", deployment_id=deployment_id)

    async def pause_deployment(self, deployment_id: str) -> dict:
        return await self._deployment_req("POST", f"/api/v1/deployments/{deployment_id}/pause", deployment_id=deployment_id)

    async def resume_deployment(self, deployment_id: str) -> dict:
        return await self._deployment_req("POST", f"/api/v1/deployments/{deployment_id}/resume", deployment_id=deployment_id)

    async def scale_deployment(self, deployment_id: str, num_instances: int) -> dict:
        return await self._deployment_req("PUT", f"/api/v1/deployments/{deployment_id}/scale", deployment_id=deployment_id, params={"num_instances": num_instances})

    async def rollback_deployment(self, deployment_id: str, checkpoint_path: str) -> dict:
        return await self._deployment_req("POST", f"/api/v1/deployments/{deployment_id}/rollback", deployment_id=deployment_id, params={"checkpoint_path": checkpoint_path})

    async def get_deployment_metrics(self, deployment_id: str, time_range: str = "24h") -> dict:
        return await self._deployment_req("GET", f"/api/v1/deployments/{deployment_id}/metrics", deployment_id=deployment_id, params={"time_range": time_range})

    async def get_deployment_logs(
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
        return await self._deployment_req("GET", f"/api/v1/deployments/{deployment_id}/logs", deployment_id=deployment_id, params=params)

    async def get_deployment_health_full(self, deployment_id: str) -> dict:
        return await self._deployment_req("GET", f"/api/v1/deployments/{deployment_id}/health", deployment_id=deployment_id)

    # ------------------------------------------------------------------
    # Hub endpoints
    # ------------------------------------------------------------------

    async def list_hub_models(self, **filter_params: Any) -> dict:
        return await self._hub_req("GET", "/api/v1/hub/models", params=filter_params)

    async def get_hub_model(self, model_id: str) -> dict:
        return await self._hub_req("GET", f"/api/v1/hub/models/{model_id}", model_id=model_id)

    async def create_hub_model(self, payload: dict) -> dict:
        return await self._hub_req("POST", "/api/v1/hub/models", json_body=payload)

    async def update_hub_model(self, model_id: str, payload: dict) -> dict:
        return await self._hub_req("PUT", f"/api/v1/hub/models/{model_id}", model_id=model_id, json_body=payload)

    async def delete_hub_model(self, model_id: str) -> None:
        await self._hub_req("DELETE", f"/api/v1/hub/models/{model_id}", model_id=model_id)

    async def list_hub_model_files(self, model_id: str) -> dict:
        return await self._hub_req("GET", f"/api/v1/hub/models/{model_id}/files", model_id=model_id)

    async def download_hub_model(self, model_id: str, file_id: str | None = None) -> dict:
        params = {"file_id": file_id} if file_id else None
        return await self._hub_req("GET", f"/api/v1/hub/models/{model_id}/download", model_id=model_id, params=params)

    async def list_hub_model_versions(self, model_id: str) -> list:
        return await self._hub_req("GET", f"/api/v1/hub/models/{model_id}/versions", model_id=model_id)

    async def create_hub_model_version(self, model_id: str, payload: dict) -> dict:
        return await self._hub_req("POST", f"/api/v1/hub/models/{model_id}/versions", model_id=model_id, json_body=payload)

    async def star_hub_model(self, model_id: str) -> dict:
        return await self._hub_req("POST", f"/api/v1/hub/models/{model_id}/star", model_id=model_id)

    async def unstar_hub_model(self, model_id: str) -> dict:
        return await self._hub_req("DELETE", f"/api/v1/hub/models/{model_id}/star", model_id=model_id)

    async def fork_hub_model(self, model_id: str) -> dict:
        return await self._hub_req("POST", f"/api/v1/hub/models/{model_id}/fork", model_id=model_id)

    async def list_hub_model_reviews(self, model_id: str, page: int = 1, limit: int = 20) -> dict:
        return await self._hub_req("GET", f"/api/v1/hub/models/{model_id}/reviews", model_id=model_id, params={"page": page, "limit": limit})

    async def add_hub_model_review(self, model_id: str, payload: dict) -> dict:
        return await self._hub_req("POST", f"/api/v1/hub/models/{model_id}/reviews", model_id=model_id, json_body=payload)

    async def use_hub_model_in_studio(self, model_id: str) -> dict:
        return await self._hub_req("POST", f"/api/v1/hub/models/{model_id}/use-in-studio", model_id=model_id)

    async def list_hub_categories(self) -> list:
        return await self._hub_req("GET", "/api/v1/hub/categories")

    async def get_hub_featured(self) -> list:
        return await self._hub_req("GET", "/api/v1/hub/featured")

    async def get_hub_trending(self, days: int = 7) -> list:
        return await self._hub_req("GET", "/api/v1/hub/trending", params={"days": days})

    async def list_hub_starred(self, sort_by: str = "date_starred", page: int = 1, limit: int = 20) -> dict:
        return await self._hub_req("GET", "/api/v1/hub/starred", params={"sort_by": sort_by, "page": page, "limit": limit})

    # ------------------------------------------------------------------
    # Project endpoints
    # ------------------------------------------------------------------

    async def list_projects(self, **filter_params: Any) -> dict:
        return await self._project_req("GET", "/api/v1/projects", params=filter_params)

    async def get_project(self, project_id: str) -> dict:
        return await self._project_req("GET", f"/api/v1/projects/{project_id}", project_id=project_id)

    async def create_project(self, payload: dict) -> dict:
        return await self._project_req("POST", "/api/v1/projects", json_body=payload)

    async def update_project(self, project_id: str, payload: dict) -> dict:
        return await self._project_req("PUT", f"/api/v1/projects/{project_id}", project_id=project_id, json_body=payload)

    async def delete_project(self, project_id: str) -> None:
        await self._project_req("DELETE", f"/api/v1/projects/{project_id}", project_id=project_id)

    async def duplicate_project(self, project_id: str, title: str | None = None) -> dict:
        body = {"title": title} if title else None
        return await self._project_req("POST", f"/api/v1/projects/{project_id}/duplicate", project_id=project_id, json_body=body)

    async def save_architecture(self, project_id: str, payload: dict) -> dict:
        return await self._project_req("POST", f"/api/v1/projects/{project_id}/architecture", project_id=project_id, json_body=payload)

    async def import_dag(self, payload: dict) -> dict:
        return await self._project_req("POST", "/api/v1/projects/import", json_body=payload)

    async def import_dag_existing(self, project_id: str, payload: dict) -> dict:
        return await self._project_req("POST", f"/api/v1/projects/{project_id}/import", project_id=project_id, json_body=payload)

    async def bulk_delete_projects(self, project_ids: list[str]) -> dict:
        return await self._project_req("POST", "/api/v1/projects/bulk-delete", json_body={"project_ids": project_ids})

    async def link_dataset(self, project_id: str, dataset_id: str, role: str) -> dict:
        return await self._project_req("POST", f"/api/v1/projects/{project_id}/datasets", project_id=project_id, json_body={"dataset_id": dataset_id, "role": role})

    async def get_project_datasets(self, project_id: str) -> dict:
        return await self._project_req("GET", f"/api/v1/projects/{project_id}/datasets", project_id=project_id)

    async def unlink_dataset(self, project_id: str, dataset_id: str) -> None:
        await self._project_req("DELETE", f"/api/v1/projects/{project_id}/datasets/{dataset_id}", project_id=project_id)

    # ------------------------------------------------------------------
    # Codegen endpoints
    # ------------------------------------------------------------------

    async def generate_code(self, project_id: str, payload: dict, async_mode: bool = False) -> dict:
        params = {"async": "true"} if async_mode else None
        return await self._codegen_req("POST", f"/api/v1/codegen/{project_id}/generate", json_body=payload, params=params)

    async def preview_code(self, project_id: str, framework: str, version_id: str | None = None) -> dict:
        params: dict[str, str] = {"framework": framework}
        if version_id:
            params["version_id"] = version_id
        return await self._codegen_req("GET", f"/api/v1/codegen/{project_id}/preview", params=params)

    async def validate_architecture(self, project_id: str, version_id: str | None = None) -> dict:
        params = {"version_id": version_id} if version_id else None
        return await self._codegen_req("POST", f"/api/v1/codegen/{project_id}/validate", params=params)

    async def download_code_zip(
        self,
        project_id: str,
        framework: str,
        version_id: str | None = None,
        dest_path: Path | str | None = None,
    ) -> Path | bytes:
        params: dict[str, str] = {"framework": framework}
        if version_id:
            params["version_id"] = version_id
        resp = await self._request("GET", f"/api/v1/codegen/{project_id}/download", params=params)
        raise_for_codegen(resp)
        if dest_path:
            dest = Path(dest_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
            return dest
        return resp.content

    async def get_code_status(self, project_id: str, task_id: str) -> dict:
        return await self._codegen_req("GET", f"/api/v1/codegen/{project_id}/status/{task_id}")

    # ------------------------------------------------------------------
    # Dataset upload endpoints
    # ------------------------------------------------------------------

    async def upload_dataset(
        self,
        file_path: str | Path,
        name: str,
        dataset_type: str,
        format: str,
        description: str | None = None,
        visibility: str = "private",
        license: str | None = None,
    ) -> dict:
        fields: dict[str, str] = {
            "name": name,
            "type": dataset_type,
            "format": format,
            "visibility": visibility,
        }
        if description:
            fields["description"] = description
        if license:
            fields["license"] = license

        fp = Path(file_path)
        with open(fp, "rb") as fh:
            files = {"file": (fp.name, fh, "application/octet-stream")}
            resp = await self._request(
                "POST", "/api/v1/datasets/upload", data=fields, files=files, timeout=None,
            )
        raise_for_upload(resp)
        return resp.json()

    async def upload_dataset_from_url(
        self,
        url: str,
        name: str,
        dataset_type: str,
        format: str,
        description: str | None = None,
        visibility: str = "private",
    ) -> dict:
        body: dict[str, str] = {
            "url": url,
            "name": name,
            "type": dataset_type,
            "format": format,
            "visibility": visibility,
        }
        if description:
            body["description"] = description
        resp = await self._request("POST", "/api/v1/datasets/upload-url", json=body)
        raise_for_upload(resp)
        return resp.json()

    async def get_dataset_task_status(self, task_id: str) -> dict:
        resp = await self._request("GET", f"/api/v1/datasets/tasks/{task_id}")
        raise_for_task(resp, task_id)
        return resp.json()


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _raise_for_job(resp: httpx.Response, job_id: str) -> None:
    """Map training-job response errors (mirrors sync client._raise_for_job)."""
    if resp.is_success:
        return
    code = resp.status_code
    if code == 401:
        raise AuthError("Authentication failed: invalid or expired API key")
    if code == 404:
        raise TrainingJobNotFoundError(job_id)
    raise APIError(code, resp.text)


def _parse_cd(header: str | None) -> str:
    """Extract filename from Content-Disposition header."""
    import re

    if not header:
        return "data"
    m = re.search(r'filename="([^"]+)"', header) or re.search(r"filename=([^\s;]+)", header)
    return m.group(1) if m else "data"
