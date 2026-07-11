"""Async projects client methods."""

from __future__ import annotations

from pathlib import Path

import httpx

from dagnam._core.aio.base import BaseAsyncDagnamClient, content_disposition_safe_name
from dagnam._core.client.common import quote_path_segment, raise_for_project, response_json_value
from dagnam._core.exceptions import APIError
from dagnam._types import (
    FormData,
    JsonObject,
    JsonValue,
    QueryParams,
    QueryValue,
    UploadFiles,
    ensure_json_object,
)


class AsyncProjectsMixin(BaseAsyncDagnamClient):
    """Async Projects resource methods."""

    async def _project_req(
        self,
        method: str,
        path: str,
        *,
        project_id: str | None = None,
        params: QueryParams | None = None,
        json_body: JsonValue = None,
        data: FormData | None = None,
        files: UploadFiles | None = None,
    ) -> JsonValue | str | None:
        resp = await self._request(
            method, path, params=params, json=json_body, data=data, files=files
        )
        raise_for_project(resp, project_id)
        if not resp.content:
            return None
        try:
            return response_json_value(resp)
        except ValueError:
            return resp.text

    async def list_projects(self, **filter_params: QueryValue) -> JsonObject | str | None:
        value = await self._project_req("GET", "/api/v1/projects", params=filter_params)
        if isinstance(value, dict):
            return ensure_json_object(value)
        if isinstance(value, str) or value is None:
            return value
        raise TypeError(f"Expected JSON object, got {type(value).__name__}")

    async def get_project(self, project_id: str) -> JsonObject:
        return ensure_json_object(
            await self._project_req(
                "GET", f"/api/v1/projects/{quote_path_segment(project_id)}", project_id=project_id
            )
        )

    async def create_project(self, payload: JsonObject) -> JsonObject:
        return ensure_json_object(
            await self._project_req("POST", "/api/v1/projects", json_body=payload)
        )

    async def update_project(self, project_id: str, payload: JsonObject) -> JsonObject:
        return ensure_json_object(
            await self._project_req(
                "PUT",
                f"/api/v1/projects/{quote_path_segment(project_id)}",
                project_id=project_id,
                json_body=payload,
            )
        )

    async def delete_project(self, project_id: str) -> None:
        await self._project_req(
            "DELETE", f"/api/v1/projects/{quote_path_segment(project_id)}", project_id=project_id
        )

    async def duplicate_project(self, project_id: str, title: str | None = None) -> JsonObject:
        body: JsonObject | None = {"title": title} if title else None
        return ensure_json_object(
            await self._project_req(
                "POST",
                f"/api/v1/projects/{quote_path_segment(project_id)}/duplicate",
                project_id=project_id,
                json_body=body,
            )
        )

    async def save_architecture(self, project_id: str, payload: JsonObject) -> JsonObject:
        return ensure_json_object(
            await self._project_req(
                "POST",
                f"/api/v1/projects/{quote_path_segment(project_id)}/save",
                project_id=project_id,
                json_body=payload,
            )
        )

    async def import_dag(self, payload: JsonObject) -> JsonObject:
        return ensure_json_object(
            await self._project_req("POST", "/api/v1/projects/import", json_body=payload)
        )

    async def import_dag_existing(self, project_id: str, payload: JsonObject) -> JsonObject:
        return ensure_json_object(
            await self._project_req(
                "POST",
                f"/api/v1/projects/{quote_path_segment(project_id)}/import",
                project_id=project_id,
                json_body=payload,
            )
        )

    async def bulk_delete_projects(self, project_ids: list[str]) -> JsonObject:
        return ensure_json_object(
            await self._project_req(
                "POST",
                "/api/v1/projects/bulk-delete",
                json_body={"project_ids": [str(project_id) for project_id in project_ids]},
            )
        )

    async def link_dataset(self, project_id: str, dataset_id: str, role: str) -> JsonObject:
        return ensure_json_object(
            await self._project_req(
                "POST",
                f"/api/v1/projects/{quote_path_segment(project_id)}/datasets",
                project_id=project_id,
                json_body={"dataset_id": dataset_id, "role": role},
            )
        )

    async def get_project_datasets(self, project_id: str) -> JsonObject:
        return ensure_json_object(
            await self._project_req(
                "GET",
                f"/api/v1/projects/{quote_path_segment(project_id)}/datasets",
                project_id=project_id,
            )
        )

    async def unlink_dataset(self, project_id: str, dataset_id: str) -> None:
        await self._project_req(
            "DELETE",
            (
                f"/api/v1/projects/{quote_path_segment(project_id)}"
                f"/datasets/{quote_path_segment(dataset_id)}"
            ),
            project_id=project_id,
        )

    # ---------------------------------------------------------------- versions

    async def list_project_versions(self, project_id: str, **filters: QueryValue) -> JsonObject:
        return ensure_json_object(
            await self._project_req(
                "GET",
                f"/api/v1/projects/{quote_path_segment(project_id)}/versions",
                project_id=project_id,
                params=filters,
            )
        )

    async def get_project_version(self, project_id: str, version_id: str) -> JsonObject:
        return ensure_json_object(
            await self._project_req(
                "GET",
                (
                    f"/api/v1/projects/{quote_path_segment(project_id)}"
                    f"/versions/{quote_path_segment(version_id)}"
                ),
                project_id=project_id,
            )
        )

    async def compare_project_versions(
        self, project_id: str, version_a: str, version_b: str
    ) -> JsonObject:
        return ensure_json_object(
            await self._project_req(
                "GET",
                f"/api/v1/projects/{quote_path_segment(project_id)}/versions/compare",
                project_id=project_id,
                params={"version_a": version_a, "version_b": version_b},
            )
        )

    async def restore_project_version(self, project_id: str, version_id: str) -> JsonObject:
        return ensure_json_object(
            await self._project_req(
                "POST",
                (
                    f"/api/v1/projects/{quote_path_segment(project_id)}"
                    f"/restore/{quote_path_segment(version_id)}"
                ),
                project_id=project_id,
            )
        )

    async def delete_project_version(self, project_id: str, version_id: str) -> None:
        await self._project_req(
            "DELETE",
            (
                f"/api/v1/projects/{quote_path_segment(project_id)}"
                f"/versions/{quote_path_segment(version_id)}"
            ),
            project_id=project_id,
        )

    async def get_latest_project_version(self, project_id: str) -> JsonObject:
        return ensure_json_object(
            await self._project_req(
                "GET",
                f"/api/v1/projects/{quote_path_segment(project_id)}/latest",
                project_id=project_id,
            )
        )

    # --------------------------------------------------------------- thumbnail

    async def upload_project_thumbnail(self, project_id: str, file_path: str | Path) -> JsonObject:
        """Upload a project thumbnail image. ``POST /api/v1/projects/{id}/thumbnail`` (multipart)."""
        path = Path(file_path)
        if not path.is_file():  # noqa: ASYNC240 - one-shot local stat before opening, not I/O-bound
            raise FileNotFoundError(f"No such file: {path}")
        with open(path, "rb") as fh:
            files = {"file": (path.name, fh, "application/octet-stream")}
            return ensure_json_object(
                await self._project_req(
                    "POST",
                    f"/api/v1/projects/{quote_path_segment(project_id)}/thumbnail",
                    project_id=project_id,
                    files=files,
                )
            )

    async def download_project_thumbnail(self, project_id: str, dest_dir: str | Path) -> Path:
        """Stream-download a project's thumbnail image into ``dest_dir``.

        ``GET /api/v1/projects/{id}/thumbnail`` returns the raw image bytes. The
        saved filename is taken from the ``Content-Disposition`` header and
        reduced to a bare basename, so a hostile header cannot escape ``dest_dir``.
        """
        url = f"{self.api_url}/api/v1/projects/{quote_path_segment(project_id)}/thumbnail"
        try:
            async with self._client.stream("GET", url, headers=self._headers()) as resp:
                if not resp.is_success:
                    await resp.aread()  # populate the body for the error message
                    raise_for_project(resp, project_id)
                name = content_disposition_safe_name(
                    resp.headers.get("content-disposition"),
                    default=f"{project_id}-thumbnail.png",
                )
                dest = Path(dest_dir) / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as fh:
                    async for chunk in resp.aiter_bytes():
                        fh.write(chunk)
                return dest
        except httpx.ConnectError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc
