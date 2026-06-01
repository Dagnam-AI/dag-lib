"""Async projects client methods."""

from __future__ import annotations

from dagnam._core.aio.base import BaseAsyncDagnamClient
from dagnam._core.client.common import quote_path_segment, raise_for_project, response_json_value
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
        return ensure_json_object(await self._project_req(
            "GET", f"/api/v1/projects/{quote_path_segment(project_id)}", project_id=project_id
        ))

    async def create_project(self, payload: JsonObject) -> JsonObject:
        return ensure_json_object(
            await self._project_req("POST", "/api/v1/projects", json_body=payload)
        )

    async def update_project(self, project_id: str, payload: JsonObject) -> JsonObject:
        return ensure_json_object(await self._project_req(
            "PUT",
            f"/api/v1/projects/{quote_path_segment(project_id)}",
            project_id=project_id,
            json_body=payload,
        ))

    async def delete_project(self, project_id: str) -> None:
        await self._project_req(
            "DELETE", f"/api/v1/projects/{quote_path_segment(project_id)}", project_id=project_id
        )

    async def duplicate_project(self, project_id: str, title: str | None = None) -> JsonObject:
        body: JsonObject | None = {"title": title} if title else None
        return ensure_json_object(await self._project_req(
            "POST",
            f"/api/v1/projects/{quote_path_segment(project_id)}/duplicate",
            project_id=project_id,
            json_body=body,
        ))

    async def save_architecture(self, project_id: str, payload: JsonObject) -> JsonObject:
        return ensure_json_object(await self._project_req(
            "POST",
            f"/api/v1/projects/{quote_path_segment(project_id)}/save",
            project_id=project_id,
            json_body=payload,
        ))

    async def import_dag(self, payload: JsonObject) -> JsonObject:
        return ensure_json_object(
            await self._project_req("POST", "/api/v1/projects/import", json_body=payload)
        )

    async def import_dag_existing(self, project_id: str, payload: JsonObject) -> JsonObject:
        return ensure_json_object(await self._project_req(
            "POST",
            f"/api/v1/projects/{quote_path_segment(project_id)}/import",
            project_id=project_id,
            json_body=payload,
        ))

    async def bulk_delete_projects(self, project_ids: list[str]) -> JsonObject:
        return ensure_json_object(await self._project_req(
            "POST",
            "/api/v1/projects/bulk-delete",
            json_body={"project_ids": [str(project_id) for project_id in project_ids]},
        ))

    async def link_dataset(self, project_id: str, dataset_id: str, role: str) -> JsonObject:
        return ensure_json_object(await self._project_req(
            "POST",
            f"/api/v1/projects/{quote_path_segment(project_id)}/datasets",
            project_id=project_id,
            json_body={"dataset_id": dataset_id, "role": role},
        ))

    async def get_project_datasets(self, project_id: str) -> JsonObject:
        return ensure_json_object(await self._project_req(
            "GET",
            f"/api/v1/projects/{quote_path_segment(project_id)}/datasets",
            project_id=project_id,
        ))

    async def unlink_dataset(self, project_id: str, dataset_id: str) -> None:
        await self._project_req(
            "DELETE",
            (
                f"/api/v1/projects/{quote_path_segment(project_id)}"
                f"/datasets/{quote_path_segment(dataset_id)}"
            ),
            project_id=project_id,
        )
