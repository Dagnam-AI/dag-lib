"""Async projects client methods."""

from __future__ import annotations

from typing import Any

from dagnam._core.client.common import raise_for_project


class AsyncProjectsMixin:
    """Async Projects resource methods."""

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
        resp = await self._request(
            method, path, params=params, json=json_body, data=data, files=files
        )
        raise_for_project(resp, project_id)
        if not resp.content:
            return None
        try:
            return resp.json()
        except ValueError:
            return resp.text

    async def list_projects(self, **filter_params: Any) -> dict:
        return await self._project_req("GET", "/api/v1/projects", params=filter_params)

    async def get_project(self, project_id: str) -> dict:
        return await self._project_req(
            "GET", f"/api/v1/projects/{project_id}", project_id=project_id
        )

    async def create_project(self, payload: dict) -> dict:
        return await self._project_req("POST", "/api/v1/projects", json_body=payload)

    async def update_project(self, project_id: str, payload: dict) -> dict:
        return await self._project_req(
            "PUT", f"/api/v1/projects/{project_id}", project_id=project_id, json_body=payload
        )

    async def delete_project(self, project_id: str) -> None:
        await self._project_req("DELETE", f"/api/v1/projects/{project_id}", project_id=project_id)

    async def duplicate_project(self, project_id: str, title: str | None = None) -> dict:
        body = {"title": title} if title else None
        return await self._project_req(
            "POST",
            f"/api/v1/projects/{project_id}/duplicate",
            project_id=project_id,
            json_body=body,
        )

    async def save_architecture(self, project_id: str, payload: dict) -> dict:
        return await self._project_req(
            "POST",
            f"/api/v1/projects/{project_id}/architecture",
            project_id=project_id,
            json_body=payload,
        )

    async def import_dag(self, payload: dict) -> dict:
        return await self._project_req("POST", "/api/v1/projects/import", json_body=payload)

    async def import_dag_existing(self, project_id: str, payload: dict) -> dict:
        return await self._project_req(
            "POST",
            f"/api/v1/projects/{project_id}/import",
            project_id=project_id,
            json_body=payload,
        )

    async def bulk_delete_projects(self, project_ids: list[str]) -> dict:
        return await self._project_req(
            "POST", "/api/v1/projects/bulk-delete", json_body={"project_ids": project_ids}
        )

    async def link_dataset(self, project_id: str, dataset_id: str, role: str) -> dict:
        return await self._project_req(
            "POST",
            f"/api/v1/projects/{project_id}/datasets",
            project_id=project_id,
            json_body={"dataset_id": dataset_id, "role": role},
        )

    async def get_project_datasets(self, project_id: str) -> dict:
        return await self._project_req(
            "GET", f"/api/v1/projects/{project_id}/datasets", project_id=project_id
        )

    async def unlink_dataset(self, project_id: str, dataset_id: str) -> None:
        await self._project_req(
            "DELETE", f"/api/v1/projects/{project_id}/datasets/{dataset_id}", project_id=project_id
        )
