"""Synchronous projects client methods."""

from __future__ import annotations

from typing import Any

from dagnam._core.client.base import _ALLOW_REDIRECTS, _TIMEOUT, APIError, requests
from dagnam._core.client.common import quote_path_segment


class ProjectsClientMixin:
    """Projects resource methods for DagnamClient."""

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
        from dagnam._core.client.common import raise_for_project

        url = f"{self.api_url}{path}"
        try:
            resp = requests.request(
                method,
                url,
                headers=self._headers(),
                params=params,
                json=json_body,
                data=data,
                files=files,
                timeout=timeout,
                allow_redirects=_ALLOW_REDIRECTS,
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
        return self._project_request(
            "GET", f"/api/v1/projects/{quote_path_segment(project_id)}", project_id=project_id
        )

    def create_project(self, payload: dict) -> dict:
        return self._project_request("POST", "/api/v1/projects", json_body=payload)

    def update_project(self, project_id: str, payload: dict) -> dict:
        return self._project_request(
            "PUT",
            f"/api/v1/projects/{quote_path_segment(project_id)}",
            project_id=project_id,
            json_body=payload,
        )

    def delete_project(self, project_id: str) -> None:
        self._project_request(
            "DELETE", f"/api/v1/projects/{quote_path_segment(project_id)}", project_id=project_id
        )

    def duplicate_project(self, project_id: str, title: str | None = None) -> dict:
        body = {"title": title} if title else None
        return self._project_request(
            "POST",
            f"/api/v1/projects/{quote_path_segment(project_id)}/duplicate",
            project_id=project_id,
            json_body=body,
        )

    def save_architecture(self, project_id: str, payload: dict) -> dict:
        return self._project_request(
            "POST",
            f"/api/v1/projects/{quote_path_segment(project_id)}/architecture",
            project_id=project_id,
            json_body=payload,
        )

    def import_dag(self, payload: dict) -> dict:
        return self._project_request("POST", "/api/v1/projects/import", json_body=payload)

    def import_dag_existing(self, project_id: str, payload: dict) -> dict:
        return self._project_request(
            "POST",
            f"/api/v1/projects/{quote_path_segment(project_id)}/import",
            project_id=project_id,
            json_body=payload,
        )

    def bulk_delete_projects(self, project_ids: list[str]) -> dict:
        return self._project_request(
            "POST", "/api/v1/projects/bulk-delete", json_body={"project_ids": project_ids}
        )

    def link_dataset(self, project_id: str, dataset_id: str, role: str) -> dict:
        return self._project_request(
            "POST",
            f"/api/v1/projects/{quote_path_segment(project_id)}/datasets",
            project_id=project_id,
            json_body={"dataset_id": dataset_id, "role": role},
        )

    def get_project_datasets(self, project_id: str) -> dict:
        return self._project_request(
            "GET",
            f"/api/v1/projects/{quote_path_segment(project_id)}/datasets",
            project_id=project_id,
        )

    def unlink_dataset(self, project_id: str, dataset_id: str) -> None:
        self._project_request(
            "DELETE",
            (
                f"/api/v1/projects/{quote_path_segment(project_id)}"
                f"/datasets/{quote_path_segment(dataset_id)}"
            ),
            project_id=project_id,
        )
