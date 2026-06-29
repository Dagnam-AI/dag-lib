"""Synchronous projects client methods."""

from __future__ import annotations

from dagnam._core.client.base import (
    ALLOW_REDIRECTS,
    DEFAULT_TIMEOUT,
    APIError,
    BaseDagnamClient,
    requests,
)
from dagnam._core.client.common import quote_path_segment, requests_query_params
from dagnam._types import FormData, JsonObject, JsonValue, QueryParams, QueryValue, UploadFiles


class ProjectsClientMixin(BaseDagnamClient):
    """Projects resource methods for DagnamClient."""

    def _project_request(
        self,
        method: str,
        path: str,
        *,
        project_id: str | None = None,
        params: QueryParams | None = None,
        json_body: JsonValue = None,
        data: FormData | None = None,
        files: UploadFiles | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> JsonValue | str | None:
        from dagnam._core.client.common import raise_for_project, response_json_value

        url = f"{self.api_url}{path}"
        try:
            resp = requests.request(
                method,
                url,
                headers=self._headers(),
                params=requests_query_params(params),
                json=json_body,
                data=data,
                files=files,
                timeout=timeout,
                allow_redirects=ALLOW_REDIRECTS,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        raise_for_project(resp, project_id)

        if not resp.content:
            return None
        try:
            return response_json_value(resp)
        except ValueError:
            return resp.text

    def list_projects(self, **filter_params: QueryValue) -> JsonObject | str | None:
        value = self._project_request("GET", "/api/v1/projects", params=filter_params)
        if isinstance(value, dict | str) or value is None:
            return value
        raise TypeError(f"Expected JSON object, got {type(value).__name__}")

    def get_project(self, project_id: str) -> JsonObject:
        value = self._project_request(
            "GET", f"/api/v1/projects/{quote_path_segment(project_id)}", project_id=project_id
        )
        if isinstance(value, dict):
            return value
        raise TypeError(f"Expected JSON object, got {type(value).__name__}")

    def create_project(self, payload: JsonObject) -> JsonObject:
        value = self._project_request("POST", "/api/v1/projects", json_body=payload)
        if isinstance(value, dict):
            return value
        raise TypeError(f"Expected JSON object, got {type(value).__name__}")

    def update_project(self, project_id: str, payload: JsonObject) -> JsonObject:
        value = self._project_request(
            "PUT",
            f"/api/v1/projects/{quote_path_segment(project_id)}",
            project_id=project_id,
            json_body=payload,
        )
        if isinstance(value, dict):
            return value
        raise TypeError(f"Expected JSON object, got {type(value).__name__}")

    def delete_project(self, project_id: str) -> None:
        self._project_request(
            "DELETE", f"/api/v1/projects/{quote_path_segment(project_id)}", project_id=project_id
        )

    def duplicate_project(self, project_id: str, title: str | None = None) -> JsonObject:
        body: JsonObject | None = {"title": title} if title else None
        value = self._project_request(
            "POST",
            f"/api/v1/projects/{quote_path_segment(project_id)}/duplicate",
            project_id=project_id,
            json_body=body,
        )
        if isinstance(value, dict):
            return value
        raise TypeError(f"Expected JSON object, got {type(value).__name__}")

    def save_architecture(self, project_id: str, payload: JsonObject) -> JsonObject:
        value = self._project_request(
            "POST",
            f"/api/v1/projects/{quote_path_segment(project_id)}/save",
            project_id=project_id,
            json_body=payload,
        )
        if isinstance(value, dict):
            return value
        raise TypeError(f"Expected JSON object, got {type(value).__name__}")

    def import_dag(self, payload: JsonObject) -> JsonObject:
        value = self._project_request("POST", "/api/v1/projects/import", json_body=payload)
        if isinstance(value, dict):
            return value
        raise TypeError(f"Expected JSON object, got {type(value).__name__}")

    def import_dag_existing(self, project_id: str, payload: JsonObject) -> JsonObject:
        value = self._project_request(
            "POST",
            f"/api/v1/projects/{quote_path_segment(project_id)}/import",
            project_id=project_id,
            json_body=payload,
        )
        if isinstance(value, dict):
            return value
        raise TypeError(f"Expected JSON object, got {type(value).__name__}")

    def bulk_delete_projects(self, project_ids: list[str]) -> JsonObject:
        value = self._project_request(
            "POST",
            "/api/v1/projects/bulk-delete",
            json_body={"project_ids": [str(project_id) for project_id in project_ids]},
        )
        if isinstance(value, dict):
            return value
        raise TypeError(f"Expected JSON object, got {type(value).__name__}")

    def link_dataset(self, project_id: str, dataset_id: str, role: str) -> JsonObject:
        value = self._project_request(
            "POST",
            f"/api/v1/projects/{quote_path_segment(project_id)}/datasets",
            project_id=project_id,
            json_body={"dataset_id": dataset_id, "role": role},
        )
        if isinstance(value, dict):
            return value
        raise TypeError(f"Expected JSON object, got {type(value).__name__}")

    def get_project_datasets(self, project_id: str) -> JsonObject:
        value = self._project_request(
            "GET",
            f"/api/v1/projects/{quote_path_segment(project_id)}/datasets",
            project_id=project_id,
        )
        if isinstance(value, dict):
            return value
        raise TypeError(f"Expected JSON object, got {type(value).__name__}")

    def unlink_dataset(self, project_id: str, dataset_id: str) -> None:
        self._project_request(
            "DELETE",
            (
                f"/api/v1/projects/{quote_path_segment(project_id)}"
                f"/datasets/{quote_path_segment(dataset_id)}"
            ),
            project_id=project_id,
        )

    # ---------------------------------------------------------------- versions

    def list_project_versions(self, project_id: str, **filters: QueryValue) -> JsonObject:
        """List a project's architecture versions. ``GET /api/v1/projects/{id}/versions``."""
        return self._expect_object(
            self._project_request(
                "GET",
                f"/api/v1/projects/{quote_path_segment(project_id)}/versions",
                project_id=project_id,
                params=filters,
            )
        )

    def get_project_version(self, project_id: str, version_id: str) -> JsonObject:
        """Get one architecture version. ``GET /api/v1/projects/{id}/versions/{version_id}``."""
        return self._expect_object(
            self._project_request(
                "GET",
                (
                    f"/api/v1/projects/{quote_path_segment(project_id)}"
                    f"/versions/{quote_path_segment(version_id)}"
                ),
                project_id=project_id,
            )
        )

    def compare_project_versions(
        self, project_id: str, version_a: str, version_b: str
    ) -> JsonObject:
        """Compare two architecture versions.

        ``GET /api/v1/projects/{id}/versions/compare?version_a=&version_b=``.
        """
        return self._expect_object(
            self._project_request(
                "GET",
                f"/api/v1/projects/{quote_path_segment(project_id)}/versions/compare",
                project_id=project_id,
                params={"version_a": version_a, "version_b": version_b},
            )
        )

    def restore_project_version(self, project_id: str, version_id: str) -> JsonObject:
        """Restore a project to a prior version (creates a new current version).

        ``POST /api/v1/projects/{id}/restore/{version_id}`` — note the path is
        ``/restore/{version_id}``, a sibling of ``/versions``, not
        ``/versions/{version_id}/restore``.
        """
        return self._expect_object(
            self._project_request(
                "POST",
                (
                    f"/api/v1/projects/{quote_path_segment(project_id)}"
                    f"/restore/{quote_path_segment(version_id)}"
                ),
                project_id=project_id,
            )
        )

    def delete_project_version(self, project_id: str, version_id: str) -> None:
        """Delete one architecture version. ``DELETE /api/v1/projects/{id}/versions/{version_id}``."""
        self._project_request(
            "DELETE",
            (
                f"/api/v1/projects/{quote_path_segment(project_id)}"
                f"/versions/{quote_path_segment(version_id)}"
            ),
            project_id=project_id,
        )

    def get_latest_project_version(self, project_id: str) -> JsonObject:
        """Get the current (latest) architecture version. ``GET /api/v1/projects/{id}/latest``."""
        return self._expect_object(
            self._project_request(
                "GET",
                f"/api/v1/projects/{quote_path_segment(project_id)}/latest",
                project_id=project_id,
            )
        )
