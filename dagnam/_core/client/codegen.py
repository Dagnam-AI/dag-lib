"""Synchronous codegen client methods."""

from __future__ import annotations

from pathlib import Path

from dagnam._types import JsonObject, JsonValue, QueryParams, ensure_json_object
from dagnam._core.client.base import (
    ALLOW_REDIRECTS,
    APIError,
    BaseDagnamClient,
    DEFAULT_TIMEOUT,
    requests,
)
from dagnam._core.client.common import quote_path_segment


def _requests_params(params: QueryParams | None) -> dict[str, str] | None:
    if params is None:
        return None
    return {key: str(value) for key, value in params.items() if value is not None}


class CodegenClientMixin(BaseDagnamClient):
    """Codegen resource methods for DagnamClient."""

    def _codegen_request(
        self,
        method: str,
        path: str,
        *,
        params: QueryParams | None = None,
        json_body: JsonValue = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> JsonValue | str | None:
        from dagnam._core.client.common import raise_for_codegen, response_json_value

        url = f"{self.api_url}{path}"
        try:
            resp = requests.request(
                method,
                url,
                headers=self._headers(),
                params=_requests_params(params),
                json=json_body,
                timeout=timeout,
                allow_redirects=ALLOW_REDIRECTS,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        raise_for_codegen(resp)

        if not resp.content:
            return None
        try:
            return response_json_value(resp)
        except ValueError:
            return resp.text

    def generate_code(
        self,
        project_id: str,
        payload: JsonObject | None = None,
        async_mode: bool = False,
        *,
        framework: str = "pytorch",
        version_id: str | None = None,
        options: JsonObject | None = None,
    ) -> JsonObject:
        if payload is None:
            payload = {"framework": framework}
            if version_id is not None:
                payload["version_id"] = version_id
            if options is not None:
                payload["options"] = options
        params: QueryParams | None = {"async_mode": "true"} if async_mode else None
        return ensure_json_object(self._codegen_request(
            "POST",
            f"/api/v1/projects/{quote_path_segment(project_id)}/generate-code",
            json_body=payload,
            params=params,
        ))

    def preview_code(
        self, project_id: str, framework: str, version_id: str | None = None
    ) -> JsonObject | str | None:
        params: QueryParams = {"framework": framework}
        if version_id:
            params["version_id"] = version_id
        value = self._codegen_request(
            "GET", f"/api/v1/projects/{quote_path_segment(project_id)}/code-preview", params=params
        )
        if isinstance(value, dict):
            return ensure_json_object(value)
        if isinstance(value, str) or value is None:
            return value
        raise TypeError(f"Expected JSON object, got {type(value).__name__}")

    def validate_code(self, project_id: str, version_id: str | None = None) -> JsonObject:
        params: QueryParams | None = {"version_id": version_id} if version_id else None
        return ensure_json_object(self._codegen_request(
            "POST", f"/api/v1/projects/{quote_path_segment(project_id)}/validate", params=params
        ))

    def validate_architecture(self, project_id: str, version_id: str | None = None) -> JsonObject:
        return self.validate_code(project_id, version_id=version_id)

    def download_code(
        self,
        project_id: str,
        framework: str = "pytorch",
        version_id: str | None = None,
        dest_path: Path | str | None = None,
    ) -> Path | bytes:
        from dagnam._core.client.common import raise_for_codegen

        params: dict[str, str] = {"framework": framework}
        if version_id:
            params["version_id"] = version_id
        url = f"{self.api_url}/api/v1/projects/{quote_path_segment(project_id)}/download-code"
        try:
            resp = requests.get(
                url,
                headers=self._headers(),
                params=params,
                stream=bool(dest_path),
                timeout=DEFAULT_TIMEOUT,
                allow_redirects=ALLOW_REDIRECTS,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        raise_for_codegen(resp)

        if dest_path:
            return self._stream_response_to_file(resp, Path(dest_path))
        return resp.content

    def download_code_zip(
        self,
        project_id: str,
        framework: str,
        version_id: str | None = None,
        dest_path: Path | str | None = None,
    ) -> Path | bytes:
        return self.download_code(
            project_id,
            framework=framework,
            version_id=version_id,
            dest_path=dest_path,
        )

    def get_code_status(self, project_id: str, task_id: str) -> JsonObject:
        return ensure_json_object(self._codegen_request(
            "GET",
            (
                f"/api/v1/projects/{quote_path_segment(project_id)}"
                f"/code-status/{quote_path_segment(task_id)}"
            ),
        ))
