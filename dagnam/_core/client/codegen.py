"""Synchronous codegen client methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dagnam._core.client.base import _ALLOW_REDIRECTS, _TIMEOUT, APIError, requests
from dagnam._core.client.common import quote_path_segment


class CodegenClientMixin:
    """Codegen resource methods for DagnamClient."""

    def _codegen_request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json_body: Any = None,
        timeout: int = _TIMEOUT,
    ) -> dict | list | None:
        from dagnam._core.client.common import raise_for_codegen

        url = f"{self.api_url}{path}"
        try:
            resp = requests.request(
                method,
                url,
                headers=self._headers(),
                params=params,
                json=json_body,
                timeout=timeout,
                allow_redirects=_ALLOW_REDIRECTS,
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

    def generate_code(
        self,
        project_id: str,
        payload: dict | None = None,
        async_mode: bool = False,
        *,
        framework: str = "pytorch",
        version_id: str | None = None,
        options: dict | None = None,
    ) -> dict:
        if payload is None:
            payload = {"framework": framework}
            if version_id is not None:
                payload["version_id"] = version_id
            if options is not None:
                payload["options"] = options
        params = {"async_mode": "true"} if async_mode else None
        return self._codegen_request(
            "POST",
            f"/api/v1/projects/{quote_path_segment(project_id)}/generate-code",
            json_body=payload,
            params=params,
        )

    def preview_code(self, project_id: str, framework: str, version_id: str | None = None) -> dict:
        params: dict[str, str] = {"framework": framework}
        if version_id:
            params["version_id"] = version_id
        return self._codegen_request(
            "GET", f"/api/v1/projects/{quote_path_segment(project_id)}/code-preview", params=params
        )

    def validate_code(self, project_id: str, version_id: str | None = None) -> dict:
        params = {"version_id": version_id} if version_id else None
        return self._codegen_request(
            "POST", f"/api/v1/projects/{quote_path_segment(project_id)}/validate", params=params
        )

    def validate_architecture(self, project_id: str, version_id: str | None = None) -> dict:
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
                timeout=_TIMEOUT,
                allow_redirects=_ALLOW_REDIRECTS,
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

    def get_code_status(self, project_id: str, task_id: str) -> dict:
        return self._codegen_request(
            "GET",
            (
                f"/api/v1/projects/{quote_path_segment(project_id)}"
                f"/code-status/{quote_path_segment(task_id)}"
            ),
        )
