"""Async codegen client methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dagnam._core.client.common import raise_for_codegen


class AsyncCodegenMixin:
    """Async Codegen resource methods."""

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

    async def generate_code(
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
        return await self._codegen_req(
            "POST",
            f"/api/v1/projects/{project_id}/generate-code",
            json_body=payload,
            params=params,
        )

    async def preview_code(
        self, project_id: str, framework: str, version_id: str | None = None
    ) -> dict:
        params: dict[str, str] = {"framework": framework}
        if version_id:
            params["version_id"] = version_id
        return await self._codegen_req(
            "GET", f"/api/v1/projects/{project_id}/code-preview", params=params
        )

    async def validate_code(self, project_id: str, version_id: str | None = None) -> dict:
        params = {"version_id": version_id} if version_id else None
        return await self._codegen_req(
            "POST", f"/api/v1/projects/{project_id}/validate", params=params
        )

    async def validate_architecture(self, project_id: str, version_id: str | None = None) -> dict:
        return await self.validate_code(project_id, version_id=version_id)

    async def download_code(
        self,
        project_id: str,
        framework: str = "pytorch",
        version_id: str | None = None,
        dest_path: Path | str | None = None,
    ) -> Path | bytes:
        params: dict[str, str] = {"framework": framework}
        if version_id:
            params["version_id"] = version_id
        resp = await self._request(
            "GET", f"/api/v1/projects/{project_id}/download-code", params=params
        )
        raise_for_codegen(resp)
        if dest_path:
            dest = Path(dest_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
            return dest
        return resp.content

    async def download_code_zip(
        self,
        project_id: str,
        framework: str,
        version_id: str | None = None,
        dest_path: Path | str | None = None,
    ) -> Path | bytes:
        return await self.download_code(
            project_id,
            framework=framework,
            version_id=version_id,
            dest_path=dest_path,
        )

    async def get_code_status(self, project_id: str, task_id: str) -> dict:
        return await self._codegen_req(
            "GET", f"/api/v1/projects/{project_id}/code-status/{task_id}"
        )
