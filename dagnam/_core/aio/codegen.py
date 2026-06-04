"""Async codegen client methods."""

from __future__ import annotations

import asyncio
from pathlib import Path

from dagnam._core.aio.base import BaseAsyncDagnamClient
from dagnam._core.client.common import quote_path_segment, raise_for_codegen, response_json_value
from dagnam._types import JsonObject, JsonValue, QueryParams, ensure_json_object


class AsyncCodegenMixin(BaseAsyncDagnamClient):
    """Async Codegen resource methods."""

    async def _codegen_req(
        self,
        method: str,
        path: str,
        *,
        params: QueryParams | None = None,
        json_body: JsonValue = None,
    ) -> JsonValue | str | None:
        resp = await self._request(method, path, params=params, json=json_body)
        raise_for_codegen(resp)
        if not resp.content:
            return None
        try:
            return response_json_value(resp)
        except ValueError:
            return resp.text

    async def generate_code(
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
        return ensure_json_object(
            await self._codegen_req(
                "POST",
                f"/api/v1/projects/{quote_path_segment(project_id)}/generate-code",
                json_body=payload,
                params=params,
            )
        )

    async def preview_code(
        self, project_id: str, framework: str, version_id: str | None = None
    ) -> JsonObject | str | None:
        params: QueryParams = {"framework": framework}
        if version_id:
            params["version_id"] = version_id
        value = await self._codegen_req(
            "GET", f"/api/v1/projects/{quote_path_segment(project_id)}/code-preview", params=params
        )
        if isinstance(value, dict):
            return ensure_json_object(value)
        if isinstance(value, str) or value is None:
            return value
        raise TypeError(f"Expected JSON object, got {type(value).__name__}")

    async def validate_code(self, project_id: str, version_id: str | None = None) -> JsonObject:
        params: QueryParams | None = {"version_id": version_id} if version_id else None
        return ensure_json_object(
            await self._codegen_req(
                "POST", f"/api/v1/projects/{quote_path_segment(project_id)}/validate", params=params
            )
        )

    async def validate_architecture(
        self, project_id: str, version_id: str | None = None
    ) -> JsonObject:
        return await self.validate_code(project_id, version_id=version_id)

    async def download_code(
        self,
        project_id: str,
        framework: str = "pytorch",
        version_id: str | None = None,
        dest_path: Path | str | None = None,
    ) -> Path | bytes:
        params: QueryParams = {"framework": framework}
        if version_id:
            params["version_id"] = version_id
        resp = await self._request(
            "GET", f"/api/v1/projects/{quote_path_segment(project_id)}/download-code", params=params
        )
        raise_for_codegen(resp)
        if dest_path:
            dest = Path(dest_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(dest.write_bytes, resp.content)
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

    async def get_code_status(self, project_id: str, task_id: str) -> JsonObject:
        return ensure_json_object(
            await self._codegen_req(
                "GET",
                (
                    f"/api/v1/projects/{quote_path_segment(project_id)}"
                    f"/code-status/{quote_path_segment(task_id)}"
                ),
            )
        )
