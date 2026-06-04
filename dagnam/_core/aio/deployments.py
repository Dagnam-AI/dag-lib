"""Async deployments client methods."""

from __future__ import annotations

from dagnam._core.aio.base import BaseAsyncDagnamClient
from dagnam._core.client.common import (
    quote_path_segment,
    raise_for_deployment,
    response_json_value,
)
from dagnam._types import JsonObject, JsonValue, QueryParams, ensure_json_object


class AsyncDeploymentsMixin(BaseAsyncDagnamClient):
    """Async Deployments resource methods."""

    async def _deployment_req(
        self,
        method: str,
        path: str,
        *,
        deployment_id: str | None = None,
        params: QueryParams | None = None,
        json_body: JsonValue = None,
        timeout: int | None = None,
    ) -> JsonValue | str | None:
        resp = await self._request(method, path, params=params, json=json_body, timeout=timeout)
        raise_for_deployment(resp, deployment_id or "deployment")
        if not resp.content:
            return None
        try:
            return response_json_value(resp)
        except ValueError:
            return resp.text

    async def list_deployments(
        self,
        *,
        page: int = 1,
        limit: int = 20,
        status_filter: str | None = None,
        platform: str | None = None,
        project_id: str | None = None,
        search: str | None = None,
    ) -> JsonObject:
        params: dict[str, str | int] = {"page": page, "limit": limit}
        if status_filter is not None:
            params["status"] = status_filter
        if platform is not None:
            params["platform"] = platform
        if project_id is not None:
            params["project_id"] = project_id
        if search is not None:
            params["search"] = search
        return ensure_json_object(
            await self._deployment_req("GET", "/api/v1/deployments", params=params)
        )

    async def get_deployment(self, deployment_id: str) -> JsonObject:
        return ensure_json_object(
            await self._deployment_req(
                "GET",
                f"/api/v1/deployments/{quote_path_segment(deployment_id)}",
                deployment_id=deployment_id,
            )
        )

    async def create_deployment(self, payload: JsonObject) -> JsonObject:
        return ensure_json_object(
            await self._deployment_req("POST", "/api/v1/deployments", json_body=payload)
        )

    async def update_deployment(self, deployment_id: str, payload: JsonObject) -> JsonObject:
        return ensure_json_object(
            await self._deployment_req(
                "PUT",
                f"/api/v1/deployments/{quote_path_segment(deployment_id)}",
                deployment_id=deployment_id,
                json_body=payload,
            )
        )

    async def delete_deployment(self, deployment_id: str) -> JsonObject | None:
        value = await self._deployment_req(
            "DELETE",
            f"/api/v1/deployments/{quote_path_segment(deployment_id)}",
            deployment_id=deployment_id,
        )
        if value is None:
            return None
        return ensure_json_object(value)

    async def pause_deployment(self, deployment_id: str) -> JsonObject:
        return ensure_json_object(
            await self._deployment_req(
                "POST",
                f"/api/v1/deployments/{quote_path_segment(deployment_id)}/pause",
                deployment_id=deployment_id,
            )
        )

    async def resume_deployment(self, deployment_id: str) -> JsonObject:
        return ensure_json_object(
            await self._deployment_req(
                "POST",
                f"/api/v1/deployments/{quote_path_segment(deployment_id)}/resume",
                deployment_id=deployment_id,
            )
        )

    async def scale_deployment(self, deployment_id: str, num_instances: int) -> JsonObject:
        return ensure_json_object(
            await self._deployment_req(
                "PUT",
                f"/api/v1/deployments/{quote_path_segment(deployment_id)}/scale",
                deployment_id=deployment_id,
                params={"num_instances": num_instances},
            )
        )

    async def rollback_deployment(self, deployment_id: str, checkpoint_path: str) -> JsonObject:
        return ensure_json_object(
            await self._deployment_req(
                "POST",
                f"/api/v1/deployments/{quote_path_segment(deployment_id)}/rollback",
                deployment_id=deployment_id,
                params={"checkpoint_path": checkpoint_path},
            )
        )

    async def get_deployment_metrics(
        self, deployment_id: str, time_range: str = "24h"
    ) -> JsonObject:
        return ensure_json_object(
            await self._deployment_req(
                "GET",
                f"/api/v1/deployments/{quote_path_segment(deployment_id)}/metrics",
                deployment_id=deployment_id,
                params={"time_range": time_range},
            )
        )

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
    ) -> JsonObject:
        params: dict[str, str | int] = {"page": page, "limit": limit}
        if level is not None:
            params["level"] = level
        if search is not None:
            params["search"] = search
        if start_time is not None:
            params["start_time"] = start_time
        if end_time is not None:
            params["end_time"] = end_time
        return ensure_json_object(
            await self._deployment_req(
                "GET",
                f"/api/v1/deployments/{quote_path_segment(deployment_id)}/logs",
                deployment_id=deployment_id,
                params=params,
            )
        )

    async def get_deployment_health_full(self, deployment_id: str) -> JsonObject:
        return ensure_json_object(
            await self._deployment_req(
                "GET",
                f"/api/v1/deployments/{quote_path_segment(deployment_id)}/health",
                deployment_id=deployment_id,
            )
        )
