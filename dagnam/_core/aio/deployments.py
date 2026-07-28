"""Async deployments client methods."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
from httpx_sse import aconnect_sse

from dagnam._core.aio.base import SSE_READ_TIMEOUT, BaseAsyncDagnamClient
from dagnam._core.client.base import scrub_secret_params
from dagnam._core.client.common import (
    quote_path_segment,
    raise_for_deployment,
    response_json_value,
    stream_query_params,
)
from dagnam._core.exceptions import APIError, ResponseError
from dagnam._core.sse import (
    TERMINAL_DEPLOYMENT_EVENTS,
    SSEEvent,
    aiter_with_reconnect,
    parse_raw_event,
)
from dagnam._types import (
    JsonArray,
    JsonObject,
    JsonValue,
    QueryParams,
    ensure_json_array,
    ensure_json_object,
)


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
        idempotent: bool = False,
    ) -> JsonValue | str | None:
        resp = await self._request(
            method,
            path,
            params=params,
            json=json_body,
            timeout=timeout,
            raise_for=lambda r: raise_for_deployment(r, deployment_id or "deployment"),
            idempotent=idempotent,
        )
        if not resp.content:
            return None
        try:
            return response_json_value(resp)
        except ResponseError:
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
            await self._deployment_req(
                "POST", "/api/v1/deployments", json_body=payload, idempotent=True
            )
        )

    async def estimate_cost(self, payload: JsonObject) -> JsonObject:
        """Estimate deployment cost. ``POST /api/v1/deployments/estimate-cost``."""
        return ensure_json_object(
            await self._deployment_req(
                "POST", "/api/v1/deployments/estimate-cost", json_body=payload
            )
        )

    async def validate_deployment(self, payload: JsonObject) -> JsonObject:
        """Validate a deployment config. ``POST /api/v1/deployments/validate``."""
        return ensure_json_object(
            await self._deployment_req("POST", "/api/v1/deployments/validate", json_body=payload)
        )

    async def list_deployment_platforms(self) -> JsonArray:
        """List serving platforms. ``GET /api/v1/deployments-platforms`` (hyphenated sibling)."""
        return ensure_json_array(await self._deployment_req("GET", "/api/v1/deployments-platforms"))

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

    async def rollback_deployment(self, deployment_id: str, checkpoint_id: str) -> JsonObject:
        return ensure_json_object(
            await self._deployment_req(
                "POST",
                f"/api/v1/deployments/{quote_path_segment(deployment_id)}/rollback",
                deployment_id=deployment_id,
                params={"checkpoint_id": checkpoint_id},
            )
        )

    async def retry_deployment(self, deployment_id: str) -> JsonObject:
        """Retry a failed/stuck deployment. ``POST /api/v1/deployments/{id}/retry`` (no body)."""
        return ensure_json_object(
            await self._deployment_req(
                "POST",
                f"/api/v1/deployments/{quote_path_segment(deployment_id)}/retry",
                deployment_id=deployment_id,
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

    async def collect_deployment_metrics(
        self, deployment_id: str, backfill_minutes: int = 60
    ) -> JsonObject:
        """Trigger an immediate metrics collection (with first-time backfill).

        POST /api/v1/deployments/{id}/metrics/collect
        """
        return ensure_json_object(
            await self._deployment_req(
                "POST",
                f"/api/v1/deployments/{quote_path_segment(deployment_id)}/metrics/collect",
                deployment_id=deployment_id,
                params={"backfill_minutes": backfill_minutes},
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

    async def mint_deployment_stream_token(self, deployment_id: str) -> str:
        """Mint a short-lived stream-access token for one deployment's SSE stream."""
        body = ensure_json_object(
            await self._deployment_req(
                "POST",
                f"/api/v1/deployments/{quote_path_segment(deployment_id)}/stream-access-token",
                deployment_id=deployment_id,
            )
        )
        return str(body["token"])

    async def _open_deployment_stream(
        self, deployment_id: str, cursor: str | None
    ) -> AsyncIterator[SSEEvent]:
        """One connection's worth of deployment events (see the training twin)."""
        token = await self.mint_deployment_stream_token(deployment_id)
        dep_path = quote_path_segment(deployment_id)
        url = f"{self.api_url}/api/v1/streaming/deployments/{dep_path}/stream"
        headers = {"Accept": "text/event-stream"}
        if cursor:
            headers["Last-Event-ID"] = cursor
        try:
            async with aconnect_sse(
                self._client,
                "GET",
                url,
                params=stream_query_params(token),
                headers=headers,
                timeout=httpx.Timeout(self.timeout, read=SSE_READ_TIMEOUT),
            ) as event_source:
                response = event_source.response
                if not 200 <= response.status_code < 300:
                    await response.aread()
                    raise_for_deployment(response, deployment_id)
                async for sse in event_source.aiter_sse():
                    yield parse_raw_event(sse)
        except httpx.ConnectError as exc:
            raise APIError(0, f"Connection failed: {scrub_secret_params(str(exc))}") from exc
        except httpx.ConnectTimeout as exc:
            raise APIError(0, f"Request timed out: {scrub_secret_params(str(exc))}") from exc

    def stream_deployment_events(
        self, deployment_id: str, last_event_id: str | None = None
    ) -> AsyncIterator[SSEEvent]:
        """Yield parsed SSE events for a deployment, reconnecting transparently.

        Async counterpart to the sync ``open_deployment_stream``. A dropped
        connection is reconnected with a freshly minted token and the preserved
        ``Last-Event-ID`` cursor; the stream ends only on a terminal event, or
        raises ``StreamError`` after repeated failures — so a drop is never
        mistaken for the deployment finishing.

        ``GET /api/v1/streaming/deployments/{deployment_id}/stream?token=...``
        """
        return aiter_with_reconnect(
            lambda cursor: self._open_deployment_stream(deployment_id, cursor),
            terminal_events=TERMINAL_DEPLOYMENT_EVENTS,
            transient_errors=(httpx.TransportError, ConnectionError, OSError),
            resource_label=f"deployment stream {deployment_id}",
            last_event_id=last_event_id,
        )
