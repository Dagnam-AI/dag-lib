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
    response_json_object,
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
        idempotency_key: str | None = None,
    ) -> JsonValue | str | None:
        resp = await self._request(
            method,
            path,
            params=params,
            json=json_body,
            timeout=timeout,
            raise_for=lambda r: raise_for_deployment(r, deployment_id or "deployment"),
            idempotent=idempotent,
            idempotency_key=idempotency_key,
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

    async def _post_created_deployment(
        self, path: str, body: JsonObject, *, idempotency_key: str | None = None
    ) -> JsonObject:
        """POST a create route; restore the one-time key after a replay (see the sync twin)."""
        resp = await self._request(
            "POST",
            path,
            json=body,
            raise_for=lambda r: raise_for_deployment(r, "deployment"),
            idempotent=True,
            idempotency_key=idempotency_key,
        )
        created = response_json_object(resp)
        if resp.headers.get("Idempotency-Replayed") == "true" and created.get("api_key") is None:
            created.update(await self.rotate_deployment_key(str(created["id"])))
        return created

    async def create_deployment(self, payload: JsonObject) -> JsonObject:
        return await self._post_created_deployment("/api/v1/deployments", payload)

    async def deploy_model_version(
        self,
        model_version_id: str,
        *,
        name: str | None = None,
        project_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> JsonObject:
        """Deploy a registry model version. ``POST /api/v1/deployments/from-model-version``.

        Returns the deployment with its one-time ``api_key``; an
        ``Idempotency-Key`` is minted when none is given (see the sync twin).
        """
        body: JsonObject = {"model_version_id": model_version_id}
        if name is not None:
            body["name"] = name
        if project_id is not None:
            body["project_id"] = project_id
        return await self._post_created_deployment(
            "/api/v1/deployments/from-model-version", body, idempotency_key=idempotency_key
        )

    async def rotate_deployment_key(self, deployment_id: str) -> JsonObject:
        """Replace the deployment's key. ``POST /api/v1/deployments/{id}/rotate-key``."""
        return ensure_json_object(
            await self._deployment_req(
                "POST",
                f"/api/v1/deployments/{quote_path_segment(deployment_id)}/rotate-key",
                deployment_id=deployment_id,
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

    async def set_deployment_warm(self, deployment_id: str, warm: bool) -> JsonObject:
        """Pin or release a warm container. ``PATCH /api/v1/deployments/{id}/capacity``."""
        return ensure_json_object(
            await self._deployment_req(
                "PATCH",
                f"/api/v1/deployments/{quote_path_segment(deployment_id)}/capacity",
                deployment_id=deployment_id,
                json_body={"warm": warm},
            )
        )

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

    async def get_deployment_revisions(
        self, deployment_id: str, *, page: int = 1, limit: int = 50
    ) -> JsonArray:
        """GET /api/v1/deployments/{id}/revisions — revision history, newest first."""
        return ensure_json_array(
            await self._deployment_req(
                "GET",
                f"/api/v1/deployments/{quote_path_segment(deployment_id)}/revisions",
                deployment_id=deployment_id,
                params={"page": page, "limit": limit},
            )
        )

    async def create_deployment_revision(
        self,
        deployment_id: str,
        payload: JsonObject,
        *,
        idempotency_key: str | None = None,
    ) -> JsonObject:
        """POST /api/v1/deployments/{id}/revisions — roll out a new model version.

        An ``Idempotency-Key`` is required by the route and minted when absent.
        """
        return ensure_json_object(
            await self._deployment_req(
                "POST",
                f"/api/v1/deployments/{quote_path_segment(deployment_id)}/revisions",
                deployment_id=deployment_id,
                json_body=payload,
                idempotent=True,
                idempotency_key=idempotency_key,
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
        url = f"{self.api_url}/api/v1/deployments/{dep_path}/stream"
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

        ``GET /api/v1/deployments/{deployment_id}/stream?token=...``
        """
        return aiter_with_reconnect(
            lambda cursor: self._open_deployment_stream(deployment_id, cursor),
            terminal_events=TERMINAL_DEPLOYMENT_EVENTS,
            transient_errors=(httpx.TransportError, ConnectionError, OSError),
            resource_label=f"deployment stream {deployment_id}",
            last_event_id=last_event_id,
        )
