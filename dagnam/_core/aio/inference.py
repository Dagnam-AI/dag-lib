"""Async inference client methods."""

from __future__ import annotations

from collections.abc import AsyncIterator
import json

import httpx
from httpx_sse import aconnect_sse

from dagnam._core.aio.base import SSE_READ_TIMEOUT, BaseAsyncDagnamClient
from dagnam._core.client.base import scrub_secret_params
from dagnam._core.client.common import (
    quote_path_segment,
    raise_for_deployment,
    stream_query_params,
)
from dagnam._core.exceptions import APIError
from dagnam._core.sse import (
    TERMINAL_INFERENCE_EVENTS,
    SSEEvent,
    aiter_sse_once,
    parse_raw_event,
)
from dagnam._types import JsonArray, JsonObject


class AsyncInferenceMixin(BaseAsyncDagnamClient):
    """Async Inference resource methods."""

    async def predict(
        self, deployment_id: str, inputs: JsonObject, timeout: int | None = None
    ) -> JsonObject:
        resp = await self._request(
            "POST",
            f"/api/v1/inference/{quote_path_segment(deployment_id)}/predict",
            json=inputs,
            headers=self._headers(),
            timeout=timeout,
        )
        raise_for_deployment(resp, deployment_id)
        return resp.json()

    async def predict_batch(
        self, deployment_id: str, inputs: JsonArray, timeout: int | None = None
    ) -> JsonArray:
        resp = await self._request(
            "POST",
            f"/api/v1/inference/{quote_path_segment(deployment_id)}/predict/batch",
            json={"inputs": inputs},
            headers=self._headers(),
            timeout=timeout,
        )
        raise_for_deployment(resp, deployment_id)
        return resp.json()

    async def mint_inference_stream_token(self, deployment_id: str) -> str:
        """Mint a short-lived stream-access token for streaming inference."""
        resp = await self._request(
            "POST",
            f"/api/v1/inference/{quote_path_segment(deployment_id)}/stream-access-token",
            headers=self._headers(),
        )
        raise_for_deployment(resp, deployment_id)
        return str(resp.json()["token"])

    async def _open_inference_stream(
        self, deployment_id: str, inputs: JsonObject
    ) -> AsyncIterator[SSEEvent]:
        """One (and only) connection's worth of streaming-predict events."""
        token = await self.mint_inference_stream_token(deployment_id)
        dep_path = quote_path_segment(deployment_id)
        url = f"{self.api_url}/api/v1/inference/{dep_path}/predict/stream"
        params = {**stream_query_params(token), "input": json.dumps(inputs)}
        try:
            async with aconnect_sse(
                self._client,
                "GET",
                url,
                params=params,
                headers={"Accept": "text/event-stream"},
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

    def stream_predict(self, deployment_id: str, inputs: JsonObject) -> AsyncIterator[SSEEvent]:
        """Yield streaming-predict SSE events (``token`` … ``complete``/``error``).

        Single-shot by design: an inference stream is not resumable, so a
        transport drop raises ``StreamError`` instead of reconnecting (a
        reconnect would replay generation from the start).
        """
        return aiter_sse_once(
            lambda: self._open_inference_stream(deployment_id, inputs),
            terminal_events=TERMINAL_INFERENCE_EVENTS,
            transient_errors=(httpx.TransportError, ConnectionError, OSError),
            resource_label=f"inference stream {deployment_id}",
        )

    async def deployment_health(self, deployment_id: str) -> JsonObject:
        resp = await self._request(
            "GET",
            f"/api/v1/inference/{quote_path_segment(deployment_id)}/health",
            headers=self._headers(),
        )
        raise_for_deployment(resp, deployment_id)
        return resp.json()

    async def schema(self, deployment_id: str, timeout: int | None = None) -> JsonObject:
        """Return a deployment's inference schema. ``GET /api/v1/inference/{id}/schema``."""
        resp = await self._request(
            "GET",
            f"/api/v1/inference/{quote_path_segment(deployment_id)}/schema",
            headers=self._headers(),
            timeout=timeout,
        )
        raise_for_deployment(resp, deployment_id)
        return resp.json()
