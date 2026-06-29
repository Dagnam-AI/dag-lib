"""Async training client methods.

Async mirror of ``dagnam._core.client.training.TrainingClientMixin``: job
lifecycle (create / register-local / token / get / list / cancel / bulk-delete),
logs + metrics reads, local metrics-event upload, and a single-connection
Server-Sent-Events stream.

The connection/timeout wrapping that the sync mixin performs per call is handled
once by the shared ``_request`` transport, so the request helpers below only map
the response body. The SSE stream uses :func:`httpx_sse.aconnect_sse` (the async
counterpart to the sync ``sseclient`` path) and decodes events through the shared
:func:`dagnam._core.sse.parse_raw_event`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import cast

import httpx
from httpx_sse import aconnect_sse

from dagnam._core.aio.base import BaseAsyncDagnamClient, raise_for_job_response
from dagnam._core.client.common import (
    quote_path_segment,
    raise_for_generic,
    response_json_object,
    response_json_value,
)
from dagnam._core.exceptions import APIError, TrainingJobNotFoundError
from dagnam._core.sse import SSEEvent, parse_raw_event
from dagnam._types import (
    JsonObject,
    JsonValue,
    QueryParams,
    QueryValue,
    ensure_json_object,
)


class AsyncTrainingMixin(BaseAsyncDagnamClient):
    """Async Training resource methods for AsyncDagnamClient."""

    async def _training_req(
        self,
        method: str,
        path: str,
        *,
        job_id: str | None = None,
        params: QueryParams | None = None,
        json_body: JsonValue = None,
    ) -> JsonValue | str | None:
        """Issue an authenticated training request and decode the body.

        Maps 404 to :class:`TrainingJobNotFoundError` only when a ``job_id`` is
        supplied (collection routes have no job to miss); other non-2xx codes are
        mapped by :func:`raise_for_generic` (401 → auth, 402/413 → quota, else
        :class:`APIError` carrying the backend detail).
        """
        resp = await self._request(method, path, params=params, json=json_body)
        raise_for_generic(resp, TrainingJobNotFoundError if job_id else None, job_id)
        if not resp.content:
            return None
        try:
            return response_json_value(resp)
        except ValueError:
            return resp.text

    async def create_training_job(self, payload: JsonObject) -> JsonObject:
        """Create a platform training job. ``POST /api/v1/training/jobs``."""
        return ensure_json_object(
            await self._training_req("POST", "/api/v1/training/jobs", json_body=payload)
        )

    async def register_local_run(
        self,
        *,
        project_id: str,
        framework: str,
        config: JsonObject,
        max_duration_seconds: int | None = None,
    ) -> JsonObject:
        """Create a local-execution run that is never enqueued remotely."""
        payload: JsonObject = {
            "project_id": str(project_id),
            "framework": framework,
            "execution_mode": "local",
            "config": config,
        }
        if max_duration_seconds is not None:
            payload["max_duration_seconds"] = max_duration_seconds
        return ensure_json_object(
            await self._training_req("POST", "/api/v1/training/jobs", json_body=payload)
        )

    async def mint_run_token(self, job_id: str) -> JsonObject:
        """Mint or refresh a short-lived upload token for one run."""
        return ensure_json_object(
            await self._training_req(
                "POST",
                f"/api/v1/training/jobs/{quote_path_segment(job_id)}/stream-token",
                job_id=job_id,
            )
        )

    async def get_training_job(self, job_id: str) -> JsonObject:
        """Fetch one training job. ``GET /api/v1/training/jobs/{id}``."""
        return ensure_json_object(
            await self._training_req(
                "GET",
                f"/api/v1/training/jobs/{quote_path_segment(job_id)}",
                job_id=job_id,
            )
        )

    async def list_training_jobs(self, **filters: QueryValue) -> JsonObject:
        """List training jobs for the credential. ``GET /api/v1/training/jobs``."""
        return ensure_json_object(
            await self._training_req("GET", "/api/v1/training/jobs", params=filters)
        )

    async def cancel_training_job(self, job_id: str) -> JsonObject:
        """Cancel a non-terminal job. ``POST /api/v1/training/jobs/{id}/cancel``."""
        return ensure_json_object(
            await self._training_req(
                "POST",
                f"/api/v1/training/jobs/{quote_path_segment(job_id)}/cancel",
                job_id=job_id,
            )
        )

    async def bulk_delete_training_jobs(self, job_ids: list[str]) -> JsonObject:
        """Delete multiple jobs. ``POST /api/v1/training/jobs/bulk-delete``."""
        return ensure_json_object(
            await self._training_req(
                "POST",
                "/api/v1/training/jobs/bulk-delete",
                json_body={"job_ids": [str(job_id) for job_id in job_ids]},
            )
        )

    async def get_training_logs(self, job_id: str, **filters: QueryValue) -> JsonObject:
        """Fetch paginated training logs for one job."""
        return ensure_json_object(
            await self._training_req(
                "GET",
                f"/api/v1/training/jobs/{quote_path_segment(job_id)}/logs",
                job_id=job_id,
                params=filters,
            )
        )

    async def get_training_metrics(self, job_id: str, **filters: QueryValue) -> JsonObject:
        """Fetch paginated training metrics for one job."""
        return ensure_json_object(
            await self._training_req(
                "GET",
                f"/api/v1/training/jobs/{quote_path_segment(job_id)}/metrics",
                job_id=job_id,
                params=filters,
            )
        )

    async def get_training_metrics_summary(self, job_id: str) -> JsonObject:
        """Fetch aggregate training metrics for one job."""
        return ensure_json_object(
            await self._training_req(
                "GET",
                f"/api/v1/training/jobs/{quote_path_segment(job_id)}/metrics/summary",
                job_id=job_id,
            )
        )

    async def upload_training_events(
        self,
        job_id: str,
        events: list[dict],
        *,
        source: dict | None = None,
    ) -> JsonObject:
        """Upload local training metrics events for an attached job."""
        if not events:
            return {"accepted": 0, "duplicates": 0}

        from importlib.metadata import PackageNotFoundError, version

        try:
            sdk_version = version("dagnam")
        except PackageNotFoundError:
            sdk_version = "0+unknown"

        # ``events``/``source`` are caller-supplied, by-contract JSON-serializable
        # dicts (the signature mirrors the sync client); cast at this boundary so
        # the strictly-typed ``_request`` transport accepts them.
        payload = cast(
            "JsonValue",
            {
                "events": events,
                "source": source or {"kind": "local_attach", "sdk_version": sdk_version},
            },
        )
        resp = await self._request(
            "POST",
            f"/api/v1/training/jobs/{quote_path_segment(job_id)}/metrics/events",
            json=payload,
        )
        raise_for_job_response(resp, job_id)
        return response_json_object(resp)

    async def stream_training_events(
        self, job_id: str, last_event_id: str | None = None
    ) -> AsyncIterator[SSEEvent]:
        """Yield parsed SSE events for a training job over a single connection.

        Async counterpart to the sync ``open_training_stream`` (which returns the
        raw streaming response for the caller to parse); this decodes events for
        the caller via the shared :func:`parse_raw_event`. There is no
        auto-reconnect — reconnection is a future resource-layer concern.

        ``GET /api/v1/streaming/training-jobs/{job_id}/stream?api_key=...``
        """
        job_path = quote_path_segment(job_id)
        url = f"{self.api_url}/api/v1/streaming/training-jobs/{job_path}/stream"
        headers = {"Accept": "text/event-stream"}
        if last_event_id:
            headers["Last-Event-ID"] = last_event_id
        try:
            async with aconnect_sse(
                self._client,
                "GET",
                url,
                params={"api_key": self.api_key},
                headers=headers,
                timeout=self.timeout,
            ) as event_source:
                response = event_source.response
                if not 200 <= response.status_code < 300:
                    await response.aread()
                    raise_for_job_response(response, job_id)
                async for sse in event_source.aiter_sse():
                    yield parse_raw_event(sse)
        except httpx.ConnectError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc
