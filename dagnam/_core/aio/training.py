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
from pathlib import Path
from typing import cast

import httpx
from httpx_sse import aconnect_sse

from dagnam._core.aio.base import (
    BaseAsyncDagnamClient,
    content_disposition_safe_name,
    raise_for_job_response,
)
from dagnam._core.client.common import (
    quote_path_segment,
    raise_for_generic,
    response_json_object,
    response_json_value,
    stream_query_params,
)
from dagnam._core.exceptions import APIError, TrainingJobNotFoundError
from dagnam._core.sse import (
    TERMINAL_TRAINING_EVENTS,
    SSEEvent,
    aiter_with_reconnect,
    parse_raw_event,
)
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

    async def mint_training_stream_token(self, job_id: str) -> str:
        """Mint a short-lived stream-access token for one training job's SSE stream."""
        body = ensure_json_object(
            await self._training_req(
                "POST",
                f"/api/v1/training/jobs/{quote_path_segment(job_id)}/stream-access-token",
                job_id=job_id,
            )
        )
        return str(body["token"])

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

    async def restart_training_job(self, job_id: str) -> JsonObject:
        """Restart a terminal job. ``POST /api/v1/training/jobs/{id}/restart``."""
        return ensure_json_object(
            await self._training_req(
                "POST",
                f"/api/v1/training/jobs/{quote_path_segment(job_id)}/restart",
                job_id=job_id,
            )
        )

    async def restore_from_checkpoint(self, job_id: str, checkpoint_id: str) -> JsonObject:
        """Restart a job from one of its checkpoints.

        ``POST /api/v1/training/jobs/{job_id}/checkpoints/{checkpoint_id}/restore``.
        """
        return ensure_json_object(
            await self._training_req(
                "POST",
                f"/api/v1/training/jobs/{quote_path_segment(job_id)}"
                f"/checkpoints/{quote_path_segment(checkpoint_id)}/restore",
                job_id=job_id,
            )
        )

    async def estimate_training_resources(self, config: JsonObject) -> JsonObject:
        """Estimate compute cost for a training config.

        ``POST /api/v1/training/estimate-resources`` (a collection route: the
        body is a ``TrainingConfig`` dict and there is no job to miss).
        """
        return ensure_json_object(
            await self._training_req(
                "POST", "/api/v1/training/estimate-resources", json_body=config
            )
        )

    async def get_allowed_strategies(self) -> JsonObject:
        """List distribution strategies available to the credential.

        ``GET /api/v1/training/allowed-strategies`` returns a flat
        ``dict[str, bool]`` mapping each strategy label to its availability.
        """
        return ensure_json_object(
            await self._training_req("GET", "/api/v1/training/allowed-strategies")
        )

    async def _stream_training_download(
        self, url: str, dest_dir: str | Path, *, job_id: str, default: str
    ) -> Path:
        """Stream a job download to a basename-only file inside ``dest_dir``.

        Shared by :meth:`download_training_code` and :meth:`download_dag`: maps a
        404 to :class:`TrainingJobNotFoundError`, reduces the
        ``Content-Disposition`` filename to a bare basename so a hostile header
        can never escape ``dest_dir``, and streams the body chunk by chunk.
        """
        try:
            async with self._client.stream("GET", url, headers=self._headers()) as resp:
                if not resp.is_success:
                    await resp.aread()  # populate the body for the error message
                    raise_for_generic(resp, TrainingJobNotFoundError, job_id)
                name = content_disposition_safe_name(
                    resp.headers.get("content-disposition"), default=default
                )
                dest = Path(dest_dir) / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as fh:
                    async for chunk in resp.aiter_bytes():
                        fh.write(chunk)
                return dest
        except httpx.ConnectError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

    async def download_training_code(self, job_id: str, dest_dir: str | Path) -> Path:
        """Stream the generated training-code ZIP to a file inside ``dest_dir``.

        ``GET /api/v1/training/jobs/{id}/download-code``. Async mirror of the
        sync ``download_training_code``.
        """
        url = f"{self.api_url}/api/v1/training/jobs/{quote_path_segment(job_id)}/download-code"
        return await self._stream_training_download(
            url, dest_dir, job_id=job_id, default=f"{job_id}-code.zip"
        )

    async def download_dag(self, job_id: str, dest_dir: str | Path) -> Path:
        """Stream a job's DAG JSON to a file inside ``dest_dir``.

        ``GET /api/v1/training/jobs/{id}/dag``. Async mirror of the sync
        ``download_dag``.
        """
        url = f"{self.api_url}/api/v1/training/jobs/{quote_path_segment(job_id)}/dag"
        return await self._stream_training_download(
            url, dest_dir, job_id=job_id, default=f"{job_id}-dag.json"
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

    async def _open_training_stream(
        self, job_id: str, cursor: str | None
    ) -> AsyncIterator[SSEEvent]:
        """One connection's worth of training events.

        Mints a fresh stream token, connects, and yields parsed events. A
        connect-time ConnectError/timeout is translated to ``APIError`` (which
        the reconnect loop treats as non-transient and surfaces immediately); a
        mid-stream transport drop propagates as an ``httpx.TransportError`` so
        :func:`aiter_with_reconnect` reconnects.
        """
        token = await self.mint_training_stream_token(job_id)
        job_path = quote_path_segment(job_id)
        url = f"{self.api_url}/api/v1/streaming/training-jobs/{job_path}/stream"
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
        except httpx.ConnectTimeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

    def stream_training_events(
        self, job_id: str, last_event_id: str | None = None
    ) -> AsyncIterator[SSEEvent]:
        """Yield parsed SSE events for a training job, reconnecting transparently.

        Async counterpart to the sync ``open_training_stream``. A dropped
        connection (LB idle timeout, network blip, or the server closing when a
        short-lived stream token expires) is reconnected with a freshly minted
        token and the preserved ``Last-Event-ID`` cursor, so a multi-hour stream
        survives; it ends only on a terminal event, or raises ``StreamError``
        after repeated failures. A dropped stream is therefore never mistaken
        for normal completion.

        ``GET /api/v1/streaming/training-jobs/{job_id}/stream?token=...``
        """
        return aiter_with_reconnect(
            lambda cursor: self._open_training_stream(job_id, cursor),
            terminal_events=TERMINAL_TRAINING_EVENTS,
            transient_errors=(httpx.TransportError, ConnectionError, OSError),
            resource_label=f"training stream {job_id}",
            last_event_id=last_event_id,
        )
