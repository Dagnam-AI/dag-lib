"""Synchronous training client methods."""

from __future__ import annotations

from pathlib import Path

from dagnam._core.client.base import (
    ALLOW_REDIRECTS,
    DEFAULT_TIMEOUT,
    SSE_READ_TIMEOUT,
    STREAM_CONNECT_TIMEOUT,
    APIError,
    BaseDagnamClient,
    content_disposition_safe_name,
    is_success_response,
    requests,
    safe_error_body_from_response,
    scrub_secret_params,
)
from dagnam._core.client.common import (
    quote_path_segment,
    raise_for_generic,
    raise_for_training_job,
    requests_query_params,
    response_json_object,
    response_json_value,
    stream_query_params,
)
from dagnam._core.exceptions import AuthError, ResponseError, TrainingJobNotFoundError
from dagnam._types import JsonObject, JsonValue, QueryParams, QueryValue


class TrainingClientMixin(BaseDagnamClient):
    """Training resource methods for DagnamClient."""

    def _training_request(
        self,
        method: str,
        path: str,
        *,
        job_id: str | None = None,
        params: QueryParams | None = None,
        json_body: JsonValue = None,
        timeout: int = DEFAULT_TIMEOUT,
        idempotent: bool = False,
    ) -> JsonValue | str | None:
        """Issue an authenticated training-job request and decode the body.

        Maps 404 to :class:`TrainingJobNotFoundError` only when a ``job_id`` is
        supplied (collection routes have no job to miss); 401 to
        :class:`AuthError`; every other non-2xx to :class:`APIError` carrying the
        backend's detail message (e.g. tier-limit rejections on create).

        ``idempotent=True`` mints an ``Idempotency-Key`` for the POST so a
        transient failure retries into a server-side replay instead of creating
        a duplicate job (see the backend idempotency middleware).
        """
        url = f"{self.api_url}{path}"
        resp = self._request(
            method,
            url,
            raise_for=lambda r: raise_for_generic(
                r, TrainingJobNotFoundError if job_id else None, job_id
            ),
            params=requests_query_params(params),
            json=json_body,
            timeout=timeout,
            allow_redirects=ALLOW_REDIRECTS,
            idempotent=idempotent,
        )
        if not resp.content:
            return None
        try:
            return response_json_value(resp)
        except ResponseError:
            return resp.text

    def create_training_job(self, payload: JsonObject) -> JsonObject:
        """Create a platform training job. ``POST /api/v1/training/jobs``."""
        return self._expect_object(
            self._training_request(
                "POST", "/api/v1/training/jobs", json_body=payload, idempotent=True
            )
        )

    def register_local_run(
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
        return self._expect_object(
            self._training_request("POST", "/api/v1/training/jobs", json_body=payload)
        )

    def mint_run_token(self, job_id: str) -> JsonObject:
        """Mint or refresh a short-lived upload token for one run."""
        return self._expect_object(
            self._training_request(
                "POST",
                f"/api/v1/training/jobs/{quote_path_segment(job_id)}/stream-token",
                job_id=job_id,
            )
        )

    def mint_training_stream_token(self, job_id: str) -> str:
        """Mint a short-lived stream-access token for one training job's SSE stream."""
        body = self._expect_object(
            self._training_request(
                "POST",
                f"/api/v1/training/jobs/{quote_path_segment(job_id)}/stream-access-token",
                job_id=job_id,
            )
        )
        return str(body["token"])

    def get_training_job(self, job_id: str) -> JsonObject:
        """Fetch one training job. ``GET /api/v1/training/jobs/{id}``."""
        return self._expect_object(
            self._training_request(
                "GET",
                f"/api/v1/training/jobs/{quote_path_segment(job_id)}",
                job_id=job_id,
            )
        )

    def list_training_jobs(self, **filters: QueryValue) -> JsonObject:
        """List training jobs for the credential. ``GET /api/v1/training/jobs``."""
        return self._expect_object(
            self._training_request("GET", "/api/v1/training/jobs", params=filters)
        )

    def cancel_training_job(self, job_id: str) -> JsonObject:
        """Cancel a non-terminal job. ``POST /api/v1/training/jobs/{id}/cancel``."""
        return self._expect_object(
            self._training_request(
                "POST",
                f"/api/v1/training/jobs/{quote_path_segment(job_id)}/cancel",
                job_id=job_id,
            )
        )

    def restart_training_job(self, job_id: str) -> JsonObject:
        """Restart a terminal job. ``POST /api/v1/training/jobs/{id}/restart``."""
        return self._expect_object(
            self._training_request(
                "POST",
                f"/api/v1/training/jobs/{quote_path_segment(job_id)}/restart",
                job_id=job_id,
            )
        )

    def restore_from_checkpoint(self, job_id: str, checkpoint_id: str) -> JsonObject:
        """Restart a job from one of its checkpoints.

        ``POST /api/v1/training/jobs/{job_id}/checkpoints/{checkpoint_id}/restore``.
        """
        return self._expect_object(
            self._training_request(
                "POST",
                f"/api/v1/training/jobs/{quote_path_segment(job_id)}"
                f"/checkpoints/{quote_path_segment(checkpoint_id)}/restore",
                job_id=job_id,
            )
        )

    def estimate_training_resources(self, config: JsonObject) -> JsonObject:
        """Estimate compute cost for a training config.

        ``POST /api/v1/training/estimate-resources`` (a collection route: the
        body is a ``TrainingConfig`` dict and there is no job to miss).
        """
        return self._expect_object(
            self._training_request("POST", "/api/v1/training/estimate-resources", json_body=config)
        )

    def get_allowed_strategies(self) -> JsonObject:
        """List distribution strategies available to the credential.

        ``GET /api/v1/training/allowed-strategies`` returns a flat
        ``{strategy_label: available}`` map plus a registry-driven
        ``required_tiers`` ``{strategy_label: tier}`` entry naming the minimum
        tier that unlocks each *lockable* strategy (free strategies omitted).
        The whole object is returned untouched.
        """
        return self._expect_object(
            self._training_request("GET", "/api/v1/training/allowed-strategies")
        )

    def download_training_code(self, job_id: str, dest_dir: str | Path) -> Path:
        """Stream the generated training-code ZIP to a file inside ``dest_dir``.

        ``GET /api/v1/training/jobs/{id}/download-code``. The saved filename is
        taken from the response's ``Content-Disposition`` header and reduced to
        a bare basename (see ``content_disposition_safe_name``), so a hostile or
        malformed header can never write outside ``dest_dir``. The body is
        streamed straight to disk, never buffered in memory.
        """
        url = f"{self.api_url}/api/v1/training/jobs/{quote_path_segment(job_id)}/download-code"
        resp = self._get_stream(url)
        raise_for_generic(resp, TrainingJobNotFoundError, job_id)
        name = content_disposition_safe_name(
            resp.headers.get("Content-Disposition"), default=f"{job_id}-code.zip"
        )
        return self._stream_response_to_file(resp, Path(dest_dir) / name)

    def download_dag(self, job_id: str, dest_dir: str | Path) -> Path:
        """Stream a job's DAG JSON to a file inside ``dest_dir``.

        ``GET /api/v1/training/jobs/{id}/dag``. Mirrors
        :meth:`download_training_code`: the filename comes from the
        ``Content-Disposition`` header, reduced to a bare basename.
        """
        url = f"{self.api_url}/api/v1/training/jobs/{quote_path_segment(job_id)}/dag"
        resp = self._get_stream(url)
        raise_for_generic(resp, TrainingJobNotFoundError, job_id)
        name = content_disposition_safe_name(
            resp.headers.get("Content-Disposition"), default=f"{job_id}-dag.json"
        )
        return self._stream_response_to_file(resp, Path(dest_dir) / name)

    def bulk_delete_training_jobs(self, job_ids: list[str]) -> JsonObject:
        """Delete multiple jobs. ``POST /api/v1/training/jobs/bulk-delete``."""
        return self._expect_object(
            self._training_request(
                "POST",
                "/api/v1/training/jobs/bulk-delete",
                json_body={"job_ids": [str(job_id) for job_id in job_ids]},
            )
        )

    def get_training_logs(self, job_id: str, **filters: QueryValue) -> JsonObject:
        """Fetch paginated training logs for one job."""
        return self._expect_object(
            self._training_request(
                "GET",
                f"/api/v1/training/jobs/{quote_path_segment(job_id)}/logs",
                job_id=job_id,
                params=filters,
            )
        )

    def get_training_metrics(self, job_id: str, **filters: QueryValue) -> JsonObject:
        """Fetch paginated training metrics for one job."""
        return self._expect_object(
            self._training_request(
                "GET",
                f"/api/v1/training/jobs/{quote_path_segment(job_id)}/metrics",
                job_id=job_id,
                params=filters,
            )
        )

    def get_training_metrics_summary(self, job_id: str) -> JsonObject:
        """Fetch aggregate training metrics for one job."""
        return self._expect_object(
            self._training_request(
                "GET",
                f"/api/v1/training/jobs/{quote_path_segment(job_id)}/metrics/summary",
                job_id=job_id,
            )
        )

    def upload_training_events(
        self,
        job_id: str,
        events: list[dict],
        *,
        source: dict | None = None,
    ) -> dict:
        """Upload local training metrics events for an attached job."""
        if not events:
            return {"accepted": 0, "duplicates": 0}

        from importlib.metadata import PackageNotFoundError, version

        try:
            sdk_version = version("dagnam")
        except PackageNotFoundError:
            sdk_version = "0+unknown"

        job_path = quote_path_segment(job_id)
        url = f"{self.api_url}/api/v1/training/jobs/{job_path}/metrics/events"
        resp = self._request(
            "POST",
            url,
            raise_for=lambda r: raise_for_training_job(r, job_id),
            json={
                "events": events,
                "source": source or {"kind": "local_attach", "sdk_version": sdk_version},
            },
            headers={"Content-Type": "application/json"},
            allow_redirects=ALLOW_REDIRECTS,
        )
        return response_json_object(resp)

    def open_training_stream(
        self, job_id: str, last_event_id: str | None = None
    ) -> requests.Response:
        """Open an SSE stream for a training job.

        GET /api/v1/streaming/training-jobs/{job_id}/stream?token=...

        Returns the raw streaming Response; the caller is responsible for
        wrapping it (e.g. via sseclient-py) and closing it.
        """
        token = self.mint_training_stream_token(job_id)
        job_path = quote_path_segment(job_id)
        url = f"{self.api_url}/api/v1/streaming/training-jobs/{job_path}/stream"
        params = stream_query_params(token)
        headers = {"Accept": "text/event-stream"}
        if last_event_id:
            headers["Last-Event-ID"] = last_event_id
        try:
            resp = requests.get(
                url,
                params=params,
                headers=headers,
                stream=True,
                timeout=(STREAM_CONNECT_TIMEOUT, SSE_READ_TIMEOUT),
                allow_redirects=ALLOW_REDIRECTS,
            )
        # The SSE stream token rides in ``params`` (never in ``url``), so it is
        # requests/urllib3 that embeds the composed ``?token=…`` URL in the
        # exception text — scrub it before it reaches the error message.
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {scrub_secret_params(str(exc))}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {scrub_secret_params(str(exc))}") from exc

        if not is_success_response(resp):
            code = resp.status_code
            if code == 401:
                raise AuthError("Authentication failed: stream token rejected")
            if code == 404:
                raise TrainingJobNotFoundError(job_id)
            raise APIError(code, safe_error_body_from_response(resp))
        return resp
