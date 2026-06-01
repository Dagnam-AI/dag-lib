"""Synchronous training client methods."""

from __future__ import annotations

from dagnam._core.client.base import (
    ALLOW_REDIRECTS,
    DEFAULT_TIMEOUT,
    APIError,
    BaseDagnamClient,
    is_success_response,
    requests,
    safe_error_body_from_response,
)
from dagnam._core.client.common import (
    quote_path_segment,
    raise_for_generic,
    requests_query_params,
    response_json_object,
    response_json_value,
)
from dagnam._core.exceptions import AuthError, TrainingJobNotFoundError
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
    ) -> JsonValue | str | None:
        """Issue an authenticated training-job request and decode the body.

        Maps 404 to :class:`TrainingJobNotFoundError` only when a ``job_id`` is
        supplied (collection routes have no job to miss); 401 to
        :class:`AuthError`; every other non-2xx to :class:`APIError` carrying the
        backend's detail message (e.g. tier-limit rejections on create).
        """
        url = f"{self.api_url}{path}"
        try:
            resp = requests.request(
                method,
                url,
                headers=self._headers(),
                params=requests_query_params(params),
                json=json_body,
                timeout=timeout,
                allow_redirects=ALLOW_REDIRECTS,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        raise_for_generic(resp, TrainingJobNotFoundError if job_id else None, job_id)

        if not resp.content:
            return None
        try:
            return response_json_value(resp)
        except ValueError:
            return resp.text

    def create_training_job(self, payload: JsonObject) -> JsonObject:
        """Create a platform training job. ``POST /api/v1/training/jobs``."""
        return self._expect_object(
            self._training_request("POST", "/api/v1/training/jobs", json_body=payload)
        )

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

        try:
            from importlib.metadata import PackageNotFoundError, version

            try:
                sdk_version = version("dagnam")
            except PackageNotFoundError:
                sdk_version = "0+unknown"

            job_path = quote_path_segment(job_id)
            url = f"{self.api_url}/api/v1/training/jobs/{job_path}/metrics/events"
            resp = requests.post(
                url,
                headers={
                    **self._headers(),
                    "Content-Type": "application/json",
                },
                json={
                    "events": events,
                    "source": source
                    or {
                        "kind": "local_attach",
                        "sdk_version": sdk_version,
                    },
                },
                timeout=DEFAULT_TIMEOUT,
                allow_redirects=ALLOW_REDIRECTS,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        self.raise_for_job_response(resp, job_id)
        return response_json_object(resp)

    def open_training_stream(
        self, job_id: str, last_event_id: str | None = None
    ) -> requests.Response:
        """Open an SSE stream for a training job.

        GET /api/v1/streaming/training-jobs/{job_id}/stream?api_key=...

        Returns the raw streaming Response; the caller is responsible for
        wrapping it (e.g. via sseclient-py) and closing it.
        """
        job_path = quote_path_segment(job_id)
        url = f"{self.api_url}/api/v1/streaming/training-jobs/{job_path}/stream"
        params = {"api_key": self.api_key}
        headers = {"Accept": "text/event-stream"}
        if last_event_id:
            headers["Last-Event-ID"] = last_event_id
        try:
            resp = requests.get(
                url,
                params=params,
                headers=headers,
                stream=True,
                timeout=DEFAULT_TIMEOUT,
                allow_redirects=ALLOW_REDIRECTS,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        if not is_success_response(resp):
            code = resp.status_code
            if code == 401:
                raise AuthError("Authentication failed: invalid or expired API key")
            if code == 404:
                raise TrainingJobNotFoundError(job_id)
            raise APIError(code, safe_error_body_from_response(resp))
        return resp
