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
from dagnam._core.client.common import quote_path_segment, response_json_object
from dagnam._core.exceptions import AuthError, TrainingJobNotFoundError


class TrainingClientMixin(BaseDagnamClient):
    """Training resource methods for DagnamClient."""

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
