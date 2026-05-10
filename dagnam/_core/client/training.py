"""Synchronous training client methods."""

from __future__ import annotations

from dagnam._core.client.base import (
    _TIMEOUT,
    APIError,
    requests,
)
from dagnam._core.exceptions import AuthError, TrainingJobNotFoundError


class TrainingClientMixin:
    """Training resource methods for DagnamClient."""

    def open_training_stream(
        self, job_id: str, last_event_id: str | None = None
    ) -> requests.Response:
        """Open an SSE stream for a training job.

        GET /api/v1/streaming/training-jobs/{job_id}/stream?api_key=...

        Returns the raw streaming Response; the caller is responsible for
        wrapping it (e.g. via sseclient-py) and closing it.
        """
        url = f"{self.api_url}/api/v1/streaming/training-jobs/{job_id}/stream"
        params = {"api_key": self.api_key}
        headers = {"Accept": "text/event-stream"}
        if last_event_id:
            headers["Last-Event-ID"] = last_event_id
        try:
            resp = requests.get(url, params=params, headers=headers, stream=True, timeout=_TIMEOUT)
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        if not resp.ok:
            code = resp.status_code
            if code == 401:
                raise AuthError("Authentication failed: invalid or expired API key")
            if code == 404:
                raise TrainingJobNotFoundError(job_id)
            raise APIError(code, resp.text)
        return resp
