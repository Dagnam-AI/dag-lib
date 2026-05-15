"""Synchronous checkpoints client methods."""

from __future__ import annotations

from pathlib import Path

from dagnam._core.client.base import (
    _ALLOW_REDIRECTS,
    _TIMEOUT,
    APIError,
    _is_success_response,
    _safe_error_body,
    requests,
)
from dagnam._core.client.common import quote_path_segment
from dagnam._core.exceptions import AuthError, CheckpointNotFoundError


class CheckpointsClientMixin:
    """Checkpoints resource methods for DagnamClient."""

    def list_checkpoints(self, job_id: str) -> list[dict]:
        """GET /api/v1/training/jobs/{job_id}/checkpoints"""
        job_path = quote_path_segment(job_id)
        url = f"{self.api_url}/api/v1/training/jobs/{job_path}/checkpoints"
        try:
            resp = requests.get(
                url,
                headers=self._headers(),
                timeout=_TIMEOUT,
                allow_redirects=_ALLOW_REDIRECTS,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc
        self._raise_for_job(resp, job_id)
        return resp.json()

    def download_checkpoint_stream(
        self, job_id: str, checkpoint_id: str, dest_path: Path
    ) -> tuple[Path, str | None]:
        """Stream-download a checkpoint file to dest_path.

        GET /api/v1/training/jobs/{job_id}/checkpoints/{checkpoint_id}/download

        Returns (dest_path, expected_sha256) — the caller must verify.
        """
        job_path = quote_path_segment(job_id)
        checkpoint_path = quote_path_segment(checkpoint_id)
        url = (
            f"{self.api_url}/api/v1/training/jobs/{job_path}/checkpoints/{checkpoint_path}/download"
        )
        resp = self._get_stream(url)

        if not _is_success_response(resp):
            code = resp.status_code
            if code == 401:
                raise AuthError("Authentication failed: invalid or expired API key")
            if code == 404:
                raise CheckpointNotFoundError(checkpoint_id)
            raise APIError(code, _safe_error_body(resp))

        expected_checksum = resp.headers.get("X-Checksum-SHA256")
        written = self._stream_response_to_file(resp, Path(dest_path))
        return written, expected_checksum
