"""Synchronous checkpoints client methods."""

from __future__ import annotations

from pathlib import Path

from dagnam._core.client.base import (
    _TIMEOUT,
    APIError,
    requests,
)
from dagnam._core.exceptions import AuthError, CheckpointNotFoundError


class CheckpointsClientMixin:
    """Checkpoints resource methods for DagnamClient."""

    def list_checkpoints(self, job_id: str) -> list[dict]:
        """GET /api/v1/training/jobs/{job_id}/checkpoints"""
        url = f"{self.api_url}/api/v1/training/jobs/{job_id}/checkpoints"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=_TIMEOUT)
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
        url = f"{self.api_url}/api/v1/training/jobs/{job_id}/checkpoints/{checkpoint_id}/download"
        resp = self._get_stream(url)

        if not resp.ok:
            code = resp.status_code
            if code == 401:
                raise AuthError("Authentication failed: invalid or expired API key")
            if code == 404:
                raise CheckpointNotFoundError(checkpoint_id)
            raise APIError(code, resp.text)

        expected_checksum = resp.headers.get("X-Checksum-SHA256")
        written = self._stream_response_to_file(resp, Path(dest_path))
        return written, expected_checksum
