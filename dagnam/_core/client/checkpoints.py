"""Synchronous checkpoints client methods."""

from __future__ import annotations

from pathlib import Path

from dagnam._core.client.base import (
    ALLOW_REDIRECTS,
    DEFAULT_TIMEOUT,
    APIError,
    BaseDagnamClient,
    is_redirect_response,
    is_success_response,
    requests,
    safe_error_body_from_response,
)
from dagnam._core.client.common import quote_path_segment
from dagnam._core.exceptions import AuthError, CheckpointNotFoundError
from dagnam._types import JsonObject, ensure_json_array


class CheckpointsClientMixin(BaseDagnamClient):
    """Checkpoints resource methods for DagnamClient."""

    def list_checkpoints(self, job_id: str) -> list[JsonObject]:
        """GET /api/v1/training/jobs/{job_id}/checkpoints"""
        job_path = quote_path_segment(job_id)
        url = f"{self.api_url}/api/v1/training/jobs/{job_path}/checkpoints"
        try:
            resp = requests.get(
                url,
                headers=self._headers(),
                timeout=DEFAULT_TIMEOUT,
                allow_redirects=ALLOW_REDIRECTS,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc
        self.raise_for_job_response(resp, job_id)
        return [item for item in ensure_json_array(resp.json()) if isinstance(item, dict)]

    def download_checkpoint_stream(
        self, job_id: str, checkpoint_id: str, dest_path: Path
    ) -> tuple[Path, str | None]:
        """Stream-download a checkpoint file to dest_path.

        GET /api/v1/training/jobs/{job_id}/checkpoints/{checkpoint_id}/download

        The backend may either stream the file bytes directly (local storage) or
        respond with a 307/308 redirect whose ``Location`` is a presigned
        object-storage URL serving the bytes. Both are handled; the presigned URL
        is fetched WITHOUT the API key.

        Returns (dest_path, expected_sha256) — the caller must verify.
        """
        job_path = quote_path_segment(job_id)
        checkpoint_path = quote_path_segment(checkpoint_id)
        url = (
            f"{self.api_url}/api/v1/training/jobs/{job_path}/checkpoints/{checkpoint_path}/download"
        )
        resp = self._get_stream(url)

        # The checksum may ride on the redirect response or the final body
        # response; prefer the one on whichever response carries the bytes,
        # falling back to the redirect's header.
        expected_checksum = resp.headers.get("X-Checksum-SHA256")

        if is_redirect_response(resp):
            location = resp.headers["Location"]
            resp.close()
            resp = self._get_stream_no_auth(location)
            expected_checksum = resp.headers.get("X-Checksum-SHA256") or expected_checksum

        if not is_success_response(resp):
            code = resp.status_code
            if code == 401:
                raise AuthError("Authentication failed: invalid or expired API key")
            if code == 404:
                raise CheckpointNotFoundError(checkpoint_id)
            raise APIError(code, safe_error_body_from_response(resp))

        written = self._stream_response_to_file(resp, Path(dest_path))
        return written, expected_checksum
