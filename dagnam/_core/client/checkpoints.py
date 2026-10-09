"""Synchronous checkpoints client methods."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import BinaryIO

from dagnam._core.client.base import (
    ALLOW_REDIRECTS,
    APIError,
    BaseDagnamClient,
    is_redirect_response,
    is_success_response,
    safe_error_body_from_response,
)
from dagnam._core.client.common import (
    quote_path_segment,
    raise_for_training_job,
    raise_for_training_state,
    response_json_object,
)
from dagnam._core.exceptions import AuthError, CheckpointNotFoundError
from dagnam._types import JsonObject, ensure_json_array

PUSH_TIMEOUT = (10, 900)
"""Connect and read timeouts, in seconds, for a checkpoint push.

The platform answers only after it has stored the bytes, and it holds the run's one push slot
for 900 s, so there is nothing to wait for past that.
"""


class CheckpointsClientMixin(BaseDagnamClient):
    """Checkpoints resource methods for DagnamClient."""

    def list_checkpoints(self, job_id: str) -> list[JsonObject]:
        """GET /api/v1/training/jobs/{job_id}/checkpoints"""
        job_path = quote_path_segment(job_id)
        url = f"{self.api_url}/api/v1/training/jobs/{job_path}/checkpoints"
        resp = self._request(
            "GET",
            url,
            raise_for=lambda r: raise_for_training_job(r, job_id),
            allow_redirects=ALLOW_REDIRECTS,
        )
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

    def push_checkpoint(
        self,
        job_id: str,
        body: BinaryIO | Iterable[bytes],
        *,
        epoch: int,
        step: int,
        name: str,
    ) -> JsonObject:
        """Send a run's checkpoint, raw, while the run is still going.

        ``POST /api/v1/training/jobs/{job_id}/checkpoints/push?epoch=&step=&name=``. ``body`` is
        an open binary file, or a sized iterable of bytes such as
        :class:`~dagnam._core.tar_stream.DirectoryTar` for a directory; either is streamed, never
        read into memory, and goes with a ``Content-Length`` (the platform refuses a push without
        one). ``name`` supplies the stored file's extension. Authenticates with the run token the
        client was built with, so it can push for its own job only.

        Returns ``{"checkpoint_id", "epoch", "step", "size_bytes", "sha256"}``. Not retried: the
        body is consumed, and the run's next checkpoint is the retry. A 409 raises
        :class:`~dagnam.TrainingStateError` (``reason`` ``not_accepting_checkpoints`` when the
        job is no longer running), a 413 :class:`~dagnam.PayloadTooLargeError`; a 429, 5xx or
        transport failure an :class:`~dagnam.APIError`.
        """
        resp = self._request(
            "POST",
            f"{self.api_url}/api/v1/training/jobs/{quote_path_segment(job_id)}/checkpoints/push",
            raise_for=lambda r: raise_for_training_state(r, job_id),
            params={"epoch": epoch, "step": step, "name": name},
            data=body,
            headers={"Content-Type": "application/octet-stream"},
            timeout=PUSH_TIMEOUT,
            allow_redirects=ALLOW_REDIRECTS,
            retry=False,
        )
        return response_json_object(resp)
