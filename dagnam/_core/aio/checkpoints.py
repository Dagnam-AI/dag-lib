"""Async checkpoints client methods."""

from __future__ import annotations

from pathlib import Path

import httpx

from dagnam._core.aio.base import BaseAsyncDagnamClient
from dagnam._core.client.base import scrub_secret_params
from dagnam._core.client.common import (
    quote_path_segment,
    raise_for_training_job,
    safe_response_text,
)
from dagnam._core.exceptions import APIError, AuthError, CheckpointNotFoundError
from dagnam._types import JsonObject, ensure_json_array, ensure_json_object


class AsyncCheckpointsMixin(BaseAsyncDagnamClient):
    """Async Checkpoints resource methods."""

    async def list_checkpoints(self, job_id: str) -> list[JsonObject]:
        resp = await self._request(
            "GET",
            f"/api/v1/training/jobs/{quote_path_segment(job_id)}/checkpoints",
            raise_for=lambda r: raise_for_training_job(r, job_id),
        )
        return [ensure_json_object(item) for item in ensure_json_array(resp.json())]

    async def _raise_for_checkpoint(self, resp: httpx.Response, checkpoint_id: str) -> None:
        await resp.aread()  # populate the body for the error message
        code = resp.status_code
        if code == 401:
            raise AuthError("Authentication failed: invalid or expired API key")
        if code == 404:
            raise CheckpointNotFoundError(checkpoint_id)
        raise APIError(code, safe_response_text(resp))

    async def download_checkpoint(
        self, job_id: str, checkpoint_id: str, dest_path: Path
    ) -> tuple[Path, str | None]:
        """Download a checkpoint file to ``dest_path``.

        The backend may either stream the bytes directly (local storage) or
        respond with a 307/308 redirect whose ``Location`` is a presigned
        object-storage URL serving the bytes. Both paths are streamed straight to
        disk (never buffered in memory — a large checkpoint would OOM) and the
        presigned URL is fetched WITHOUT the API key.

        Returns ``(dest_path, expected_sha256)``. **The caller MUST verify the
        digest** — unlike the sync ``dagnam.resources.checkpoints.download_checkpoint``
        wrapper, this low-level async method does not compute or compare the
        checksum itself, and a checkpoint is ``torch.load``'d downstream (a pickle
        code-execution sink). If ``expected_sha256`` is ``None`` the server sent no
        checksum (e.g. an S3 presigned redirect); a security-sensitive caller
        should refuse an unverified checkpoint rather than load it. Compute the
        local digest with ``dagnam.data.cache.compute_file_checksum``.
        """
        path = (
            f"/api/v1/training/jobs/{quote_path_segment(job_id)}"
            f"/checkpoints/{quote_path_segment(checkpoint_id)}/download"
        )
        url = f"{self.api_url}{path}"
        dest = Path(dest_path)
        # The checksum may ride on the redirect response or the final body
        # response; prefer the one on whichever response carries the bytes.
        location: str | None = None
        expected_checksum: str | None = None
        try:
            async with self._client.stream("GET", url, headers=self._headers()) as resp:
                expected_checksum = resp.headers.get("x-checksum-sha256")
                if resp.is_redirect and resp.headers.get("location"):
                    location = resp.headers["location"]
                else:
                    if not resp.is_success:
                        await self._raise_for_checkpoint(resp, checkpoint_id)
                    await self._stream_response_to_file(resp, dest)
                    return dest, expected_checksum

            # Presigned redirect: stream the object-storage URL without auth
            # (the signature is in the query string, never a forwarded header).
            async with self._client.stream("GET", location) as resp:
                expected_checksum = resp.headers.get("x-checksum-sha256") or expected_checksum
                if not resp.is_success:
                    await self._raise_for_checkpoint(resp, checkpoint_id)
                await self._stream_response_to_file(resp, dest)
        # The presigned signature rides in the URL query; scrub it out of any
        # transport-error text before it reaches the message.
        except httpx.ConnectError as exc:
            raise APIError(0, f"Connection failed: {scrub_secret_params(str(exc))}") from exc
        except httpx.TimeoutException as exc:
            raise APIError(0, f"Request timed out: {scrub_secret_params(str(exc))}") from exc
        return dest, expected_checksum
