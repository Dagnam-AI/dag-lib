"""Async checkpoints client methods."""

from __future__ import annotations

from pathlib import Path

from dagnam._types import JsonObject, ensure_json_array, ensure_json_object
from dagnam._core.aio.base import BaseAsyncDagnamClient, raise_for_job_response
from dagnam._core.client.common import quote_path_segment, safe_response_text
from dagnam._core.exceptions import APIError, AuthError, CheckpointNotFoundError


class AsyncCheckpointsMixin(BaseAsyncDagnamClient):
    """Async Checkpoints resource methods."""

    async def list_checkpoints(self, job_id: str) -> list[JsonObject]:
        resp = await self._request(
            "GET", f"/api/v1/training/jobs/{quote_path_segment(job_id)}/checkpoints"
        )
        raise_for_job_response(resp, job_id)
        return [ensure_json_object(item) for item in ensure_json_array(resp.json())]

    async def download_checkpoint(
        self, job_id: str, checkpoint_id: str, dest_path: Path
    ) -> tuple[Path, str | None]:
        resp = await self._request(
            "GET",
            (
                f"/api/v1/training/jobs/{quote_path_segment(job_id)}"
                f"/checkpoints/{quote_path_segment(checkpoint_id)}/download"
            ),
        )
        if not resp.is_success:
            code = resp.status_code
            if code == 401:
                raise AuthError("Authentication failed: invalid or expired API key")
            if code == 404:
                raise CheckpointNotFoundError(checkpoint_id)
            raise APIError(code, safe_response_text(resp))
        expected_checksum = resp.headers.get("x-checksum-sha256")
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return dest, expected_checksum
