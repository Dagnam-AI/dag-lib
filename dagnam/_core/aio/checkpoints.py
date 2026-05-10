"""Async checkpoints client methods."""

from __future__ import annotations

from pathlib import Path

from dagnam._core.aio.base import _raise_for_job
from dagnam._core.exceptions import APIError, AuthError, CheckpointNotFoundError


class AsyncCheckpointsMixin:
    """Async Checkpoints resource methods."""

    async def list_checkpoints(self, job_id: str) -> list[dict]:
        resp = await self._request("GET", f"/api/v1/training/jobs/{job_id}/checkpoints")
        _raise_for_job(resp, job_id)
        return resp.json()

    async def download_checkpoint(
        self, job_id: str, checkpoint_id: str, dest_path: Path
    ) -> tuple[Path, str | None]:
        resp = await self._request(
            "GET",
            f"/api/v1/training/jobs/{job_id}/checkpoints/{checkpoint_id}/download",
        )
        if not resp.is_success:
            code = resp.status_code
            if code == 401:
                raise AuthError("Authentication failed: invalid or expired API key")
            if code == 404:
                raise CheckpointNotFoundError(checkpoint_id)
            raise APIError(code, resp.text)
        expected_checksum = resp.headers.get("x-checksum-sha256")
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return dest, expected_checksum
