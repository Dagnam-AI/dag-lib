"""Checkpoint download with local caching.

Caches checkpoints under ``~/.dagnam/checkpoints/{job_id}/{checkpoint_id}.pt``
using the same integrity + LRU eviction model as the dataset cache, with
a separate root so dataset and checkpoint budgets don't collide.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from dagnam._core.client import DagnamClient
from dagnam._core.config import get_config_value
from dagnam._core.exceptions import CheckpointNotFoundError, ChecksumError
from dagnam._core.resolver import resolve_client
from dagnam._types import JsonObject
from dagnam.data.cache import cache_dir_name, compute_file_checksum, evict_lru, touch_cache

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_CACHE_DIR: Path = Path.home() / ".dagnam" / "checkpoints"


def pick_latest(checkpoints: list[JsonObject]) -> JsonObject:
    if not checkpoints:
        raise CheckpointNotFoundError("<no checkpoints for job>")

    def sort_key(c: JsonObject) -> tuple[int, int, str]:
        epoch = c.get("epoch", -1)
        step = c.get("step", -1)
        created_at = c.get("created_at", "")
        return (
            epoch if isinstance(epoch, int) else -1,
            step if isinstance(step, int) else -1,
            created_at if isinstance(created_at, str) else "",
        )

    return sorted(checkpoints, key=sort_key)[-1]


def pick_best(checkpoints: list[JsonObject]) -> JsonObject:
    """Return the newest checkpoint marked best, falling back to latest."""
    best = [checkpoint for checkpoint in checkpoints if checkpoint.get("is_best")]
    return pick_latest(best or checkpoints)


def download_checkpoint(
    job_id: str,
    checkpoint_id: Optional[str] = None,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    prefer_best: bool = False,
) -> Path:
    """Download (or fetch from cache) a checkpoint file.

    If ``checkpoint_id`` is None, the latest checkpoint is selected by
    epoch/step. Set ``prefer_best`` to select the best checkpoint first,
    falling back to the latest. Returns the local Path.

    Raises:
        TrainingJobNotFoundError: Job does not exist.
        CheckpointNotFoundError: Job has no matching checkpoint.
        ChecksumError: Downloaded file does not match server-reported SHA-256.
    """
    resolved = resolve_client(client, api_key, api_url)

    if checkpoint_id is None:
        checkpoints = resolved.list_checkpoints(job_id)
        picked = pick_best(checkpoints) if prefer_best else pick_latest(checkpoints)
        checkpoint_id = str(picked["id"])

    base = Path(cache_dir) if cache_dir is not None else DEFAULT_CHECKPOINT_CACHE_DIR
    job_dir = base / cache_dir_name(job_id)
    dest = job_dir / f"{cache_dir_name(checkpoint_id)}.pt"

    # Cache hit: bump access time and return. Integrity was verified at download.
    if dest.exists():
        touch_cache(job_id, base_dir=base)
        return dest

    job_dir.mkdir(parents=True, exist_ok=True)
    local_path, expected_sha = resolved.download_checkpoint_stream(job_id, checkpoint_id, dest)

    if expected_sha:
        actual = compute_file_checksum(local_path)
        if actual != expected_sha:
            try:
                local_path.unlink()
            except OSError:
                pass
            raise ChecksumError(
                f"Checkpoint checksum mismatch: expected {expected_sha}, got {actual}"
            )

    # Mark just-downloaded entry as most-recently-used BEFORE eviction so it
    # cannot evict itself when the budget is tight.
    touch_cache(job_id, base_dir=base)

    # Best-effort LRU eviction. Log and continue on failure; a download
    # succeeded and the caller should not see disk-maintenance errors.
    max_bytes = get_config_value("max_checkpoint_cache_size", None) or get_config_value(
        "max_cache_size", None
    )
    if max_bytes is not None:
        try:
            evict_lru(
                max_size_bytes=max_bytes if isinstance(max_bytes, int) else None,
                base_dir=base,
            )
        except Exception as exc:
            logger.warning("Checkpoint LRU eviction failed: %s", exc)

    return local_path
