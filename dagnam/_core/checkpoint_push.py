"""Best-effort push of a mid-run checkpoint from a training script to the platform.

A run on the platform's own compute loses its container when it ends, so a checkpoint that is
only on its disk cannot be resumed. :func:`push_checkpoint` hands the newest one to the platform
while the run is still going. It is for the training loop, which must never be stopped by it:
whatever goes wrong is reported through ``log`` and answered with ``None``.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Protocol

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import APIError, PayloadTooLargeError, TrainingStateError
from dagnam._core.resolver import resolve_client
from dagnam._core.tar_stream import DirectoryTar

_NOT_ACCEPTING = "not_accepting_checkpoints"
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")
_MAX_NAME = 120

# The platform said it will take no more checkpoints from this run (it is not running any
# more): stop asking for the rest of the process, quietly.
_stopped = False


class _Log(Protocol):
    def __call__(self, level: str, message: str) -> None: ...


def upload_name(path: Path, *, directory: bool) -> str:
    """A name the platform accepts (``[A-Za-z0-9._-]{1,120}``); it only reads the extension."""
    name = _UNSAFE_NAME_CHARS.sub("_", path.name) or "checkpoint"
    if directory:
        name += ".tar"
    return name[-_MAX_NAME:]


def _why(exc: Exception) -> str:
    """What to say about a failure: a status or a type, never a URL, a token or a body."""
    if isinstance(exc, PayloadTooLargeError):
        return "HTTP 413"
    if isinstance(exc, APIError):
        return f"HTTP {exc.status_code}" if exc.status_code else "network error"
    return type(exc).__name__


def _send(client: DagnamClient, job_id: str, path: Path, epoch: int, step: int) -> str | None:
    if path.is_dir():
        tar = DirectoryTar(path)
        saved = client.push_checkpoint(
            job_id, tar, epoch=epoch, step=step, name=upload_name(path, directory=True)
        )
    else:
        with path.open("rb") as source:
            saved = client.push_checkpoint(
                job_id, source, epoch=epoch, step=step, name=upload_name(path, directory=False)
            )
    checkpoint_id = saved.get("checkpoint_id")
    return checkpoint_id if isinstance(checkpoint_id, str) else None


def push_checkpoint(
    path: str | os.PathLike[str], *, epoch: int, step: int, log: _Log
) -> str | None:
    """Send ``path`` (a file, or a directory as a tar) as this run's newest checkpoint.

    Returns the platform's checkpoint id, or ``None`` when nothing was saved: outside a platform
    run (``DAGNAM_INTERNAL`` set, or no ``DAGNAM_JOB_ID``), or when the push failed. A failure is
    a ``WARNING`` through ``log`` and the run carries on; the next checkpoint is another chance.
    Only ``Exception`` is caught, so Ctrl+C still stops the run.
    """
    global _stopped
    job_id = os.environ.get("DAGNAM_JOB_ID")
    if os.environ.get("DAGNAM_INTERNAL") or not job_id or _stopped:
        return None
    try:
        return _send(resolve_client(), job_id, Path(os.path.abspath(path)), epoch, step)
    except Exception as exc:
        if isinstance(exc, TrainingStateError) and exc.reason == _NOT_ACCEPTING:
            _stopped = True
        else:
            log(
                level="WARNING",
                message=(
                    f"Checkpoint for epoch {epoch}, step {step} was not saved to the platform "
                    f"({_why(exc)}); training continues."
                ),
            )
        return None
