"""Best-effort push of a mid-run checkpoint from a training script to the platform.

A run on the platform's own compute loses its container when it ends, so a checkpoint that is
only on its disk cannot be resumed. :func:`push_checkpoint` hands the newest one to the platform
while the run is still going. It is for the training loop, which must never be stopped by it:
whatever goes wrong is reported through ``log`` and answered with ``None``.

A refusal is remembered for the rest of the process, because every push sends its whole body
before the answer is read and the next checkpoint would be refused the same way: a rejected
credential, a missing route or a too-large checkpoint stops the pushing after one warning, and a
rate limit pauses it for as long as the platform asks.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import time
from typing import Protocol

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import (
    APIError,
    AuthError,
    PayloadTooLargeError,
    QuotaExceededError,
    TrainingJobNotFoundError,
    TrainingStateError,
)
from dagnam._core.tar_stream import DirectoryTar, FileChangedError, FileStream

_NOT_ACCEPTING = "not_accepting_checkpoints"
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")
_MAX_NAME = 120
_DEFAULT_API_URL = "https://api.dagnam.ai"
_DEFAULT_RETRY_AFTER = 300.0
_MAX_RETRY_AFTER = 3600.0
# Statuses that are about this one request, not about the run's standing with the platform: a
# request timeout and "another push is in flight". Every other 4xx will repeat on the next push.
_TRANSIENT_CLIENT_ERRORS = frozenset({408, 409})
_LOGGER = logging.getLogger("dagnam.checkpoint_push")

# The platform refused a push for a reason that will not change: stop asking for the rest of the
# process. Set after the one warning, or none for ``not_accepting_checkpoints`` (the job is simply
# not running any more).
_stopped = False
# After a rate limit: no push before this ``time.monotonic()`` instant.
_not_before = 0.0


class _Log(Protocol):
    def __call__(self, level: str, message: str) -> None: ...


def upload_name(path: Path, *, directory: bool) -> str:
    """A name the platform accepts (``[A-Za-z0-9._-]{1,120}``); it only reads the extension."""
    name = _UNSAFE_NAME_CHARS.sub("_", path.name) or "checkpoint"
    if directory:
        name += ".tar"
    return name[-_MAX_NAME:]


def _status(exc: Exception) -> int | None:
    """The HTTP status a failure stands for, or ``None`` for a transport or local failure."""
    if isinstance(exc, PayloadTooLargeError):
        return 413
    if isinstance(exc, QuotaExceededError):  # also InsufficientCreditsError; not an APIError
        return 402
    if isinstance(exc, AuthError):
        return 401
    if isinstance(exc, TrainingJobNotFoundError):
        return 404
    if isinstance(exc, APIError):
        return exc.status_code or None
    return None


def _why(exc: Exception) -> str:
    """What to say about a failure: a status or a type, never a URL, a token or a body."""
    if isinstance(exc, FileChangedError):
        return "a checkpoint file changed while it was being sent"
    status = _status(exc)
    if status is not None:
        return f"HTTP {status}"
    if isinstance(exc, APIError):
        return "network error"
    return type(exc).__name__


def _retry_after(exc: Exception) -> float:
    header = exc.retry_after_header if isinstance(exc, APIError) else None
    try:
        seconds = float(header) if header is not None else _DEFAULT_RETRY_AFTER
    except ValueError:
        seconds = _DEFAULT_RETRY_AFTER
    return min(max(seconds, 1.0), _MAX_RETRY_AFTER)


def _send(client: DagnamClient, job_id: str, path: Path, epoch: int, step: int) -> str | None:
    real = Path(os.path.realpath(path))
    body = DirectoryTar(real) if real.is_dir() else FileStream(real)
    if not len(body) or (isinstance(body, DirectoryTar) and not body.files):
        # Nothing to resume from, and an empty body would go out chunked, which the route refuses.
        _LOGGER.debug("checkpoint push skipped: nothing to send")
        return None
    name = upload_name(Path(os.path.abspath(path)), directory=isinstance(body, DirectoryTar))
    try:
        saved = client.push_checkpoint(job_id, body, epoch=epoch, step=step, name=name)
    except Exception as exc:
        # An HTTP client wraps what a body raises as a connection error; the file is the cause.
        if body.failure is not None:
            raise body.failure from exc
        raise
    checkpoint_id = saved.get("checkpoint_id")
    return checkpoint_id if isinstance(checkpoint_id, str) else None


def _remember(exc: Exception, epoch: int, step: int, log: _Log) -> None:
    """Remember what the platform's answer means for later pushes, and warn if that is news."""
    global _stopped, _not_before
    status = _status(exc)
    if isinstance(exc, TrainingStateError) and exc.reason == _NOT_ACCEPTING:
        _stopped = True
        return
    if status == 429:
        _not_before = time.monotonic() + _retry_after(exc)
        return
    final = status is not None and 400 <= status < 500 and status not in _TRANSIENT_CLIENT_ERRORS
    _stopped = _stopped or final
    log(
        level="WARNING",
        message=(
            f"Checkpoint for epoch {epoch}, step {step} was not saved to the platform "
            f"({_why(exc)}); "
            + (
                "no more checkpoints will be pushed from this run."
                if final
                else "training continues."
            )
        ),
    )


def push_checkpoint(
    path: str | os.PathLike[str], *, epoch: int, step: int, log: _Log
) -> str | None:
    """Send ``path`` (a regular file, or a directory as a tar) as this run's newest checkpoint.

    Returns the platform's checkpoint id, or ``None`` when nothing was saved: outside a platform
    run (``DAGNAM_INTERNAL`` set, or no ``DAGNAM_JOB_ID`` / ``DAGNAM_API_KEY``), when there is
    nothing to send, while the platform has asked for a pause or refused for good, or when the
    push failed. The credential is the run token in the environment, never a key set with
    ``dagnam.configure()``. A failure is a ``WARNING`` through ``log`` and the run carries on; a
    file that changes while it is being sent is a failure, never a padded or cut checkpoint.
    Only ``Exception`` is caught, so Ctrl+C still stops the run.
    """
    job_id = os.environ.get("DAGNAM_JOB_ID")
    token = os.environ.get("DAGNAM_API_KEY")
    if os.environ.get("DAGNAM_INTERNAL") or not job_id or not token:
        return None
    if _stopped or time.monotonic() < _not_before:
        return None
    try:
        client = DagnamClient(os.environ.get("DAGNAM_API_URL") or _DEFAULT_API_URL, token)
        return _send(client, job_id, Path(path), epoch, step)
    except Exception as exc:
        _remember(exc, epoch, step, log)
        return None
