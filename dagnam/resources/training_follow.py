"""A training stream that also ends when the job is paused.

Split from :mod:`dagnam.resources.training` to keep that module small. ``stream_training`` and
``get_training_job`` are looked up on that module at call time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import time
from typing import Optional

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import DagnamError
from dagnam._core.sse import SSEEvent, is_pause
from dagnam.resources import training
from dagnam.resources.training import TrainingEvent

PAUSE_CHECK_SECONDS = 60.0
"""How often :func:`follow_training` reads the job's status while a stream is open."""


def follow_training(
    job_id: str,
    *,
    include_heartbeats: bool = False,
    check_every: Optional[float] = None,
    clock: Callable[[], float] = time.monotonic,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> Iterator[TrainingEvent]:
    """Like :func:`stream_training`, but it also ends when the job is paused.

    A pause is recognised from the stream (a ``paused`` event, a ``status`` event whose
    ``new_status`` is ``paused``, or a ``stream_end`` whose ``reason`` is ``paused``) and from
    the job itself: its status is read before the stream is opened, so a job that is already
    paused is reported at once, and then at most every ``check_every`` seconds as events or
    heartbeats arrive, so a stream that says nothing about the pause cannot hold the caller
    for ever. A pause found that way is yielded as a ``paused`` event whose ``data["message"]``
    is the job's ``error_message``. A status that cannot be read is ignored: the stream decides.
    """

    def paused_job() -> TrainingEvent | None:
        try:
            job = training.get_training_job(job_id, client=client, api_key=api_key, api_url=api_url)
        except DagnamError:
            return None
        if job.get("status") != "paused":
            return None
        message = job.get("error_message")
        return SSEEvent(
            event="paused", data={"message": message if isinstance(message, str) else ""}
        )

    interval = PAUSE_CHECK_SECONDS if check_every is None else check_every
    found = paused_job()
    if found is not None:
        yield found
        return
    checked = clock()
    for event in training.stream_training(
        job_id,
        include_heartbeats=True,
        client=client,
        api_key=api_key,
        api_url=api_url,
    ):
        if event.event != "heartbeat" or include_heartbeats or is_pause(event):
            yield event
        if is_pause(event):
            return
        if clock() - checked >= interval:
            checked = clock()
            found = paused_job()
            if found is not None:
                yield found
                return


__all__ = ["PAUSE_CHECK_SECONDS", "follow_training"]
