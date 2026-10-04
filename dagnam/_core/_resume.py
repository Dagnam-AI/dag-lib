"""Resumable creates: an ``Idempotency-Key`` derived from the request, not minted per call.

The platform caches a create's terminal answer under its key for 24 hours and
replays it to the same key and body. A random key protects one call: a process
that dies after the platform committed asks again under a new key and gets a
second resource. A key derived from the request itself is the same in every
process, so the second ask replays the first answer.

Two things keep that from swallowing an ask that is meant:

- The key is a digest of the method, the URL and the body, so a different
  request is a different create. It is only right for a body that identifies
  its create -- one naming ids minted for this caller alone.
- The platform caches a refusal as it does a success, so a key that never
  changed would replay a refusal for a day after its cause was fixed (credits
  topped up, say). A replayed refusal therefore answers an *earlier* ask, and
  is stepped past: the next key in the sequence is either a fresh ask or the
  create an interrupted process lost. A refusal heard first-hand is the answer.
- A replayed success can name a resource that has been deleted since (the owner
  removed the orphan an interrupted run left). It is therefore confirmed by a read
  of the resource, and one that is gone is stepped past like a replayed refusal.
- A replay can come back without its body, pointing at the resource it made
  (:func:`created_body` reads it back).
- The platform answers 409 while the first ask of a key is still in progress, and
  holds that marker for up to two minutes if the handler died with the process. An
  immediate rerun is not an error: it waits, under the same key, for the marker to
  clear, and then the platform either replays the finished answer or runs the ask.
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from typing import Any
import uuid

from dagnam._core.exceptions import APIError, DagnamError

RESUME_ATTEMPTS = 8
"""How many replayed refusals a resumable create steps past before it asks under a random key."""
IN_PROGRESS_SECONDS = 120.0
"""How long a create waits out a 409 "in progress": the platform's marker outlives a dead handler this long."""
IN_PROGRESS_POLL = 5.0
CONFLICT = 409
NOT_FOUND = 404
IN_PROGRESS_MARKER = "idempotency_in_progress"
"""The ``error`` a platform that marks its "in progress" answer sends in the body."""
IN_PROGRESS_TEXT = "A request with this Idempotency-Key is in progress"
"""The ``detail`` every platform sends for it; older ones send nothing else."""
REPLAYED = "Idempotency-Replayed"


def content_key(method: str, url: str, body: Any, attempt: int) -> str:
    """The key of a resumable create: a digest of the request and of its place in line.

    ``attempt`` is how many replayed refusals were stepped past to get here.
    """
    request = f"{method} {url}\n{attempt}\n{json.dumps(body, allow_nan=False)}"
    return f"dagnam-{hashlib.sha256(request.encode('utf-8')).hexdigest()}"


def idempotency_in_progress(response: Any) -> bool:
    """Whether a response is the platform's "the first ask of this key is still in progress".

    A 409 whose body carries the marker, or whose ``detail`` is exactly the text older
    platforms send -- never a looser match: any other 409 is a real conflict and is not
    waited on.
    """
    if response.status_code != CONFLICT:
        return False
    try:
        body = response.json()
    except ValueError:
        return False
    return isinstance(body, dict) and (
        body.get("error") == IN_PROGRESS_MARKER or body.get("detail") == IN_PROGRESS_TEXT
    )


def replay_pointer(response: Any, body: dict[str, Any]) -> str | None:
    """The id a replay that dropped its body points at: its ``Location``, else ``resource_id``."""
    location = str(response.headers.get("Location") or "").rstrip("/").rsplit("/", 1)[-1]
    if location:
        return location
    found = body.get("resource_id")
    return found if isinstance(found, str) else None


def created_body(
    response: Any,
    body: dict[str, Any],
    read: Callable[[str], dict[str, Any]],
    id_key: str = "id",
) -> dict[str, Any]:
    """What a create answered with, even when the platform replayed it without the body.

    A replay whose cached body was dropped carries only a pointer to the resource it made
    (same key, the resource exists). Such an answer -- replayed, and without the resource's own
    ``id_key`` -- is read back through ``read``; any other answer is returned as it came, and
    so is a pointer-less one, which nothing can be done about here.
    """
    if response.headers.get(REPLAYED) != "true" or id_key in body:
        return body
    pointer = replay_pointer(response, body)
    return body if pointer is None else read(pointer)


def gone(*typed: type[DagnamError]) -> Callable[[DagnamError], bool]:
    """A read's way of saying "no such thing": one of the typed not-found errors, or a plain 404."""
    return lambda exc: (
        isinstance(exc, typed) or (isinstance(exc, APIError) and exc.status_code == NOT_FOUND)
    )


def confirmed_by[R](
    read: Callable[[str], object],
    missing: Callable[[DagnamError], bool],
    id_key: str = "id",
) -> Callable[[Any], bool]:
    """A ``confirm`` for :func:`resume_create`: read back what a replayed create answered with.

    A replay that dropped its body names its resource through its pointer
    (:func:`replay_pointer`), which is read the same way. ``True`` when the resource is
    there -- or the answer names none, which a read cannot settle. ``False`` only when the
    read says it is gone (``missing``): the replay answers a resource deleted since, and the
    create steps to its next key. Any other failure of the read is raised: a platform that
    cannot be asked proves nothing, and trusting the replay blind is what this exists to stop.
    """

    def confirm(response: Any) -> bool:
        body = response.json()
        if not isinstance(body, dict):
            return True
        item_id = body.get(id_key)
        if id_key not in body and response.headers.get(REPLAYED) == "true":
            item_id = replay_pointer(response, body)
        if not isinstance(item_id, str):
            return True
        try:
            read(item_id)
        except DagnamError as exc:
            if missing(exc):
                return False
            raise
        return True

    return confirm


def resume_create[T](
    send: Callable[[str], T],
    seen: Callable[[], tuple[bool, bool]],
    method: str,
    url: str,
    body: Any,
    *,
    sleep: Callable[[float], None],
    confirm: Callable[[T], bool] | None = None,
) -> T:
    """Send a create under its own keys in order, stepping past the answers that are not this ask's.

    ``send(key)`` issues the request and raises what the platform refused it
    with; ``seen()`` says whether that last answer came from the replay cache and
    whether it was an "in progress" one (:func:`idempotency_in_progress`). A replayed refusal, or a replayed success that
    ``confirm`` says names something gone, moves to the next key. A 409 waits
    (``sleep``) up to :data:`IN_PROGRESS_SECONDS` and asks again under the same
    key. Past :data:`RESUME_ATTEMPTS` the ask goes out under a random key, so it
    can never be stuck behind its own history.
    """
    attempt, waited = 0, 0.0
    while attempt < RESUME_ATTEMPTS:
        try:
            result = send(content_key(method, url, body, attempt))
        except DagnamError:
            replayed, in_progress = seen()
            if replayed:
                attempt += 1
            elif in_progress and waited < IN_PROGRESS_SECONDS:
                sleep(IN_PROGRESS_POLL)
                waited += IN_PROGRESS_POLL
            else:
                raise
            continue
        if confirm is not None and seen()[0] and not confirm(result):
            attempt += 1
            continue
        return result
    return send(str(uuid.uuid4()))


__all__ = [
    "IN_PROGRESS_MARKER",
    "IN_PROGRESS_POLL",
    "IN_PROGRESS_SECONDS",
    "IN_PROGRESS_TEXT",
    "RESUME_ATTEMPTS",
    "confirmed_by",
    "content_key",
    "created_body",
    "gone",
    "idempotency_in_progress",
    "replay_pointer",
    "resume_create",
]
