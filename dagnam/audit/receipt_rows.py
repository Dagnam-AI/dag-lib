"""What one receipt row means to this client: the single place that reads ``status`` and ``code``.

For a published audit the platform's ``delete`` and ``cancel`` are the only paths that touch
its resources; this client calls one, shows the receipt, and records what the rows say. It
never acts on a row: it only decides what each one MEANS (:func:`decide`), and what the
command's exit status is (:func:`exit_status`). Nothing here, or anywhere else in this client,
reads a row's ``reason``, except :func:`_legacy` for a platform that sends no ``code``.

Statuses (the platform's words, stable):

* ``deleted`` / ``already_absent``: nothing is left. ``stopped``: a cancel stopped it.
* ``kept``: the platform left it on purpose and the audit is deleted around it. Any code.
* ``blocked``: one of the audit's own resources is still there. Any code: the client does
  not retry it, whatever it is.
* anything else (a newer platform): shown as sent, never acted on, and it fails the
  command, because this client cannot promise what it cannot read.

The ``already_stopped`` code (status ``stopped``) says a cancel found the run or endpoint
already terminal, paused or stopped: it is done, and it marks nothing, so a finished run
stays resumable.

A platform that predates ``code`` sends none. Its ``blocked`` row is read once, here
(:func:`_legacy`): the four reasons it used for a decision it made on purpose are kept, any
other reason is a leftover. The day production ships ``code`` that function is deleted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

DELETED = "deleted"
ALREADY_ABSENT = "already_absent"
STOPPED = "stopped"
KEPT = "kept"
BLOCKED = "blocked"
ALREADY_STOPPED = "already_stopped"
"""The ``code`` of a ``stopped`` row for something a cancel found already terminal or paused."""
NOT_REMOVED = "not_removed"
"""The ``code`` of a ``blocked`` row this client wrote itself: its reason is for people, never read."""
NOT_ANSWERED = "not_answered"
"""The ``code`` of the ``audit`` row this client writes when the platform did not answer."""
PLATFORM_ONLY = "platform_only"
"""The ``code`` of a registry version this client cannot purge: the platform has no route for it."""
LEGACY_FINISHED_PREFIX = "Cannot cancel job with status "
"""What a platform without ``code`` said of a run a cancel found already ended (a ``blocked`` row)."""
LEGACY_KEPT_PREFIXES = (
    "project was not created by this audit",
    "project holds artifacts this audit did not record",
    "registry entry holds versions this audit did not record",
    "a live deployment serves these weights",
)
"""The reasons a platform without ``code`` gave a ``blocked`` row it kept on purpose."""


def blocked(kind: str, item_id: object, reason: str, code: str = NOT_REMOVED) -> dict[str, Any]:
    """A ``blocked`` row this client writes: the audit's own resource is still there."""
    return {"kind": kind, "id": item_id, "status": BLOCKED, "code": code, "reason": reason}


class Verdict(StrEnum):
    """What a row means for the command that read it."""

    DONE = "done"
    """Nothing is left of it to delete or stop."""
    KEPT = "kept"
    """The platform kept it on purpose: final, untouched."""
    LEFT = "left"
    """One of the audit's own resources is still there."""
    UNKNOWN = "unknown"
    """A status this client does not know: shown, never acted on, and it fails the command."""


@dataclass(frozen=True, slots=True)
class Decision:
    """A row's verdict, and whether a cancel may mark the local run or endpoint with it."""

    verdict: Verdict
    marks: bool = True


def _legacy(row: Mapping[str, Any]) -> Decision:
    """A ``blocked`` row from a platform that sends no ``code``."""
    reason = str(row.get("reason") or "")
    if reason.startswith(LEGACY_KEPT_PREFIXES):
        return Decision(Verdict.KEPT)
    if row.get("kind") == "training_job" and reason.startswith(LEGACY_FINISHED_PREFIX):
        return Decision(
            Verdict.DONE, marks=False
        )  # what ``already_stopped`` says, in its old words
    return Decision(Verdict.LEFT)


def decide(row: Mapping[str, Any]) -> Decision:
    """What one receipt row means, from its ``status`` and ``code`` only."""
    status = row.get("status")
    if status in (DELETED, ALREADY_ABSENT, STOPPED):
        return Decision(Verdict.DONE, marks=row.get("code") != ALREADY_STOPPED)
    if status == KEPT:
        return Decision(Verdict.KEPT)
    if status != BLOCKED:
        return Decision(Verdict.UNKNOWN)
    return _legacy(row) if row.get("code") is None else Decision(Verdict.LEFT)


def exit_status(rows: list[dict[str, Any]], audit_status: object, *, verb: str) -> int:
    """The command's exit status, in one place: 1 iff something of the audit's own is left.

    Left means a ``blocked`` row (the platform's, or this client's: a platform that did not
    answer, a local file it could not remove), a row whose status this client cannot read,
    or, for a delete, a platform that answers that it did not delete the audit
    (``audit_status: halted``). A ``kept`` row never fails a command.
    """
    verdicts = {decide(row).verdict for row in rows}
    left = bool(verdicts & {Verdict.LEFT, Verdict.UNKNOWN})
    return int(left or not _status_ok(audit_status, verb))


def _status_ok(audit_status: object, verb: str) -> bool:
    """Whether the platform's own ``audit_status`` leaves the command finished.

    Absent is the older platform. A delete is finished only when the audit is ``deleted``; a
    cancel leaves it ``halted`` (``deleted`` too, for one the tombstone answered). Anything else,
    ``halted`` after a delete included, is a status this client cannot call finished.
    """
    return (
        audit_status is None
        or audit_status == "deleted"
        or (verb == "cancel" and audit_status == "halted")
    )


__all__ = [
    "ALREADY_ABSENT",
    "ALREADY_STOPPED",
    "BLOCKED",
    "DELETED",
    "KEPT",
    "LEGACY_FINISHED_PREFIX",
    "LEGACY_KEPT_PREFIXES",
    "NOT_ANSWERED",
    "NOT_REMOVED",
    "PLATFORM_ONLY",
    "STOPPED",
    "Decision",
    "Verdict",
    "blocked",
    "decide",
    "exit_status",
]
