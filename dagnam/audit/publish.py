"""Mirror a local audit into the account as it runs.

The run stays the source of truth: everything here is a best-effort echo of
what ``state.json`` already records, so the website can show a run in progress
without the run depending on the website. Every call is wrapped -- an
``APIError`` is logged and the body queued to go out with the next one -- and
:meth:`Publisher.halt` is the only thing that ever tells the server a run
stopped. ``dagnam audit run --local-only`` builds the publisher with
``client=None`` and every method returns before it can open a socket.

The server's candidate lifecycle is a state machine
(``uploading -> splitting -> pii_check -> submitting -> training -> deploying
-> replaying -> scored``), so :data:`STATUS_BY_STEP` maps each step of
``orchestrate.STEPS`` onto it in order; a step that recorded an error
publishes ``failed`` (or ``pii_disagreement``) instead, which every state
allows.

Each candidate remembers the last step the account acknowledged
(``StepState.published_step``), so a step whose patch never landed -- a
blip on the last one, a Ctrl+C mid-send -- goes out on the next run.

A run resumes its audit once, before anything else (:meth:`Publisher.resume`),
and every publish after that says ``resume: false``: a cancel on the website at
any later point is the 409 "audit is halted" the next publish gets, and a
delete is the uniform 404 -- either one ends the run. A platform without the
resume route gets the older behaviour: the first publish resumes.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from importlib.metadata import PackageNotFoundError, version
import logging
from pathlib import Path
from typing import Any

from dagnam._core.auth import web_url_from_api_url
from dagnam._core.exceptions import APIError
from dagnam._types import JsonObject
from dagnam.audit.publish_body import (
    DONE_BY_STEP,
    ID_MAX,
    MAX_WORKLOADS,
    MEASURED_FROM,
    REPLAY_STEP,
    SCORED_BY,
    SOURCE,
    STATUS_BY_STEP,
    audit_body,
    capped,
    patch_body,
    status_for,
    student_cost,
    too_many_selected,
    workload_body,
)
from dagnam.audit.state import AuditState, StepState, save_state
from dagnam.audit.steps import CONFLICT_STATUS, PlatformClient, StepContext, answer_field

_LOGGER = logging.getLogger("dagnam.audit.publish")

UNKNOWN_VERSION = "0+unknown"
"""What a source checkout with no installed distribution reports as its version."""

AUDIT_HALTED = "audit is halted"
"""The 409 a publish that does not resume gets from an audit the account halted."""
CANCELLED = "cancelled"
DELETED = "deleted"
"""Why the account ended the audit under a live run; the run halts with the same reason."""
HALTED = "halted"
"""The audit's status while the account holds it halted."""
NOT_FOUND = 404

DROP_STATUSES = frozenset({400, NOT_FOUND, CONFLICT_STATUS, 422})
"""Logged once with the server's reason and never resent.

400/422 is a body this client should never have sent; 409 is a transition the
candidate's lifecycle refuses, and a refused transition does not become legal
by asking again -- the back-fill of a resumed run relies on that. A 404 whose
audit still exists names an artifact the owner deleted, and will not reappear.
"""
MAX_PENDING = 20
"""Cap on the resend queue; the oldest unsent step is dropped rather than grow it forever."""
AUDIT_PATH = "audits"  # the site route one published audit is watched on


def installed_version() -> str:
    """The installed ``dagnam`` version, or a marker when the package is not installed."""
    try:
        return version("dagnam")
    except PackageNotFoundError:
        return UNKNOWN_VERSION


def silent(line: str) -> None:
    """The default ``announce``: this package prints nothing the CLI did not ask it to."""


def audit_url(api_url: str, audit_id: str) -> str:
    """Where a person watches this audit: the site that matches the configured API.

    ``dagnam login`` derives its "sign in at" link the same way
    (:func:`dagnam._core.auth.web_url_from_api_url`); an API host with no known
    site -- a private deployment -- links to the API host itself rather than
    guessing at a domain that may not exist.
    """
    return f"{web_url_from_api_url(api_url) or api_url.rstrip('/')}/{AUDIT_PATH}/{audit_id}"


class Publisher:
    """Echoes one audit into the account. Never raises: a publish failure is a warning.

    ``client=None`` (``--local-only``) makes every method a no-op that opens no
    socket, and so does a run whose ``create_audit`` never succeeded: there is
    nothing to attach a candidate or a step to.
    """

    def __init__(
        self,
        client: PlatformClient | None,
        state: AuditState,
        announce: Callable[[str], None] = silent,
    ) -> None:
        self._client = client
        self._state = state
        self._announce = announce
        """Where the "watch it at ..." line goes; the CLI passes its own printer."""
        self._entries: dict[str, Mapping[str, Any]] = {}
        """Workload id -> its scan entry, for the serving cost a scored candidate carries."""
        self._pending: list[tuple[StepState, JsonObject]] = []
        """(the candidate's state, body) patches not yet sent, oldest first."""
        self._resume = True
        """Each request asks to resume until the account applies one, or :meth:`resume` succeeds."""
        self.create_failed: str | None = None
        """Why ``create_audit`` failed, once it did: the run halts before it uploads anything."""
        self.stopped: str | None = None
        """``cancelled``/``deleted`` once the account ended the audit mid-run: nothing more goes."""

    # -- the audit ------------------------------------------------------------

    def follow(self, state: AuditState) -> None:
        """Mirror ``state`` from here on: the one ``run_audit`` loaded, not the CLI's copy."""
        self._state = state

    def resume(self) -> None:
        """Un-halt the audit once, before this run waits on anything.

        Every publish after it says ``resume: false``, so a cancel that lands
        during the run's first long wait sticks. An audit the account deleted
        stops the run here; on a platform without the route (or any other
        failure) the first publish resumes the audit instead.
        """
        target = self._target()
        if target is None:
            return
        client, audit_id = target
        try:
            client.resume_audit(audit_id)
        except Exception as exc:
            if isinstance(exc, APIError) and exc.status_code == NOT_FOUND and self.gone():
                return
            _LOGGER.warning(
                "audit publish: resume_audit failed (%s); the first publish resumes it", exc
            )
            return
        self._resume = False

    def gone(self) -> bool:
        """Whether the account deleted this audit (then publishing ends as ``deleted``).

        Every audit 404 is the same uniform answer -- a deleted audit, an artifact
        the owner deleted, a route an older platform lacks -- so a read settles it.
        """
        target = self._target()
        if target is None:
            return False
        client, audit_id = target
        try:
            client.get_audit(audit_id)
        except APIError as exc:
            if exc.status_code != NOT_FOUND:
                return False
            _LOGGER.warning(
                "audit publish: the platform has no audit with this id for this key; stopping"
            )
            self.stopped = DELETED
            return True
        return False

    def start(
        self,
        audit_dir: Path,
        scan: Mapping[str, Any],
        selected: Collection[str],
        *,
        floor: float,
        max_credits: int,
        sdk_version: str,
    ) -> None:
        """Publish the scan (once), remember its workloads; sets ``state.audit_id`` or ``create_failed``.

        A resumed run calls this too (it reads the workload numbers here) but returns before the
        POST once the state already carries an ``audit_id``.
        """
        entries = [e for e in scan.get("workloads", []) if isinstance(e, Mapping) and "id" in e]
        self._entries = {str(entry["id"]): entry for entry in entries}
        client = self._client
        if client is None or self._state.audit_id is not None or not entries:
            return
        refusal = too_many_selected(sum(1 for entry in entries if str(entry["id"]) in selected))
        if refusal is not None:
            _LOGGER.warning("audit publish: %s", refusal)
            # Said out loud: the person is waiting for a link that is not coming.
            self._guard("the workload-cap notice", lambda: self._announce(refusal))
            return
        published = capped(entries, selected)
        nonce = self._state.project_nonce

        def create() -> JsonObject:
            pending = self._state.pending_audit
            if pending is None:
                pending = audit_body(
                    audit_dir,
                    scan,
                    published,
                    selected,
                    project_id=self._state.project_id,
                    floor=floor,
                    max_credits=max_credits,
                    sdk_version=sdk_version,
                )
                # On disk before it is sent, and sent as written every time: the platform
                # replays a create only for the same body.
                self._state.pending_audit = pending
                save_state(audit_dir, self._state)
            # Not through ``_send``: there is no audit yet for the account to have ended.
            return client.create_audit({**pending, "resume": self._resume}, resume_nonce=nonce)

        try:
            created = create()
            audit_id = answer_field(created, "id", "the audit create")
        except Exception as exc:
            # The run halts: what it uploads before its audit exists is untagged. A body the
            # platform did not refuse (a timeout, a 502) may have landed, so it is re-sent as written.
            _LOGGER.warning("audit publish: create_audit failed (%s); the run halts", exc)
            self.create_failed = str(exc)
            if isinstance(exc, APIError) and exc.status_code in DROP_STATUSES - {CONFLICT_STATUS}:
                self._state.pending_audit = None
            return
        self._resume = False
        self._state.audit_id = audit_id
        self._state.pending_audit = None
        link = f"published: {audit_id} — watch it at {audit_url(client.api_url, audit_id)}"
        # Guarded like every other publish: a console that cannot encode the
        # line must not end a run that has already published its scan.
        self._guard("the audit's link", lambda: self._announce(link))

    def halt(self, reason: str) -> None:
        """Tell the account the run stopped short, and why -- unless it already shows a halt.

        Whatever is queued goes first. A halt the account already holds -- one
        the previous run published, or a cancel that landed since this run's
        last publish -- keeps its reason: ``budget`` must never overwrite the
        owner's ``cancelled``.
        """
        target = self._target()
        if target is None:
            return
        client, audit_id = target
        self._flush(client, audit_id)
        if self.stopped is not None:
            return
        current = self._guard("get_audit", lambda: client.get_audit(audit_id))
        if current is None or current.get("status") != HALTED:
            self._guard("halt_audit", lambda: client.halt_audit(audit_id, reason))

    # -- candidates -----------------------------------------------------------

    def candidate(self, ctx: StepContext, step: StepState) -> None:
        """Open this candidate on the server once; the id is remembered in the state."""
        target = self._target()
        if target is None or step.published_candidate_id is not None:
            return
        client, audit_id = target
        body: JsonObject = {
            "workload_id": ctx.workload_id[:ID_MAX],
            "kind": ctx.spec.kind.value,
            "recipe_key": ctx.spec.recipe_key,
        }
        created = self._guard(
            "create_audit_candidate",
            lambda: self._send(
                lambda resume: client.create_audit_candidate(audit_id, {**body, "resume": resume})
            ),
        )
        if created is not None:
            step.published_candidate_id = str(created["id"])

    def backfill(self, ctx: StepContext, step: StepState) -> None:
        """Publish the steps this candidate finished that the account never acknowledged.

        Everything after ``step.published_step``: all of it when the audit
        only reached the account on a resumed run, the one patch a blip or a
        Ctrl+C stranded on an audit that was published all along, and nothing
        on a fresh run. A step the server refuses (409) is dropped, so a
        candidate that is further along than this walk expects settles at its
        real status instead of being retried forever.

        A candidate an earlier run left with a recorded error ends at that
        error: the steps it finished keep the status they earned, and the
        first one it never finished -- the step it died on -- carries the
        terminal ``failed``. Without the distinction the whole trail would read
        ``failed`` and the audit would say the candidate died at ``upload``.
        """
        if self._target() is None:
            return
        terminal = step.error is not None and not step.scored
        names = list(DONE_BY_STEP)
        acknowledged = step.published_step
        if acknowledged is not None and terminal and not DONE_BY_STEP[acknowledged](step):
            return  # the step it died on is what the account acknowledged last
        start = 0 if acknowledged is None else names.index(acknowledged) + 1
        for step_name in names[start:]:
            done = DONE_BY_STEP[step_name]
            if done(step):
                # Only a terminal candidate needs the override: for every other
                # one ``status_for`` already reads the right status off the step.
                self.step(
                    ctx, step_name, step, status=STATUS_BY_STEP[step_name] if terminal else None
                )
            elif terminal:
                self.step(ctx, step_name, step)
                return

    def step(
        self, ctx: StepContext, step_name: str, step: StepState, *, status: str | None = None
    ) -> None:
        """Record one step of a candidate, resending anything an earlier call could not.

        ``status`` overrides what :func:`status_for` would read off the step;
        only :meth:`backfill` passes it, to publish a step that *succeeded* on
        a candidate whose later error is already recorded.
        """
        target = self._target()
        if target is None:
            return
        self.candidate(ctx, step)
        candidate_id = step.published_candidate_id
        if candidate_id is None:
            return
        patch = self._guard(
            f"the {step_name} body",
            lambda: patch_body(step_name, step, status, self._serving_cost(ctx)),
        )
        if patch is None:
            return
        if step_name == "replay_and_score":
            self._queue(step, {"step": REPLAY_STEP, "status": "replaying"})
        self._queue(step, patch)
        self._flush(*target)

    def flush(self) -> None:
        """Send whatever is still queued: the end of a run must not strand its last step."""
        target = self._target()
        if target is not None:
            self._flush(*target)

    def _serving_cost(self, ctx: StepContext) -> float | None:
        """What serving this candidate costs a month at the workload's own volume."""
        entry = self._entries.get(ctx.workload_id)
        if entry is None or ctx.spec.serving_rate_key is None:
            return None
        return student_cost(ctx.spec.serving_rate_key, entry)

    # -- transport ------------------------------------------------------------

    def _target(self) -> tuple[PlatformClient, str] | None:
        """The client and audit to publish against, or ``None`` when there is nothing to publish to."""
        client, audit_id = self._client, self._state.audit_id
        if client is None or audit_id is None or self.stopped is not None:
            return None
        return client, audit_id

    def _send[T](self, call: Callable[[bool], T]) -> T | None:
        """One request, told whether to resume the audit; ``None`` once the account ended it.

        The resume is spent by :meth:`resume` or, on an older platform, by the
        first request the account applies; after that a 409 "audit is halted"
        is a cancel, and a 404 that :meth:`gone` settles is a delete.
        """
        try:
            result = call(self._resume)
        except APIError as exc:
            if exc.status_code == CONFLICT_STATUS and exc.message == AUDIT_HALTED:
                _LOGGER.warning("audit publish: the audit was cancelled in your account; stopping")
                self.stopped = CANCELLED
                return None
            if exc.status_code == NOT_FOUND and self.gone():
                return None
            raise
        self._resume = False
        return result

    def _queue(self, step: StepState, body: JsonObject) -> None:
        self._pending.append((step, body))
        if len(self._pending) > MAX_PENDING:
            dropped = self._pending.pop(0)
            _LOGGER.warning("audit publish: dropping the unsent step %r", dropped[1].get("step"))

    def _flush(self, client: PlatformClient, audit_id: str) -> None:
        """Send the queued patches in order, stopping at the first that will not go.

        Order matters -- the server's transitions are monotonic -- so a failure
        leaves everything behind it queued for the next call rather than
        letting a later step overtake an earlier one. A body the server calls a
        client bug is dropped instead: resending it would only fail again.
        Either way the candidate's ``published_step`` moves past what the
        server answered, so a resumed run never resends it.
        """
        while self._pending and self.stopped is None:
            owner, body = self._pending[0]
            try:
                self._patch_once(client, audit_id, str(owner.published_candidate_id), body)
            except APIError as exc:
                if exc.status_code not in DROP_STATUSES:
                    self._retrying(body, exc)
                    return
                _LOGGER.warning(
                    "audit publish: step %r was refused by the server (HTTP %d); not retried",
                    body["step"],
                    exc.status_code,
                )
            except Exception as exc:
                self._retrying(body, exc)
                return
            if self.stopped is not None:
                return
            self._pending.pop(0)
            if body["step"] != REPLAY_STEP:
                owner.published_step = str(body["step"])

    def _patch_once(
        self, client: PlatformClient, audit_id: str, candidate_id: str, body: JsonObject
    ) -> None:
        self._send(
            lambda resume: client.patch_audit_candidate(
                audit_id, candidate_id, {**body, "resume": resume}
            )
        )

    @staticmethod
    def _retrying(body: JsonObject, exc: Exception) -> None:
        """One transient failure: the body stays queued in front of the next step's."""
        _LOGGER.warning(
            "audit publish: step %r failed (%s); will retry with the next step", body["step"], exc
        )

    def _guard[T](self, what: str, run: Callable[[], T]) -> T | None:
        """Run one publish step; a failure is a warning, never the end of the run.

        This wraps the body-building too, not only the call, so nothing in the
        publisher -- a scan report it cannot map, a step name it does not know
        -- can raise into the run.
        """
        try:
            return run()
        except Exception as exc:
            # Publishing is an echo of what the run already recorded on disk, so
            # nothing it can go wrong at -- a refusal, a dead network, a body
            # this client should not have built -- is worth ending a run over.
            _LOGGER.warning("audit publish: %s failed (%s); the run continues", what, exc)
            return None


__all__ = [
    "AUDIT_HALTED",
    "AUDIT_PATH",
    "CANCELLED",
    "DELETED",
    "DONE_BY_STEP",
    "MAX_PENDING",
    "MAX_WORKLOADS",
    "MEASURED_FROM",
    "REPLAY_STEP",
    "SCORED_BY",
    "SOURCE",
    "STATUS_BY_STEP",
    "UNKNOWN_VERSION",
    "Publisher",
    "audit_url",
    "installed_version",
    "silent",
    "status_for",
    "too_many_selected",
    "workload_body",
]
