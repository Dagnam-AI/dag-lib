"""``state.json``: every platform artifact the audit created, per workload and candidate.

The file is written atomically after every step, so a crash or a Ctrl+C
leaves the state the previous step reached and the next ``run`` resumes from
it. Deployment keys never appear here -- :mod:`dagnam.audit.secrets` holds
them; the state records only a ``key_ref``. One command at a time holds the
directory (:func:`lock_audit`): two runs over it would each see no ``run_id``
and each pay for one.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path
from typing import Any

import filelock

from dagnam._core.exceptions import DagnamError
from dagnam._types import JsonValue
from dagnam.audit.candidates import CandidateKind
from dagnam.audit.workspace import check_writable, read_regular, write_atomic

SCHEMA = "dagnam.audit.state/1"
STATE_FILE = "state.json"
LOCK_FILE = "state.json.lock"


class AuditBusyError(DagnamError):
    """Another ``dagnam audit`` command holds this audit directory."""


class StateFileError(DagnamError, ValueError):
    """``state.json`` cannot be used (not JSON, not this schema, a field of the wrong type).

    A command that meets it stops before it touches anything, with the file named, so a damaged
    record is never read as "nothing recorded".
    """


class AuditDeletedError(DagnamError):
    """The audit directory holds a deleted audit: there is nothing left to run or cancel."""


DELETED_STATE: dict[str, JsonValue] = {"reason": "deleted"}
"""``halted`` of a deleted audit. Only the platform's own answer writes it (a delete it completed);
a 404 never does: it says only that this key cannot see the audit."""


@dataclass(slots=True)
class StepState:
    """One candidate's progress; every key is ``None`` until its step ran."""

    dataset_id: str | None = None
    version_id: str | None = None
    split_task_id: str | None = None
    split_done: bool | None = None
    pii_task_id: str | None = None
    pii_agrees: bool | None = None
    base: str | None = None
    run_id: str | None = None
    training_job_id: str | None = None
    run_status: str | None = None
    model_version_id: str | None = None
    deployment_id: str | None = None
    key_ref: str | None = None
    deploy_status: str | None = None
    scored: bool | None = None
    agreement: dict[str, JsonValue] | None = None
    latency: dict[str, JsonValue] | None = None
    training_cost_credits: float | None = None
    replay_cost_credits: float | None = None
    error: str | None = None
    published_candidate_id: str | None = None
    published_step: str | None = None
    """The last step the account acknowledged; a resumed run publishes what came after it."""
    workload_id: str | None = field(default=None, compare=False)
    kind: CandidateKind | None = field(default=None, compare=False)
    """Which candidate this step belongs to; set when the state creates or reads it.

    An active step's place in ``workloads`` already says so, and that is all
    ``state.json`` records for one. The step carries it too so that a forced
    rescan, which moves the bare steps into ``retired``, does not lose which
    workload a run still billing was for. It is not part of what a step *did*,
    so it is left out of equality; ``None`` on a step retired before the state
    kept it.
    """


_IDENTITY = frozenset({"workload_id", "kind"})
"""The keys that say which candidate a step is: written for a retired step only."""
_STEP_KEYS = frozenset(f.name for f in fields(StepState)) - _IDENTITY
"""The keys of what a step did: every key an active step is written with."""


def _step_json(step: StepState) -> dict[str, Any]:
    """What a step did, as it is written under its workload and candidate."""
    return {key: value for key, value in asdict(step).items() if key in _STEP_KEYS}


@dataclass(slots=True)
class AuditState:
    """The whole audit: project, published audit, price table, candidates, and why it halted.

    ``audit_id`` is the account-side audit ``dagnam audit run`` published this
    run to (``None`` for a ``--local-only`` run, and for one whose first publish
    failed): it is what makes a resumed run patch the audit it already created
    rather than open a second one, and what routes ``cancel``/``delete``
    through the server.
    """

    schema: str = SCHEMA
    project_id: str | None = None
    project_nonce: str | None = None
    """This audit directory's own random value, saved before anything is created under it.

    The project create's idempotency key comes from it: that body is only a
    title, the same for every audit directory of that name, so nothing in it can
    tell a replay of this audit's project from another's. A create that died
    after the platform made the project is asked again under this nonce and
    replays its own; a different directory has its own and never does. The audit
    create and each dataset upload are keyed by it the same way. ``None`` in a
    state written before it existed; a run mints one first thing.
    """
    pending_audit: dict[str, JsonValue] | None = None
    """The body of an audit create no answer has come for, saved before it was sent.

    The platform replays a create only for the same key *and* the same body, so a
    rerun that has to ask again after a lost create sends this very body (whatever
    ``--max-credits`` or ``--floor`` it has now), and finds the audit the first ask
    made instead of opening a second. Cleared once the audit is created.
    """
    audit_id: str | None = None
    tagged: bool = False
    """Every resource of this state carries the audit's id: the audit was created by a client that
    sends it, over resources it claimed or created itself. ``False`` for an audit an older client
    published, whose untagged resources only a claim can make the audit's."""
    claim_pending: bool = False
    """The audit was created over resources an earlier unpublished run made, and the claim for
    them has not been answered yet; a run asks again before it goes on."""
    unclaimed_ids: list[str] = field(default_factory=list)
    """Ids the platform refused to claim. This directory created them, so ``audit delete`` and
    ``audit cancel`` handle them directly, as they would for an unpublished audit."""
    confirmed_gone: list[str] = field(default_factory=list)
    """Ids a delete of this directory has seen deleted. They prove this key can see the account's
    resources, so a later walk that finds nothing is a finished one, not another account's."""
    confirmed_by: str | None = None
    """Who confirmed them (the client's ``identity``). Another identity's memory is no proof: a
    different key, or host, may see none of the account that made these."""
    kept_ids: list[str] = field(default_factory=list)
    """Ids the platform said are not this audit's to take (a ``kept`` row, a claim it refused).

    Recorded from a receipt or a claim answer; no listing offers them as the audit's own, and an
    unpublished walk never deletes them. The platform decides about a published audit's resources
    and this client never acts on them, so this is a record of the decision, not a guard.
    """
    price_table_version: str | None = None
    workloads: dict[str, dict[CandidateKind, StepState]] = field(default_factory=dict)
    halted: dict[str, JsonValue] | None = None
    retired_cost_credits: float = 0.0
    """Spend and conservative reservations captured before old replay files are removed."""
    retired: list[StepState] = field(default_factory=list)
    """Superseded local candidates, kept for cancellation, deletion and ``audit status``."""

    def candidate(self, workload_id: str, kind: CandidateKind) -> StepState:
        """The step state for one candidate, created empty on first access."""
        return self.workloads.setdefault(workload_id, {}).setdefault(
            kind, StepState(workload_id=workload_id, kind=kind)
        )

    def all_steps(self) -> Generator[StepState]:
        """Active and retired candidates, so cleanup never loses a remote handle."""
        for candidates in self.workloads.values():
            yield from candidates.values()
        yield from self.retired

    def to_json(self) -> dict[str, Any]:
        """The on-disk shape; a retired step also says which candidate it had been."""
        return {
            "schema": self.schema,
            "project_id": self.project_id,
            "project_nonce": self.project_nonce,
            "pending_audit": self.pending_audit,
            "audit_id": self.audit_id,
            "tagged": self.tagged,
            "claim_pending": self.claim_pending,
            "unclaimed_ids": self.unclaimed_ids,
            "confirmed_gone": self.confirmed_gone,
            "confirmed_by": self.confirmed_by,
            "kept_ids": self.kept_ids,
            "price_table_version": self.price_table_version,
            "workloads": {
                workload_id: {
                    "candidates": {
                        kind.value: _step_json(step) for kind, step in candidates.items()
                    }
                }
                for workload_id, candidates in self.workloads.items()
            },
            "halted": self.halted,
            "retired": [
                {"workload_id": step.workload_id, "kind": step.kind, **_step_json(step)}
                for step in self.retired
            ],
            "retired_cost_credits": self.retired_cost_credits,
        }


def _object(value: object, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StateFileError(f"state.json: {what} must be a JSON object")
    return value


def _from_json(raw: object) -> AuditState:
    data = _object(raw, "the document")
    schema = data.get("schema")
    if schema != SCHEMA:
        raise StateFileError(
            f"state.json: unknown schema {schema!r}; this version of dagnam understands {SCHEMA!r}"
        )
    workloads: dict[str, dict[CandidateKind, StepState]] = {}
    for workload_id, entry in _object(data.get("workloads", {}), "workloads").items():
        candidates = _object(
            _object(entry, f"workloads[{workload_id}]").get("candidates", {}), "candidates"
        )
        workloads[workload_id] = {}
        for kind, step in candidates.items():
            step_fields = _object(step, f"workloads[{workload_id}].candidates[{kind}]")
            unknown = sorted(set(step_fields) - _STEP_KEYS)
            if unknown:
                raise StateFileError(
                    f"state.json: unknown step keys {unknown} under {workload_id}/{kind}"
                )
            workloads[workload_id][CandidateKind(kind)] = StepState(
                **step_fields, workload_id=workload_id, kind=CandidateKind(kind)
            )
    retired = data.get("retired", [])
    if not isinstance(retired, list):
        raise StateFileError("state.json: retired must be a JSON array")
    retired_steps: list[StepState] = []
    for index, raw_step in enumerate(retired):
        step_fields = dict(_object(raw_step, f"retired[{index}]"))
        unknown = sorted(set(step_fields) - _STEP_KEYS - _IDENTITY)
        if unknown:
            raise StateFileError(f"state.json: unknown step keys {unknown} under retired[{index}]")
        # Absent from a state written before retired steps kept their candidate.
        kind = step_fields.pop("kind", None)
        retired_steps.append(
            StepState(**step_fields, kind=None if kind is None else CandidateKind(kind))
        )
    lists: dict[str, list[str]] = {}
    for key in ("kept_ids", "unclaimed_ids", "confirmed_gone"):
        found = data.get(key, [])
        if not isinstance(found, list):
            raise StateFileError(f"state.json: {key} must be a JSON array")
        lists[key] = [str(i) for i in found]
    confirmed_by = data.get("confirmed_by")
    if confirmed_by is not None and not isinstance(confirmed_by, str):
        raise StateFileError("state.json: confirmed_by must be a string")
    halted = data.get("halted")
    pending = data.get("pending_audit")
    return AuditState(
        project_id=data.get("project_id"),
        project_nonce=data.get("project_nonce"),
        pending_audit=_object(pending, "pending_audit") if pending is not None else None,
        audit_id=data.get("audit_id"),
        tagged=data.get("tagged") is True,
        claim_pending=data.get("claim_pending") is True,
        unclaimed_ids=lists["unclaimed_ids"],
        confirmed_gone=lists["confirmed_gone"],
        confirmed_by=confirmed_by,
        kept_ids=lists["kept_ids"],
        price_table_version=data.get("price_table_version"),
        workloads=workloads,
        halted=_object(halted, "halted") if halted is not None else None,
        retired=retired_steps,
        retired_cost_credits=data.get("retired_cost_credits", 0.0),
    )


def load_state(audit_dir: Path) -> AuditState:
    """Read ``audit_dir/state.json``; a missing file is a fresh audit.

    Raises:
        StateFileError: (a ``ValueError``) the file is not JSON, not this version's schema (the
            message names ``dagnam.audit.state/1``), or a step carries a key no step writes.
    """
    path = audit_dir / STATE_FILE
    if not path.exists():
        return AuditState()
    try:
        raw = json.loads(read_regular(path))
    except json.JSONDecodeError as exc:
        raise StateFileError(f"state.json: not valid JSON ({exc})") from None
    try:
        return _from_json(raw)
    except StateFileError:
        raise
    except (ValueError, TypeError, KeyError) as exc:  # e.g. a candidate no version of dagnam knows
        raise StateFileError(f"state.json: {type(exc).__name__}: {exc}") from None


@contextmanager
def lock_audit(audit_dir: Path) -> Generator[None]:
    """Hold ``audit_dir`` for this process until the block ends; the OS drops it if the process dies.

    Raises:
        AuditBusyError: another command holds it -- a second ``audit run`` over
            the same directory, or a ``delete`` under a live run.
        FileNotFoundError: there is no such directory; a mistyped one is never created.
        UnsafeWorkloadsError: the lock file or the directory is a link, or the lock file is
            not a regular file (opening a pipe would block for ever). Older ``filelock``
            releases open it with ``O_TRUNC`` and follow the link, which would empty whatever
            it points at.
    """
    if not audit_dir.is_dir():
        raise FileNotFoundError(f"{audit_dir} is not an audit directory: run `dagnam audit scan`")
    check_writable(audit_dir / LOCK_FILE)
    lock = filelock.FileLock(str(audit_dir / LOCK_FILE), timeout=0)
    try:
        lock.acquire()
    except filelock.Timeout:
        raise AuditBusyError(
            f"{audit_dir} is in use by another `dagnam audit` command; let it finish first"
        ) from None
    try:
        yield
    finally:
        lock.release()


def save_state(audit_dir: Path, state: AuditState) -> None:
    """Write the state atomically (``.tmp`` + ``os.replace``) so a reader never sees a partial file."""
    audit_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(audit_dir / STATE_FILE, json.dumps(state.to_json(), indent=2, ensure_ascii=False))


__all__ = [
    "LOCK_FILE",
    "SCHEMA",
    "STATE_FILE",
    "AuditBusyError",
    "AuditState",
    "StateFileError",
    "StepState",
    "load_state",
    "lock_audit",
    "save_state",
]
