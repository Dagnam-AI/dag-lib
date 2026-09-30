"""``state.json``: every platform artifact the audit created, per workload and candidate (spec D9).

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
from dagnam.audit.workspace import write_atomic

SCHEMA = "dagnam.audit.state/1"
STATE_FILE = "state.json"
LOCK_FILE = "state.json.lock"


class AuditBusyError(DagnamError):
    """Another ``dagnam audit`` command holds this audit directory."""


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


_STEP_KEYS = frozenset(f.name for f in fields(StepState))


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
    audit_id: str | None = None
    price_table_version: str | None = None
    workloads: dict[str, dict[CandidateKind, StepState]] = field(default_factory=dict)
    halted: dict[str, JsonValue] | None = None
    retired_cost_credits: float = 0.0
    """Spend and conservative reservations captured before old replay files are removed."""
    retired: list[StepState] = field(default_factory=list)
    """Superseded local candidates, kept only for cancellation and deletion."""

    def candidate(self, workload_id: str, kind: CandidateKind) -> StepState:
        """The step state for one candidate, created empty on first access."""
        return self.workloads.setdefault(workload_id, {}).setdefault(kind, StepState())

    def all_steps(self) -> Generator[StepState]:
        """Active and retired candidates, so cleanup never loses a remote handle."""
        for candidates in self.workloads.values():
            yield from candidates.values()
        yield from self.retired

    def to_json(self) -> dict[str, Any]:
        """The on-disk shape of spec section 7."""
        return {
            "schema": self.schema,
            "project_id": self.project_id,
            "audit_id": self.audit_id,
            "price_table_version": self.price_table_version,
            "workloads": {
                workload_id: {
                    "candidates": {kind.value: asdict(step) for kind, step in candidates.items()}
                }
                for workload_id, candidates in self.workloads.items()
            },
            "halted": self.halted,
            "retired": [asdict(step) for step in self.retired],
            "retired_cost_credits": self.retired_cost_credits,
        }


def _object(value: object, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"state.json: {what} must be a JSON object")
    return value


def _from_json(raw: object) -> AuditState:
    data = _object(raw, "the document")
    schema = data.get("schema")
    if schema != SCHEMA:
        raise ValueError(
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
                raise ValueError(
                    f"state.json: unknown step keys {unknown} under {workload_id}/{kind}"
                )
            workloads[workload_id][CandidateKind(kind)] = StepState(**step_fields)
    retired = data.get("retired", [])
    if not isinstance(retired, list):
        raise ValueError("state.json: retired must be a JSON array")
    retired_steps: list[StepState] = []
    for index, raw_step in enumerate(retired):
        step_fields = _object(raw_step, f"retired[{index}]")
        unknown = sorted(set(step_fields) - _STEP_KEYS)
        if unknown:
            raise ValueError(f"state.json: unknown step keys {unknown} under retired[{index}]")
        retired_steps.append(StepState(**step_fields))
    halted = data.get("halted")
    return AuditState(
        project_id=data.get("project_id"),
        audit_id=data.get("audit_id"),
        price_table_version=data.get("price_table_version"),
        workloads=workloads,
        halted=_object(halted, "halted") if halted is not None else None,
        retired=retired_steps,
        retired_cost_credits=data.get("retired_cost_credits", 0.0),
    )


def load_state(audit_dir: Path) -> AuditState:
    """Read ``audit_dir/state.json``; a missing file is a fresh audit.

    Raises:
        ValueError: the file is not this version's schema (the message names
            ``dagnam.audit.state/1``) or a step carries a key no step writes.
    """
    path = audit_dir / STATE_FILE
    if not path.exists():
        return AuditState()
    return _from_json(json.loads(path.read_text(encoding="utf-8")))


@contextmanager
def lock_audit(audit_dir: Path) -> Generator[None]:
    """Hold ``audit_dir`` for this process until the block ends; the OS drops it if the process dies.

    Raises:
        AuditBusyError: another command holds it -- a second ``audit run`` over
            the same directory, or a ``delete`` under a live run.
        FileNotFoundError: there is no such directory; a mistyped one is never created.
    """
    if not audit_dir.is_dir():
        raise FileNotFoundError(f"{audit_dir} is not an audit directory: run `dagnam audit scan`")
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
    "StepState",
    "load_state",
    "lock_audit",
    "save_state",
]
