"""``state.json``: every platform artifact the audit created, per workload and candidate (spec D9).

The file is written atomically after every step, so a crash or a Ctrl+C
leaves the state the previous step reached and the next ``run`` resumes from
it. Deployment keys never appear here -- :mod:`dagnam.audit.secrets` holds
them; the state records only a ``key_ref``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path
from typing import Any

from dagnam._types import JsonValue
from dagnam.audit.candidates import CandidateKind
from dagnam.audit.workspace import write_atomic

SCHEMA = "dagnam.audit.state/1"
STATE_FILE = "state.json"


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


_STEP_KEYS = frozenset(f.name for f in fields(StepState))


@dataclass(slots=True)
class AuditState:
    """The whole audit: project, price table, per-workload candidates, and why it halted."""

    schema: str = SCHEMA
    project_id: str | None = None
    price_table_version: str | None = None
    workloads: dict[str, dict[CandidateKind, StepState]] = field(default_factory=dict)
    halted: dict[str, JsonValue] | None = None

    def candidate(self, workload_id: str, kind: CandidateKind) -> StepState:
        """The step state for one candidate, created empty on first access."""
        return self.workloads.setdefault(workload_id, {}).setdefault(kind, StepState())

    def to_json(self) -> dict[str, Any]:
        """The on-disk shape of spec section 7."""
        return {
            "schema": self.schema,
            "project_id": self.project_id,
            "price_table_version": self.price_table_version,
            "workloads": {
                workload_id: {
                    "candidates": {kind.value: asdict(step) for kind, step in candidates.items()}
                }
                for workload_id, candidates in self.workloads.items()
            },
            "halted": self.halted,
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
    halted = data.get("halted")
    return AuditState(
        project_id=data.get("project_id"),
        price_table_version=data.get("price_table_version"),
        workloads=workloads,
        halted=_object(halted, "halted") if halted is not None else None,
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


def save_state(audit_dir: Path, state: AuditState) -> None:
    """Write the state atomically (``.tmp`` + ``os.replace``) so a reader never sees a partial file."""
    audit_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(audit_dir / STATE_FILE, json.dumps(state.to_json(), indent=2, ensure_ascii=False))


__all__ = ["SCHEMA", "STATE_FILE", "AuditState", "StepState", "load_state", "save_state"]
