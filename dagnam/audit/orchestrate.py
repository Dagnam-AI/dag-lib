"""``run_audit``: the resumable frontier over workloads x candidates (spec U4, D9).

One loop, no branching on kind: for each selected workload, for each
candidate in :data:`~dagnam.audit.candidates.CANDIDATES`, run the step list.
The state is saved after every step, so an interrupt or a crash leaves the
last completed step on disk and the next call resumes from it -- and a
``KeyboardInterrupt`` is never caught. A hard failure inside a step is
recorded as ``halted: error`` before it propagates.

Verdicts and prices belong to the scan report and Task 5's economics: this
module reads ``scan-report.json`` for each workload's structure class and
verdict, and records agreement, latency and credits spent for the report
to price; it never computes a cost itself.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import json
from pathlib import Path
import time

from dagnam.audit.candidates import CANDIDATES
from dagnam.audit.secrets import SecretStore
from dagnam.audit.state import AuditState, load_state, save_state
from dagnam.audit.steps import (
    DEPLOY_TIMEOUT_SECONDS,
    RUN_TIMEOUT_SECONDS,
    PlatformClient,
    Step,
    StepContext,
    error_code,
)
from dagnam.audit.steps_data import pii_scan, resolve_version, split, upload, wait_pii, wait_split
from dagnam.audit.steps_serve import (
    create_deployment,
    create_revision,
    replay_and_score,
    wait_active,
)
from dagnam.audit.steps_train import resolve_model_version, submit, wait_run
from dagnam.audit.structure import StructureClass
from dagnam.audit.thresholds import FLOOR_JSON, FLOOR_LABEL

SCAN_REPORT = "scan-report.json"
AUDITED_VERDICTS = frozenset({"candidate", "marginal"})
"""Verdict statuses the audit runs when no workload list is given (spec section 7a)."""
DEFAULT_MAX_CREDITS = 500

STEPS: tuple[Step, ...] = (
    upload,
    resolve_version,
    split,
    wait_split,
    pii_scan,
    wait_pii,
    submit,
    wait_run,
    resolve_model_version,
    create_deployment,
    create_revision,
    wait_active,
    replay_and_score,
)
"""Every step, in order; each skips itself once its state key is set."""
LONG_WAITS: frozenset[Step] = frozenset({wait_run, wait_active})
"""Steps ``wait=False`` returns before, leaving the submitted work to a later ``run``."""
WORKLOAD_STOPPING = frozenset({"pii_disagreement"})
"""Error codes that stop the rest of the workload's candidates, not just the one."""

FLOOR_BY_STRUCTURE: dict[StructureClass, float] = {
    StructureClass.ENUM_LABEL: FLOOR_LABEL,
    StructureClass.SHORT_SPAN: FLOOR_LABEL,
    StructureClass.JSON_OBJECT: FLOOR_JSON,
}
"""Default quality floor per structure class (spec section 8); free text is never audited."""

type ScanWorkload = tuple[str, StructureClass, str]


def _scan(audit_dir: Path) -> tuple[str | None, list[ScanWorkload]]:
    """``(price_table_version, [(id, structure_class, verdict status)])`` from ``scan-report.json``."""
    path = audit_dir / SCAN_REPORT
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `dagnam audit scan` first")
    report = json.loads(path.read_text(encoding="utf-8"))
    version = report.get("price_table_version")
    workloads = [
        (
            str(entry["id"]),
            StructureClass(str(entry["structure_class"])),
            str((entry.get("verdict") or {}).get("status", "")),
        )
        for entry in report.get("workloads", [])
    ]
    return (version if isinstance(version, str) else None), workloads


def _select(
    audit_dir: Path, scanned: Sequence[ScanWorkload], workloads: Sequence[str] | None
) -> list[tuple[str, StructureClass]]:
    """The workloads to run: the ones named, else every audited verdict with derived files."""
    by_id = {workload_id: cls for workload_id, cls, _ in scanned}
    if workloads is None:
        chosen = [
            workload_id
            for workload_id, _, verdict in scanned
            if verdict in AUDITED_VERDICTS and (audit_dir / "workloads" / workload_id).is_dir()
        ]
    else:
        unknown = [w for w in workloads if w not in by_id]
        if unknown:
            raise ValueError(f"workloads not in {SCAN_REPORT}: {unknown}")
        chosen = list(workloads)
        missing = [
            w for w in chosen if not (audit_dir / "workloads" / w / "dataset.jsonl").exists()
        ]
        if missing:
            raise ValueError(f"workloads without derived data (run `dagnam audit scan`): {missing}")
    return [(workload_id, by_id[workload_id]) for workload_id in chosen]


def select_workloads(
    audit_dir: Path, workloads: Sequence[str] | None
) -> list[tuple[str, StructureClass]]:
    """The workloads ``run_audit`` would run for ``workloads``, from ``scan-report.json``."""
    return _select(audit_dir, _scan(audit_dir)[1], workloads)


def run_audit(
    audit_dir: Path,
    *,
    floor: float | None,
    workloads: Sequence[str] | None,
    max_credits: int,
    wait: bool,
    client: PlatformClient,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
    run_timeout: float = RUN_TIMEOUT_SECONDS,
    deploy_timeout: float = DEPLOY_TIMEOUT_SECONDS,
) -> AuditState:
    """Run (or resume) the frontier under ``audit_dir`` and return the state it reached.

    ``floor`` overrides the per-class default on the agreement lower bound;
    ``workloads`` names the workload ids to run (``None`` runs every
    ``candidate``/``marginal`` workload the scan derived); ``max_credits``
    halts the frontier before a submit that would exceed it, counting each
    candidate's training run and its holdout replay; ``wait=False``
    returns as soon as a step would block on a run or a deployment, and the
    next call resumes. ``sleep``/``now`` are the clock every wait uses.

    Idempotent: a second call over a finished audit makes no platform call.
    """
    state = load_state(audit_dir)
    price_table_version, scanned = _scan(audit_dir)
    selected = _select(audit_dir, scanned, workloads)
    state.price_table_version = price_table_version
    state.halted = None
    secrets = SecretStore(audit_dir)
    if state.project_id is None:
        created = client.create_project(
            {
                "title": f"workload-audit-{audit_dir.resolve().name}",
                "framework": "pytorch",
                "visibility": "private",
            }
        )
        state.project_id = str(created["id"])
    save_state(audit_dir, state)

    for workload_id, structure_class in selected:
        stop_workload = False
        for spec in CANDIDATES[structure_class]:
            step = state.candidate(workload_id, spec.kind)
            if spec.recipe_key is None or step.error or stop_workload:
                # The hosted floor is priced by the report, never run; a recorded
                # failure is terminal for this audit (delete the state to retry).
                continue
            ctx = StepContext(
                audit_dir=audit_dir,
                workload_id=workload_id,
                structure_class=structure_class,
                spec=spec,
                client=client,
                secrets=secrets,
                floor=floor if floor is not None else FLOOR_BY_STRUCTURE[structure_class],
                max_credits=max_credits,
                sleep=sleep,
                now=now,
                run_timeout=run_timeout,
                deploy_timeout=deploy_timeout,
            )
            for run_step in STEPS:
                if run_step in LONG_WAITS and not wait:
                    return state
                try:
                    state = run_step(state, ctx)
                except Exception as exc:
                    state.halted = {
                        "reason": "error",
                        "workload_id": workload_id,
                        "candidate": spec.kind.value,
                        "step": run_step.__name__,
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                    save_state(audit_dir, state)
                    raise
                save_state(audit_dir, state)
                if state.halted is not None:
                    return state
                if step.error:
                    stop_workload = error_code(step) in WORKLOAD_STOPPING
                    break
    return state


__all__ = [
    "AUDITED_VERDICTS",
    "DEFAULT_MAX_CREDITS",
    "FLOOR_BY_STRUCTURE",
    "LONG_WAITS",
    "SCAN_REPORT",
    "STEPS",
    "WORKLOAD_STOPPING",
    "run_audit",
    "select_workloads",
]
