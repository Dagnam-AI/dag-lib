"""``run_audit``: the resumable frontier over workloads x candidates (spec U4, D9).

One loop, no branching on kind: for each selected workload, for each
candidate in :data:`~dagnam.audit.candidates.CANDIDATES`, run the step list.
The state is saved after every step -- before anything is published, so a
Ctrl+C on a slow publish never loses a step that was paid for -- and the
next call resumes from it; a ``KeyboardInterrupt`` is never caught. A hard
failure inside a step is recorded as ``halted: error`` before it propagates,
and a cancel in the account ends the run as ``halted: cancelled`` (K1).

Verdicts and prices belong to the scan report and Task 5's economics: this
module reads ``scan-report.json`` for each workload's structure class and
verdict, and records agreement, latency and credits spent for the report
to price; it never computes a cost itself.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from typing import Any

from dagnam.audit.candidates import CANDIDATES
from dagnam.audit.publish import DELETED, Publisher, installed_version
from dagnam.audit.secrets import SecretStore
from dagnam.audit.state import AuditState, StepState, load_state, lock_audit, save_state
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
from dagnam.audit.steps_train import (
    credits_spent,
    projected_replay,
    resolve_model_version,
    submit,
    wait_run,
)
from dagnam.audit.structure import StructureClass
from dagnam.audit.thresholds import FLOOR_JSON, FLOOR_LABEL

SCAN_REPORT = "scan-report.json"
AUDITED_VERDICTS = frozenset({"candidate", "marginal"})
"""Verdict statuses the audit runs when no workload list is given (spec section 7a)."""
PLAN_ROUNDING = 100
"""The default ceiling is the plan's estimate rounded up to a multiple of this (spec P5)."""

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


def _scan(audit_dir: Path) -> tuple[dict[str, Any], list[ScanWorkload]]:
    """``(the scan report, [(id, structure_class, verdict status)])`` from ``scan-report.json``."""
    path = audit_dir / SCAN_REPORT
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `dagnam audit scan` first")
    report = json.loads(path.read_text(encoding="utf-8"))
    workloads = [
        (
            str(entry["id"]),
            StructureClass(str(entry["structure_class"])),
            str((entry.get("verdict") or {}).get("status", "")),
        )
        for entry in report.get("workloads", [])
    ]
    return report, workloads


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


def plan_credits(
    audit_dir: Path,
    selected: Sequence[tuple[str, StructureClass]],
    state: AuditState | None = None,
) -> int:
    """The ceiling a run without ``--max-credits`` is held to (spec P5).

    What the audit already spent, plus every trained candidate of every
    selected workload still to finish at its projected cost -- its run's
    ceiling unless it was already submitted, plus its replay -- rounded up to
    :data:`PLAN_ROUNDING`, so the listing the user confirms names it. A
    candidate that scored or stopped on an error adds nothing more.
    """
    state = AuditState() if state is None else state
    total = credits_spent(state, audit_dir)
    for workload_id, structure_class in selected:
        meta = json.loads(
            (audit_dir / "workloads" / workload_id / "meta.json").read_text(encoding="utf-8")
        )
        replay = projected_replay(int(meta["splits"]["eval_holdout"]))
        for spec in CANDIDATES[structure_class]:
            if spec.training_credits_max is None:
                continue
            step = state.workloads.get(workload_id, {}).get(spec.kind, StepState())
            if step.scored or step.error is not None:
                continue
            total += replay + (spec.training_credits_max if step.run_id is None else 0)
    return math.ceil(total / PLAN_ROUNDING) * PLAN_ROUNDING


def run_audit(
    audit_dir: Path,
    *,
    floor: float | None,
    workloads: Sequence[str] | None,
    max_credits: int,
    wait: bool,
    client: PlatformClient,
    publisher: Publisher | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
    run_timeout: float = RUN_TIMEOUT_SECONDS,
    deploy_timeout: float = DEPLOY_TIMEOUT_SECONDS,
) -> AuditState:
    """Run (or resume) the frontier under ``audit_dir`` and return the state it reached.

    ``publisher`` mirrors the run into the account as it goes; ``None``
    (``--local-only``) keeps every number on this machine. A publish failure is
    never allowed to stop the run.

    ``floor`` overrides the per-class default on the agreement lower bound;
    ``workloads`` names the workload ids to run (``None`` runs every
    ``candidate``/``marginal`` workload the scan derived); ``max_credits``
    halts the frontier before a submit or a replay that could take the credits
    spent past it, projecting each at the most it can cost; ``wait=False``
    returns as soon as a step would block on a run or a deployment, and the
    next call resumes. ``sleep``/``now`` are the clock every wait uses.

    Idempotent: a second call over a finished audit makes no platform call.
    One call at a time holds ``audit_dir``: a second raises ``AuditBusyError``.
    """
    with lock_audit(audit_dir):
        return _run_audit(
            audit_dir,
            floor=floor,
            workloads=workloads,
            max_credits=max_credits,
            wait=wait,
            client=client,
            publisher=publisher,
            sleep=sleep,
            now=now,
            run_timeout=run_timeout,
            deploy_timeout=deploy_timeout,
        )


def _run_audit(
    audit_dir: Path,
    *,
    floor: float | None,
    workloads: Sequence[str] | None,
    max_credits: int,
    wait: bool,
    client: PlatformClient,
    publisher: Publisher | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
    run_timeout: float = RUN_TIMEOUT_SECONDS,
    deploy_timeout: float = DEPLOY_TIMEOUT_SECONDS,
) -> AuditState:
    """:func:`run_audit`, with ``audit_dir`` held."""
    state = load_state(audit_dir)
    if publisher is not None:
        publisher.follow(state)
        # Before anything that can wait: a cancel landing from here on sticks (K1b),
        # and an audit the account already deleted stops the run here.
        publisher.resume()
        if publisher.stopped is not None:
            return _stopped(audit_dir, state, publisher.stopped)
    scan, scanned = _scan(audit_dir)
    selected = _select(audit_dir, scanned, workloads)
    price_table_version = scan.get("price_table_version")
    state.price_table_version = (
        price_table_version if isinstance(price_table_version, str) else None
    )
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
    if publisher is not None:
        # Without --floor each class keeps its own default, so the audit header
        # records the strictest of them; every candidate's agreement carries
        # the floor it was actually held to.
        publisher.start(
            audit_dir,
            scan,
            [workload_id for workload_id, _ in selected],
            floor=floor if floor is not None else max(FLOOR_BY_STRUCTURE.values()),
            max_credits=max_credits,
            sdk_version=installed_version(),
        )
    save_state(audit_dir, state)

    for workload_id, structure_class in selected:
        stop_workload = False
        for spec in CANDIDATES[structure_class]:
            step = state.candidate(workload_id, spec.kind)
            if spec.recipe_key is None or stop_workload:
                # The hosted floor is priced by the report, never run; a workload
                # a stopping error ended runs no further candidate.
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
            if publisher is not None:
                publisher.candidate(ctx, step)
                # A step the account never acknowledged -- all of them when the
                # audit only reached it now -- goes out before the next one;
                # without this the candidate would sit at its last acknowledged
                # status until a step it has not reached yet moves it.
                publisher.backfill(ctx, step)
                if publisher.stopped is not None:
                    return _stopped(audit_dir, state, publisher.stopped)
            if step.error:
                # A recorded failure is terminal for this audit (delete the state
                # to retry) -- but the publisher above still had to see it, or a
                # run that reached the account late would leave the candidate out
                # of the audit entirely. Saved because that publish is what
                # gave the candidate its id: unsaved, the next run opens a
                # second candidate for the same failure.
                save_state(audit_dir, state)
                continue
            for run_step in STEPS:
                if run_step in LONG_WAITS and not wait:
                    return _finish(audit_dir, state, publisher)
                before = asdict(step)
                try:
                    state = run_step(state, ctx)
                except Exception as exc:
                    if publisher is not None and publisher.gone():
                        # The owner deleted the audit and the platform took the
                        # job or the endpoint this step was polling with it.
                        return _stopped(audit_dir, state, DELETED)
                    state.halted = {
                        "reason": "error",
                        "workload_id": workload_id,
                        "candidate": spec.kind.value,
                        "step": run_step.__name__,
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                    save_state(audit_dir, state)
                    _halt(publisher, "error")
                    raise
                # On disk before it is published: the publish is a network call
                # that can hang, and a step it loses must not be paid for twice.
                save_state(audit_dir, state)
                if publisher is not None and state.halted is None and asdict(step) != before:
                    # Only a step that recorded something is published: a resumed
                    # run walks every step again and the ones already done change
                    # nothing. A step that halted the run did not do what its
                    # name says (the budget check refuses the submit), so it is
                    # the halt that is published, never the step it stopped.
                    publisher.step(ctx, run_step.__name__, step)
                    save_state(audit_dir, state)  # what the account acknowledged
                    if publisher.stopped is not None:
                        return _stopped(audit_dir, state, publisher.stopped)
                if state.halted is not None:
                    _halt(publisher, str(state.halted["reason"]))
                    return state
                if step.error:
                    stop_workload = error_code(step) in WORKLOAD_STOPPING
                    break
    return _finish(audit_dir, state, publisher)


def _halt(publisher: Publisher | None, reason: str) -> None:
    """Tell the account the run stopped short; a ``None`` publisher keeps it local."""
    if publisher is not None:
        publisher.halt(reason)


def _finish(audit_dir: Path, state: AuditState, publisher: Publisher | None) -> AuditState:
    """End a run that was not halted: flush what the account has not heard yet, and save it."""
    if publisher is None:
        return state
    publisher.flush()
    save_state(audit_dir, state)
    return state if publisher.stopped is None else _stopped(audit_dir, state, publisher.stopped)


def _stopped(audit_dir: Path, state: AuditState, reason: str) -> AuditState:
    """Stop where the account's cancel or delete caught the run: nothing more is run or published."""
    state.halted = {"reason": reason}
    save_state(audit_dir, state)
    return state


__all__ = [
    "AUDITED_VERDICTS",
    "FLOOR_BY_STRUCTURE",
    "LONG_WAITS",
    "PLAN_ROUNDING",
    "SCAN_REPORT",
    "STEPS",
    "WORKLOAD_STOPPING",
    "plan_credits",
    "run_audit",
    "select_workloads",
]
