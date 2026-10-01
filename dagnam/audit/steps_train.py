"""Training steps: pick the base, submit the recipe run, follow it, find the version it pushed.

The credit budget is enforced here, before a submit: the frontier halts with
``halted: budget`` rather than start a run it cannot pay for (spec U4, P5). What
it counts is a candidate's whole cost -- the most its training run can charge
plus the metered predictions of its holdout replay, which is the larger half.
The platform's own preflight still runs on every submit; its verdicts are
recorded per candidate and the frontier continues (spec section 9).
"""

from __future__ import annotations

import math
from pathlib import Path

from dagnam._core._retry import parse_retry_after
from dagnam._core.exceptions import APIError, LROFailedError, QuotaExceededError
from dagnam._types import JsonMapping, JsonObject
from dagnam.audit.candidates import CANDIDATES, CandidateKind
from dagnam.audit.state import AuditState, StepState
from dagnam.audit.steps import (
    StepContext,
    answered_rows,
    replay_file,
    required,
    string_field,
    wait_for,
)

CAPACITY_STATUS = 503
"""A capacity refusal: retried on the server's ``Retry-After`` until ``run_timeout``."""
CAPACITY_RETRY_DEFAULT_SECONDS = 60.0
REJECTED_STATUS = frozenset({400, 422})
"""Contract/preflight rejections: recorded as ``rejected_preflight``; the frontier continues."""
RUN_COMPLETED = "completed"
RUN_FAILED = frozenset({"failed", "cancelled", "timeout"})
RUN_SETTLED = frozenset({RUN_COMPLETED, *RUN_FAILED})
"""A run in one of these has stopped, and its recorded cost is what it charged."""
REPLAY_CREDITS_PER_ROW = 1
"""Every served prediction is metered at one credit."""
REPLAY_MARGIN = 1.1
"""P5: a replay is budgeted 10% over its row count."""

_TRAINING_CEILING: dict[CandidateKind, int] = {
    spec.kind: spec.training_credits_max or 0 for specs in CANDIDATES.values() for spec in specs
}


def _parameter_count(base: JsonObject) -> int | None:
    value = base.get("parameter_count")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def pick_base(ctx: StepContext) -> JsonObject | None:
    """The smallest ungated catalog base of the candidate's family within ``max_params``."""
    candidates: list[tuple[int, JsonObject]] = []
    for entry in ctx.client.list_foundation_catalog(limit=50):
        if not isinstance(entry, dict) or entry.get("family") != ctx.spec.base_family:
            continue
        params = _parameter_count(entry)
        if entry.get("gated") or params is None:
            continue
        if ctx.spec.max_params is not None and params > ctx.spec.max_params:
            continue
        candidates.append((params, entry))
    return min(candidates, key=lambda pair: pair[0])[1] if candidates else None


def projected_replay(rows: int) -> int:
    """What replaying ``rows`` holdout rows is budgeted at, in whole credits."""
    return math.ceil(rows * REPLAY_CREDITS_PER_ROW * REPLAY_MARGIN)


def _replay_spent(step: StepState, answers: Path | None) -> float:
    """What a candidate's replay spent: measured, else projected -- never 0 because unread.

    A replay whose balance read failed is counted at its projection over the
    calls it made; one a Ctrl+C cut short that never resumed, at its
    projection over the rows its answers file shows answered.
    """
    if step.replay_cost_credits is not None:
        return step.replay_cost_credits
    if step.scored:
        calls = (step.latency or {}).get("calls")
        return projected_replay(calls if isinstance(calls, int) else 0)
    return projected_replay(0 if answers is None else answered_rows(answers))


def credits_spent(state: AuditState, audit_dir: Path | None = None) -> float:
    """Training plus replay credits over every candidate, counting what may still come.

    A settled run costs what was recorded for it (the charge the server
    reported, else its estimate). A run still going can cost up to its
    recipe's ceiling, so it is counted at no less than that -- the budget must
    hold with every submitted run finishing at its dearest. ``audit_dir`` is
    where an interrupted replay's answers are read from.
    """
    total = state.retired_cost_credits
    for workload_id, candidates in state.workloads.items():
        for kind, step in candidates.items():
            training = step.training_cost_credits or 0.0
            if step.run_id is not None and step.run_status not in RUN_SETTLED:
                training = max(training, _TRAINING_CEILING[kind])
            answers = None if audit_dir is None else replay_file(audit_dir, workload_id, kind)
            total += training + _replay_spent(step, answers)
    return total


def over_budget(state: AuditState, ctx: StepContext, cost: float, next_step: str) -> bool:
    """Halt with ``budget`` when spending ``cost`` more would pass ``max_credits`` (spec P5)."""
    spent = credits_spent(state, ctx.audit_dir)
    if spent + cost <= ctx.max_credits:
        return False
    state.halted = {
        "reason": "budget",
        "spent_credits": spent,
        "max_credits": ctx.max_credits,
        "next": next_step,
    }
    return True


def _create_run(ctx: StepContext, payload: JsonObject) -> JsonObject:
    """Submit, retrying a 503 capacity refusal on ``Retry-After`` until ``run_timeout``.

    The client already retries transient statuses within its short budget;
    this loop is the long horizon a single-GPU queue needs.
    """
    deadline = ctx.now() + ctx.run_timeout
    while True:
        try:
            return ctx.client.create_foundation_run(payload)
        except APIError as exc:
            if exc.status_code != CAPACITY_STATUS or ctx.now() >= deadline:
                raise
            delay = parse_retry_after(exc.retry_after_header, cap=ctx.run_timeout)
            ctx.sleep(delay if delay is not None else CAPACITY_RETRY_DEFAULT_SECONDS)


def submit(state: AuditState, ctx: StepContext) -> AuditState:
    """Submit the recipe run for this candidate; done once ``run_id`` is set.

    Before submitting, the budget check projects this candidate at its own
    whole cost -- the most its run can charge plus its replay, one credit a
    holdout row and 10% more -- and halts the audit with ``halted: budget``
    when that would take the credits spent past ``max_credits``.
    """
    step = ctx.step(state)
    if step.run_id is not None:
        return state
    ceiling = _TRAINING_CEILING[ctx.spec.kind]
    if over_budget(state, ctx, ceiling + projected_replay(ctx.holdout_rows()), ctx.label):
        return state
    base = pick_base(ctx)
    if base is None:
        step.error = (
            f"no_base: no ungated {ctx.spec.base_family!r} base in the catalog "
            f"within {ctx.spec.max_params} parameters"
        )
        return state
    payload: JsonObject = {
        "project_id": required(state.project_id, "project_id"),
        "base_catalog_entry_id": str(base["id"]),
        "dataset_version_id": required(step.version_id, "version_id"),
        "recipe_key": required(ctx.spec.recipe_key, "recipe_key"),
        "hyperparameters": {},
        "dataset_field_bindings": {},
    }
    try:
        run = _create_run(ctx, payload)
    except QuotaExceededError as exc:
        state.halted = {"reason": "budget", "detail": str(exc), "next": ctx.label}
        return state
    except APIError as exc:
        if exc.status_code not in REJECTED_STATUS:
            raise
        step.error = f"rejected_preflight: {exc.message}"
        return state
    step.base = string_field(base, "display_name") or str(base["id"])
    step.run_id = str(run["run_id"])
    step.training_job_id = string_field(run, "training_job_id")
    step.run_status = string_field(run, "status") or "queued"
    estimate = run.get("credits_estimate_max")
    if isinstance(estimate, int | float) and not isinstance(estimate, bool):
        step.training_cost_credits = float(estimate)
    return state


def wait_run(state: AuditState, ctx: StepContext) -> AuditState:
    """Follow the run to a terminal status; a failure is recorded, the frontier continues.

    Either way the run's cost becomes what the server says it charged
    (``credits_consumed``) when it says -- the estimate is only a ceiling.
    """
    step = ctx.step(state)
    if step.run_status == RUN_COMPLETED:
        return state
    if step.run_status in RUN_FAILED:
        # Terminal already, and nothing recorded it as an error: `dagnam audit
        # cancel` stopped the run between two `audit run`s. Polling a job that
        # will never move again is the one way a resumed audit hangs.
        step.error = f"run_{step.run_status}: the run was already terminal when this audit resumed"
        return state
    run_id = required(step.run_id, "run_id")
    seen: list[JsonMapping] = []

    def poll() -> JsonMapping:
        seen.append(ctx.client.get_foundation_run(run_id))
        return seen[-1]

    try:
        wait_for(
            ctx,
            poll,
            success={RUN_COMPLETED},
            failure=RUN_FAILED,
            timeout=ctx.run_timeout,
            name=f"run {run_id}",
        )
    except LROFailedError as exc:
        step.run_status = exc.state
        step.error = f"run_{exc.state}: {exc.detail or 'no reason given'}"
    else:
        step.run_status = RUN_COMPLETED
    measured = seen[-1].get("credits_consumed")
    if isinstance(measured, int | float) and not isinstance(measured, bool):
        step.training_cost_credits = float(measured)
    return state


def resolve_model_version(state: AuditState, ctx: StepContext) -> AuditState:
    """Record the model version the completed run pushed; done once ``model_version_id`` is set.

    Only the run's own answer counts (K2): the account's newest version may be
    another run's -- a Studio retrain of a sibling workload -- so a run that
    does not name one is a recorded error, never a guess from the registry.
    """
    step = ctx.step(state)
    if step.model_version_id is not None:
        return state
    run = ctx.client.get_foundation_run(required(step.run_id, "run_id"))
    found = string_field(run, "model_version_id")
    if found is None:
        step.error = "no_model_version: the server did not report the version this run pushed"
        return state
    step.model_version_id = found
    return state


__all__ = [
    "CAPACITY_RETRY_DEFAULT_SECONDS",
    "CAPACITY_STATUS",
    "REJECTED_STATUS",
    "REPLAY_CREDITS_PER_ROW",
    "REPLAY_MARGIN",
    "RUN_COMPLETED",
    "RUN_FAILED",
    "RUN_SETTLED",
    "credits_spent",
    "over_budget",
    "pick_base",
    "projected_replay",
    "resolve_model_version",
    "submit",
    "wait_run",
]
