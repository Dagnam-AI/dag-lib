"""Training steps: pick the base, submit the recipe run, follow it, find the version it pushed.

The credit budget is enforced here, before a submit: the frontier halts with
``halted: budget`` rather than start a run it cannot pay for (spec U4). What
it counts is a candidate's whole cost -- the training run plus the metered
predictions of its holdout replay, which is the larger half. The
platform's own preflight still runs on every submit; its verdicts are recorded
per candidate and the frontier continues (spec section 9).
"""

from __future__ import annotations

from datetime import UTC, datetime

from dagnam._core._retry import parse_retry_after
from dagnam._core.exceptions import APIError, LROFailedError, QuotaExceededError
from dagnam._types import JsonObject
from dagnam.audit.state import AuditState
from dagnam.audit.steps import StepContext, required, string_field, wait_for

CAPACITY_STATUS = 503
"""A capacity refusal: retried on the server's ``Retry-After`` until ``run_timeout``."""
CAPACITY_RETRY_DEFAULT_SECONDS = 60.0
REJECTED_STATUS = frozenset({400, 422})
"""Contract/preflight rejections: recorded as ``rejected_preflight``; the frontier continues."""
RUN_COMPLETED = "completed"
RUN_FAILED = frozenset({"failed", "cancelled", "timeout"})


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


def credits_spent(state: AuditState) -> tuple[float, float]:
    """``(total, largest)`` over every candidate's training plus measured replay credits."""
    costs = [
        (step.training_cost_credits or 0.0) + (step.replay_cost_credits or 0.0)
        for candidates in state.workloads.values()
        for step in candidates.values()
    ]
    return sum(costs), max(costs, default=0.0)


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

    Before submitting, the budget check assumes the next candidate costs at
    least as much as the most expensive one so far -- training plus the
    credits its holdout replay was measured to burn: ``spent + largest >
    max_credits`` (or ``spent >= max_credits``) halts the audit with
    ``halted: budget``.
    """
    step = ctx.step(state)
    if step.run_id is not None:
        return state
    spent, largest = credits_spent(state)
    if spent >= ctx.max_credits or spent + largest > ctx.max_credits:
        state.halted = {
            "reason": "budget",
            "spent_credits": spent,
            "max_credits": ctx.max_credits,
            "next": ctx.label,
        }
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
    """Follow the run to a terminal status; a failure is recorded, the frontier continues."""
    step = ctx.step(state)
    if step.run_status == RUN_COMPLETED:
        return state
    run_id = required(step.run_id, "run_id")
    try:
        run = wait_for(
            ctx,
            lambda: ctx.client.get_foundation_run(run_id),
            success={RUN_COMPLETED},
            failure=RUN_FAILED,
            timeout=ctx.run_timeout,
            name=f"run {run_id}",
        )
    except LROFailedError as exc:
        step.run_status = exc.state
        step.error = f"run_{exc.state}: {exc.detail or 'no reason given'}"
        return state
    step.run_status = RUN_COMPLETED
    measured = run.get("credits_consumed")
    if isinstance(measured, int | float) and not isinstance(measured, bool):
        step.training_cost_credits = float(measured)
    return state


def _instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _newest_version_since(ctx: StepContext, created_at: str | None) -> str | None:
    """The registry version this account pushed most recently after ``created_at``.

    The run read carries no version id (G0), so this is the same scan the G0
    driver used: every entry the account owns, newest version created after
    the run was.
    """
    since = _instant(created_at) if created_at else None
    newest: tuple[datetime, str] | None = None
    for entry in ctx.client.list_model_entries(limit=100):
        if not isinstance(entry, dict):
            continue
        for version in ctx.client.list_model_versions(str(entry["id"])):
            if not isinstance(version, dict):
                continue
            stamp = string_field(version, "created_at")
            if stamp is None:
                continue
            when = _instant(stamp)
            if since is not None and when < since:
                continue
            if newest is None or when > newest[0]:
                newest = (when, str(version["id"]))
    return newest[1] if newest else None


def resolve_model_version(state: AuditState, ctx: StepContext) -> AuditState:
    """Find the model version the completed run pushed; done once ``model_version_id`` is set."""
    step = ctx.step(state)
    if step.model_version_id is not None:
        return state
    run = ctx.client.get_foundation_run(required(step.run_id, "run_id"))
    found = (
        string_field(run, "model_version_id")
        or string_field(run, "pushed_version_id")
        or _newest_version_since(ctx, string_field(run, "created_at"))
    )
    if found is None:
        step.error = "no_model_version: the run completed but no registry version was pushed"
        return state
    step.model_version_id = found
    return state


__all__ = [
    "CAPACITY_RETRY_DEFAULT_SECONDS",
    "CAPACITY_STATUS",
    "REJECTED_STATUS",
    "RUN_COMPLETED",
    "RUN_FAILED",
    "credits_spent",
    "pick_base",
    "resolve_model_version",
    "submit",
    "wait_run",
]
