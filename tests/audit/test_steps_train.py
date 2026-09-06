"""Training steps: base choice, the budget gate, capacity retry, run follow-up, the pushed version."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from tests.audit._platform import Clock, FakePlatform

from dagnam._core.exceptions import APIError, QuotaExceededError
from dagnam.audit.candidates import HEAD_TUNE, SFT_SMALL, CandidateKind
from dagnam.audit.state import AuditState, StepState
from dagnam.audit.steps import StepContext
from dagnam.audit.steps_train import (
    _instant,
    _newest_version_since,
    credits_spent,
    pick_base,
    resolve_model_version,
    submit,
    wait_run,
)


def _ready(kind: CandidateKind = CandidateKind.HEAD_TUNE) -> AuditState:
    state = AuditState(project_id="proj-1")
    state.workloads["w1"] = {
        kind: StepState(dataset_id="ds-1", version_id="ds-1-v2", split_done=True)
    }
    return state


def test_pick_base_is_the_smallest_ungated_sized_base_of_the_family(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    platform.bases.append("junk")
    platform.bases.append({"id": "bert-bool", "family": "bert", "parameter_count": True})
    head = pick_base(make_ctx(spec=HEAD_TUNE))
    assert head is not None
    assert head["id"] == "bert-small"
    small = pick_base(make_ctx(spec=SFT_SMALL))
    assert small is not None
    assert small["id"] == "qwen-05b"
    platform.bases = [b for b in platform.bases if isinstance(b, dict) and b["id"] != "qwen-05b"]
    assert pick_base(make_ctx(spec=SFT_SMALL)) is None  # qwen-7b is over max_params


def test_credits_spent_sums_every_candidate() -> None:
    state = AuditState()
    assert credits_spent(state) == (0.0, 0.0)
    state.candidate("w1", CandidateKind.HEAD_TUNE).training_cost_credits = 100.0
    state.candidate("w2", CandidateKind.SFT_SMALL).training_cost_credits = 250.0
    state.candidate("w2", CandidateKind.HOSTED_FLOOR)
    assert credits_spent(state) == (350.0, 250.0)


def test_submit_sends_the_frozen_payload_and_records_the_run(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    ctx = make_ctx()
    state = submit(_ready(), ctx)
    step = ctx.step(state)
    assert platform.submitted == [
        {
            "project_id": "proj-1",
            "base_catalog_entry_id": "bert-small",
            "dataset_version_id": "ds-1-v2",
            "recipe_key": "head-tune-text-classification@1.1",
            "hyperparameters": {},
            "dataset_field_bindings": {},
        }
    ]
    assert (step.base, step.run_id, step.training_job_id) == ("BERT small", "run-1", "job-1")
    assert step.run_status == "queued"
    assert step.training_cost_credits == 100.0
    submit(state, ctx)
    assert platform.submits == 1


def test_submit_without_an_estimate_or_display_name(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    platform.credits_estimate_max = None
    ctx = make_ctx(spec=SFT_SMALL)
    step = ctx.step(submit(_ready(CandidateKind.SFT_SMALL), ctx))
    assert step.base == "qwen-05b"
    assert step.training_cost_credits is None
    platform.credits_estimate_max = True
    ctx2 = make_ctx(workload_id="w2", spec=SFT_SMALL)
    state = AuditState(project_id="proj-1")
    state.workloads["w2"] = {CandidateKind.SFT_SMALL: StepState(version_id="v")}
    assert ctx2.step(submit(state, ctx2)).training_cost_credits is None


def test_submit_halts_on_budget_before_calling_the_platform(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    state = _ready()
    state.candidate("w0", CandidateKind.SFT_SMALL).training_cost_credits = 400.0
    ctx = make_ctx(max_credits=500)
    submit(state, ctx)
    assert state.halted == {
        "reason": "budget",
        "spent_credits": 400.0,
        "max_credits": 500,
        "next": "w1/head_tune",
    }
    state.halted = None
    submit(state, make_ctx(max_credits=400))
    assert state.halted is not None
    assert state.halted["reason"] == "budget"
    assert platform.call_log == []


def test_submit_records_no_base(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    platform.bases = []
    ctx = make_ctx()
    step = ctx.step(submit(_ready(), ctx))
    assert step.error == "no_base: no ungated 'bert' base in the catalog within None parameters"
    assert platform.submits == 0


def test_submit_402_halts_the_budget_and_422_is_a_recorded_rejection(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    ctx = make_ctx()
    platform.submit_errors = [QuotaExceededError("plan limit")]
    state = submit(_ready(), ctx)
    assert state.halted == {"reason": "budget", "detail": "plan limit", "next": "w1/head_tune"}

    platform.submit_errors = [APIError(422, "dataset too small")]
    step = ctx.step(submit(_ready(), ctx))
    assert step.error == "rejected_preflight: dataset too small"
    assert step.run_id is None

    platform.submit_errors = [APIError(500, "boom")]
    with pytest.raises(APIError, match="boom"):
        submit(_ready(), ctx)


def test_capacity_503_is_retried_after_retry_after(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, clock: Clock
) -> None:
    platform.submit_errors = [
        APIError(503, "busy", retry_after_header="7"),
        APIError(503, "busy"),
    ]
    ctx = make_ctx()
    step = ctx.step(submit(_ready(), ctx))
    assert step.run_id == "run-1"
    assert clock.sleeps == [7.0, 60.0]
    assert platform.call_log.count("create_foundation_run") == 3


def test_capacity_503_past_the_run_timeout_propagates(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, clock: Clock
) -> None:
    platform.submit_errors = [
        APIError(503, "busy", retry_after_header="100"),
        APIError(503, "busy"),
    ]
    ctx = make_ctx(run_timeout=50.0)
    with pytest.raises(APIError, match="busy"):
        submit(_ready(), ctx)
    assert clock.sleeps == [50.0]


def test_wait_run_follows_to_completion_and_takes_the_measured_credits(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    platform.run_extra = {"credits_consumed": 42}
    ctx = make_ctx()
    state = wait_run(submit(_ready(), ctx), ctx)
    step = ctx.step(state)
    assert step.run_status == "completed"
    assert step.training_cost_credits == 42.0
    assert platform.call_log.count("get_foundation_run") == 3
    wait_run(state, ctx)
    assert platform.call_log.count("get_foundation_run") == 3


def test_wait_run_keeps_the_estimate_when_nothing_was_measured(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    platform.run_extra = {"credits_consumed": True}
    ctx = make_ctx()
    assert ctx.step(wait_run(submit(_ready(), ctx), ctx)).training_cost_credits == 100.0


def test_wait_run_records_a_failed_run(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    platform.run_final_status = "failed"
    platform.run_error = "CUDA out of memory"
    ctx = make_ctx()
    step = ctx.step(wait_run(submit(_ready(), ctx), ctx))
    assert step.run_status == "failed"
    assert step.error == "run_failed: CUDA out of memory"
    platform.run_final_status = "cancelled"
    platform.run_error = None
    step = ctx.step(wait_run(submit(_ready(), ctx), ctx))
    assert step.error == "run_cancelled: no reason given"


def test_instant_parses_z_naive_and_aware() -> None:
    assert _instant("2026-09-06T10:00:00Z") == datetime(2026, 9, 6, 10, tzinfo=UTC)
    assert _instant("2026-09-06T10:00:00") == datetime(2026, 9, 6, 10, tzinfo=UTC)
    assert _instant("2026-09-06T12:00:00+02:00") == datetime(2026, 9, 6, 10, tzinfo=UTC)


def test_newest_version_since_scans_the_registry(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    ctx = make_ctx()
    assert _newest_version_since(ctx, "2026-09-06T10:00:00Z") == "mv-pushed"
    assert _newest_version_since(ctx, None) == "mv-pushed"
    assert _newest_version_since(ctx, "2026-09-06T11:00:00Z") is None
    platform.pushes_version = False
    assert _newest_version_since(ctx, None) == "mv-2020"


def test_resolve_model_version_prefers_the_run_field_then_scans(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    ctx = make_ctx()
    platform.run_polls = 0
    platform.run_extra = {"model_version_id": "mv-from-run"}
    state = resolve_model_version(wait_run(submit(_ready(), ctx), ctx), ctx)
    assert ctx.step(state).model_version_id == "mv-from-run"
    calls = len(platform.call_log)
    resolve_model_version(state, ctx)
    assert len(platform.call_log) == calls

    platform.run_extra = {"pushed_version_id": "mv-pushed-field"}
    state = _ready()
    assert (
        ctx.step(resolve_model_version(wait_run(submit(state, ctx), ctx), ctx)).model_version_id
        == "mv-pushed-field"
    )

    platform.run_extra = {}
    state = _ready()
    assert (
        ctx.step(resolve_model_version(wait_run(submit(state, ctx), ctx), ctx)).model_version_id
        == "mv-pushed"
    )


def test_resolve_model_version_records_a_missing_push(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    platform.pushes_version = False
    platform.run_polls = 0
    ctx = make_ctx()
    step = ctx.step(resolve_model_version(wait_run(submit(_ready(), ctx), ctx), ctx))
    assert step.model_version_id is None
    assert step.error == "no_model_version: the run completed but no registry version was pushed"
