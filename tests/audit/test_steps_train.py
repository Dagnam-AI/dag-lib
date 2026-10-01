"""Training steps: base choice, the budget gate, capacity retry, run follow-up, the pushed version."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from tests.audit._platform import Clock, FakePlatform

from dagnam._core.exceptions import APIError, QuotaExceededError
from dagnam.audit.candidates import HEAD_TUNE, SFT_SMALL, TRAINING_CREDITS_MAX, CandidateKind
from dagnam.audit.state import AuditState, StepState
from dagnam.audit.steps import StepContext, replay_file
from dagnam.audit.steps_train import (
    credits_spent,
    pick_base,
    projected_replay,
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


def test_credits_spent_counts_a_settled_run_at_its_charge_and_a_live_one_at_its_ceiling() -> None:
    """B8: a run still going can cost up to its recipe ceiling, whatever the server first quoted."""
    state = AuditState()
    assert credits_spent(state) == 0.0
    head = state.candidate("w1", CandidateKind.HEAD_TUNE)
    head.run_id, head.run_status = "run-1", "completed"
    head.training_cost_credits, head.replay_cost_credits = 7.0, 396.0
    live = state.candidate("w2", CandidateKind.SFT_SMALL)
    live.run_id, live.run_status, live.training_cost_credits = "run-2", "running", 4.0
    state.candidate("w2", CandidateKind.HOSTED_FLOOR).replay_cost_credits = 4.0
    assert credits_spent(state) == 7.0 + 396.0 + TRAINING_CREDITS_MAX + 4.0
    live.training_cost_credits = 500.0
    assert credits_spent(state) == 7.0 + 396.0 + 500.0 + 4.0
    live.run_status = "failed"
    live.training_cost_credits = None
    assert credits_spent(state) == 7.0 + 396.0 + 4.0


def test_a_replay_is_projected_at_a_credit_a_row_plus_ten_percent() -> None:
    assert [projected_replay(n) for n in (0, 3, 4, 10, 8_000)] == [0, 4, 5, 11, 8_800]


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
            "recipe_key": "head-tune-text-classification@1.2",
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


def test_the_budget_projection_counts_the_previous_replay(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    """20 credits of training would fit; the 396-credit replay it implies does not."""
    state = _ready()
    done = state.candidate("w0", CandidateKind.SFT_SMALL)
    done.training_cost_credits = 10.0
    done.replay_cost_credits = 396.0
    submit(state, make_ctx(max_credits=500))
    assert state.halted == {
        "reason": "budget",
        "spent_credits": 406.0,
        "max_credits": 500,
        "next": "w1/head_tune",
    }
    assert platform.call_log == []


def test_the_first_submit_is_held_to_the_ceiling(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    """B8: `--max-credits 10` never submits a run whose own ceiling is 120 credits."""
    state = submit(_ready(), make_ctx(max_credits=10))
    assert state.halted == {
        "reason": "budget",
        "spent_credits": 0.0,
        "max_credits": 10,
        "next": "w1/head_tune",
    }
    assert platform.call_log == []


def test_the_next_submit_is_projected_at_its_own_ceiling_not_the_largest_so_far(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    """B8: a cheap first candidate no longer vouches for a dearer second one.

    9 spent, then 120 for this run's ceiling and 5 for its 4-row replay: 134.
    """
    state = _ready()
    done = state.candidate("w0", CandidateKind.HEAD_TUNE)
    done.run_id, done.run_status = "run-0", "completed"
    done.training_cost_credits, done.replay_cost_credits = 5.0, 4.0
    submit(state, make_ctx(max_credits=133))
    assert state.halted is not None
    assert state.halted["spent_credits"] == 9.0
    assert platform.submits == 0
    state.halted = None
    submit(state, make_ctx(max_credits=134))
    assert state.halted is None
    assert platform.submits == 1


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


def test_a_failed_run_is_charged_what_the_server_reports_not_its_estimate(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    """B12: the estimate is a ceiling, not a charge -- a run that died early cost what it used."""
    platform.run_final_status = "failed"
    platform.run_extra = {"credits_consumed": 3}
    ctx = make_ctx()
    step = ctx.step(wait_run(submit(_ready(), ctx), ctx))
    assert step.run_status == "failed"
    assert step.training_cost_credits == 3.0


def test_resolve_model_version_takes_the_version_the_run_reports(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    ctx = make_ctx()
    platform.run_polls = 0
    state = resolve_model_version(wait_run(submit(_ready(), ctx), ctx), ctx)
    assert ctx.step(state).model_version_id == "mv-pushed"
    calls = len(platform.call_log)
    resolve_model_version(state, ctx)
    assert len(platform.call_log) == calls


@pytest.mark.parametrize("reported", [{}, {"model_version_id": None}])
def test_a_run_that_reports_no_version_is_an_error_never_a_registry_guess(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, reported: dict[str, None]
) -> None:
    """B7: the account's newest version can be another run's -- a Studio retrain, say.

    So a completed run that does not name the version it pushed is recorded as
    such, and nothing reads the registry to guess.
    """
    platform.pushes_version = False
    platform.run_extra = dict(reported)
    platform.run_polls = 0
    ctx = make_ctx()
    state = wait_run(submit(_ready(), ctx), ctx)
    before = list(platform.call_log)
    step = ctx.step(resolve_model_version(state, ctx))
    assert step.model_version_id is None
    assert step.error == ("no_model_version: the server did not report the version this run pushed")
    assert platform.call_log == [*before, "get_foundation_run"]


def test_a_replay_whose_cost_was_never_read_counts_at_its_projection(tmp_path: Path) -> None:
    """M2: a failed balance read left `replay_cost_credits` unset, which the budget read as 0."""
    state = AuditState()
    scored = state.candidate("w1", CandidateKind.HEAD_TUNE)
    scored.scored = True
    scored.latency = {"p50": 1.0, "p95": 2.0, "calls": 200, "errors": 0}
    assert credits_spent(state) == projected_replay(200)

    # A replay a Ctrl+C cut short, whose candidate never resumed: two rows answered.
    cut = state.candidate("w2", CandidateKind.SFT_SMALL)
    cut.deployment_id = "dep-2"
    answers = replay_file(tmp_path, "w2", CandidateKind.SFT_SMALL)
    answers.parent.mkdir(parents=True)
    answers.write_text(
        '{"deployment_id": "dep-2", "balance_before": 100}\n'
        '{"row": 0, "answer": "a", "ms": 1.0}\n'
        '{"row": 1, "answer": null, "ms": 0.0}\n'
        '{"row": 2, "answer": "b", "ms": 1.0}\n',
        encoding="utf-8",
    )
    assert credits_spent(state, tmp_path) == projected_replay(200) + projected_replay(2)
    assert credits_spent(state) == projected_replay(200)  # no directory: nothing to read


def test_a_replay_file_a_crash_cut_short_never_breaks_the_budget(tmp_path: Path) -> None:
    """R4: a head line cut mid-write made every budget check -- and the listing -- raise."""
    state = AuditState()
    cut = state.candidate("w2", CandidateKind.SFT_SMALL)
    cut.deployment_id = "dep-2"
    answers = replay_file(tmp_path, "w2", CandidateKind.SFT_SMALL)
    answers.parent.mkdir(parents=True)
    answers.write_text(
        '{"deployment_id": "de\n{"row": 0, "answer": "a", "ms": 1.0}\n', encoding="utf-8"
    )
    assert credits_spent(state, tmp_path) == projected_replay(1)  # its answers still count
    answers.write_text("", encoding="utf-8")
    assert credits_spent(state, tmp_path) == 0.0


def test_credits_spent_keeps_retired_spend_in_the_directory_budget() -> None:
    state = AuditState(retired_cost_credits=17.0)
    assert credits_spent(state) == 17.0
    state.candidate("w1", CandidateKind.HEAD_TUNE).training_cost_credits = 3.0
    assert credits_spent(state) == 20.0
