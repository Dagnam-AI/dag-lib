"""A replay the account cannot pay for stops as a budget halt, never as a verdict on the candidate."""

from __future__ import annotations

from collections.abc import Callable
import json

from tests.audit._chat import OutOfCreditsError, last_user, serve_chat, teacher
from tests.audit._platform import FakePlatform
from tests.typing_helpers import RequestsMocker

from dagnam.audit.state import AuditState, StepState
from dagnam.audit.steps import StepContext
from dagnam.audit.steps_serve import (
    create_deployment,
    create_revision,
    replay_and_score,
    wait_active,
)
from dagnam.audit.steps_train import credits_spent, projected_replay

HALT = {
    "reason": "budget",
    "detail": (
        "the account ran out of credits during the w1/head_tune replay; nothing was lost: add "
        "credits, then run `dagnam audit run` again to resume where it stopped"
    ),
    "next": "w1/head_tune replay",
}


def _served(ctx: StepContext) -> AuditState:
    state = AuditState(project_id="proj-1")
    state.workloads["w1"] = {
        ctx.spec.kind: StepState(
            training_job_id="job-1", run_status="completed", model_version_id="mv-1"
        )
    }
    for step in (create_deployment, create_revision, wait_active):
        state = step(state, ctx)
    return state


def _rows_on_disk(ctx: StepContext) -> list[int]:
    """The row numbers the answers file holds, after its head line."""
    lines = (ctx.workload_dir / "replay-head_tune.jsonl").read_text(encoding="utf-8").splitlines()
    return sorted(json.loads(line)["row"] for line in lines[1:])


def test_a_402_mid_replay_halts_on_budget_and_scores_nothing(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, requests_mock: RequestsMocker
) -> None:
    def answer(messages: list[dict[str, str]]) -> str:
        if last_user(messages) == "ticket 17":
            raise OutOfCreditsError
        return teacher(messages)

    serve_chat(requests_mock, answer)
    ctx = make_ctx()
    state = _served(ctx)

    replay_and_score(state, ctx)

    step = ctx.step(state)
    assert state.halted == HALT
    assert (step.scored, step.error, step.agreement, step.latency) == (None, None, None, None)
    assert step.replay_cost_credits is None
    # The answered rows stay on disk; the refused one is neither answered nor failed.
    assert set(_rows_on_disk(ctx)) <= {0, 2, 3}
    assert platform.balance_reads == 1  # the one before the first attempt; no new read to halt


def test_resuming_after_the_credits_are_added_sends_the_refused_row_and_scores(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, requests_mock: RequestsMocker
) -> None:
    funded = False

    def answer(messages: list[dict[str, str]]) -> str:
        if last_user(messages) == "ticket 17" and not funded:
            raise OutOfCreditsError
        return teacher(messages)

    seen = serve_chat(requests_mock, answer)
    ctx = make_ctx()
    state = _served(ctx)
    replay_and_score(state, ctx)
    assert state.halted is not None

    answered = set(_rows_on_disk(ctx))
    funded = True
    state.halted = None  # what the next `audit run` does first
    del seen[:]
    replay_and_score(state, ctx)

    step = ctx.step(state)
    # Exactly the rows without an answer -- the refused one among them -- are sent again.
    assert sorted(last_user(s["body"]["messages"]) for s in seen) == [
        f"ticket {16 + i}" for i in range(4) if i not in answered
    ]
    assert 1 not in answered
    assert state.halted is None
    assert step.scored is True
    assert step.error is None
    assert step.agreement is not None
    assert step.agreement["n"] == 4
    assert step.latency is not None
    assert (step.latency["calls"], step.latency["errors"]) == (4, 0)


def test_an_empty_account_from_the_first_call_halts_with_nothing_recorded_as_failed(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, requests_mock: RequestsMocker
) -> None:
    def answer(messages: list[dict[str, str]]) -> str:
        raise OutOfCreditsError

    serve_chat(requests_mock, answer)
    ctx = make_ctx()
    state = _served(ctx)

    replay_and_score(state, ctx)

    assert state.halted == HALT
    assert _rows_on_disk(ctx) == []
    step = ctx.step(state)
    assert (step.scored, step.error) == (None, None)


def test_a_halt_rewrites_only_the_head_and_leaves_every_later_byte_alone(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, requests_mock: RequestsMocker
) -> None:
    """A resumed replay's file: a head, two answers and a line a crash cut short.

    The 402 forgets the opening balance in the head and nothing else -- the answers
    stay, and so does the torn last line, byte for byte.
    """

    def answer(messages: list[dict[str, str]]) -> str:
        raise OutOfCreditsError

    serve_chat(requests_mock, answer)
    ctx = make_ctx()
    state = _served(ctx)
    path = ctx.workload_dir / "replay-head_tune.jsonl"
    kept = (
        '{"row": 0, "answer": "a", "ms": 1.5}\n'
        '{"row": 1, "answer": "b\u00e9", "ms": 2.25}\n'
        '{"row": 2, "answer": "c", "ms'
    )
    path.write_text('{"deployment_id": "dep-1", "balance_before": 1000}\n' + kept, encoding="utf-8")

    replay_and_score(state, ctx)

    assert state.halted == HALT
    head, _, rest = path.read_bytes().partition(b"\n")
    assert json.loads(head) == {"deployment_id": "dep-1", "balance_before": None}
    assert rest == kept.encode("utf-8")


def test_a_top_up_between_the_halt_and_the_resume_leaves_the_replay_cost_unknown(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, requests_mock: RequestsMocker
) -> None:
    """The balance rises by what the user added, so before/after would read the replay as free.

    The halt forgets the opening balance, keeping every answered row, so the resumed
    replay's cost is unknown and the budget counts it at its projection.
    """
    funded = False

    def answer(messages: list[dict[str, str]]) -> str:
        if last_user(messages) == "ticket 17" and not funded:
            raise OutOfCreditsError
        return teacher(messages)

    serve_chat(requests_mock, answer)
    ctx = make_ctx()
    state = _served(ctx)
    replay_and_score(state, ctx)
    assert state.halted is not None
    path = ctx.workload_dir / "replay-head_tune.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {"deployment_id": "dep-1", "balance_before": None}
    assert sorted(json.loads(line)["row"] for line in lines[1:]) == _rows_on_disk(ctx)

    funded = True
    platform.grant_credits(1_000)  # the balance read after the resume is far above the first
    state.halted = None
    replay_and_score(state, ctx)

    step = ctx.step(state)
    assert step.scored is True
    assert step.replay_cost_credits is None
    assert credits_spent(state, ctx.audit_dir) == projected_replay(4)
