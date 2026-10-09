"""The holdout replay: scored through the endpoint, costed from the balance, resumable."""

from __future__ import annotations

from collections.abc import Callable
import itertools
import json
from pathlib import Path
import threading
from typing import Any

import pytest
from tests.audit._chat import last_user, serve_chat, teacher
from tests.audit._platform import FakePlatform
from tests.typing_helpers import RequestsMocker

from dagnam._core.exceptions import APIError
from dagnam.audit.candidates import SFT_SMALL, CandidateKind
from dagnam.audit.frontier import replay_holdout
from dagnam.audit.readers.messages import effective_response
from dagnam.audit.secrets import SECRETS_FILE
from dagnam.audit.state import AuditState, StepState
from dagnam.audit.steps import StepContext
from dagnam.audit.steps_serve import (
    create_deployment,
    create_revision,
    replay_and_score,
    wait_active,
)
from dagnam.audit.structure import StructureClass
from dagnam.audit.workspace import write_workload

WRITTEN_TIMEOUT = 30.0


def _written_down(monkeypatch: pytest.MonkeyPatch, count: int) -> threading.Event:
    """An event set once ``count`` answers of the next replay are in its answers file.

    It wraps the replay's own ``on_result`` -- the callback that writes and
    flushes each answer -- and is set only after that callback returned for
    the ``count``-th, so a call that waits on it is interrupted with those
    rows provably on disk. A sleep only made that likely.
    """
    written = threading.Event()

    def counting(endpoint: Any, rows: Any, *, on_result: Callable[[int, str | None, float], None]):
        landed = itertools.count(1)

        def heard(position: int, reply: str | None, ms: float) -> None:
            on_result(position, reply, ms)
            if next(landed) == count:
                written.set()

        return replay_holdout(endpoint, rows, on_result=heard)

    monkeypatch.setattr("dagnam.audit.steps_serve.replay_holdout", counting)
    return written


def _trained(kind: CandidateKind = CandidateKind.HEAD_TUNE) -> AuditState:
    state = AuditState(project_id="proj-1")
    state.workloads["w1" if kind is CandidateKind.HEAD_TUNE else "w2"] = {
        kind: StepState(training_job_id="job-1", run_status="completed", model_version_id="mv-1")
    }
    return state


def _served(state: AuditState, ctx: StepContext) -> AuditState:
    for step in (create_deployment, create_revision, wait_active):
        state = step(state, ctx)
    return state


def test_replay_and_score_labels_through_the_endpoint(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, requests_mock: RequestsMocker
) -> None:
    seen = serve_chat(requests_mock, teacher)
    ctx = make_ctx()
    state = replay_and_score(_served(_trained(), ctx), ctx)
    step = ctx.step(state)
    assert step.scored is True
    assert step.error is None
    assert step.agreement is not None
    assert step.agreement["metric"] == "exact"
    assert step.agreement["value"] == 1.0
    assert step.agreement["n"] == 4
    assert step.agreement["floor"] == 0.97
    assert step.agreement["passes_floor"] is False  # four rows cannot clear 0.97 on the lower bound
    assert step.latency is not None
    assert (step.latency["calls"], step.latency["errors"]) == (4, 0)
    assert all(s["authorization"] == "Bearer dk-secret" for s in seen)
    assert all(s["body"]["model"] == "dep-1" for s in seen)
    # Completion order, not row order: the replay runs four calls concurrently.
    assert sorted(last_user(s["body"]["messages"]) for s in seen) == [
        f"ticket {i}" for i in range(16, 20)
    ]
    replay_and_score(state, ctx)
    assert len(seen) == 4


def test_replay_measures_its_credit_cost_from_the_account_balance(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, requests_mock: RequestsMocker
) -> None:
    serve_chat(requests_mock, teacher)
    platform.replay_charge = 396
    ctx = make_ctx()
    step = ctx.step(replay_and_score(_served(_trained(), ctx), ctx))
    assert step.replay_cost_credits == 396.0  # 1000 before, 604 after
    assert platform.call_log.count("get_credit_balance") == 2


def test_replay_cost_never_reads_as_a_refund(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, requests_mock: RequestsMocker
) -> None:
    serve_chat(requests_mock, teacher)
    platform.replay_charge = -396  # a grant landed mid-replay: 604 before, 1000 after
    ctx = make_ctx()
    assert ctx.step(replay_and_score(_served(_trained(), ctx), ctx)).replay_cost_credits == 0.0


@pytest.mark.parametrize("failing_read", [0, 1])
def test_a_failed_balance_read_leaves_the_cost_unknown_and_the_candidate_scored(
    make_ctx: Callable[..., StepContext],
    platform: FakePlatform,
    requests_mock: RequestsMocker,
    failing_read: int,
) -> None:
    serve_chat(requests_mock, teacher)
    platform.credit_errors = [None, None]
    platform.credit_errors[failing_read] = APIError(500, "credits unavailable")
    ctx = make_ctx()
    step = ctx.step(replay_and_score(_served(_trained(), ctx), ctx))
    assert step.replay_cost_credits is None
    assert step.scored is True
    assert step.error is None


def test_replay_and_score_json_fields(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, requests_mock: RequestsMocker
) -> None:
    def wrong_product(messages: list[dict[str, str]]) -> str:
        return json.dumps({"order_id": last_user(messages).split()[1], "product": "y"})

    serve_chat(requests_mock, wrong_product)
    ctx = make_ctx(
        workload_id="w2", spec=SFT_SMALL, structure_class=StructureClass.JSON_OBJECT, floor=0.5
    )
    step = ctx.step(replay_and_score(_served(_trained(CandidateKind.SFT_SMALL), ctx), ctx))
    assert step.agreement is not None
    assert step.agreement["metric"] == "field_f1"
    assert step.agreement["field_precision"] == 0.5
    assert step.agreement["passes_floor"] is False


def test_replay_marks_an_unreliable_endpoint(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, requests_mock: RequestsMocker
) -> None:
    serve_chat(requests_mock, lambda m: None if last_user(m) == "ticket 17" else teacher(m))
    ctx = make_ctx()
    step = ctx.step(replay_and_score(_served(_trained(), ctx), ctx))
    assert step.scored is True
    assert step.error == "unreliable: 1 of 4 replay calls failed"
    assert step.agreement is not None
    assert step.agreement["n"] == 3
    serve_chat(requests_mock, lambda m: None)
    step = ctx.step(replay_and_score(_served(_trained(), ctx), ctx))
    assert step.error == "unreliable: 4 of 4 replay calls failed"
    assert step.agreement is not None
    assert step.agreement["n"] == 0


def test_an_interrupted_replay_resumes_where_it_stopped_and_counts_both_halves(
    make_ctx: Callable[..., StepContext],
    platform: FakePlatform,
    requests_mock: RequestsMocker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Ctrl+C mid-replay loses neither the answers already served nor what they cost.

    The rerun sends only the row that never answered, and the cost runs from
    the balance read before the FIRST attempt -- the old rerun read a fresh
    "before" and so dropped every credit the interrupted attempt had burned.
    """
    sent: list[str] = []
    written = _written_down(monkeypatch, 3)
    turn = threading.Lock()

    def answer(messages: list[dict[str, str]]) -> str:
        with turn:
            sent.append(last_user(messages))
            last = len(sent) == 4
        if last:
            # The first three are written down before the last one fails.
            assert written.wait(timeout=WRITTEN_TIMEOUT)
            raise KeyboardInterrupt
        return teacher(messages)

    serve_chat(requests_mock, answer)
    ctx = make_ctx()
    state = _served(_trained(), ctx)
    with pytest.raises(KeyboardInterrupt):
        replay_and_score(state, ctx)
    step = ctx.step(state)
    assert (step.scored, step.replay_cost_credits) == (None, None)
    cache = ctx.workload_dir / "replay-head_tune.jsonl"
    assert len(cache.read_text(encoding="utf-8").splitlines()) == 1 + 3
    with cache.open("a", encoding="utf-8") as handle:
        handle.write('{"row": 3, "answ')  # and the process died mid-line

    step = ctx.step(replay_and_score(state, ctx))
    assert sent[4:] == [sent[3]]  # only the interrupted row is sent again
    assert step.scored is True
    assert step.agreement is not None
    assert step.agreement["n"] == 4
    assert step.latency is not None
    assert step.latency["calls"] == 4
    assert platform.balance_reads == 2  # once before the first attempt, once after the last
    assert step.replay_cost_credits == 4.0


def test_a_resumed_replay_retries_the_rows_that_failed(
    make_ctx: Callable[..., StepContext],
    platform: FakePlatform,
    requests_mock: RequestsMocker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An outage the user stopped with Ctrl+C must not freeze its failures into `unreliable`."""
    sent: list[str] = []
    written = _written_down(monkeypatch, 3)
    turn = threading.Lock()

    def answer(messages: list[dict[str, str]]) -> str | None:
        with turn:
            sent.append(last_user(messages))
            position = len(sent)
        if position == 4:
            assert written.wait(timeout=WRITTEN_TIMEOUT)
            raise KeyboardInterrupt
        return None if position <= 4 and last_user(messages) == "ticket 17" else teacher(messages)

    serve_chat(requests_mock, answer)
    ctx = make_ctx()
    state = _served(_trained(), ctx)
    with pytest.raises(KeyboardInterrupt):
        replay_and_score(state, ctx)
    step = ctx.step(replay_and_score(state, ctx))
    assert sorted(sent[4:]) == sorted({"ticket 17", sent[3]})
    assert step.error is None
    assert step.latency is not None
    assert (step.latency["calls"], step.latency["errors"]) == (4, 0)


def test_a_replay_whose_file_a_crash_cut_short_starts_over_at_an_unknown_cost(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, requests_mock: RequestsMocker
) -> None:
    """The head names neither the endpoint nor the opening balance any more.

    So none of its answers can be trusted and every row is sent again, and what
    the lost attempt spent is unknown -- the cost is left unknown (the budget
    counts it at its projection) rather than measured from a fresh balance.
    """
    seen = serve_chat(requests_mock, teacher)
    ctx = make_ctx()
    state = _served(_trained(), ctx)
    answers = ctx.workload_dir / "replay-head_tune.jsonl"
    answers.write_text('{"deployment_id": "de\n{"row": 0, "answer": "a", "ms": 1}\n')
    step = ctx.step(replay_and_score(state, ctx))
    assert len(seen) == 4
    assert step.scored is True
    assert step.replay_cost_credits is None
    assert json.loads(answers.read_text(encoding="utf-8").splitlines()[0]) == {
        "deployment_id": "dep-1",
        "balance_before": None,
    }


def test_a_replay_of_another_deployment_starts_over(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, requests_mock: RequestsMocker
) -> None:
    """Answers are kept per endpoint: a candidate redeployed since has none of them yet."""
    seen = serve_chat(requests_mock, teacher)
    ctx = make_ctx()
    replay_and_score(_served(_trained(), ctx), ctx)
    step = ctx.step(replay_and_score(_served(_trained(), ctx), ctx))
    assert step.deployment_id == "dep-2"
    assert len(seen) == 8
    assert step.agreement is not None
    assert step.agreement["n"] == 4


def test_an_interrupted_replay_counts_what_it_already_answered(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, requests_mock: RequestsMocker
) -> None:
    """The answered rows count as spent, and only the rows left to send are projected.

    150 for training, 4 for the three rows already answered, 2 for the last one:
    156 fits a 157 ceiling -- projecting the whole holdout again on top of the
    answered rows (159) would refuse a replay that cannot pass it.
    """
    serve_chat(requests_mock, teacher)
    ctx = make_ctx(max_credits=157)
    state = _served(_trained(), ctx)
    ctx.step(state).training_cost_credits = 150.0
    answers = ctx.workload_dir / "replay-head_tune.jsonl"
    answers.write_text(
        '{"deployment_id": "dep-1", "balance_before": 1000}\n'
        + "".join(f'{{"row": {i}, "answer": "a", "ms": 1.0}}\n' for i in range(3)),
        encoding="utf-8",
    )
    replay_and_score(state, ctx)
    assert state.halted is None
    assert ctx.step(state).scored is True


def test_an_answer_after_a_cut_off_line_lands_on_a_line_of_its_own(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, requests_mock: RequestsMocker
) -> None:
    """A crash left the last line short; appending onto it would merge it with the next answer.

    The merged line parses as nothing, so the answer was lost on disk and a later
    resume paid for it again. The fragment keeps its bytes and is still skipped.
    """
    serve_chat(requests_mock, teacher)
    ctx = make_ctx()
    state = _served(_trained(), ctx)
    path = ctx.workload_dir / "replay-head_tune.jsonl"
    head = '{"deployment_id": "dep-1", "balance_before": 1000}\n'
    kept = '{"row": 0, "answer": "a", "ms": 1.0}\n{"row": 1, "answer": "b", "ms'
    path.write_text(head + kept, encoding="utf-8")

    replay_and_score(state, ctx)

    text = path.read_text(encoding="utf-8")
    assert text.startswith(head + kept + "\n")
    parsed = [json.loads(line) for line in text.splitlines() if line.endswith("}")]
    assert sorted(row["row"] for row in parsed[1:]) == [0, 1, 2, 3]
    assert ctx.step(state).scored is True


def test_a_replay_that_could_pass_the_ceiling_is_never_started(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, requests_mock: RequestsMocker
) -> None:
    """125 of 128 credits spent, and a 4-row replay is budgeted at 5: it must not start."""
    seen = serve_chat(requests_mock, teacher)
    ctx = make_ctx(max_credits=128)
    state = _served(_trained(), ctx)
    ctx.step(state).training_cost_credits = 125.0
    replay_and_score(state, ctx)
    assert state.halted == {
        "reason": "budget",
        "spent_credits": 125.0,
        "max_credits": 128,
        "next": "w1/head_tune replay",
    }
    assert seen == []
    assert "get_credit_balance" not in platform.call_log
    assert ctx.step(state).scored is None


ROUTES = ("transfer_to_billing", "transfer_to_shipping", "transfer_to_returns", "escalate")


def test_a_router_that_misroutes_7_percent_of_calls_does_not_pass(
    make_ctx: Callable[..., StepContext],
    platform: FakePlatform,
    requests_mock: RequestsMocker,
    audit_dir: Path,
) -> None:
    """A tool call's `{}` arguments used to be a free field on every row.

    Micro-F1 was then (1 + routing accuracy) / 2: at 93% routing the lower bound
    over 1,000 rows was 0.952, a REPLACE at the 0.95 JSON floor. The truths go to
    the contract's `score_json` exactly as the scan derived them -- the scan's
    own `effective_response` -- and it scores a wrong tool as a wrong row.
    """
    n = 1_000
    truths = [
        effective_response("", ({"function": {"name": ROUTES[i % 4], "arguments": "{}"}},))
        for i in range(n)
    ]
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"route {i}"},
                {"role": "assistant", "content": t},
            ]
        }
        for i, t in enumerate(truths)
    ]
    stats = {
        "format_key": "chat-messages",
        "redact": {"counts": {}, "pass_list": [], "rows_changed": 0},
    }
    write_workload(
        audit_dir, "w7", [*rows, rows[0]], {"train": [n], "eval_holdout": list(range(n))}, stats
    )

    def router(messages: list[dict[str, str]]) -> str:
        i = int(last_user(messages).split()[1])
        misrouted = i % 100 < 7
        return truths[i + 1 if misrouted else i]  # the next row's tool: a different route

    serve_chat(requests_mock, router)
    ctx = make_ctx(
        workload_id="w7",
        spec=SFT_SMALL,
        structure_class=StructureClass.JSON_OBJECT,
        floor=0.95,
        max_credits=5_000,  # 1,000 replayed rows are budgeted at 1,100
    )
    state = _trained(CandidateKind.SFT_SMALL)
    state.workloads["w7"] = state.workloads.pop("w2")
    step = ctx.step(replay_and_score(_served(state, ctx), ctx))
    assert step.agreement is not None
    assert step.agreement["n"] == n
    assert step.agreement["value"] == pytest.approx(0.93)
    assert step.agreement["passes_floor"] is False


def test_replay_needs_the_key(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, audit_dir: Path
) -> None:
    platform.deployment_key = None
    ctx = make_ctx()
    step = ctx.step(replay_and_score(_served(_trained(), ctx), ctx))
    assert step.scored is None
    assert step.error is not None
    assert step.error.startswith("no_key:")

    platform.deployment_key = "dk-2"
    state = _served(_trained(), ctx)
    (audit_dir / SECRETS_FILE).write_text("{}")
    assert ctx.step(replay_and_score(state, ctx)).error is not None
