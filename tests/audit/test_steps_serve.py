"""Serving steps: the deployment shape G0 proved, the key in the secret store, the replay and score."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path

from dagnam_contracts.prompts import render_chat_prompt
import pytest
from tests.audit._platform import Clock, FakePlatform, last_user, serve_chat, teacher
from tests.typing_helpers import RequestsMocker

from dagnam.audit.candidates import SFT_SMALL, CandidateKind
from dagnam.audit.secrets import SECRETS_FILE
from dagnam.audit.state import AuditState, StepState
from dagnam.audit.steps import StepContext
from dagnam.audit.steps_serve import (
    _revision_status,
    create_deployment,
    create_revision,
    holdout,
    parse_chat_prompt,
    replay_and_score,
    wait_active,
)
from dagnam.audit.structure import StructureClass


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


def test_parse_chat_prompt_inverts_the_contract_rendering() -> None:
    turns = [{"role": "user", "content": "hi\nthere\n"}, {"role": "tool", "content": "42"}]
    rendered = render_chat_prompt(turns, system="Be terse")
    parsed = parse_chat_prompt(rendered)
    assert parsed == [{"role": "system", "content": "Be terse"}, *turns]
    assert render_chat_prompt(parsed[1:], system=parsed[0]["content"]) == rendered
    assert parse_chat_prompt("") == []
    assert parse_chat_prompt("tail of a cut turn\n<|user|>\nnext\n") == [
        {"role": "user", "content": "tail of a cut turn"},
        {"role": "user", "content": "next"},
    ]
    assert parse_chat_prompt("no markers at all") == [
        {"role": "user", "content": "no markers at all"}
    ]


def test_holdout_rows_for_both_formats(make_ctx: Callable[..., StepContext]) -> None:
    labels = holdout(make_ctx())
    assert len(labels) == 4
    assert labels[0] == (
        [
            {"role": "system", "content": "Classify the ticket"},
            {"role": "user", "content": "ticket 16"},
        ],
        "b",
    )
    chats = holdout(make_ctx(workload_id="w2", spec=SFT_SMALL))
    assert chats[1][0] == [
        {"role": "system", "content": "Extract"},
        {"role": "user", "content": "order 17"},
    ]
    assert json.loads(chats[1][1]) == {"order_id": "17", "product": "x"}


def test_create_deployment_uses_the_g0_shape_and_stores_the_key(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, audit_dir: Path
) -> None:
    ctx = make_ctx()
    state = create_deployment(_trained(), ctx)
    step = ctx.step(state)
    assert platform.deployments == [
        {
            "name": "audit-w1-head_tune",
            "project_id": "proj-1",
            "checkpoint_path": "training-job://job-1",
            "platform": "vllm",
            "deployment_type": "text",
            "instance_type": "modal-serverless",
            "num_instances": 1,
            "auto_scaling_enabled": False,
            "training_job_id": "job-1",
        }
    ]
    assert (step.deployment_id, step.key_ref) == ("dep-1", "w1/head_tune")
    assert json.loads((audit_dir / SECRETS_FILE).read_text()) == {"w1/head_tune": "dk-secret"}
    assert "dk-secret" not in json.dumps(state.to_json())
    create_deployment(state, ctx)
    assert platform.call_log == ["create_deployment"]


def test_create_deployment_without_a_key_leaves_no_key_ref(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    platform.deployment_key = None
    ctx = make_ctx()
    assert ctx.step(create_deployment(_trained(), ctx)).key_ref is None


def test_create_revision_is_serverless_and_idempotent(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    ctx = make_ctx()
    state = create_revision(create_deployment(_trained(), ctx), ctx)
    assert platform.revisions == [
        {
            "model_version_id": "mv-1",
            "capacity_mode": "serverless",
            "capacity_policy": {"min_replicas": 0, "max_replicas": 1},
            "deployment_id": "dep-1",
            "key": "audit-w1/head_tune-mv-1",
        }
    ]
    assert ctx.step(state).deploy_status == "deploying"
    create_revision(state, ctx)
    assert platform.call_log.count("create_deployment_revision") == 1


def test_revision_status_reads_the_newest_revision(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = make_ctx()
    listings = iter(
        [
            [],
            ["junk"],
            [{"id": "r"}],
            [{"id": "r", "status": "failed", "failure_reason": "no gpu"}],
            [{"id": "r", "is_active": True}],
        ]
    )
    monkeypatch.setattr(platform, "get_deployment_revisions", lambda deployment_id: next(listings))
    assert _revision_status(ctx, "d") == {"status": "deploying"}
    assert _revision_status(ctx, "d") == {"status": "deploying"}
    assert _revision_status(ctx, "d") == {"status": "deploying", "error_message": None}
    assert _revision_status(ctx, "d") == {"status": "failed", "error_message": "no gpu"}
    assert _revision_status(ctx, "d") == {"status": "running"}


def test_wait_active_reaches_running_and_then_skips(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    platform.revision_polls = 3
    ctx = make_ctx()
    state = _served(_trained(), ctx)
    assert ctx.step(state).deploy_status == "running"
    assert platform.call_log.count("get_deployment_revisions") == 3
    wait_active(state, ctx)
    assert platform.call_log.count("get_deployment_revisions") == 3


def test_wait_active_records_a_failed_revision(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    platform.revision_final = "failed"
    ctx = make_ctx()
    step = ctx.step(_served(_trained(), ctx))
    assert (step.deploy_status, step.error) == ("failed", "deploy_failed: no gpu")
    assert platform.paused == []


def test_wait_active_timeout_pauses_the_deployment(
    make_ctx: Callable[..., StepContext], platform: FakePlatform, clock: Clock
) -> None:
    platform.revision_final = "stuck"
    ctx = make_ctx(deploy_timeout=5.0)
    step = ctx.step(_served(_trained(), ctx))
    assert (step.deploy_status, step.error) == (
        "timeout",
        "deploy_timeout: not active after 5s; paused",
    )
    assert platform.paused == ["dep-1"]
    assert clock.t >= 5.0


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
