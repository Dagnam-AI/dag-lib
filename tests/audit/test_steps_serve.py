"""Serving steps: the proven deployment shape, the key in the secret store, the replay and score."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path

from dagnam_contracts.prompts import parse_chat_prompt, render_chat_prompt
import pytest
from tests.audit._platform import Clock, FakePlatform

from dagnam.audit.candidates import SFT_SMALL, CandidateKind
from dagnam.audit.secrets import SECRETS_FILE
from dagnam.audit.state import AuditState, StepState
from dagnam.audit.steps import StepContext
from dagnam.audit.steps_serve import (
    _revision_status,
    create_deployment,
    create_revision,
    holdout,
    wait_active,
)


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


def test_a_published_audits_endpoint_is_created_naming_the_audit(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    state = _trained()
    state.audit_id = "audit-1"
    create_deployment(state, make_ctx())
    assert platform.deployments[0]["audit_id"] == "audit-1"


def test_an_unpublished_audits_endpoint_carries_no_audit_id(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    create_deployment(_trained(), make_ctx())
    assert "audit_id" not in platform.deployments[0]


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


def test_wait_active_timeout_of_a_published_audit_pauses_nothing_the_audits_cancel_does(
    make_ctx: Callable[..., StepContext], platform: FakePlatform
) -> None:
    """Only the platform's own cancel pauses a published audit's endpoint."""
    platform.revision_final = "stuck"
    ctx = make_ctx(deploy_timeout=5.0)
    state = _trained()
    state.audit_id = "audit-1"

    step = ctx.step(_served(state, ctx))

    assert step.deploy_status == "timeout"
    assert step.error == "deploy_timeout: not active after 5s; `dagnam audit cancel` stops it"
    assert platform.paused == []
    assert "pause_deployment" not in platform.call_log


def test_wait_active_timeout_tolerates_a_deployment_that_cannot_be_paused(
    make_ctx: Callable[..., StepContext],
    platform: FakePlatform,
    clock: Clock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A revision that never activated leaves the deployment unpausable; the timeout still records."""
    from dagnam._core.exceptions import DeploymentStateError

    def refuse(deployment_id: str) -> None:
        raise DeploymentStateError("Invalid status transition from not_provisioned to paused")

    platform.revision_final = "stuck"
    monkeypatch.setattr(platform, "pause_deployment", refuse)
    ctx = make_ctx(deploy_timeout=5.0)
    step = ctx.step(_served(_trained(), ctx))
    assert (step.deploy_status, step.error) == (
        "timeout",
        "deploy_timeout: not active after 5s; it could not be paused",
    )
    assert platform.paused == []
    assert clock.t >= 5.0
