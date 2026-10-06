"""What the ``audit cancel`` / ``audit delete`` CLI tests share: a recorded run, its platform, its receipts."""

from __future__ import annotations

from pathlib import Path

from tests.audit import _receipts as r
from tests.audit._cleanup import FakeCleanup

from dagnam._types import JsonObject
from dagnam.audit.candidates import CandidateKind
from dagnam.audit.state import AuditState, StepState, save_state
from dagnam.audit.workspace import write_workload

HEAD, SFT, HOSTED = CandidateKind.HEAD_TUNE, CandidateKind.SFT_SMALL, CandidateKind.HOSTED_FLOOR


def cli_state() -> AuditState:
    state = AuditState(project_id="proj-1")
    state.workloads["w1"] = {
        HOSTED: StepState(),
        HEAD: StepState(
            dataset_id="ds-1",
            run_id="run-1",
            training_job_id="job-1",
            run_status="running",
            deployment_id="dep-1",
            deploy_status="deploying",
        ),
    }
    state.workloads["w2"] = {
        SFT: StepState(
            dataset_id="ds-2",
            run_id="run-2",
            training_job_id="job-2",
            run_status="completed",
            model_version_id="mv-2",
            deployment_id="dep-2",
            deploy_status="paused",
            scored=True,
            agreement={"metric": "exact", "value": 0.9876, "ci95": [0.97, 1.0], "n": 200},
            training_cost_credits=42.0,
        ),
    }
    return state


def published_dir(tmp_path: Path) -> Path:
    """An audit dir whose run published, so ``cancel`` / ``delete`` go through the account."""
    root = tmp_path / "audit"
    state = cli_state()
    state.audit_id = "audit-1"
    state.tagged = True  # published by a client that names the audit on every create
    state.workloads["w1"][HEAD].key_ref = "w1/head_tune"
    state.workloads["w2"][SFT].deploy_status = "running"  # scored, and still serving
    save_state(root, state)
    rows = [{"input": "a", "label": "x"}, {"input": "b", "label": "y"}]
    write_workload(root, "w1", rows, {"train": [0], "eval_holdout": [1]}, {"format_key": "x"})
    return root


def platform_with_everything(*, receipt: JsonObject | None = None) -> FakeCleanup:
    """Every artifact of :func:`cli_state` still on the platform, the account answering ``receipt``."""
    fake = FakeCleanup(
        deployment=["dep-1", "dep-2"],
        model=["mv-2"],
        job=["job-1", "job-2"],
        dataset=["ds-1", "ds-2"],
        project=["proj-1"],
    )
    if receipt is not None:
        fake.server_receipt = dict(receipt)
    return fake


CANCEL_RECEIPT = r.receipt(
    r.row("training_job", "job-1", "stopped"),
    r.row("deployment", "dep-1", "blocked", "refused", "not provisioned"),
    r.row("deployment", "dep-2", "stopped"),
    schema=r.SCHEMA_CANCELLED,
)
"""The account's own cancel receipt, written through verbatim: a cancel stops artifacts, so it is
not the ``deleted/1`` document a delete returns."""
DELETE_RECEIPT = r.receipt(
    r.row("deployment", "dep-1", "deleted"), r.row("project", "proj-1", "deleted")
)


def kept_on_purpose_receipt() -> JsonObject:
    """What the platform answers when everything of the audit is gone but what is not the audit's."""
    return r.receipt(
        r.row("deployment", "dep-1", "deleted"),
        r.row("project", "proj-1", "kept", "project_held", r.PROJECT_HELD),
        r.row("model_entry", "entry-2", "kept", "weights_served", r.WEIGHTS_SERVED),
        r.row("model_version", "mv-2", "kept", "weights_served", r.WEIGHTS_SERVED),
    )
