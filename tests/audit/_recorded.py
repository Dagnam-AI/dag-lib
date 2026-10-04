"""The audit state the cleanup tests walk: what a finished run recorded, and what is still live."""

from __future__ import annotations

from dagnam.audit.candidates import CandidateKind
from dagnam.audit.state import AuditState, StepState

HEAD, SFT, HOSTED = CandidateKind.HEAD_TUNE, CandidateKind.SFT_SMALL, CandidateKind.HOSTED_FLOOR


def recorded_state() -> AuditState:
    """Two trained candidates (a dataset, a run, a version and an endpoint each) under one project."""
    state = AuditState(project_id="proj-1")
    state.workloads["w1"] = {
        HOSTED: StepState(),
        HEAD: StepState(
            dataset_id="ds-1",
            training_job_id="job-1",
            model_version_id="mv-1",
            deployment_id="dep-1",
            key_ref="w1/head_tune",
        ),
    }
    state.workloads["w2"] = {
        SFT: StepState(
            dataset_id="ds-2",
            training_job_id="job-2",
            model_version_id="mv-2",
            deployment_id="dep-2",
        ),
    }
    state.workloads["w3"] = {HEAD: StepState(dataset_id="ds-1")}  # a shared id is deleted once
    return state


def live_state() -> AuditState:
    """One run still queued, and a scored candidate whose endpoint is up."""
    state = AuditState(project_id="proj-1")
    state.workloads["w1"] = {
        HEAD: StepState(dataset_id="ds-1", training_job_id="job-1", run_status="queued")
    }
    state.workloads["w2"] = {
        SFT: StepState(
            training_job_id="job-2",
            run_status="completed",
            deployment_id="dep-2",
            deploy_status="running",
            scored=True,
        )
    }
    return state
