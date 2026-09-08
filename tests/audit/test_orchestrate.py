"""run_audit: idempotent, resumable, budget-gated, and never swallowing a Ctrl+C."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import json
from pathlib import Path
import socket
from typing import Any

import pytest
from tests.audit._platform import Clock, FakePlatform, last_user, serve_chat, teacher
from tests.audit.conftest import SCAN_REPORT as REPORT
from tests.typing_helpers import RequestsMocker

from dagnam._core.exceptions import APIError, QuotaExceededError
from dagnam.audit import build_dataset, discover_workloads
from dagnam.audit.candidates import CandidateKind
from dagnam.audit.orchestrate import SCAN_REPORT, run_audit
from dagnam.audit.record import TraceRecord
from dagnam.audit.state import STATE_FILE, AuditState, load_state

HEAD, SFT, HOSTED = CandidateKind.HEAD_TUNE, CandidateKind.SFT_SMALL, CandidateKind.HOSTED_FLOOR


@pytest.fixture
def run(
    audit_dir: Path, platform: FakePlatform, clock: Clock, requests_mock: RequestsMocker
) -> Callable[..., AuditState]:
    serve_chat(requests_mock, teacher)

    def call(**overrides: Any) -> AuditState:
        settings: dict[str, Any] = {
            "floor": None,
            "workloads": None,
            "max_credits": 500,
            "wait": True,
            "client": platform,
            "sleep": clock.sleep,
            "now": clock.now,
        }
        settings.update(overrides)
        return run_audit(audit_dir, **settings)

    return call


def test_full_frontier_then_each_step_is_idempotent(
    run: Callable[..., AuditState], audit_dir: Path, platform: FakePlatform
) -> None:
    state = run()

    assert state.project_id == "proj-1"
    assert state.price_table_version == "2026-09"
    assert state.halted is None
    assert set(state.workloads) == {"w1", "w2"}
    head, sft = state.workloads["w1"][HEAD], state.workloads["w2"][SFT]
    assert state.workloads["w1"][HOSTED].dataset_id is None
    for step in (head, sft):
        assert step.split_done is True
        assert step.pii_agrees is True
        assert step.scored is True
        assert step.run_status == "completed"
        assert step.model_version_id == "mv-pushed"
        assert step.deploy_status == "running"
        assert step.training_cost_credits == 100.0
        assert step.error is None
    assert (head.version_id, sft.version_id) == ("ds-1-v2", "ds-2-v2")
    assert (head.base, sft.base) == ("BERT small", "qwen-05b")
    assert [p["recipe_key"] for p in platform.submitted] == [
        "head-tune-text-classification@1.1",
        "qlora-sft-chat@1.1",
    ]
    assert head.agreement is not None
    assert head.agreement["floor"] == 0.97
    assert sft.agreement is not None
    assert sft.agreement["floor"] == 0.95
    assert sft.agreement["value"] == 1.0
    assert (head.key_ref, sft.key_ref) == ("w1/head_tune", "w2/sft_small")
    assert load_state(audit_dir) == state

    calls_after_first = platform.call_log[:]
    second = run()
    assert platform.call_log == calls_after_first
    assert second == state


def test_budget_halts_before_the_next_submit(
    run: Callable[..., AuditState], platform: FakePlatform
) -> None:
    platform.credits_estimate_max = 400
    state = run(max_credits=500)
    assert state.halted is not None
    assert state.halted["reason"] == "budget"
    assert state.halted["next"] == "w2/sft_small"
    assert platform.submits == 1
    assert state.workloads["w1"][HEAD].scored is True
    assert state.workloads["w2"][SFT].run_id is None

    resumed = run(max_credits=1000)
    assert resumed.halted is None
    assert platform.submits == 2
    assert resumed.workloads["w2"][SFT].scored is True


def test_402_from_the_platform_halts_the_budget(
    run: Callable[..., AuditState], platform: FakePlatform
) -> None:
    platform.submit_errors = [QuotaExceededError("out of credits")]
    state = run()
    assert state.halted == {"reason": "budget", "detail": "out of credits", "next": "w1/head_tune"}
    assert state.workloads["w1"][HEAD].pii_agrees is True


def test_capacity_503_is_retried_after_retry_after(
    run: Callable[..., AuditState], platform: FakePlatform, clock: Clock
) -> None:
    platform.submit_errors = [APIError(503, "at capacity", retry_after_header="7")]
    state = run()
    assert state.halted is None
    assert 7.0 in clock.sleeps
    assert platform.submits == 2


def test_preflight_rejection_marks_the_candidate_and_the_frontier_continues(
    run: Callable[..., AuditState], platform: FakePlatform
) -> None:
    platform.submit_errors = [APIError(422, "needs 64 rows")]
    state = run()
    assert state.workloads["w1"][HEAD].error == "rejected_preflight: needs 64 rows"
    assert state.workloads["w1"][HEAD].deployment_id is None
    assert state.workloads["w2"][SFT].scored is True
    assert platform.submits == 1


def test_pii_disagreement_stops_that_workload_and_the_other_continues(
    run: Callable[..., AuditState], platform: FakePlatform
) -> None:
    platform.pii_counts = {"ds-1": {"PII_EMAIL": 1}}
    state = run()
    head = state.workloads["w1"][HEAD]
    assert head.error is not None
    assert head.error.startswith("pii_disagreement")
    assert head.run_id is None
    assert state.workloads["w2"][SFT].scored is True
    assert platform.submits == 1


def test_run_failure_is_recorded_and_the_frontier_continues(
    run: Callable[..., AuditState], platform: FakePlatform
) -> None:
    platform.run_final_status = "failed"
    platform.run_error = "worker died"
    state = run()
    assert state.workloads["w1"][HEAD].error == "run_failed: worker died"
    assert state.workloads["w2"][SFT].error == "run_failed: worker died"
    assert platform.submits == 2
    assert platform.deployments == []


def test_keyboard_interrupt_propagates_with_the_state_saved(
    run: Callable[..., AuditState], platform: FakePlatform, audit_dir: Path
) -> None:
    platform.submit_errors = [KeyboardInterrupt()]
    with pytest.raises(KeyboardInterrupt):
        run()
    saved = load_state(audit_dir)
    assert saved.halted is None
    assert saved.workloads["w1"][HEAD].pii_agrees is True
    assert saved.workloads["w1"][HEAD].run_id is None

    resumed = run()
    assert resumed.workloads["w1"][HEAD].scored is True
    assert platform.call_log.count("upload_dataset") == 2  # w1 once, w2 once


def test_a_crash_mid_step_is_recorded_and_the_rerun_resumes_from_the_saved_state(
    run: Callable[..., AuditState], platform: FakePlatform, audit_dir: Path
) -> None:
    platform.submit_errors = [RuntimeError("socket reset")]
    with pytest.raises(RuntimeError, match="socket reset"):
        run()
    on_disk = json.loads((audit_dir / STATE_FILE).read_text())
    assert on_disk["halted"] == {
        "reason": "error",
        "workload_id": "w1",
        "candidate": "head_tune",
        "step": "submit",
        "detail": "RuntimeError: socket reset",
    }
    before = len(platform.call_log)

    state = run()
    assert state.halted is None
    assert state.workloads["w1"][HEAD].scored is True
    rerun_calls = platform.call_log[before:]
    assert rerun_calls.count("upload_dataset") == 1  # w2 only; w1's data steps were on disk
    assert rerun_calls[0] == "list_foundation_catalog"


def test_no_wait_returns_after_submitting(
    run: Callable[..., AuditState], platform: FakePlatform
) -> None:
    state = run(wait=False)
    head = state.workloads["w1"][HEAD]
    assert head.run_id == "run-1"
    assert head.run_status == "queued"
    assert "get_foundation_run" not in platform.call_log
    assert "w2" not in state.workloads

    finished = run(wait=True)
    assert finished.workloads["w1"][HEAD].scored is True
    assert finished.workloads["w2"][SFT].scored is True


def test_explicit_workload_selection(
    run: Callable[..., AuditState], platform: FakePlatform
) -> None:
    state = run(workloads=["w2"])
    assert set(state.workloads) == {"w2"}
    with pytest.raises(ValueError, match=r"not in scan-report.json: \['nope'\]"):
        run(workloads=["nope"])
    with pytest.raises(ValueError, match=r"without derived data.*\['w5'\]"):
        run(workloads=["w5"])


def test_missing_scan_report_and_an_unversioned_price_table(
    run: Callable[..., AuditState], audit_dir: Path, tmp_path: Path, platform: FakePlatform
) -> None:
    report = dict(REPORT)
    report["price_table_version"] = 3
    report["workloads"] = [{"id": "w1", "structure_class": "enum_label"}]
    (audit_dir / SCAN_REPORT).write_text(json.dumps(report))
    state = run()
    assert state.price_table_version is None
    assert state.workloads == {}  # no verdict -> not audited unless named

    (audit_dir / SCAN_REPORT).unlink()
    with pytest.raises(FileNotFoundError, match="dagnam audit scan"):
        run()


def test_floor_override_is_recorded_on_every_agreement(run: Callable[..., AuditState]) -> None:
    state = run(floor=0.5)
    for step in (state.workloads["w1"][HEAD], state.workloads["w2"][SFT]):
        assert step.agreement is not None
        assert step.agreement["floor"] == 0.5


def test_deploy_failure_is_recorded_and_the_rest_continues(
    run: Callable[..., AuditState], platform: FakePlatform
) -> None:
    platform.revision_final = "failed"
    state = run()
    assert state.workloads["w1"][HEAD].error == "deploy_failed: no gpu"
    assert state.workloads["w2"][SFT].error == "deploy_failed: no gpu"


def test_unreliable_endpoint_is_scored_and_flagged(
    audit_dir: Path, platform: FakePlatform, clock: Clock, requests_mock: RequestsMocker
) -> None:
    serve_chat(requests_mock, lambda m: None if last_user(m) == "ticket 18" else teacher(m))
    state = run_audit(
        audit_dir,
        floor=None,
        workloads=["w1"],
        max_credits=500,
        wait=True,
        client=platform,
        sleep=clock.sleep,
        now=clock.now,
    )
    head = state.workloads["w1"][HEAD]
    assert head.error == "unreliable: 1 of 4 replay calls failed"
    assert head.latency is not None
    assert (head.latency["calls"], head.latency["errors"]) == (4, 1)


def test_no_socket_is_opened_by_scan_paths(
    monkeypatch: pytest.MonkeyPatch, langfuse_records: Sequence[TraceRecord]
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("scan opened a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    found = discover_workloads(langfuse_records, window_days=30)
    assert found
    records = [langfuse_records[i] for i in found[0].record_indices]
    dataset = build_dataset(
        records, structure_class=found[0].structure_class.value, max_seq_length=512
    )
    assert dataset.rows


def test_a_cancelled_candidate_is_skipped_wherever_the_cancel_caught_it(
    run: Callable[..., AuditState], audit_dir: Path, platform: FakePlatform
) -> None:
    """`audit cancel` paused the deployment; resuming must not poll it back to life."""
    from dagnam.audit.cleanup import mark_cancelled
    from dagnam.audit.state import save_state

    state = run()
    head = state.workloads["w1"][HEAD]
    head.scored, head.deploy_status = None, "deploying"  # the cancel caught it at wait_active
    mark_cancelled(state)
    save_state(audit_dir, state)
    platform.call_log.clear()

    run()

    assert platform.call_log == []
    assert load_state(audit_dir).workloads == state.workloads
    # w2 scored before the cancel, so it kept its result and lost its endpoint.
    assert state.workloads["w2"][SFT].deploy_status == "paused"
    assert state.workloads["w2"][SFT].error is None
