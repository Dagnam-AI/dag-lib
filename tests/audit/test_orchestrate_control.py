"""run_audit: the project it creates, the directory it holds, and what stops a live run."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import shutil
from typing import Any

import pytest
from tests.audit._chat import Clock, serve_chat, teacher
from tests.audit._platform import FakePlatform
from tests.typing_helpers import RequestsMocker

from dagnam._core.exceptions import APIError, DeploymentNotFoundError, FoundationRunNotFoundError
from dagnam._types import JsonObject
from dagnam.audit.candidates import CandidateKind
from dagnam.audit.cleanup import CANCELLED_ERROR, settle_cancel
from dagnam.audit.orchestrate import (
    plan_credits,
    run_audit,
    run_audit_held,
)
from dagnam.audit.publish import Publisher
from dagnam.audit.state import (
    AuditBusyError,
    AuditState,
    load_state,
    lock_audit,
    save_state,
)
from dagnam.audit.structure import StructureClass

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


def test_the_project_is_created_under_a_nonce_saved_before_the_create(
    run: Callable[..., AuditState],
    audit_dir: Path,
    platform: FakePlatform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A create that dies after the platform made it is asked again under the same nonce."""
    on_disk: list[str | None] = []
    real = platform.create_project

    def create(payload: JsonObject, *, resume_nonce: str | None = None) -> JsonObject:
        on_disk.append(load_state(audit_dir).project_nonce)  # what a crash right now would leave
        if len(on_disk) == 1:
            raise KeyboardInterrupt
        return real(payload, resume_nonce=resume_nonce)

    monkeypatch.setattr(platform, "create_project", create)
    with pytest.raises(KeyboardInterrupt):
        run(workloads=["w1"])
    assert load_state(audit_dir).project_id is None
    run(workloads=["w1"])

    assert on_disk[0] is not None
    assert on_disk == [on_disk[0], on_disk[0]]  # saved first, and the same one the next run reads
    assert platform.project_nonces == [on_disk[0]]
    assert load_state(audit_dir).project_id == "proj-1"
    run(workloads=["w1"])
    assert platform.call_log.count("create_project") == 1


def test_two_audit_directories_of_one_name_never_share_a_project_nonce(
    audit_dir: Path, tmp_path: Path, platform: FakePlatform, clock: Clock
) -> None:
    """The create's body is only a title, the same for both: the nonce is what tells them apart."""
    other = tmp_path / "elsewhere" / audit_dir.name
    shutil.copytree(audit_dir, other)
    for directory in (audit_dir, other):
        run_audit(
            directory,
            floor=None,
            workloads=["w1"],
            max_credits=500,
            wait=False,
            client=platform,
            sleep=clock.sleep,
            now=clock.now,
        )
    first, second = platform.project_nonces
    assert first is not None
    assert second is not None
    assert first != second


def test_a_caller_that_holds_the_directory_runs_without_taking_it_again(
    audit_dir: Path, platform: FakePlatform, clock: Clock, requests_mock: RequestsMocker
) -> None:
    """``audit run`` holds the directory from its listing on; the run itself must not fight it."""
    serve_chat(requests_mock, teacher)
    with lock_audit(audit_dir):
        state = run_audit_held(
            audit_dir,
            floor=None,
            workloads=["w1"],
            max_credits=500,
            wait=True,
            client=platform,
            sleep=clock.sleep,
            now=clock.now,
        )
    assert state.workloads["w1"][HEAD].scored is True


def test_the_last_patch_goes_out_even_when_its_first_send_fails(
    run: Callable[..., AuditState],
    audit_dir: Path,
    platform: FakePlatform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the last candidate's last step nothing followed to carry it out."""
    real_patch = platform.patch_audit_candidate
    failed: list[bool] = []

    def patch(audit_id: str, candidate_id: str, payload: JsonObject) -> JsonObject:
        if payload.get("status") == "scored" and not failed:
            failed.append(True)
            raise APIError(503, "blip")
        return real_patch(audit_id, candidate_id, payload)

    monkeypatch.setattr(platform, "patch_audit_candidate", patch)
    state = run(workloads=["w1"], publisher=Publisher(platform, AuditState()))
    assert [body["status"] for _, body in platform.patches][-1] == "scored"
    assert state.workloads["w1"][HEAD].published_step == "replay_and_score"
    assert load_state(audit_dir).workloads["w1"][HEAD].published_step == "replay_and_score"


def test_a_cancel_on_the_website_stops_a_live_run(
    run: Callable[..., AuditState],
    audit_dir: Path,
    platform: FakePlatform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The next publish meets the halt, and the run stops there -- nothing more is paid for."""
    real_create = platform.create_foundation_run

    def create(payload: JsonObject) -> JsonObject:
        created = real_create(payload)
        platform.audit_halted = platform.submits == 1  # Cancel pressed while w1 trained
        return created

    monkeypatch.setattr(platform, "create_foundation_run", create)
    state = run(publisher=Publisher(platform, AuditState()))
    assert state.halted == {"reason": "cancelled"}
    assert load_state(audit_dir).halted == {"reason": "cancelled"}
    assert platform.submits == 1
    assert "get_foundation_run" not in platform.call_log
    assert platform.halts == []

    resumed = run(publisher=Publisher(platform, load_state(audit_dir)))
    assert resumed.halted is None
    assert platform.resumes[-1] == ("patch_audit_candidate", False)
    assert resumed.workloads["w2"][SFT].scored is True


def test_a_cancel_between_two_candidates_stops_before_the_second_opens(
    run: Callable[..., AuditState],
    platform: FakePlatform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open = platform.create_audit_candidate

    def open_candidate(audit_id: str, payload: JsonObject) -> JsonObject:
        platform.audit_halted = len(platform.candidates) == 1  # after w1 scored
        return real_open(audit_id, payload)

    monkeypatch.setattr(platform, "create_audit_candidate", open_candidate)
    state = run(publisher=Publisher(platform, AuditState()))
    assert state.halted == {"reason": "cancelled"}
    assert state.workloads["w1"][HEAD].scored is True
    assert platform.call_log.count("upload_dataset") == 1  # w2 never started


def test_the_first_submit_is_held_to_the_ceiling_too(
    run: Callable[..., AuditState], platform: FakePlatform
) -> None:
    """`--max-credits 10` used to submit a 300-credit run; nothing is submitted now."""
    platform.credits_estimate_max = 300
    state = run(max_credits=10, workloads=["w1"])
    assert platform.submits == 0
    assert state.halted is not None
    assert state.halted["reason"] == "budget"


def test_the_default_ceiling_is_the_plan_rounded_up_to_a_hundred(audit_dir: Path) -> None:
    """Each trained candidate at its 150-credit ceiling plus 5 for its 4-row replay."""
    w1 = ("w1", StructureClass.ENUM_LABEL)
    w2 = ("w2", StructureClass.JSON_OBJECT)
    assert plan_credits(audit_dir, [w1]) == 200
    assert plan_credits(audit_dir, [w1, w2]) == 400
    assert plan_credits(audit_dir, []) == 0


def test_a_second_run_over_the_same_directory_is_refused(
    run: Callable[..., AuditState], audit_dir: Path, platform: FakePlatform
) -> None:
    """Both saw no `run_id` and both submitted; last writer won `state.json`."""
    with lock_audit(audit_dir), pytest.raises(AuditBusyError):
        run()
    assert platform.call_log == []
    assert run().halted is None  # and the directory is free again once the first is done


def test_a_cancel_during_a_resumed_run_s_first_wait_sticks(
    run: Callable[..., AuditState],
    audit_dir: Path,
    platform: FakePlatform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--no-wait`, then a resumed run whose first request came after `wait_run`.

    That request carried the run's one `resume: true`, un-halted the audit the
    owner had just cancelled, and the run went on to submit a second paid run.
    """
    run(wait=False, publisher=Publisher(platform, AuditState()))
    real_poll = platform.get_foundation_run

    def poll(run_id: str) -> JsonObject:
        platform.audit_halted = True  # Cancel pressed while w1 trains
        return real_poll(run_id)

    monkeypatch.setattr(platform, "get_foundation_run", poll)
    state = run(publisher=Publisher(platform, load_state(audit_dir)))
    assert state.halted == {"reason": "cancelled"}
    assert platform.submits == 1
    assert "resume_audit" in platform.call_log
    assert all(flag is False for name, flag in platform.resumes if name != "create_audit")


def test_an_audit_deleted_during_a_live_run_stops_it(
    run: Callable[..., AuditState],
    audit_dir: Path,
    platform: FakePlatform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The owner deleted the audit; every publish is a 404 from then on."""
    real_create = platform.create_foundation_run

    def create(payload: JsonObject) -> JsonObject:
        created = real_create(payload)
        platform.audit_deleted = True
        return created

    monkeypatch.setattr(platform, "create_foundation_run", create)
    state = run(publisher=Publisher(platform, AuditState()))
    assert state.halted is not None
    assert state.halted["reason"] == "audit_not_found"  # a 404 is not the deleted state
    assert load_state(audit_dir).halted == state.halted
    assert platform.submits == 1
    assert "get_foundation_run" not in platform.call_log


def test_the_default_ceiling_counts_what_the_audit_already_spent(audit_dir: Path) -> None:
    """`--workloads w1` spent 150; `--workloads w2` was then told a 200 ceiling and halted."""
    state = AuditState()
    w1 = state.candidate("w1", HEAD)
    w1.run_id, w1.run_status, w1.scored = "run-1", "completed", True
    w1.training_cost_credits, w1.replay_cost_credits = 146.0, 4.0
    w2 = ("w2", StructureClass.JSON_OBJECT)
    assert plan_credits(audit_dir, [w2], state) == 400  # 150 spent + 155 still to come
    assert plan_credits(audit_dir, [("w1", StructureClass.ENUM_LABEL), w2], state) == 400

    live = state.candidate("w2", SFT)
    live.run_id, live.run_status = "run-2", "running"  # its ceiling is in what was spent
    assert plan_credits(audit_dir, [w2], state) == 400  # 150 + 150 live + 5 replay
    live.error = "deploy_failed: no gpu"
    assert plan_credits(audit_dir, [w2], state) == 300  # 150 + 150: nothing left to run


def test_a_delete_while_the_run_waits_on_training_ends_it_as_deleted(
    run: Callable[..., AuditState],
    audit_dir: Path,
    platform: FakePlatform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The account's delete took the job with it, so the run's own poll got the 404.

    That ended the run as `halted: error` and a traceback; it is the owner's
    delete, and the run ends as `deleted` -- quietly, with nothing more spent.
    """

    def poll(run_id: str) -> JsonObject:
        platform.audit_deleted = True
        raise FoundationRunNotFoundError(run_id)

    monkeypatch.setattr(platform, "get_foundation_run", poll)
    state = run(publisher=Publisher(platform, AuditState()))
    assert state.halted is not None
    assert state.halted["reason"] == "audit_not_found"  # a 404 is not the deleted state
    assert load_state(audit_dir).halted == state.halted
    assert platform.submits == 1
    assert "halt_audit" not in platform.call_log


def test_a_step_that_fails_under_a_live_audit_is_still_an_error(
    run: Callable[..., AuditState], platform: FakePlatform
) -> None:
    platform.submit_errors = [RuntimeError("socket reset")]
    with pytest.raises(RuntimeError, match="socket reset"):
        run(publisher=Publisher(platform, AuditState()))
    assert platform.call_log.count("get_audit") >= 1  # asked, and the audit was there


def test_a_run_over_an_audit_the_account_deleted_stops_before_any_step(
    run: Callable[..., AuditState], audit_dir: Path, platform: FakePlatform
) -> None:
    """The resume at start is where the run learns it, not its first publish."""
    run(wait=False, publisher=Publisher(platform, AuditState()))
    platform.audit_deleted = True
    platform.call_log.clear()
    state = run(publisher=Publisher(platform, load_state(audit_dir)))
    assert state.halted is not None
    assert state.halted["reason"] == "audit_not_found"
    assert platform.call_log == ["get_platform_build", "resume_audit", "get_audit"]


def test_a_deployment_a_cancel_deleted_does_not_halt_the_next_run_on_its_404(
    run: Callable[..., AuditState],
    audit_dir: Path,
    platform: FakePlatform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cancel deletes a deployment that never served; the candidate used to keep its id.

    The next ``audit run`` then waited on an endpoint the platform answers 404 for, and the
    exception ended the whole run -- every workload after it never started -- on every retry.
    """
    state = run()
    head = state.workloads["w1"][HEAD]
    head.scored, head.deploy_status = None, "deploying"  # still rolling out when the cancel came
    deleted = {
        "kind": "deployment",
        "id": head.deployment_id,
        "status": "deleted",
        "code": "deleted",
    }
    settle_cancel(state, {"schema": "dagnam.audit.cancelled/1", "entries": [deleted]})
    save_state(audit_dir, state)

    def gone(*_args: object, **_kwargs: object) -> None:
        raise DeploymentNotFoundError("dep-1")

    monkeypatch.setattr(platform, "get_deployment_revisions", gone)

    again = run()

    assert again.halted is None
    skipped = again.workloads["w1"][HEAD]
    assert (skipped.deploy_status, skipped.error) == ("deleted", CANCELLED_ERROR)
