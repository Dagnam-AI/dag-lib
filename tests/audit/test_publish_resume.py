"""The publisher under failure and on a resumed run: retries, the back-fill, the halt."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from tests.audit._platform import FakePlatform
from tests.audit._publish import AUDIT_ID, start

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import APIError
from dagnam._types import JsonObject
from dagnam.audit.candidates import CandidateKind
from dagnam.audit.cleanup import DEPLOY_PAUSED
from dagnam.audit.publish import (
    DONE_BY_STEP,
    MAX_PENDING,
    UNKNOWN_VERSION,
    Publisher,
    installed_version,
)
from dagnam.audit.state import AuditState, StepState

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tests.typing_helpers import PytestMonkeyPatch, RequestsMocker

    from dagnam.audit.steps import StepContext


# -------------------------------------------------------------- halt / resilience


def test_halt_names_the_reason_the_run_stopped(
    publisher: Publisher, platform: FakePlatform, audit_dir: Path
) -> None:
    start(publisher, audit_dir)
    publisher.halt("budget")
    assert platform.halts == [("audit-1", "budget")]


def test_a_failed_step_is_resent_with_the_next_one_and_never_raises(
    publisher: Publisher,
    platform: FakePlatform,
    audit_dir: Path,
    make_ctx: Callable[..., StepContext],
    caplog: pytest.LogCaptureFixture,
) -> None:
    start(publisher, audit_dir)
    ctx = make_ctx()
    platform.publish_errors["patch_audit_candidate"] = [APIError(0, "Connection failed")]

    publisher.step(ctx, "upload", StepState())
    assert platform.patches == []
    assert "'upload' failed (API error 0: Connection failed); will retry" in caplog.text
    assert "not retried" not in caplog.text

    publisher.step(ctx, "split", StepState(version_id="ver-1"))
    assert [body["step"] for _, body in platform.patches] == ["upload", "split"]


def test_a_body_the_server_calls_a_client_bug_is_logged_once_and_dropped(
    publisher: Publisher,
    platform: FakePlatform,
    audit_dir: Path,
    make_ctx: Callable[..., StepContext],
    caplog: pytest.LogCaptureFixture,
) -> None:
    start(publisher, audit_dir)
    ctx = make_ctx()
    platform.publish_errors["patch_audit_candidate"] = [
        APIError(422, "extra keys not permitted"),
        APIError(409, "candidate cannot move from 'scored' to 'splitting'"),
    ]

    publisher.step(ctx, "upload", StepState())
    publisher.step(ctx, "split", StepState())
    publisher.step(ctx, "pii_scan", StepState())

    # Neither the bad body nor the refused transition is ever resent.
    assert [body["step"] for _, body in platform.patches] == ["pii_scan"]
    # A drop reads differently from a retry: nothing is coming back for these.
    assert "step 'upload' was refused by the server (HTTP 422); not retried" in caplog.text
    assert "step 'split' was refused by the server (HTTP 409); not retried" in caplog.text
    assert "will retry with the next step" not in caplog.text


def test_the_resend_queue_is_capped_rather_than_grown_forever(
    publisher: Publisher,
    platform: FakePlatform,
    audit_dir: Path,
    make_ctx: Callable[..., StepContext],
    caplog: pytest.LogCaptureFixture,
) -> None:
    start(publisher, audit_dir)
    ctx, step = make_ctx(), StepState()
    platform.publish_errors["patch_audit_candidate"] = [
        APIError(0, "down") for _ in range(MAX_PENDING + 5)
    ]

    for _ in range(MAX_PENDING + 2):
        publisher.step(ctx, "upload", step)

    assert "dropping the unsent step" in caplog.text


def test_a_failure_that_is_not_an_api_error_is_still_only_a_warning(
    publisher: Publisher,
    platform: FakePlatform,
    audit_dir: Path,
    state: AuditState,
    make_ctx: Callable[..., StepContext],
    caplog: pytest.LogCaptureFixture,
) -> None:
    platform.publish_errors["create_audit"] = [TypeError("not JSON serializable")]
    start(publisher, audit_dir)
    assert state.audit_id is None

    state.audit_id = AUDIT_ID
    platform.publish_errors["patch_audit_candidate"] = [TypeError("not JSON serializable")]
    publisher.step(make_ctx(), "upload", StepState())
    assert platform.patches == []
    assert caplog.text.count("the run continues") == 1  # create_audit
    assert caplog.text.count("will retry with the next step") == 1  # the patch


# ----------------------------------------------------------------- local only


def test_a_publisher_without_a_client_records_nothing(
    audit_dir: Path, state: AuditState, make_ctx: Callable[..., StepContext]
) -> None:
    publisher = Publisher(None, state)
    ctx, step = make_ctx(), StepState()

    start(publisher, audit_dir)
    publisher.candidate(ctx, step)
    publisher.step(ctx, "upload", step)
    publisher.halt("cancelled")

    assert state.audit_id is None
    assert step.published_candidate_id is None


def test_follow_points_the_publisher_at_the_state_the_run_loaded(
    publisher: Publisher, platform: FakePlatform, audit_dir: Path
) -> None:
    live = AuditState(project_id="proj-9")
    publisher.follow(live)
    start(publisher, audit_dir)
    assert live.audit_id == "audit-1"
    assert platform.audits[0]["project_id"] == "proj-9"


# --------------------------------------------------------------- sdk version


def test_installed_version_reads_the_distribution(monkeypatch: PytestMonkeyPatch) -> None:
    assert installed_version() == __import__("dagnam").__version__

    def missing(_name: str) -> str:
        from importlib.metadata import PackageNotFoundError

        raise PackageNotFoundError(_name)

    monkeypatch.setattr("dagnam.audit.publish.version", missing)
    assert installed_version() == UNKNOWN_VERSION


def test_the_state_round_trips_the_published_ids(tmp_path: Path) -> None:
    from dagnam.audit.state import load_state, save_state

    state = AuditState(project_id="proj-1", audit_id=AUDIT_ID)
    state.candidate("w1", CandidateKind.HEAD_TUNE).published_candidate_id = "cand-1"
    save_state(tmp_path, state)

    read = load_state(tmp_path)
    assert read.audit_id == AUDIT_ID
    assert read.candidate("w1", CandidateKind.HEAD_TUNE).published_candidate_id == "cand-1"


# ---------------------------------------------------------------- the back-fill


def _completed_upload(step: StepState) -> StepState:
    step.dataset_id, step.version_id = "ds-1", "ver-1"
    return step


def test_a_run_whose_first_publish_failed_backfills_what_it_already_did(
    platform: FakePlatform, state: AuditState, audit_dir: Path, make_ctx: Callable[..., StepContext]
) -> None:
    """The audit lands on the second run; its candidates must not sit at `uploading`."""
    publisher = Publisher(platform, state)
    step = _completed_upload(StepState())

    start(publisher, audit_dir)
    ctx = make_ctx()
    publisher.candidate(ctx, step)
    publisher.backfill(ctx, step)

    assert [body["step"] for _, body in platform.patches] == ["upload", "resolve_version"]
    assert [body["status"] for _, body in platform.patches] == ["uploading", "uploading"]
    assert platform.patches[-1][1]["dataset_version_id"] == "ver-1"


def test_a_transition_the_backfill_gets_wrong_is_dropped_not_retried_forever(
    platform: FakePlatform, state: AuditState, audit_dir: Path, make_ctx: Callable[..., StepContext]
) -> None:
    publisher = Publisher(platform, state)
    step = _completed_upload(StepState())
    start(publisher, audit_dir)
    platform.publish_errors["patch_audit_candidate"] = [APIError(409, "cannot move from 'scored'")]

    ctx = make_ctx()
    publisher.candidate(ctx, step)
    publisher.backfill(ctx, step)

    # The refused patch is gone, not queued in front of the one after it.
    assert [body["step"] for _, body in platform.patches] == ["resolve_version"]


def test_the_backfill_is_silent_on_a_fresh_run_and_on_a_resumed_published_one(
    platform: FakePlatform, state: AuditState, audit_dir: Path, make_ctx: Callable[..., StepContext]
) -> None:
    fresh = Publisher(platform, state)
    start(fresh, audit_dir)
    ctx = make_ctx()
    fresh.backfill(ctx, StepState())  # nothing is done yet
    assert platform.patches == []

    resumed_state = AuditState(project_id="proj-1", audit_id=AUDIT_ID)
    resumed = Publisher(platform, resumed_state)
    start(resumed, audit_dir)  # returns before the POST: the audit already exists
    acknowledged = _completed_upload(StepState(published_candidate_id="cand-1"))
    acknowledged.published_step = "resolve_version"
    resumed.backfill(ctx, acknowledged)
    assert platform.patches == []


def test_a_step_whose_patch_never_landed_goes_out_on_the_next_run(
    platform: FakePlatform, audit_dir: Path, make_ctx: Callable[..., StepContext]
) -> None:
    """B10: the queue dies with the process, the acknowledgement mark does not."""
    step = _completed_upload(StepState(published_candidate_id="cand-1"))
    step.published_step = "upload"  # `resolve_version`'s patch was lost to a blip or a Ctrl+C
    resumed = Publisher(platform, AuditState(project_id="proj-1", audit_id=AUDIT_ID))
    resumed.backfill(make_ctx(), step)
    assert [body["step"] for _, body in platform.patches] == ["resolve_version"]
    assert step.published_step == "resolve_version"


def test_the_last_patch_of_a_run_is_flushed_before_the_run_ends(
    publisher: Publisher,
    platform: FakePlatform,
    audit_dir: Path,
    make_ctx: Callable[..., StepContext],
) -> None:
    """B10: nothing comes after the last step to carry a failed patch out with it."""
    start(publisher, audit_dir)
    step = StepState()
    platform.publish_errors["patch_audit_candidate"] = [APIError(503, "blip")]
    publisher.step(make_ctx(), "upload", step)
    assert platform.patches == []
    assert step.published_step is None
    publisher.flush()
    assert [body["step"] for _, body in platform.patches] == ["upload"]
    assert step.published_step == "upload"


def test_an_acknowledged_failure_is_not_published_again(
    platform: FakePlatform, audit_dir: Path, make_ctx: Callable[..., StepContext]
) -> None:
    step = _completed_upload(StepState(published_candidate_id="cand-1"))
    step.split_task_id, step.error = "split-1", "split_failed: the task never started"
    step.published_step = "wait_split"  # the step it died on, published `failed`
    resumed = Publisher(platform, AuditState(project_id="proj-1", audit_id=AUDIT_ID))
    resumed.backfill(make_ctx(), step)
    assert platform.patches == []


# ------------------------------------------------------ K1: resume, and a cancel that sticks


def test_only_the_first_request_of_a_run_asks_to_resume(
    publisher: Publisher,
    platform: FakePlatform,
    audit_dir: Path,
    make_ctx: Callable[..., StepContext],
) -> None:
    platform.publish_errors["create_audit"] = [APIError(503, "down")]
    start(publisher, audit_dir)  # never reached the account: the resume is not spent
    start(publisher, audit_dir)
    step = StepState()
    publisher.step(make_ctx(), "upload", step)
    publisher.step(make_ctx(), "split", step)
    assert platform.resumes == [
        ("create_audit", True),
        ("create_audit_candidate", False),
        ("patch_audit_candidate", False),
        ("patch_audit_candidate", False),
    ]


def test_a_cancel_on_the_website_ends_the_publishing_and_says_so(
    platform: FakePlatform, make_ctx: Callable[..., StepContext], caplog: pytest.LogCaptureFixture
) -> None:
    """P6: the halted 409 a non-resuming publish gets is the website's cancel."""
    publisher = Publisher(platform, AuditState(project_id="proj-1", audit_id=AUDIT_ID))
    step = StepState(published_candidate_id="cand-1")
    publisher.step(make_ctx(), "upload", step)
    assert publisher.stopped is None

    platform.audit_halted = True  # somebody pressed Cancel on the audit page
    publisher.step(make_ctx(), "resolve_version", step)
    assert publisher.stopped == "cancelled"
    assert "the audit was cancelled in your account" in caplog.text
    assert step.published_step == "upload"

    publisher.step(make_ctx(), "split", step)
    publisher.candidate(make_ctx(workload_id="w2"), StepState())
    publisher.flush()
    publisher.halt("error")
    assert [body["step"] for _, body in platform.patches] == ["upload"]
    assert platform.halts == []


def test_the_first_request_of_a_new_run_resumes_a_halted_audit(
    platform: FakePlatform, make_ctx: Callable[..., StepContext]
) -> None:
    platform.audit_halted = True
    publisher = Publisher(platform, AuditState(project_id="proj-1", audit_id=AUDIT_ID))
    step = StepState(published_candidate_id="cand-1")
    publisher.step(make_ctx(), "upload", step)
    publisher.step(make_ctx(), "resolve_version", step)
    assert publisher.stopped is None
    assert [flag for _, flag in platform.resumes] == [True, False]
    assert platform.audit_halted is False


def test_every_step_of_the_frontier_has_a_done_guard() -> None:
    from dagnam.audit.orchestrate import STEPS

    assert list(DONE_BY_STEP) == [run_step.__name__ for run_step in STEPS]


# ------------------------------------ the halt a resumed run inherits from the last one


def _halted(state: AuditState) -> AuditState:
    state.audit_id, state.halted = AUDIT_ID, {"reason": "budget"}
    return state


def test_the_halt_the_previous_run_published_is_not_sent_again(
    platform: FakePlatform, state: AuditState
) -> None:
    """The account already shows this audit halted; repeating it says nothing new."""
    platform.audit_halted = True
    publisher = Publisher(platform, _halted(state))
    publisher.halt("budget")
    assert platform.halts == []

    publisher.follow(_halted(AuditState(project_id="proj-1")))
    publisher.halt("error")
    assert platform.halts == []


def test_a_halt_after_this_run_published_anything_is_sent(
    platform: FakePlatform, state: AuditState, make_ctx: Callable[..., StepContext]
) -> None:
    """A patch the server applies resumes the audit, so the next halt is a new one."""
    publisher = Publisher(platform, _halted(state))
    step = StepState(published_candidate_id="cand-1")

    publisher.step(make_ctx(), "upload", step)
    publisher.halt("budget")

    assert platform.halts == [(AUDIT_ID, "budget")]
    publisher.halt("budget")  # and only once
    assert platform.halts == [(AUDIT_ID, "budget")]


def test_a_run_that_never_halted_publishes_its_first_halt(
    publisher: Publisher, platform: FakePlatform, audit_dir: Path
) -> None:
    start(publisher, audit_dir)
    publisher.halt("error")
    publisher.halt("error")
    assert platform.halts == [("audit-1", "error")]


def test_the_wait_active_guard_mirrors_the_step_s_own_or_scored(
    platform: FakePlatform, state: AuditState, audit_dir: Path, make_ctx: Callable[..., StepContext]
) -> None:
    """A candidate that scored and was paused since has finished `wait_active`.

    The step returns early on `step.scored` as well as on `running` -- it was
    live once, and `dagnam audit cancel` pauses that endpoint -- so a guard
    that asked only for `running` made the back-fill publish one `deploying`
    pulse the run itself never sends.
    """
    scored_then_paused = StepState(
        deployment_id="dep-1", deploy_status=DEPLOY_PAUSED, scored=True, published_candidate_id="c1"
    )
    assert DONE_BY_STEP["wait_active"](scored_then_paused)

    publisher = Publisher(platform, state)
    start(publisher, audit_dir)
    publisher.backfill(make_ctx(), scored_then_paused)

    steps = [body["step"] for _, body in platform.patches]
    assert steps.index("wait_active") < steps.index("replay")


def test_the_backfill_of_an_errored_candidate_ends_at_its_terminal_failure(
    platform: FakePlatform, state: AuditState, audit_dir: Path, make_ctx: Callable[..., StepContext]
) -> None:
    """The steps it finished keep their own status; the one it died on carries the error.

    Marking every finished step `failed` would say the candidate died at
    `upload`; the step that failed is the first one it never finished.
    """
    step = _completed_upload(StepState())
    step.split_task_id, step.error = "split-1", "split_failed: the task never started"

    publisher = Publisher(platform, state)
    start(publisher, audit_dir)
    publisher.candidate(make_ctx(), step)
    publisher.backfill(make_ctx(), step)

    assert [(body["step"], body["status"]) for _, body in platform.patches] == [
        ("upload", "uploading"),
        ("resolve_version", "uploading"),
        ("split", "splitting"),
        ("wait_split", "failed"),
    ]
    assert platform.patches[-1][1]["error"] == "split_failed: the task never started"


def test_the_platform_s_own_halted_409_reads_as_the_cancel(
    requests_mock: RequestsMocker, make_ctx: Callable[..., StepContext]
) -> None:
    """The wire form, through the real client: FastAPI's ``{"detail": "audit is halted"}``."""
    requests_mock.patch(
        "https://x/api/v1/audits/audit-1/candidates/cand-1",
        status_code=409,
        json={"detail": "audit is halted"},
    )
    publisher = Publisher(
        DagnamClient("https://x", "k"), AuditState(project_id="proj-1", audit_id=AUDIT_ID)
    )
    publisher.step(make_ctx(), "upload", StepState(published_candidate_id="cand-1"))
    assert publisher.stopped == "cancelled"
    assert requests_mock.last_request.json() == {
        "step": "upload",
        "status": "uploading",
        "resume": True,
    }


# ------------------------------------------- K1b: resume up front; a delete or a cancel stops


def test_a_run_resumes_its_audit_once_up_front_and_never_again(
    platform: FakePlatform, make_ctx: Callable[..., StepContext]
) -> None:
    """N1: the resume is spent before any wait, so a cancel landing during one sticks."""
    platform.audit_halted = True  # the last run stopped on its budget
    publisher = Publisher(platform, AuditState(project_id="proj-1", audit_id=AUDIT_ID))
    publisher.resume()
    assert platform.audit_halted is False

    platform.audit_halted = True  # the owner cancels while this run trains
    publisher.step(make_ctx(), "wait_run", StepState(published_candidate_id="cand-1"))
    assert publisher.stopped == "cancelled"
    assert platform.resumes == [("patch_audit_candidate", False)]


def test_an_older_platform_without_the_resume_route_resumes_on_the_first_publish(
    platform: FakePlatform, make_ctx: Callable[..., StepContext], caplog: pytest.LogCaptureFixture
) -> None:
    platform.resume_route = False
    platform.audit_halted = True
    publisher = Publisher(platform, AuditState(project_id="proj-1", audit_id=AUDIT_ID))
    publisher.resume()
    step = StepState(published_candidate_id="cand-1")
    publisher.step(make_ctx(), "upload", step)
    publisher.step(make_ctx(), "resolve_version", step)
    assert publisher.stopped is None
    assert [flag for _, flag in platform.resumes] == [True, False]
    assert "the first publish resumes it" in caplog.text


def test_a_run_with_nothing_published_yet_has_nothing_to_resume_or_lose(
    platform: FakePlatform, state: AuditState
) -> None:
    publisher = Publisher(platform, state)
    publisher.resume()
    assert publisher.gone() is False
    assert platform.call_log == []


def test_an_audit_deleted_in_the_account_stops_the_run(
    platform: FakePlatform, make_ctx: Callable[..., StepContext], caplog: pytest.LogCaptureFixture
) -> None:
    """Backend review N3: every publish 404s after a delete; the old queue retried forever."""
    publisher = Publisher(platform, AuditState(project_id="proj-1", audit_id=AUDIT_ID))
    platform.audit_deleted = True
    publisher.step(make_ctx(), "upload", StepState(published_candidate_id="cand-1"))
    assert publisher.stopped == "deleted"
    assert "the audit was deleted in your account" in caplog.text
    publisher.candidate(make_ctx(workload_id="w2"), StepState())
    publisher.halt("error")
    assert platform.call_log.count("patch_audit_candidate") == 1
    assert "halt_audit" not in platform.call_log


def test_a_read_that_fails_for_another_reason_is_not_a_deleted_audit(
    platform: FakePlatform,
) -> None:
    platform.publish_errors["get_audit"] = [APIError(503, "busy")]
    publisher = Publisher(platform, AuditState(project_id="proj-1", audit_id=AUDIT_ID))
    assert publisher.gone() is False
    assert publisher.stopped is None


def test_a_404_for_one_artifact_is_dropped_not_taken_for_a_deleted_audit(
    platform: FakePlatform,
    make_ctx: Callable[..., StepContext],
    monkeypatch: PytestMonkeyPatch,
) -> None:
    """A patch naming a deployment the owner deleted is the same uniform 404: the read settles it."""
    real_patch = platform.patch_audit_candidate
    refused: list[bool] = []

    def patch(audit_id: str, candidate_id: str, payload: JsonObject) -> JsonObject:
        if not refused:
            refused.append(True)
            raise APIError(404, "Audit not found")
        return real_patch(audit_id, candidate_id, payload)

    monkeypatch.setattr(platform, "patch_audit_candidate", patch)
    publisher = Publisher(platform, AuditState(project_id="proj-1", audit_id=AUDIT_ID))
    step = StepState(published_candidate_id="cand-1")
    publisher.step(make_ctx(), "upload", step)
    publisher.step(make_ctx(), "resolve_version", step)
    assert publisher.stopped is None
    assert [body["step"] for _, body in platform.patches] == ["resolve_version"]


def test_a_halt_never_overwrites_the_account_s_cancel(
    platform: FakePlatform, make_ctx: Callable[..., StepContext]
) -> None:
    """M1: the owner cancelled, then the next submit failed the budget check -- nothing queued.

    The account's halt reason is the owner's `cancelled`; a `budget` halt over it
    would tell them the run ran out of money.
    """
    publisher = Publisher(platform, AuditState(project_id="proj-1", audit_id=AUDIT_ID))
    publisher.resume()
    platform.audit_halted = True
    publisher.halt("budget")
    assert platform.halts == []

    # And when it is the halt's own flush that meets the cancel: a patch a blip
    # left queued goes out first, gets the 409, and nothing overwrites it.
    publisher = Publisher(platform, AuditState(project_id="proj-1", audit_id=AUDIT_ID))
    publisher.resume()
    platform.publish_errors["patch_audit_candidate"] = [APIError(503, "blip")]
    publisher.step(make_ctx(), "upload", StepState(published_candidate_id="cand-1"))
    platform.audit_halted = True
    publisher.halt("budget")
    assert publisher.stopped == "cancelled"
    assert platform.halts == []
    assert platform.call_log.count("get_audit") == 1  # the first halt's read only
