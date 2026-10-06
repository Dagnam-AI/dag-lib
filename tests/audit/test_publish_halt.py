"""The publisher's resume and halt: once up front, never over the account's own cancel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from tests.audit._platform import FakePlatform
from tests.audit._publish import AUDIT_ID, start

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import APIError
from dagnam._types import JsonObject
from dagnam.audit.cleanup import DEPLOY_PAUSED
from dagnam.audit.publish import (
    DONE_BY_STEP,
    Publisher,
)
from dagnam.audit.state import AuditState, StepState

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tests.typing_helpers import PytestMonkeyPatch, RequestsMocker

    from dagnam.audit.steps import StepContext


def _completed_upload(step: StepState) -> StepState:
    step.dataset_id, step.version_id = "ds-1", "ver-1"
    return step


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
    """The halted 409 a non-resuming publish gets is the website's cancel."""
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


def test_a_run_resumes_its_audit_once_up_front_and_never_again(
    platform: FakePlatform, make_ctx: Callable[..., StepContext]
) -> None:
    """The resume is spent before any wait, so a cancel landing during one sticks."""
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
    """Every publish 404s after a delete; the old queue retried forever."""
    publisher = Publisher(platform, AuditState(project_id="proj-1", audit_id=AUDIT_ID))
    platform.audit_deleted = True
    publisher.step(make_ctx(), "upload", StepState(published_candidate_id="cand-1"))
    assert publisher.stopped == "deleted"
    assert "the platform has no audit with this id for this key" in caplog.text
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
    """The owner cancelled, then the next submit failed the budget check -- nothing queued.

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
