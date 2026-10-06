"""The publisher under failure and on a resumed run: retries, the back-fill, the halt."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from tests.audit._platform import FakePlatform
from tests.audit._publish import AUDIT_ID, start

from dagnam._core.exceptions import APIError
from dagnam.audit.candidates import CandidateKind
from dagnam.audit.publish import (
    MAX_PENDING,
    UNKNOWN_VERSION,
    Publisher,
    installed_version,
)
from dagnam.audit.state import AuditState, StepState

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tests.typing_helpers import PytestMonkeyPatch

    from dagnam.audit.steps import StepContext


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


def test_a_create_audit_failure_halts_the_run_but_a_patch_failure_is_only_a_warning(
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
    assert publisher.create_failed == "not JSON serializable"  # the run halts before any upload

    state.audit_id = AUDIT_ID
    platform.publish_errors["patch_audit_candidate"] = [TypeError("not JSON serializable")]
    publisher.step(make_ctx(), "upload", StepState())
    assert platform.patches == []
    assert caplog.text.count("the run halts") == 1  # create_audit
    assert caplog.text.count("will retry with the next step") == 1  # the patch


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
    """The queue dies with the process, the acknowledgement mark does not."""
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
    """Nothing comes after the last step to carry a failed patch out with it."""
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
