"""A cancel: the platform's receipt is recorded for a published audit, the recorded ids are stopped for the rest."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.audit import _receipts as r
from tests.audit._cleanup import FakeCleanup, as_cleanup_client
from tests.audit._recorded import HEAD, SFT, live_state, recorded_state

from dagnam._core.exceptions import APIError
from dagnam._types import JsonObject
from dagnam.audit.cleanup import (
    CANCELLED_ERROR,
    cancel_unpublished,
    delete_audit,
    live,
    mark_cancelled,
    receipt_rows,
    recorded_ids,
    settle_cancel,
)
from dagnam.audit.receipt_rows import Verdict, decide
from dagnam.audit.secrets import SecretStore
from dagnam.audit.state import AuditState, StepState, load_state, save_state


@pytest.fixture
def platform() -> FakeCleanup:
    return FakeCleanup(
        deployment=["dep-1", "dep-2"], job=["job-1", "job-2"], dataset=["ds-1", "ds-2"]
    )


def _left(receipt: JsonObject) -> list[str]:
    """The ids a cancel's receipt says may still be running or serving."""
    return [str(row["id"]) for row in receipt_rows(receipt) if decide(row).verdict is Verdict.LEFT]


class TestPublished:
    """The platform stopped what it stopped; this client records the rows and calls nothing."""

    def test_a_stopped_row_marks_the_run_and_the_endpoint_and_a_gone_row_marks_it_gone(
        self,
    ) -> None:
        state = live_state()
        receipt = r.receipt(
            r.row("training_job", "job-1", "stopped"),
            r.row("deployment", "dep-2", "deleted"),
            schema=r.SCHEMA_CANCELLED,
        )

        assert settle_cancel(state, receipt) is receipt

        head, done = state.workloads["w1"][HEAD], state.workloads["w2"][SFT]
        assert (head.run_status, head.error) == ("cancelled", CANCELLED_ERROR)
        assert (done.deploy_status, done.error, done.scored) == ("deleted", None, True)
        assert state.halted == {"reason": "cancelled"}

    def test_an_already_stopped_run_is_not_marked_and_stays_resumable(self) -> None:
        """A run that finished before the cancel keeps the status it really has."""
        state = live_state()
        receipt = r.receipt(
            r.ALREADY_STOPPED_ROW,
            r.row("deployment", "dep-2", "stopped", "already_stopped"),
            schema=r.SCHEMA_CANCELLED,
        )

        settle_cancel(state, receipt)

        head, done = state.workloads["w1"][HEAD], state.workloads["w2"][SFT]
        assert (head.run_status, head.error) == ("queued", None)
        assert (done.deploy_status, done.error) == ("running", None)
        assert state.halted == {"reason": "cancelled"}

    @pytest.mark.parametrize(
        "entry",
        [
            r.row("training_job", "job-1", "kept", "not_in_project", r.NOT_IN_PROJECT),
            r.row("training_job", "job-1", "blocked", "refused", "Cannot cancel"),
            r.row("training_job", "job-1", "quarantined", "a_code_from_the_future"),
            r.legacy(
                r.row(
                    "training_job",
                    "job-1",
                    "blocked",
                    None,
                    "Cannot cancel job with status completed",
                )
            ),
        ],
        ids=["kept", "refused", "unknown_status", "legacy_blocked"],
    )
    def test_a_row_that_is_not_a_stop_marks_nothing(self, entry: JsonObject) -> None:
        state = live_state()

        settle_cancel(state, r.receipt(entry, schema=r.SCHEMA_CANCELLED))

        assert state.workloads["w1"][HEAD].error is None
        assert state.workloads["w1"][HEAD].run_status == "queued"

    def test_the_legacy_stopped_row_without_a_code_marks_like_a_stop(self) -> None:
        state = live_state()

        settle_cancel(
            state,
            r.receipt(
                r.legacy(r.row("training_job", "job-1", "stopped")), schema=r.SCHEMA_CANCELLED
            ),
        )

        assert state.workloads["w1"][HEAD].run_status == "cancelled"


class TestUnpublished:
    """No platform record: every recorded live run and endpoint is stopped from here."""

    def test_a_run_that_ended_while_nobody_watched_is_already_stopped_and_stays_resumable(
        self, platform: FakeCleanup
    ) -> None:
        """`--no-wait` left the run `queued`; the platform answers its cancel with a 400."""
        platform.finished = {"job-1"}
        state = live_state()

        receipt = cancel_unpublished(state, as_cleanup_client(platform))

        assert receipt["entries"] == [
            {"kind": "training_job", "id": "job-1", "status": "stopped", "code": "already_stopped"},
            {"kind": "deployment", "id": "dep-2", "status": "stopped"},
        ]
        assert receipt["schema"] == "dagnam.audit.cancelled/1"
        assert _left(receipt) == []
        head, done = state.workloads["w1"][HEAD], state.workloads["w2"][SFT]
        assert (head.run_status, head.error) == ("queued", None)
        assert (done.deploy_status, done.error) == ("paused", None)
        assert state.halted == {"reason": "cancelled"}

    def test_every_run_is_stopped_even_when_an_earlier_stop_fails(
        self, platform: FakeCleanup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Three runs are training; the first cancel answers 500. The rest were never reached."""
        platform.present["job"] |= {"job-w1", "job-w2", "job-w3"}
        state = AuditState(project_id="proj-1")
        state.workloads["w1"] = {
            HEAD: StepState(training_job_id="job-w1", run_status="running"),
            SFT: StepState(training_job_id="job-w2", run_status="running"),
        }
        state.workloads["w2"] = {HEAD: StepState(training_job_id="job-w3", run_status="running")}
        real = platform.cancel_training_job

        def flaky(job_id: str) -> JsonObject:
            if job_id == "job-w1":
                raise APIError(500, "boom")
            return real(job_id)

        monkeypatch.setattr(platform, "cancel_training_job", flaky)

        receipt = cancel_unpublished(state, as_cleanup_client(platform))

        assert [(row["id"], row["status"]) for row in receipt["entries"]] == [
            ("job-w1", "blocked"),
            ("job-w2", "stopped"),
            ("job-w3", "stopped"),
        ]
        assert receipt["entries"][0]["reason"] == "API error 500: boom"
        assert _left(receipt) == ["job-w1"]
        assert state.workloads["w1"][HEAD].error is None
        assert state.workloads["w1"][SFT].error == CANCELLED_ERROR
        assert state.workloads["w2"][HEAD].run_status == "cancelled"

    def test_an_endpoint_that_cannot_be_paused_is_blocked_and_a_gone_one_is_already_absent(
        self, platform: FakeCleanup
    ) -> None:
        platform.present["job"].discard("job-1")
        platform.unpausable = {"dep-2"}
        state = live_state()
        state.workloads["w4"] = {HEAD: StepState(deployment_id="dep-gone", deploy_status="running")}

        receipt = cancel_unpublished(state, as_cleanup_client(platform))

        assert [(row["id"], row["status"]) for row in receipt["entries"]] == [
            ("job-1", "already_absent"),
            ("dep-2", "blocked"),
            ("dep-gone", "already_absent"),
        ]
        assert _left(receipt) == ["dep-2"]
        assert state.workloads["w2"][SFT].deploy_status == "running"

    def test_retired_candidates_are_cancelled_and_deleted(self, audit_dir: Path) -> None:
        platform = FakeCleanup(
            deployment=["dep-1", "dep-2"],
            model=["mv-1", "mv-2"],
            job=["job-1", "job-2"],
            dataset=["ds-1", "ds-2"],
            project=["proj-1"],
        )
        save_state(audit_dir, recorded_state())
        SecretStore(audit_dir).store("w1/head_tune", "dk-secret")
        state = load_state(audit_dir)
        state.retired = list(state.all_steps())
        state.workloads.clear()
        assert recorded_ids(state) == recorded_ids(recorded_state())
        cancel_unpublished(state, as_cleanup_client(platform))
        assert ("cancel_training_job", "job-1") in platform.call_log
        assert ("pause_deployment", "dep-1") in platform.call_log
        assert state.retired[1].run_status == "cancelled"
        save_state(audit_dir, state)
        delete_audit(audit_dir, as_cleanup_client(platform))
        assert all(not ids for ids in platform.present.values())
        assert SecretStore(audit_dir).load("w1/head_tune") is None
        assert load_state(audit_dir).retired == state.retired


def test_a_job_the_platform_says_is_gone_is_marked_like_a_stopped_one() -> None:
    state = live_state()

    mark_cancelled(state, stopped=set(), gone={"job-1"})

    head = state.workloads["w1"][HEAD]
    assert (head.run_status, head.error) == ("cancelled", CANCELLED_ERROR)


def test_a_deployment_a_cancel_deleted_is_marked_so_the_next_run_does_not_halt_on_its_404() -> None:
    """The platform deletes one that never served instead of pausing it; its row says ``deleted``."""
    state = AuditState(project_id="proj-1")
    state.workloads["w1"] = {
        HEAD: StepState(
            training_job_id="job-1",
            run_status="completed",
            deployment_id="dep-1",
            deploy_status="deploying",
        ),
        SFT: StepState(
            training_job_id="job-2",
            run_status="completed",
            deployment_id="dep-2",
            deploy_status="running",
            scored=True,
        ),
    }
    server = r.receipt(
        r.row("deployment", "dep-1", "deleted"),
        r.row("deployment", "dep-2", "already_absent"),
        schema=r.SCHEMA_CANCELLED,
    )

    settle_cancel(state, server)

    mid, scored = state.workloads["w1"][HEAD], state.workloads["w1"][SFT]
    assert (mid.deploy_status, mid.error) == ("deleted", CANCELLED_ERROR)
    assert (scored.deploy_status, scored.error, scored.scored) == ("deleted", None, True)
    assert live(mid) == live(scored) == []
