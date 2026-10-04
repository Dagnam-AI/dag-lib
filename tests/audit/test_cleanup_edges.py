"""The edges of a teardown: a walk another caller holds, what a claim refused, a status nobody knows."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from tests.audit import _receipts as r
from tests.audit._cleanup import FakeCleanup, as_cleanup_client
from tests.audit._recorded import HEAD, recorded_state

from dagnam._core.exceptions import TeardownInProgressError
from dagnam._types import JsonObject
from dagnam.audit.cleanup import (
    TEARDOWN_POLL,
    TEARDOWN_WAIT,
    cancel_audit,
    delete_audit,
    delete_unpublished,
    receipt_rows,
)
from dagnam.audit.receipt_rows import exit_status
from dagnam.audit.secrets import SecretStore
from dagnam.audit.state import load_state, save_state


@pytest.fixture
def platform() -> FakeCleanup:
    return FakeCleanup(
        deployment=["dep-1", "dep-2"],
        model=["mv-1", "mv-2"],
        job=["job-1", "job-2"],
        dataset=["ds-1", "ds-2"],
        project=["proj-1"],
    )


@pytest.fixture
def prepared(audit_dir: Path) -> Path:
    state = recorded_state()
    state.audit_id = "audit-1"
    save_state(audit_dir, state)
    SecretStore(audit_dir).store("w1/head_tune", "dk-secret")
    return audit_dir


def _busy(retry_after: str | None = "2") -> TeardownInProgressError:
    return TeardownInProgressError(409, "another walk", retry_after_header=retry_after)


class _Sleeps:
    def __init__(self) -> None:
        self.slept: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.slept.append(seconds)


def _answers(
    platform: FakeCleanup, monkeypatch: pytest.MonkeyPatch, *failures: Exception
) -> list[str]:
    """Make the next ``delete_audit`` calls raise ``failures`` in order, then answer for real."""
    pending = list(failures)
    real = platform.delete_audit
    calls: list[str] = []

    def delete(audit_id: str) -> JsonObject:
        calls.append(audit_id)
        if pending:
            raise pending.pop(0)
        return real(audit_id)

    monkeypatch.setattr(platform, "delete_audit", delete)
    return calls


def _exit(receipt: Mapping[str, Any], verb: str) -> int:
    return exit_status(receipt_rows(receipt), receipt.get("audit_status"), verb=verb)


class TestAnotherWalkHoldsTheAudit:
    """``409 teardown_in_progress``: wait it out, then ask again; give up as not answered."""

    def test_the_wait_follows_retry_after_and_the_third_ask_is_the_answer(
        self, prepared: Path, platform: FakeCleanup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        platform.server_receipt = r.designed(r.DELETED_ROW)
        calls = _answers(platform, monkeypatch, _busy("2"), _busy("3"))
        sleeps = _Sleeps()

        receipt = delete_audit(prepared, as_cleanup_client(platform), sleep=sleeps)

        assert len(calls) == 3
        assert sleeps.slept == [2.0, 3.0]
        assert _exit(receipt, "delete") == 0

    def test_without_retry_after_it_polls_at_the_default_pace(
        self, prepared: Path, platform: FakeCleanup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _answers(platform, monkeypatch, _busy(None))
        sleeps = _Sleeps()
        delete_audit(prepared, as_cleanup_client(platform), sleep=sleeps)
        assert sleeps.slept == [TEARDOWN_POLL]

    def test_a_walk_that_never_ends_is_not_answered_after_the_bound_and_nothing_is_touched(
        self, prepared: Path, platform: FakeCleanup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _answers(platform, monkeypatch, *[_busy(None)] * 100)
        sleeps = _Sleeps()

        receipt = delete_audit(prepared, as_cleanup_client(platform), sleep=sleeps)

        assert sum(sleeps.slept) <= TEARDOWN_WAIT
        (row,) = receipt_rows(receipt)
        assert (row["status"], row["code"]) == ("blocked", "not_answered")
        assert "still running" in row["reason"]
        assert _exit(receipt, "delete") == 1
        assert load_state(prepared).halted is None
        assert (prepared / "workloads" / "w1" / "dataset.jsonl").exists()

    def test_a_cancel_waits_the_same_way(
        self, prepared: Path, platform: FakeCleanup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pending: list[Exception] = [_busy("1")]
        real = platform.cancel_audit

        def cancel(audit_id: str) -> JsonObject:
            if pending:
                raise pending.pop()
            return real(audit_id)

        monkeypatch.setattr(platform, "cancel_audit", cancel)
        platform.server_receipt = r.designed(
            r.STOPPED_ROW, status="halted", schema=r.SCHEMA_CANCELLED
        )
        sleeps = _Sleeps()

        receipt = cancel_audit(prepared, as_cleanup_client(platform), sleep=sleeps)

        assert sleeps.slept == [1.0]
        assert _exit(receipt, "cancel") == 0


class TestWhatTheClaimRefused:
    """The platform never owned these: this directory made them, so it removes them itself."""

    def _refused(self, prepared: Path, ids: list[str]) -> None:
        state = load_state(prepared)
        state.unclaimed_ids = ids
        save_state(prepared, state)

    def test_once_the_audit_is_deleted_the_refused_ids_are_deleted_directly_and_join_the_receipt(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        self._refused(prepared, ["ds-1", "job-1", "mv-1", "proj-1"])
        platform.server_receipt = r.designed(
            r.row("dataset", "ds-1", "kept", "not_created_here"),
            r.row("training_job", "job-1", "kept", "not_created_here"),
            r.row("dataset", "ds-2", "deleted"),
        )

        receipt = delete_audit(prepared, as_cleanup_client(platform))

        assert platform.present["dataset"] == set()  # ds-2 by the platform, ds-1 by this client
        assert platform.present["job"] == {"job-2"}
        assert platform.present["model"] == {"mv-2"}
        assert platform.present["deployment"] == {"dep-1", "dep-2"}  # never refused: not touched
        assert ("delete_project", "proj-1") not in platform.call_log
        assert "ds-1" not in load_state(prepared).kept_ids
        assert _exit(receipt, "delete") == 0
        assert load_state(prepared).halted == {"reason": "deleted"}

    def test_what_the_platform_kept_for_any_other_reason_is_never_deleted_from_here(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        """The owner moved it: it is theirs whatever an earlier claim said."""
        self._refused(prepared, ["ds-1", "dep-1"])
        platform.server_receipt = r.designed(
            r.row("dataset", "ds-1", "kept", "not_in_project"),
            r.row("deployment", "dep-1", "kept", "not_created_here"),
        )

        delete_audit(prepared, as_cleanup_client(platform))

        assert "ds-1" in platform.present["dataset"]
        assert "dep-1" not in platform.present["deployment"]
        assert "ds-1" in load_state(prepared).kept_ids

    def test_nothing_refused_is_deleted_while_the_platform_has_not_deleted_the_audit(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        self._refused(prepared, ["ds-1"])
        platform.server_receipt = r.designed(r.REFUSED_ROW, status="halted")

        delete_audit(prepared, as_cleanup_client(platform))

        assert platform.present["dataset"] == {"ds-1", "ds-2"}

    def test_a_claim_never_answered_leaves_everything_recorded_in_the_same_position(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        state = load_state(prepared)
        state.claim_pending = True
        save_state(prepared, state)
        platform.server_receipt = r.designed(r.DELETED_ROW)

        delete_audit(prepared, as_cleanup_client(platform))

        assert platform.present["dataset"] == set()
        assert platform.present["project"] == {"proj-1"}  # a project is never deleted from here

    def test_a_cancel_stops_what_was_refused_and_nothing_else(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        state = load_state(prepared)
        state.unclaimed_ids = ["job-1", "dep-1"]
        state.workloads["w1"][HEAD].run_status = "running"
        state.workloads["w1"][HEAD].deploy_status = "running"
        save_state(prepared, state)
        platform.server_receipt = r.designed(
            r.ALREADY_STOPPED_ROW, status="halted", schema=r.SCHEMA_CANCELLED
        )

        receipt = cancel_audit(prepared, as_cleanup_client(platform))

        assert ("cancel_training_job", "job-1") in platform.call_log
        assert ("pause_deployment", "dep-1") in platform.call_log
        assert ("pause_deployment", "dep-2") not in platform.call_log
        assert {row["id"] for row in receipt_rows(receipt)} >= {"job-1", "dep-1"}


class TestWhatThePlatformSays:
    def test_an_audit_status_nobody_knows_removes_nothing_and_exits_one(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        for odd in ("deleting", "Deleted", True):
            platform.server_receipt = {**r.designed(r.DELETED_ROW), "audit_status": odd}

            result = delete_audit(prepared, as_cleanup_client(platform))

            assert _exit(result, "delete") == 1
            assert load_state(prepared).halted is None
            assert (prepared / "workloads").exists()

    def test_a_cancel_calls_halted_normal_and_anything_else_unfinished(self) -> None:
        assert exit_status([r.STOPPED_ROW], "halted", verb="cancel") == 0
        assert exit_status([r.STOPPED_ROW], "deleted", verb="cancel") == 0
        assert exit_status([r.STOPPED_ROW], "cancelling", verb="cancel") == 1

    def test_a_deleted_audit_beside_a_deployment_that_is_still_there_keeps_its_key(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        platform.server_receipt = r.designed(
            r.row("deployment", "dep-1", "blocked", "refused", "still deploying"),
            status="deleted",
        )

        receipt = delete_audit(prepared, as_cleanup_client(platform))

        assert _exit(receipt, "delete") == 1
        assert SecretStore(prepared).load("w1/head_tune") == "dk-secret"
        assert (prepared / "workloads").exists()
        assert load_state(prepared).halted is None

    def test_rows_that_are_not_objects_are_shown_and_fail_the_command(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        platform.server_receipt = {"schema": "x", "entries": ["a", 3, None]}

        receipt = delete_audit(prepared, as_cleanup_client(platform))

        assert receipt_rows(receipt) == [{}, {}, {}]
        assert _exit(receipt, "delete") == 1
        assert load_state(prepared).halted is None


class TestAnUnpublishedAuditTheKeyCannotSee:
    def test_every_id_answering_not_found_proves_nothing_and_changes_nothing_local(
        self, audit_dir: Path
    ) -> None:
        save_state(audit_dir, recorded_state())
        SecretStore(audit_dir).store("w1/head_tune", "dk-secret")
        nothing_visible = FakeCleanup()  # another account's key: every read is a 404

        receipt = delete_unpublished(
            audit_dir, as_cleanup_client(nothing_visible), load_state(audit_dir)
        )

        (row,) = receipt_rows(receipt)
        assert (row["status"], row["code"]) == ("blocked", "not_answered")
        assert "none of the 9 ids" in row["reason"]
        assert load_state(audit_dir).halted is None
        assert SecretStore(audit_dir).load("w1/head_tune") == "dk-secret"

    def test_delete_audit_passes_the_callers_word_to_an_unpublished_walk(
        self, audit_dir: Path
    ) -> None:
        save_state(audit_dir, recorded_state())
        client = as_cleanup_client(FakeCleanup())

        assert _exit(delete_audit(audit_dir, client), "delete") == 1
        assert _exit(delete_audit(audit_dir, client, assume_gone=True), "delete") == 0
        assert load_state(audit_dir).halted == {"reason": "deleted"}

    def test_the_callers_word_that_it_is_deleted_clears_the_directory(
        self, audit_dir: Path
    ) -> None:
        save_state(audit_dir, recorded_state())

        receipt = delete_unpublished(
            audit_dir, as_cleanup_client(FakeCleanup()), load_state(audit_dir), assume_gone=True
        )

        assert _exit(receipt, "delete") == 0
        assert load_state(audit_dir).halted == {"reason": "deleted"}

    def test_a_cancel_of_ids_nobody_can_see_marks_nothing(self, audit_dir: Path) -> None:
        state = recorded_state()
        state.workloads["w1"][HEAD].run_status = "running"
        save_state(audit_dir, state)

        receipt = cancel_audit(audit_dir, as_cleanup_client(FakeCleanup()))

        assert _exit(receipt, "cancel") == 1
        assert load_state(audit_dir).workloads["w1"][HEAD].run_status == "running"
        assert load_state(audit_dir).halted is None
