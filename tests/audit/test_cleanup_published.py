"""``cancel_audit`` and ``delete_audit`` for a published audit: one call, the receipt recorded, nothing else.

The platform's cancel and delete are the only paths that touch a published audit's resources, so
every test here asserts the fake saw exactly that one audit call and no destructive call of the
client's own -- whatever the platform answered, including nothing at all. Every receipt is built from
rows in the shape the platform writes (``tests/audit/_receipts.py``).
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import Any

import pytest
from tests.audit import _receipts as r
from tests.audit._cleanup import FakeCleanup, as_cleanup_client
from tests.audit._recorded import HEAD, recorded_state

from dagnam._core.exceptions import APIError
from dagnam._types import JsonObject
from dagnam.audit.cleanup import (
    CANCELLED_FILE,
    DELETED_FILE,
    AuditDeletedError,
    cancel_audit,
    delete_audit,
    receipt_rows,
    recorded_ids,
)
from dagnam.audit.receipt_rows import exit_status
from dagnam.audit.secrets import SecretStore
from dagnam.audit.state import load_state, save_state
from dagnam.audit.workspace import write_workload


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


def _deleted(*extra: JsonObject, status: str = "deleted") -> JsonObject:
    """The platform's delete receipt: every recorded id gone, plus ``extra`` rows."""
    return r.designed(
        r.row("deployment", "dep-1", "deleted"),
        r.row("training_job", "job-1", "deleted"),
        *extra,
        status=status,
    )


def _only_the_audit_call(platform: FakeCleanup, name: str) -> None:
    assert platform.call_log == [(name, "audit-1")]


def _exit(receipt: Mapping[str, Any], verb: str) -> int:
    return exit_status(receipt_rows(receipt), receipt.get("audit_status"), verb=verb)


class TestDelete:
    def test_a_deleted_audit_is_written_as_sent_and_this_machines_files_go(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        platform.server_receipt = _deleted(r.PROJECT_SHARED_ROW)

        receipt = delete_audit(prepared, as_cleanup_client(platform))

        _only_the_audit_call(platform, "delete_audit")
        assert receipt == platform.server_receipt
        assert json.loads((prepared / DELETED_FILE).read_text(encoding="utf-8")) == receipt
        assert not (prepared / "workloads").exists()
        assert SecretStore(prepared).load("w1/head_tune") is None
        assert load_state(prepared).halted == {"reason": "deleted"}
        assert _exit(receipt, "delete") == 0

    def test_what_the_platform_kept_is_recorded_so_no_listing_offers_it_again(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        platform.server_receipt = _deleted(
            r.row("dataset", "ds-2", "kept", "not_in_project", r.NOT_IN_PROJECT),
            r.PROJECT_NOT_OURS_ROW,
        )

        delete_audit(prepared, as_cleanup_client(platform))

        state = load_state(prepared)
        assert state.kept_ids == ["ds-2", "proj-1"]
        assert recorded_ids(state)["dataset"] == ["ds-1"]
        assert recorded_ids(state)["project"] == []
        _only_the_audit_call(platform, "delete_audit")

    def test_a_halted_audit_removes_nothing_local_and_exits_one(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        platform.server_receipt = _deleted(r.REFUSED_ROW, status="halted")

        receipt = delete_audit(prepared, as_cleanup_client(platform))

        _only_the_audit_call(platform, "delete_audit")
        assert (prepared / "workloads" / "w1" / "dataset.jsonl").exists()
        assert SecretStore(prepared).load("w1/head_tune") == "dk-secret"
        assert load_state(prepared).halted is None
        assert json.loads((prepared / DELETED_FILE).read_text(encoding="utf-8")) == receipt
        assert _exit(receipt, "delete") == 1

    def test_a_halted_audit_is_exit_one_even_with_no_blocked_row_at_all(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        platform.server_receipt = _deleted(status="halted")

        receipt = delete_audit(prepared, as_cleanup_client(platform))

        assert _exit(receipt, "delete") == 1
        assert load_state(prepared).halted is None

    def test_a_refusal_is_never_retried_from_here_and_the_next_delete_asks_the_platform_again(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        platform.server_receipt = _deleted(r.REFUSED_ROW, status="halted")
        delete_audit(prepared, as_cleanup_client(platform))
        platform.server_receipt = _deleted(r.row("dataset", "ds-1", "deleted"))

        receipt = delete_audit(prepared, as_cleanup_client(platform))

        assert platform.call_log == [("delete_audit", "audit-1")] * 2
        assert _exit(receipt, "delete") == 0
        assert load_state(prepared).halted == {"reason": "deleted"}

    @pytest.mark.parametrize(
        "failure", [APIError(500, "boom"), APIError(0, "Request failed: timed out")]
    )
    def test_a_platform_that_does_not_answer_is_one_blocked_row_and_nothing_is_touched(
        self, prepared: Path, platform: FakeCleanup, failure: APIError
    ) -> None:
        platform.account_error = failure

        receipt = delete_audit(prepared, as_cleanup_client(platform))

        _only_the_audit_call(platform, "delete_audit")
        (row,) = receipt_rows(receipt)
        assert (row["kind"], row["id"], row["status"], row["code"]) == (
            "audit",
            "audit-1",
            "blocked",
            "not_answered",
        )
        assert "run it again" in row["reason"]
        assert _exit(receipt, "delete") == 1
        assert (prepared / "workloads" / "w1" / "dataset.jsonl").exists()
        assert SecretStore(prepared).load("w1/head_tune") == "dk-secret"
        state = load_state(prepared)
        assert state.halted is None
        assert recorded_ids(state)["deployment"] == ["dep-1", "dep-2"]
        assert json.loads((prepared / DELETED_FILE).read_text(encoding="utf-8")) == receipt

    def test_a_404_is_not_a_decision_nothing_is_touched_and_the_account_is_named(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        """Another account's key, or another host, is told the same: it can only mean 'not visible'."""
        platform.account_error = APIError(404, "not found")
        said: list[str] = []

        receipt = delete_audit(
            prepared, as_cleanup_client(platform), on_missing=lambda: said.append("404")
        )

        _only_the_audit_call(platform, "delete_audit")
        assert said == ["404"]
        (row,) = receipt_rows(receipt)
        assert (row["status"], row["code"]) == ("blocked", "not_answered")
        assert "no audit with this id for this key" in row["reason"]
        assert (prepared / "workloads" / "w1" / "dataset.jsonl").exists()
        assert SecretStore(prepared).load("w1/head_tune") == "dk-secret"
        assert load_state(prepared).halted is None
        assert _exit(receipt, "delete") == 1

    def test_the_right_key_after_a_404_still_deletes_everything(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        platform.account_error = APIError(404, "not found")
        delete_audit(prepared, as_cleanup_client(platform))
        platform.account_error = None
        platform.server_receipt = _deleted()

        receipt = delete_audit(prepared, as_cleanup_client(platform))

        assert _exit(receipt, "delete") == 0
        assert load_state(prepared).halted == {"reason": "deleted"}
        assert not (prepared / "workloads").exists()

    def test_a_404_with_the_callers_word_that_it_is_deleted_removes_the_local_files(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        platform.account_error = APIError(404, "not found")

        receipt = delete_audit(prepared, as_cleanup_client(platform), assume_gone=True)

        _only_the_audit_call(platform, "delete_audit")
        assert receipt_rows(receipt) == []
        assert receipt["audit_status"] == "deleted"
        assert not (prepared / "workloads").exists()
        assert load_state(prepared).halted == {"reason": "deleted"}
        assert _exit(receipt, "delete") == 0

    def test_the_callers_word_does_not_excuse_a_failure_that_is_not_a_404(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        platform.account_error = APIError(500, "boom")
        receipt = delete_audit(prepared, as_cleanup_client(platform), assume_gone=True)
        assert _exit(receipt, "delete") == 1
        assert load_state(prepared).halted is None

    def test_a_deleted_audit_asks_the_platform_nothing_and_finishes_local_leftovers(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        platform.server_receipt = _deleted()
        delete_audit(prepared, as_cleanup_client(platform))
        write_workload(
            prepared, "w9", [{"input": "a", "label": "x"}], {"train": [0]}, {"format_key": "x"}
        )

        again = delete_audit(prepared, as_cleanup_client(platform))

        _only_the_audit_call(platform, "delete_audit")
        assert receipt_rows(again) == []
        assert not (prepared / "workloads" / "w9").exists()

    @pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs a privilege on Windows")
    def test_a_local_file_behind_a_link_is_a_row_of_its_own_and_the_state_is_not_deleted(
        self, prepared: Path, platform: FakeCleanup, tmp_path: Path
    ) -> None:
        platform.server_receipt = _deleted()
        moved = tmp_path / "moved"
        (prepared / "workloads").rename(moved)
        (prepared / "workloads").symlink_to(moved, target_is_directory=True)

        receipt = delete_audit(prepared, as_cleanup_client(platform))

        local = [row for row in receipt_rows(receipt) if row["kind"] == "local_workload"]
        assert [(row["id"], row["status"], row["code"]) for row in local] == [
            ("workloads", "blocked", "not_removed")
        ]
        assert (moved / "w1" / "dataset.jsonl").exists()
        assert load_state(prepared).halted is None
        assert _exit(receipt, "delete") == 1
        assert json.loads((prepared / DELETED_FILE).read_text(encoding="utf-8")) == receipt


@pytest.mark.parametrize("answer", [{}, {"entries": None}, {"entries": "none"}, {"schema": "x"}])
def test_an_answer_that_is_not_a_receipt_is_no_answer_and_nothing_is_touched(
    prepared: Path, platform: FakeCleanup, answer: JsonObject
) -> None:
    """A proxy's 200 or another host's JSON says nothing about the audit: it is not "all gone"."""
    platform.server_receipt = answer

    for command in (delete_audit, cancel_audit):
        receipt = command(prepared, as_cleanup_client(platform))
        (row,) = receipt_rows(receipt)
        assert (row["status"], row["code"]) == ("blocked", "not_answered")
        assert "was not a receipt" in row["reason"]

    assert load_state(prepared).halted is None
    assert (prepared / "workloads" / "w1" / "dataset.jsonl").exists()


def test_an_empty_list_of_rows_is_a_receipt(prepared: Path, platform: FakeCleanup) -> None:
    platform.server_receipt = r.designed(status="deleted")
    assert receipt_rows(delete_audit(prepared, as_cleanup_client(platform))) == []
    assert load_state(prepared).halted == {"reason": "deleted"}


class TestReceiptsFromOtherPlatforms:
    """A platform that predates ``audit_status`` and ``code``, and one that is newer than this client."""

    def test_an_older_platform_that_deleted_everything_is_read_by_its_rows(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        platform.server_receipt = r.receipt(
            r.legacy(r.DELETED_ROW),
            r.legacy(r.row("project", "proj-1", "blocked", None, r.NOT_OURS)),
        )

        receipt = delete_audit(prepared, as_cleanup_client(platform))

        _only_the_audit_call(platform, "delete_audit")
        assert load_state(prepared).halted == {"reason": "deleted"}
        assert _exit(receipt, "delete") == 0

    def test_an_older_platforms_leftover_is_exit_one_nothing_local_goes_and_nothing_is_retried(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        platform.server_receipt = r.receipt(
            r.legacy(r.row("dataset", "ds-1", "blocked", None, "Cannot delete: busy"))
        )

        receipt = delete_audit(prepared, as_cleanup_client(platform))

        _only_the_audit_call(platform, "delete_audit")
        assert load_state(prepared).halted is None
        assert (prepared / "workloads").exists()
        assert _exit(receipt, "delete") == 1

    def test_an_older_platforms_second_delete_is_its_404_and_is_not_trusted_either(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        platform.account_error = APIError(404, "not found")

        receipt = delete_audit(prepared, as_cleanup_client(platform))

        assert _exit(receipt, "delete") == 1  # `--already-deleted` is how a person says it is gone
        assert load_state(prepared).halted is None

    def test_a_row_with_a_status_from_the_future_is_exit_one_and_untouched(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        platform.server_receipt = _deleted(r.row("dataset", "ds-1", "quarantined", "x"))

        receipt = delete_audit(prepared, as_cleanup_client(platform))

        _only_the_audit_call(platform, "delete_audit")
        assert _exit(receipt, "delete") == 1
        assert [row["status"] for row in receipt_rows(receipt)][-1] == "quarantined"

    def test_a_code_from_the_future_on_a_known_status_is_read_by_the_status(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        platform.server_receipt = _deleted(
            r.row("dataset", "ds-1", "kept", "a_code_from_the_future")
        )

        receipt = delete_audit(prepared, as_cleanup_client(platform))

        assert _exit(receipt, "delete") == 0
        assert load_state(prepared).kept_ids == ["ds-1"]


class TestCancel:
    def test_the_platforms_receipt_is_recorded_and_written_as_sent(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        platform.server_receipt = r.designed(
            r.row("training_job", "job-1", "stopped"),
            r.row("deployment", "dep-1", "stopped", "already_stopped"),
            status="halted",
            schema=r.SCHEMA_CANCELLED,
        )

        receipt = cancel_audit(prepared, as_cleanup_client(platform))

        _only_the_audit_call(platform, "cancel_audit")
        assert json.loads((prepared / CANCELLED_FILE).read_text(encoding="utf-8")) == receipt
        head = load_state(prepared).workloads["w1"][HEAD]
        assert head.run_status == "cancelled"  # `stopped` marks the run ...
        assert head.deploy_status is None  # ... `already_stopped` marks nothing
        assert load_state(prepared).halted == {"reason": "cancelled"}
        assert _exit(receipt, "cancel") == 0

    def test_a_blocked_row_is_exit_one_and_nothing_is_stopped_from_here(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        platform.server_receipt = r.designed(
            r.HAS_SERVED_ROW, status="halted", schema=r.SCHEMA_CANCELLED
        )

        receipt = cancel_audit(prepared, as_cleanup_client(platform))

        _only_the_audit_call(platform, "cancel_audit")
        assert _exit(receipt, "cancel") == 1

    def test_a_platform_that_does_not_answer_is_one_blocked_row_and_the_state_is_unchanged(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        platform.account_error = APIError(500, "boom")

        receipt = cancel_audit(prepared, as_cleanup_client(platform))

        _only_the_audit_call(platform, "cancel_audit")
        (row,) = receipt_rows(receipt)
        assert (row["status"], row["code"]) == ("blocked", "not_answered")
        assert receipt["schema"] == "dagnam.audit.cancelled/1"
        assert load_state(prepared).halted is None
        assert _exit(receipt, "cancel") == 1

    def test_a_404_is_not_a_decision_and_leaves_the_state_alone(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        platform.account_error = APIError(404, "not found")
        said: list[str] = []

        receipt = cancel_audit(
            prepared, as_cleanup_client(platform), on_missing=lambda: said.append("404")
        )

        _only_the_audit_call(platform, "cancel_audit")
        assert said == ["404"]
        assert load_state(prepared).halted is None
        assert _exit(receipt, "cancel") == 1

    def test_a_deleted_audit_refuses_a_cancel_without_asking_anyone(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        platform.server_receipt = _deleted()
        delete_audit(prepared, as_cleanup_client(platform))

        with pytest.raises(AuditDeletedError, match="deleted audit"):
            cancel_audit(prepared, as_cleanup_client(platform))

        _only_the_audit_call(platform, "delete_audit")

    def test_an_unpublished_audit_is_cancelled_from_here(
        self, audit_dir: Path, platform: FakeCleanup
    ) -> None:
        state = recorded_state()
        state.workloads["w1"][HEAD].run_status = "running"
        save_state(audit_dir, state)

        receipt = cancel_audit(audit_dir, as_cleanup_client(platform))

        assert ("cancel_audit", "audit-1") not in platform.call_log
        assert {row["status"] for row in receipt_rows(receipt)} >= {"stopped"}
        assert (audit_dir / CANCELLED_FILE).exists()
