"""delete_audit on this machine's own walk: every recorded id deleted and confirmed gone; idempotent."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import Any

import pytest
from tests.audit._cleanup import FakeCleanup, as_cleanup_client
from tests.audit._recorded import recorded_state

from dagnam._core.exceptions import APIError
from dagnam.audit.cleanup import DELETED_FILE, delete_audit, receipt_rows, recorded_ids
from dagnam.audit.cleanup_kinds import STILL_THERE
from dagnam.audit.secrets import SecretStore
from dagnam.audit.state import AuditState, load_state, save_state


@pytest.fixture
def platform() -> FakeCleanup:
    fake = FakeCleanup(
        deployment=["dep-1", "dep-2"],
        model=["mv-1", "mv-2"],
        job=["job-1", "job-2"],
        dataset=["ds-1", "ds-2"],
        project=["proj-1"],
    )
    return fake


@pytest.fixture
def prepared(audit_dir: Path) -> Path:
    save_state(audit_dir, recorded_state())
    SecretStore(audit_dir).store("w1/head_tune", "dk-secret")
    return audit_dir


def _blocked(receipt: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    return {
        (r["kind"], r["id"]): r["reason"]
        for r in receipt_rows(receipt)
        if r.get("status") == "blocked"
    }


def test_recorded_ids_in_deletion_order_without_duplicates() -> None:
    assert recorded_ids(recorded_state()) == {
        "deployment": ["dep-1", "dep-2"],
        "model_version": ["mv-1", "mv-2"],
        "training_job": ["job-1", "job-2"],
        "dataset": ["ds-1", "ds-2"],
        "project": ["proj-1"],
    }
    assert recorded_ids(AuditState()) == {
        "deployment": [],
        "model_version": [],
        "training_job": [],
        "dataset": [],
        "project": [],
    }


def test_delete_removes_every_recorded_id_and_writes_receipt(
    prepared: Path, platform: FakeCleanup
) -> None:
    receipt = delete_audit(prepared, as_cleanup_client(platform))

    assert receipt["schema"] == "dagnam.audit.deleted/1"
    assert receipt["deleted_at"].endswith("+00:00")
    assert receipt["entries"] == [
        {"kind": "deployment", "id": "dep-1", "status": "deleted"},
        {"kind": "deployment", "id": "dep-2", "status": "deleted"},
        {"kind": "model_version", "id": "mv-1", "status": "deleted"},
        {"kind": "model_version", "id": "mv-2", "status": "deleted"},
        {"kind": "training_job", "id": "job-1", "status": "deleted"},
        {"kind": "training_job", "id": "job-2", "status": "deleted"},
        {"kind": "dataset", "id": "ds-1", "status": "deleted"},
        {"kind": "dataset", "id": "ds-2", "status": "deleted"},
        {"kind": "project", "id": "proj-1", "status": "deleted"},
    ]
    assert json.loads((prepared / DELETED_FILE).read_text(encoding="utf-8")) == receipt
    # Each id: delete, then a read that must answer not-found; a version through its own purge route.
    assert platform.call_log[:2] == [("delete_deployment", "dep-1"), ("get_deployment", "dep-1")]
    # The purge answers its own receipt row, so it is trusted and the version is not re-read.
    assert platform.call_log[4:6] == [
        ("purge_model_version", "mv-1"),
        ("purge_model_version", "mv-2"),
    ]
    assert "delete_model_entry" not in {name for name, _ in platform.call_log}
    assert platform.call_log[-2:] == [("delete_project", "proj-1"), ("get_project", "proj-1")]
    assert all(not ids for ids in platform.present.values())
    # Local rows and the secret go last, after the platform confirmed.
    assert not (prepared / "workloads").exists()
    assert SecretStore(prepared).load("w1/head_tune") is None
    assert (prepared / "state.json").exists()


def test_delete_records_already_absent_and_a_second_delete_asks_nothing(
    prepared: Path, platform: FakeCleanup
) -> None:
    platform.present["deployment"].discard("dep-2")
    first = delete_audit(prepared, as_cleanup_client(platform))
    assert [i["status"] for i in first["entries"] if i["id"] == "dep-2"] == ["already_absent"]
    assert [i["status"] for i in first["entries"] if i["id"] != "dep-2"] == ["deleted"] * 8

    calls = len(platform.call_log)
    second = delete_audit(prepared, as_cleanup_client(platform))  # deleted: nothing is asked again
    assert second["entries"] == []
    assert len(platform.call_log) == calls


def test_an_id_that_survives_its_delete_is_blocked_and_the_rest_still_goes(
    prepared: Path, platform: FakeCleanup
) -> None:
    """It used to raise out of the walk: no receipt, and the project after it never reached."""
    platform.sticky.add("ds-2")

    receipt = delete_audit(prepared, as_cleanup_client(platform))

    assert _blocked(receipt) == {("dataset", "ds-2"): STILL_THERE}
    assert [i["status"] for i in receipt["entries"] if i["id"] != "ds-2"] == [
        *["deleted"] * 7,
        "kept",
    ]  # the project stays while anything in it does
    assert json.loads((prepared / DELETED_FILE).read_text(encoding="utf-8")) == receipt
    assert not (prepared / "workloads").exists()


def test_a_deployment_the_platform_will_not_delete_blocks_only_itself_and_keeps_its_key(
    prepared: Path, platform: FakeCleanup
) -> None:
    """A refused deployment delete is a typed state error, not an ``APIError``.

    It raised out of the delete before the training job -- first in line after
    the deployments and still billing -- was cancelled. And the endpoint is
    still up: the key that calls it, and the rows its replay reads, stay until
    a rerun has removed it.
    """
    platform.undeletable = {"dep-1"}
    platform.running = {"job-1", "job-2"}

    receipt = delete_audit(prepared, as_cleanup_client(platform))

    assert _blocked(receipt) == {
        ("deployment", "dep-1"): "Cannot delete a deployment that is still deploying"
    }
    assert ("cancel_training_job", "job-1") in platform.call_log
    assert ("cancel_training_job", "job-2") in platform.call_log
    assert [i["status"] for i in receipt["entries"] if i["id"] != "dep-1"] == [
        *["deleted"] * 7,
        "kept",
    ]  # the project stays while anything in it does
    assert json.loads((prepared / DELETED_FILE).read_text(encoding="utf-8")) == receipt
    assert (prepared / "workloads" / "w1" / "dataset.jsonl").exists()
    assert SecretStore(prepared).load("w1/head_tune") == "dk-secret"
    assert load_state(prepared).halted is None  # something is left: the next delete asks again
    assert recorded_ids(load_state(prepared))["deployment"] == ["dep-1", "dep-2"]  # for a rerun
    assert receipt["audit_status"] == "halted"

    platform.undeletable.clear()  # and the rerun finishes the job, then lets go of the key
    again = delete_audit(prepared, as_cleanup_client(platform))
    statuses = {(i["kind"], i["id"]): i["status"] for i in again["entries"]}
    assert statuses.pop(("deployment", "dep-1")) == "deleted"
    assert statuses.pop(("project", "proj-1")) == "deleted"  # nothing is left in it now
    assert set(statuses.values()) == {"already_absent"}
    assert not (prepared / "workloads").exists()
    assert SecretStore(prepared).load("w1/head_tune") is None
    assert again["audit_status"] == "deleted"
    assert load_state(prepared).halted == {"reason": "deleted"}


def test_the_training_run_holding_a_dataset_is_deleted_before_it(
    prepared: Path, platform: FakeCleanup
) -> None:
    """The platform refuses a dataset while a run of it exists, so the job goes first."""
    platform.held_by_job = {"ds-1": "job-1", "ds-2": "job-2"}

    receipt = delete_audit(prepared, as_cleanup_client(platform))

    assert [i["status"] for i in receipt["entries"]] == ["deleted"] * 9
    order = [call for call in platform.call_log if call[1] in {"job-1", "ds-1"}]
    assert order == [
        ("cancel_training_job", "job-1"),
        ("bulk_delete_training_jobs", "job-1"),
        ("get_training_job", "job-1"),
        ("delete_dataset", "ds-1"),
        ("get_dataset_meta", "ds-1"),
    ]
    assert not (prepared / "workloads").exists()


def test_a_run_still_going_is_cancelled_before_it_is_deleted(
    prepared: Path, platform: FakeCleanup
) -> None:
    """The platform deletes terminal jobs only, so a live run used to block -- and bill."""
    platform.held_by_job = {"ds-1": "job-1"}
    platform.running = {"job-1"}

    receipt = delete_audit(prepared, as_cleanup_client(platform))

    assert [i["status"] for i in receipt["entries"]] == ["deleted"] * 9
    assert platform.call_log.index(("cancel_training_job", "job-1")) < platform.call_log.index(
        ("bulk_delete_training_jobs", "job-1")
    )
    assert not (prepared / "workloads").exists()


def test_a_run_the_platform_will_neither_stop_nor_delete_blocks_it_and_its_dataset(
    prepared: Path, platform: FakeCleanup
) -> None:
    platform.held_by_job = {"ds-1": "job-1"}
    platform.running, platform.finished = {"job-1"}, {"job-1"}  # mid-transition either way

    receipt = delete_audit(prepared, as_cleanup_client(platform))

    blocked = _blocked(receipt)
    assert set(blocked) == {("training_job", "job-1"), ("dataset", "ds-1")}
    assert "Cannot delete job with status running" in blocked[("training_job", "job-1")]


def test_a_refusal_blocks_only_its_own_id_and_the_local_rows_still_go(
    prepared: Path, platform: FakeCleanup
) -> None:
    """A refusal is a receipt row, not an abort.

    The local rows and keys go regardless of what is left of a dataset, a run
    or the project -- a second delete needs only the ids `state.json` keeps, and
    a platform that refuses for good would otherwise hold the redacted rows on
    this machine forever. (An endpoint still up is the one exception.)
    """
    platform.present["job"].add("job-9")  # a run the audit never recorded still holds ds-1
    platform.held_by_job = {"ds-1": "job-9"}

    receipt = delete_audit(prepared, as_cleanup_client(platform))

    blocked = _blocked(receipt)
    assert set(blocked) == {("dataset", "ds-1")}
    assert "referenced by a training run" in blocked[("dataset", "ds-1")]
    assert [i["status"] for i in receipt["entries"] if i["id"] != "ds-1"] == [
        *["deleted"] * 7,
        "kept",
    ]  # the project stays while anything in it does
    assert json.loads((prepared / DELETED_FILE).read_text(encoding="utf-8")) == receipt
    assert not (prepared / "workloads").exists()
    assert SecretStore(prepared).load("w1/head_tune") is None
    assert not (prepared / "secrets.json").exists()
    assert load_state(prepared).halted is None  # a dataset is left: the next delete retries it
    assert recorded_ids(load_state(prepared))["dataset"] == ["ds-1", "ds-2"]  # for a rerun


def test_a_server_failure_on_one_id_is_its_row_and_the_delete_still_finishes(
    prepared: Path, platform: FakeCleanup
) -> None:
    """A fault is not a refusal, and the receipt says which it was -- but neither ends the walk."""
    platform.dataset_error = APIError(500, "boom")

    receipt = delete_audit(prepared, as_cleanup_client(platform))

    assert _blocked(receipt) == {
        ("dataset", "ds-1"): "API error 500: boom",
        ("dataset", "ds-2"): "API error 500: boom",
    }
    assert {(i["kind"], i["status"]) for i in receipt["entries"] if i["kind"] == "project"} == {
        ("project", "kept")
    }
    assert json.loads((prepared / DELETED_FILE).read_text(encoding="utf-8")) == receipt
    assert load_state(prepared).halted is None


def test_fresh_audit_dir_deletes_nothing(audit_dir: Path, platform: FakeCleanup) -> None:
    receipt = delete_audit(audit_dir, as_cleanup_client(platform))
    assert receipt["entries"] == []
    assert platform.call_log == []
    assert not (audit_dir / "workloads").exists()


def test_a_receipt_with_no_rows_reads_as_empty() -> None:
    """A receipt from a server that lists its rows under neither key is not a crash."""
    assert receipt_rows({"deleted_at": "2026-09-07T10:00:00+00:00"}) == []
    assert receipt_rows({"items": [{"kind": "project", "id": "p1"}, "junk"]}) == [
        {"kind": "project", "id": "p1"},
        {},  # a row that is not an object is shown as ``?`` and read as one nobody understands
    ]
    assert receipt_rows({"entries": [{"kind": "project", "id": "p1"}]}) == [
        {"kind": "project", "id": "p1"}
    ]


class TestLocalLeftovers:
    """Rows on this machine go through the safe remover, and a link is a leftover, never followed."""

    @pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs a privilege on Windows")
    def test_a_workload_folder_that_is_a_link_is_left_and_named_in_the_receipt(
        self, prepared: Path, platform: FakeCleanup, tmp_path: Path
    ) -> None:
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        (outside / "keep.txt").write_text("not ours", encoding="utf-8")
        (prepared / "workloads" / "w2").rename(tmp_path / "w2-moved")
        (prepared / "workloads" / "w2").symlink_to(outside, target_is_directory=True)

        receipt = delete_audit(prepared, as_cleanup_client(platform))

        local = [i for i in receipt["entries"] if i["kind"] == "local_workload"]
        assert [(i["id"], i["status"]) for i in local] == [("w2", "blocked")]
        assert "symbolic link" in local[0]["reason"]
        assert (outside / "keep.txt").read_text(encoding="utf-8") == "not ours"
        assert not (prepared / "workloads" / "w1").exists()  # the plain one still went
        assert json.loads((prepared / DELETED_FILE).read_text(encoding="utf-8")) == receipt

    @pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs a privilege on Windows")
    def test_a_workloads_folder_that_is_a_link_leaves_one_row_and_deletes_the_platform_side(
        self, prepared: Path, platform: FakeCleanup, tmp_path: Path
    ) -> None:
        moved = tmp_path / "moved"
        (prepared / "workloads").rename(moved)
        (prepared / "workloads").symlink_to(moved, target_is_directory=True)

        receipt = delete_audit(prepared, as_cleanup_client(platform))

        assert [
            (i["id"], i["status"]) for i in receipt["entries"] if i["kind"] == "local_workload"
        ] == [("workloads", "blocked")]
        assert (moved / "w1" / "dataset.jsonl").exists()
        assert all(not ids for ids in platform.present.values())

    @pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs a privilege on Windows")
    def test_a_key_file_that_is_a_link_is_left_and_named_in_the_receipt(
        self, prepared: Path, platform: FakeCleanup, tmp_path: Path
    ) -> None:
        target = tmp_path / "elsewhere.json"
        target.write_text(json.dumps({"w1/head_tune": "dk-secret"}), encoding="utf-8")
        (prepared / "secrets.json").unlink()
        (prepared / "secrets.json").symlink_to(target)

        receipt = delete_audit(prepared, as_cleanup_client(platform))

        local = [i for i in receipt["entries"] if i["kind"] == "local_keys"]
        assert [(i["id"], i["status"]) for i in local] == [("secrets.json", "blocked")]
        assert json.loads(target.read_text(encoding="utf-8")) == {"w1/head_tune": "dk-secret"}

    def test_a_folder_that_is_not_a_scans_and_names_that_are_not_ours_stay(
        self, prepared: Path, platform: FakeCleanup
    ) -> None:
        stray = prepared / "workloads" / "notes"
        stray.mkdir()
        (stray / "mine.txt").write_text("keep me", encoding="utf-8")
        (prepared / "workloads" / ".DS_Store").write_text("", encoding="utf-8")

        receipt = delete_audit(prepared, as_cleanup_client(platform))

        assert [i for i in receipt["entries"] if i["kind"].startswith("local")] == []
        assert (stray / "mine.txt").read_text(encoding="utf-8") == "keep me"
        assert (prepared / "workloads" / ".DS_Store").exists()
        assert sorted(p.name for p in (prepared / "workloads").iterdir()) == [".DS_Store", "notes"]
