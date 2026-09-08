"""delete_audit: every recorded id deleted and confirmed gone, with a receipt; idempotent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.audit._platform import FakeCleanup, as_cleanup_client

from dagnam._core.exceptions import APIError
from dagnam.audit.candidates import CandidateKind
from dagnam.audit.cleanup import DELETED_FILE, delete_audit, recorded_ids
from dagnam.audit.secrets import SecretStore
from dagnam.audit.state import AuditState, StepState, save_state

HEAD, SFT, HOSTED = CandidateKind.HEAD_TUNE, CandidateKind.SFT_SMALL, CandidateKind.HOSTED_FLOOR


def _state() -> AuditState:
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


@pytest.fixture
def platform() -> FakeCleanup:
    fake = FakeCleanup(
        deployment=["dep-1", "dep-2"],
        model=["entry-1", "entry-2"],
        job=["job-1", "job-2"],
        dataset=["ds-1", "ds-2"],
        project=["proj-1"],
    )
    fake.entry_of = {"mv-1": "entry-1", "mv-2": "entry-2"}
    return fake


@pytest.fixture
def prepared(audit_dir: Path) -> Path:
    save_state(audit_dir, _state())
    SecretStore(audit_dir).store("w1/head_tune", "dk-secret")
    return audit_dir


def test_recorded_ids_in_deletion_order_without_duplicates() -> None:
    assert recorded_ids(_state()) == {
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
    assert receipt["items"] == [
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
    # Each id: delete, then a read that must answer not-found; a version goes through its entry.
    assert platform.call_log[:2] == [("delete_deployment", "dep-1"), ("get_deployment", "dep-1")]
    assert platform.call_log[4:7] == [
        ("get_model_version", "mv-1"),
        ("delete_model_entry", "entry-1"),
        ("get_model_version", "mv-1"),
    ]
    assert platform.call_log[-2:] == [("delete_project", "proj-1"), ("get_project", "proj-1")]
    assert all(not ids for ids in platform.present.values())
    # Local rows and the secret go last, after the platform confirmed.
    assert not (prepared / "workloads").exists()
    assert SecretStore(prepared).load("w1/head_tune") is None
    assert (prepared / "state.json").exists()


def test_delete_is_idempotent_and_records_already_absent(
    prepared: Path, platform: FakeCleanup
) -> None:
    platform.present["deployment"].discard("dep-2")
    first = delete_audit(prepared, as_cleanup_client(platform))
    assert [i["status"] for i in first["items"] if i["id"] == "dep-2"] == ["already_absent"]
    assert [i["status"] for i in first["items"] if i["id"] != "dep-2"] == ["deleted"] * 8

    second = delete_audit(prepared, as_cleanup_client(platform))
    assert {i["status"] for i in second["items"]} == {"already_absent"}
    assert [i["id"] for i in second["items"]] == [i["id"] for i in first["items"]]


def test_an_id_that_survives_its_delete_is_an_error_and_no_receipt(
    prepared: Path, platform: FakeCleanup
) -> None:
    platform.sticky.add("ds-2")
    with pytest.raises(RuntimeError, match="dataset ds-2 still exists after delete"):
        delete_audit(prepared, as_cleanup_client(platform))
    assert not (prepared / DELETED_FILE).exists()
    assert (prepared / "workloads").is_dir()
    assert SecretStore(prepared).load("w1/head_tune") == "dk-secret"


def test_the_training_run_holding_a_dataset_is_deleted_before_it(
    prepared: Path, platform: FakeCleanup
) -> None:
    """Defect 24: the platform refuses a dataset while a run of it exists, so the job goes first."""
    platform.held_by_job = {"ds-1": "job-1", "ds-2": "job-2"}

    receipt = delete_audit(prepared, as_cleanup_client(platform))

    assert [i["status"] for i in receipt["items"]] == ["deleted"] * 9
    order = [call for call in platform.call_log if call[1] in {"job-1", "ds-1"}]
    assert order == [
        ("bulk_delete_training_jobs", "job-1"),
        ("get_training_job", "job-1"),
        ("delete_dataset", "ds-1"),
        ("get_dataset_meta", "ds-1"),
    ]
    assert not (prepared / "workloads").exists()


def test_a_job_the_platform_will_not_delete_blocks_it_and_its_dataset(
    prepared: Path, platform: FakeCleanup
) -> None:
    """A refusal is a receipt row, not an abort; the local rows stay for a second attempt."""
    platform.held_by_job = {"ds-1": "job-1"}
    platform.running = {"job-1"}  # the platform deletes terminal jobs only

    receipt = delete_audit(prepared, as_cleanup_client(platform))

    blocked = {(i["kind"], i["id"]): i["reason"] for i in receipt["items"] if "reason" in i}
    assert set(blocked) == {("training_job", "job-1"), ("dataset", "ds-1")}
    assert "Cannot delete job with status running" in blocked[("training_job", "job-1")]
    assert "referenced by a training run" in blocked[("dataset", "ds-1")]
    assert {(i["kind"], i["id"]) for i in receipt["items"] if i["status"] == "deleted"} == {
        ("deployment", "dep-1"),
        ("deployment", "dep-2"),
        ("model_version", "mv-1"),
        ("model_version", "mv-2"),
        ("training_job", "job-2"),
        ("dataset", "ds-2"),
        ("project", "proj-1"),
    }
    assert json.loads((prepared / DELETED_FILE).read_text(encoding="utf-8")) == receipt
    assert (prepared / "workloads").is_dir()
    assert SecretStore(prepared).load("w1/head_tune") == "dk-secret"


def test_a_server_failure_is_not_a_refusal_and_still_raises(
    prepared: Path, platform: FakeCleanup
) -> None:
    platform.dataset_error = APIError(500, "boom")
    with pytest.raises(APIError, match="API error 500"):
        delete_audit(prepared, as_cleanup_client(platform))
    assert not (prepared / DELETED_FILE).exists()


def test_fresh_audit_dir_deletes_nothing(audit_dir: Path, platform: FakeCleanup) -> None:
    receipt = delete_audit(audit_dir, as_cleanup_client(platform))
    assert receipt["items"] == []
    assert platform.call_log == []
    assert not (audit_dir / "workloads").exists()


def test_a_receipt_with_no_rows_reads_as_empty() -> None:
    """A receipt from a server that lists its rows under neither key is not a crash."""
    from dagnam.audit.cleanup import receipt_rows

    assert receipt_rows({"deleted_at": "2026-09-07T10:00:00+00:00"}) == []
    assert receipt_rows({"items": [{"kind": "project", "id": "p1"}, "junk"]}) == [
        {"kind": "project", "id": "p1"}
    ]
    assert receipt_rows({"entries": [{"kind": "project", "id": "p1"}]}) == [
        {"kind": "project", "id": "p1"}
    ]
