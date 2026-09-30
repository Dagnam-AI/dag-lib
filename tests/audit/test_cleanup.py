"""delete_audit: every recorded id deleted and confirmed gone, with a receipt; idempotent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.audit._cleanup import FakeCleanup, as_cleanup_client

from dagnam._core.exceptions import APIError
from dagnam._types import JsonObject, JsonValue
from dagnam.audit.candidates import CandidateKind
from dagnam.audit.cleanup import (
    CANCELLED_ERROR,
    DELETED_FILE,
    cancel_recorded,
    delete_audit,
    recorded_ids,
)
from dagnam.audit.secrets import SecretStore
from dagnam.audit.state import AuditState, StepState, load_state, save_state

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
    """P7: the platform deletes terminal jobs only, so a live run used to block -- and bill."""
    platform.held_by_job = {"ds-1": "job-1"}
    platform.running = {"job-1"}

    receipt = delete_audit(prepared, as_cleanup_client(platform))

    assert [i["status"] for i in receipt["items"]] == ["deleted"] * 9
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

    blocked = {(i["kind"], i["id"]): i["reason"] for i in receipt["items"] if "reason" in i}
    assert set(blocked) == {("training_job", "job-1"), ("dataset", "ds-1")}
    assert "Cannot delete job with status running" in blocked[("training_job", "job-1")]


def test_a_refusal_blocks_only_its_own_id_and_the_local_rows_still_go(
    prepared: Path, platform: FakeCleanup
) -> None:
    """A refusal is a receipt row, not an abort.

    R1: the local rows and keys go regardless -- a second delete needs only the
    ids `state.json` keeps, and a platform that refuses for good (a project it
    keeps) would otherwise hold the redacted rows on this machine forever.
    """
    platform.present["job"].add("job-9")  # a run the audit never recorded still holds ds-1
    platform.held_by_job = {"ds-1": "job-9"}

    receipt = delete_audit(prepared, as_cleanup_client(platform))

    blocked = {(i["kind"], i["id"]): i["reason"] for i in receipt["items"] if "reason" in i}
    assert set(blocked) == {("dataset", "ds-1")}
    assert "referenced by a training run" in blocked[("dataset", "ds-1")]
    assert [i["status"] for i in receipt["items"] if i["id"] != "ds-1"] == ["deleted"] * 8
    assert json.loads((prepared / DELETED_FILE).read_text(encoding="utf-8")) == receipt
    assert not (prepared / "workloads").exists()
    assert SecretStore(prepared).load("w1/head_tune") is None
    assert not (prepared / "secrets.json").exists()
    assert load_state(prepared).halted == {"reason": "deleted"}
    assert recorded_ids(load_state(prepared))["dataset"] == ["ds-1", "ds-2"]  # for a rerun


def _server_deleted(*rows: tuple[str, str, str]) -> JsonObject:
    """A published delete's receipt: every recorded id except ds-1, plus the given rows."""
    settled = (
        ("deployment", "dep-1", "deleted"),
        ("deployment", "dep-2", "deleted"),
        ("training_job", "job-1", "deleted"),
        ("training_job", "job-2", "deleted"),
        ("dataset", "ds-2", "deleted"),
    )
    entries: list[JsonValue] = [
        {"kind": kind, "id": item_id, "status": status, "reason": None}
        for kind, item_id, status in (*settled, *rows)
    ]
    return {
        "schema": "dagnam.audit.deleted/1",
        "deleted_at": "2026-09-27T10:00:00Z",
        "entries": entries,
    }


def test_a_published_delete_also_removes_what_the_server_never_heard_of(
    prepared: Path, platform: FakeCleanup
) -> None:
    """B5: the server deletes datasets only through a published version id.

    A dataset whose upload failed never had one, so the server's receipt has no
    row for it -- and a delete that trusted that receipt dropped the local rows
    and the keys and left the customer's uploaded rows on the platform.
    """
    platform.server_receipt = _server_deleted(
        ("model_version", "mv-1", "deleted"),
        ("model_version", "mv-2", "deleted"),
        ("project", "proj-1", "deleted"),
    )
    server = platform.delete_audit("audit-1")

    receipt = delete_audit(prepared, as_cleanup_client(platform), server)

    rows = {(r["kind"], r["id"]): r["status"] for r in receipt["entries"]}
    assert rows[("dataset", "ds-1")] == "deleted"
    assert len(receipt["entries"]) == 9
    assert receipt["schema"] == "dagnam.audit.deleted/1"
    assert ("delete_deployment", "dep-1") not in platform.call_log  # the server settled it
    assert platform.present["dataset"] == set()
    assert not (prepared / "workloads").exists()


def test_a_project_the_server_kept_is_never_deleted_here(
    prepared: Path, platform: FakeCleanup
) -> None:
    """N2 (D-F14): the owner reused the audit's project, so the server kept it and its models.

    The CLI used to walk the `blocked` project itself -- an unconditional soft
    delete -- and delete every registry entry it had recorded, which is exactly
    what the server declined to do.
    """
    platform.server_receipt = _server_deleted(("project", "proj-1", "blocked"))
    server = platform.delete_audit("audit-1")

    receipt = delete_audit(prepared, as_cleanup_client(platform), server)

    rows = {(r["kind"], r["id"]): r for r in receipt["entries"]}
    assert rows[("dataset", "ds-1")]["status"] == "deleted"  # still ours to remove
    assert rows[("project", "proj-1")]["status"] == "blocked"
    for version in ("mv-1", "mv-2"):
        assert rows[("model_version", version)]["status"] == "blocked"
        assert rows[("model_version", version)]["reason"] == "kept with project proj-1"
    touched = {name for name, _ in platform.call_log}
    assert touched.isdisjoint({"delete_project", "delete_model_entry", "get_model_version"})
    assert platform.present["project"] == {"proj-1"}
    assert platform.present["model"] == {"entry-1", "entry-2"}
    # R1: the project stays on the platform for good, so nothing here waits on it.
    assert not (prepared / "workloads").exists()
    assert SecretStore(prepared).load("w1/head_tune") is None


def test_weights_the_server_kept_stay_blocked_and_the_rest_is_retried_here(
    prepared: Path, platform: FakeCleanup
) -> None:
    """The server soft-deletes the entry but keeps its weights while a deployment serves them.

    Its version row is ``blocked``; retried here, the soft delete read 404 and
    the receipt said ``already_absent`` over weights still in storage. A
    ``blocked`` deployment is still the client's to retry.
    """
    served = "a live deployment serves these weights; delete it, then run the delete again"
    platform.server_receipt = _server_deleted(
        ("model_version", "mv-1", "blocked"),
        ("model_version", "mv-2", "deleted"),
        ("project", "proj-1", "deleted"),
    )
    entries = platform.server_receipt["entries"]
    assert isinstance(entries, list)
    for row in entries:
        assert isinstance(row, dict)
        if row["id"] == "mv-1":
            row["reason"] = served
        if row["id"] == "dep-2":
            row.update(status="blocked", reason="not provisioned")
    server = platform.delete_audit("audit-1")
    platform.present["model"].discard("entry-1")  # soft-deleted by the server's walk

    receipt = delete_audit(prepared, as_cleanup_client(platform), server)

    rows = {(r["kind"], r["id"]): r for r in receipt["entries"]}
    assert rows[("model_version", "mv-1")] == {
        "kind": "model_version",
        "id": "mv-1",
        "status": "blocked",
        "reason": served,
    }
    assert ("get_model_version", "mv-1") not in platform.call_log
    assert rows[("deployment", "dep-2")]["status"] == "deleted"  # retried here


def test_a_second_delete_after_the_server_deleted_the_audit_finishes_here(
    prepared: Path, platform: FakeCleanup
) -> None:
    """R1: the account already deleted the audit, so its delete now answers 404.

    The rerun walks every recorded id -- finding what is gone `already_absent` --
    and only reads the project and the registry: the account decided about those.
    """
    platform.server_receipt = _server_deleted(("project", "proj-1", "blocked"))
    first = delete_audit(prepared, as_cleanup_client(platform), platform.delete_audit("audit-1"))
    platform.call_log.clear()

    second = delete_audit(prepared, as_cleanup_client(platform), account_deleted=True)

    rows = {(r["kind"], r["id"]): r for r in second["items"]}
    for kept in (("project", "proj-1"), ("model_version", "mv-1"), ("model_version", "mv-2")):
        assert (rows[kept]["status"], rows[kept]["reason"]) == ("blocked", "kept by the account")
        del rows[kept]
    assert {row["status"] for row in rows.values()} == {"already_absent"}
    assert len(second["items"]) == len(first["entries"])


def test_a_delete_after_the_website_deleted_the_audit_never_deletes_its_project(
    prepared: Path, platform: FakeCleanup
) -> None:
    """RR1: deleted on the website (no local receipt), the project kept for the owner's work.

    The CLI found no record of what the account kept and walked the project
    itself: `delete_project` soft-deleted the owner's Studio work with it.
    """
    platform.present["deployment"].clear()  # what the account's own delete removed
    platform.present["job"].clear()
    platform.present["dataset"].discard("ds-2")

    receipt = delete_audit(prepared, as_cleanup_client(platform), account_deleted=True)

    touched = {name for name, _ in platform.call_log}
    assert touched.isdisjoint({"delete_project", "delete_model_entry"})
    rows = {(r["kind"], r["id"]): r["status"] for r in receipt["items"]}
    assert rows[("project", "proj-1")] == "blocked"
    assert rows[("model_version", "mv-1")] == "blocked"
    assert rows[("dataset", "ds-1")] == "deleted"  # never published: still ours to remove
    assert platform.present["project"] == {"proj-1"}
    assert not (prepared / "workloads").exists()
    assert SecretStore(prepared).load("w1/head_tune") is None

    platform.present["project"].clear()  # and one the account did delete reads as gone
    platform.present["model"].clear()
    again = delete_audit(prepared, as_cleanup_client(platform), account_deleted=True)
    assert {r["status"] for r in again["items"]} == {"already_absent"}


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


# ------------------------------------------------------------------- cancel


def _live_state() -> AuditState:
    state = AuditState(project_id="proj-1")
    state.workloads["w1"] = {
        HEAD: StepState(dataset_id="ds-1", training_job_id="job-1", run_status="queued")
    }
    state.workloads["w2"] = {
        SFT: StepState(
            training_job_id="job-2",
            run_status="completed",
            deployment_id="dep-2",
            deploy_status="running",
            scored=True,
        )
    }
    return state


def test_a_run_that_ended_while_nobody_watched_does_not_stop_the_cancel(
    platform: FakeCleanup,
) -> None:
    """B3: `--no-wait` left the run `queued`; the platform answers its cancel with a 400.

    That used to raise out of the command: no receipt, no marks, and every later
    candidate's endpoint left running. Now it is a `blocked` row, and the run
    -- which really did finish -- is left resumable.
    """
    platform.finished = {"job-1"}
    state = _live_state()

    receipt = cancel_recorded(state, as_cleanup_client(platform))

    assert receipt["entries"] == [
        {
            "kind": "training_job",
            "id": "job-1",
            "status": "blocked",
            "reason": "Cannot cancel job with status completed",
        },
        {"kind": "deployment", "id": "dep-2", "status": "stopped"},
    ]
    assert receipt["schema"] == "dagnam.audit.cancelled/1"
    head, done = state.workloads["w1"][HEAD], state.workloads["w2"][SFT]
    assert (head.run_status, head.error) == ("queued", None)
    assert (done.deploy_status, done.error) == ("paused", None)
    assert state.halted == {"reason": "cancelled"}


def test_a_published_cancel_stops_what_the_server_never_heard_of(platform: FakeCleanup) -> None:
    """B4: the server stops only the ids the run published, so the rest is stopped here.

    Before, every non-terminal local run was marked cancelled whatever the
    server did: a job whose `submit` patch was lost kept training and billing
    while `audit status` said `cancelled`.
    """
    state = _live_state()
    server = {
        "schema": "dagnam.audit.cancelled/1",
        "deleted_at": "2026-09-27T10:00:00+00:00",
        "entries": [{"kind": "deployment", "id": "dep-2", "status": "stopped", "reason": None}],
    }

    receipt = cancel_recorded(state, as_cleanup_client(platform), server)

    assert platform.call_log == [("cancel_training_job", "job-1")]
    assert receipt["deleted_at"] == "2026-09-27T10:00:00+00:00"
    assert receipt["entries"][-1] == {"kind": "training_job", "id": "job-1", "status": "stopped"}
    head = state.workloads["w1"][HEAD]
    assert (head.run_status, head.error) == ("cancelled", CANCELLED_ERROR)


def test_a_run_the_server_could_not_cancel_is_not_marked_cancelled(
    platform: FakeCleanup,
) -> None:
    """B4: it had already completed; skipping it on the next run would waste a paid result."""
    state = _live_state()
    server = {
        "entries": [
            {"kind": "training_job", "id": "job-1", "status": "blocked", "reason": "completed"},
            {"kind": "deployment", "id": "dep-2", "status": "stopped", "reason": None},
        ]
    }

    cancel_recorded(state, as_cleanup_client(platform), server)

    assert platform.call_log == []
    head = state.workloads["w1"][HEAD]
    assert (head.run_status, head.error) == ("queued", None)
    assert state.workloads["w2"][SFT].deploy_status == "paused"


def test_a_cancel_pauses_or_notes_every_endpoint_once(platform: FakeCleanup) -> None:
    platform.present["job"].discard("job-1")
    platform.unpausable = {"dep-2"}
    state = _live_state()
    state.workloads["w3"] = {HEAD: StepState(deployment_id="dep-2", deploy_status="deploying")}
    state.workloads["w4"] = {HEAD: StepState(deployment_id="dep-gone", deploy_status="running")}

    receipt = cancel_recorded(state, as_cleanup_client(platform))

    assert [(r["id"], r["status"]) for r in receipt["entries"]] == [
        ("job-1", "already_absent"),
        ("dep-2", "blocked"),
        ("dep-gone", "already_absent"),
    ]
    assert state.workloads["w2"][SFT].deploy_status == "running"


def test_a_server_failure_during_a_cancel_still_raises(
    platform: FakeCleanup, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(job_id: str) -> None:
        raise APIError(500, "boom")

    monkeypatch.setattr(platform, "cancel_training_job", broken)  # a fault is not a refusal
    with pytest.raises(APIError, match="boom"):
        cancel_recorded(_live_state(), as_cleanup_client(platform))
