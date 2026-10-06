"""An unpublished walk changes nothing local unless its key is shown to see the account.

Not-found is what another account's key is told for every id, so it is never proof. These cases are
the ones where something else went wrong beside the not-founds: an id whose delete and read both
failed, a stop that failed. Proof is a positive answer only the owner's account can get: a
deletion this key confirmed earlier, a delete or stop that succeeded, a keep the platform's own
purge answered, or a read of a recorded deployment or training run. A project, dataset or model
version can be public, so a read of one proves nothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from tests.audit._cleanup import FakeCleanup, as_cleanup_client
from tests.audit._recorded import HEAD, recorded_state

from dagnam._core.exceptions import APIError, DatasetNotFoundError
from dagnam.audit.cleanup import cancel_audit, delete_unpublished, receipt_rows
from dagnam.audit.receipt_rows import exit_status
from dagnam.audit.secrets import SecretStore
from dagnam.audit.state import DELETED_STATE, AuditState, StepState, load_state, save_state


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
def local(audit_dir: Path) -> Path:
    save_state(audit_dir, recorded_state())
    SecretStore(audit_dir).store("w1/head_tune", "dk-secret")
    return audit_dir


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def _same_but_receipt(a: dict[str, bytes], b: dict[str, bytes]) -> bool:
    return {k: v for k, v in a.items() if k != "deleted.json"} == {
        k: v for k, v in b.items() if k != "deleted.json"
    }


def _exit(receipt: Mapping[str, Any], verb: str = "delete") -> int:
    return exit_status(receipt_rows(receipt), receipt.get("audit_status"), verb=verb)


def _other_account() -> FakeCleanup:
    """Every id is not found; the project's own delete and read both fail (a 5xx burst)."""
    other = FakeCleanup()
    other.identity = "key-b"
    other.delete_errors = {"proj-1": APIError(503, "unavailable")}
    other.unreadable = {"proj-1": APIError(503, "unavailable")}
    return other


@pytest.mark.parametrize("partial", [True, False], ids=["after a partial delete", "fresh"])
def test_another_keys_walk_with_one_id_failing_removes_no_key_and_no_row(
    local: Path, platform: FakeCleanup, partial: bool
) -> None:
    if partial:
        platform.undeletable = {"dep-1"}
        delete_unpublished(local, as_cleanup_client(platform), load_state(local))
    before = _files(local)

    receipt = delete_unpublished(local, as_cleanup_client(_other_account()), load_state(local))

    (row,) = receipt_rows(receipt)
    assert (row["status"], row["code"]) == ("blocked", "not_answered")
    assert "--already-deleted" in row["reason"]
    assert _exit(receipt) == 1
    assert load_state(local).halted is None
    assert SecretStore(local).load("w1/head_tune") == "dk-secret"
    assert _same_but_receipt(before, _files(local))
    if partial:  # the right key afterwards still finishes it, keys and all
        platform.undeletable = set()
        again = delete_unpublished(local, as_cleanup_client(platform), load_state(local))
        assert _exit(again) == 0
        assert load_state(local).halted == DELETED_STATE
        assert platform.present["deployment"] == set()


def test_a_project_this_client_held_back_is_not_a_positive_answer(
    local: Path,
) -> None:
    """``project_held`` is written here, not by the platform: it proves nothing about the key."""
    other = FakeCleanup()
    other.identity = "key-b"
    other.delete_errors = {"ds-1": APIError(503, "unavailable")}
    other.unreadable = {"ds-1": APIError(503, "unavailable")}
    before = _files(local)

    receipt = delete_unpublished(local, as_cleanup_client(other), load_state(local))

    (row,) = receipt_rows(receipt)
    assert row["code"] == "not_answered"
    assert _same_but_receipt(before, _files(local))
    assert SecretStore(local).load("w1/head_tune") == "dk-secret"


def test_a_delete_that_got_one_positive_answer_proves_its_key_but_not_an_unanswered_id(
    local: Path, platform: FakeCleanup
) -> None:
    """Proven by what it deleted; the id it could not read still keeps the keys and rows."""
    platform.delete_errors = {"dep-1": APIError(503, "unavailable")}
    platform.unreadable = {"dep-1": APIError(503, "unavailable")}

    receipt = delete_unpublished(local, as_cleanup_client(platform), load_state(local))

    rows = {(r["kind"], r["id"]): r for r in receipt_rows(receipt)}
    assert rows[("deployment", "dep-1")]["code"] == "not_answered"
    assert rows[("dataset", "ds-1")]["status"] == "deleted"
    assert SecretStore(local).load("w1/head_tune") == "dk-secret"
    assert (local / "workloads").exists()
    assert load_state(local).halted is None
    assert _exit(receipt) == 1


def test_a_key_is_proven_by_reading_a_deployment_that_a_refusal_left_in_place(
    local: Path, platform: FakeCleanup
) -> None:
    """Nothing deleted, the endpoint refused: only the owner can read it, so the refusal stands."""
    only_a_deployment = AuditState(project_id="proj-1")
    only_a_deployment.workloads = {"w1": {HEAD: StepState(deployment_id="dep-1")}}
    save_state(local, only_a_deployment)
    platform.undeletable = {"dep-1"}

    receipt = delete_unpublished(local, as_cleanup_client(platform), load_state(local))

    row = next(r for r in receipt_rows(receipt) if r["id"] == "dep-1")
    assert (row["status"], row["code"]) == ("blocked", "not_removed")
    assert _exit(receipt) == 1
    assert SecretStore(local).load("w1/head_tune") == "dk-secret"  # an endpoint is up: keys stay


def test_a_read_of_a_dataset_that_a_refusal_left_in_place_proves_nothing(
    local: Path, platform: FakeCleanup
) -> None:
    """A dataset can be public: another account's key reads it too, so the walk is not answered."""
    only_a_dataset = AuditState(project_id="proj-1")
    only_a_dataset.workloads = {"w1": {HEAD: StepState(dataset_id="ds-1")}}
    save_state(local, only_a_dataset)
    platform.delete_errors = {"ds-1": APIError(409, "Dataset is held")}
    before = _files(local)

    receipt = delete_unpublished(local, as_cleanup_client(platform), load_state(local))

    (row,) = receipt_rows(receipt)
    assert (row["status"], row["code"]) == ("blocked", "not_answered")
    assert _exit(receipt) == 1
    assert _same_but_receipt(before, _files(local))
    assert load_state(local).halted is None


def test_a_cancel_by_another_key_with_a_failed_stop_marks_nothing(
    local: Path, platform: FakeCleanup, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = load_state(local)
    state.workloads["w1"][HEAD].deploy_status = "running"
    state.workloads["w1"][HEAD].run_status = "running"
    save_state(local, state)
    other = FakeCleanup()
    other.identity = "key-b"

    def failing(_job: str) -> object:
        raise APIError(503, "unavailable")

    monkeypatch.setattr(other, "cancel_training_job", failing)

    receipt = cancel_audit(local, as_cleanup_client(other))

    (row,) = receipt_rows(receipt)
    assert row["code"] == "not_answered"
    assert _exit(receipt, "cancel") == 1
    after = load_state(local).workloads["w1"][HEAD]
    assert (after.deploy_status, after.run_status) == ("running", "running")


def test_a_cancel_whose_stop_succeeded_is_proof_and_marks_what_stopped(
    local: Path, platform: FakeCleanup
) -> None:
    state = load_state(local)
    state.workloads["w1"][HEAD].deploy_status = "running"
    save_state(local, state)

    receipt = cancel_audit(local, as_cleanup_client(platform))

    assert {r["status"] for r in receipt_rows(receipt)} == {"stopped"}
    assert load_state(local).workloads["w1"][HEAD].deploy_status == "paused"


def test_a_cancel_of_a_finished_run_the_platform_still_knows_is_proof_too(
    local: Path, platform: FakeCleanup
) -> None:
    """A 400 "cannot cancel a finished run" is the platform answering for that run."""
    state = load_state(local)
    state.workloads["w1"][HEAD].run_status = "running"
    save_state(local, state)
    platform.finished = {"job-1"}

    receipt = cancel_audit(local, as_cleanup_client(platform))

    row = next(r for r in receipt_rows(receipt) if r["id"] == "job-1")
    assert (row["status"], row["code"]) == ("stopped", "already_stopped")
    assert _exit(receipt, "cancel") == 0


def test_a_delete_answered_not_found_is_read_again_and_a_ghost_is_not_taken_for_gone(
    local: Path, platform: FakeCleanup
) -> None:
    """One 404 is one answer; the id still reads back, so it is blocked, never ``already_absent``."""
    platform.delete_errors = {"ds-1": DatasetNotFoundError("ds-1")}

    receipt = delete_unpublished(local, as_cleanup_client(platform), load_state(local))

    row = next(r for r in receipt_rows(receipt) if r["id"] == "ds-1")
    assert row["status"] == "blocked"
    assert "still reads back" in row["reason"]
    assert ("get_dataset_meta", "ds-1") in platform.call_log


def test_a_delete_answered_not_found_whose_read_fails_is_not_answered(
    local: Path, platform: FakeCleanup
) -> None:
    platform.delete_errors = {"ds-1": DatasetNotFoundError("ds-1")}
    platform.unreadable = {"ds-1": APIError(503, "unavailable")}

    receipt = delete_unpublished(local, as_cleanup_client(platform), load_state(local))

    row = next(r for r in receipt_rows(receipt) if r["id"] == "ds-1")
    assert (row["status"], row["code"]) == ("blocked", "not_answered")
    assert SecretStore(local).load("w1/head_tune") == "dk-secret"  # nothing local goes
    assert (local / "workloads").exists()


def test_a_delete_answered_not_found_and_read_again_as_gone_is_already_absent(
    local: Path, platform: FakeCleanup
) -> None:
    platform.present["dataset"].discard("ds-1")

    receipt = delete_unpublished(local, as_cleanup_client(platform), load_state(local))

    row = next(r for r in receipt_rows(receipt) if r["id"] == "ds-1")
    assert row["status"] == "already_absent"
    assert ("get_dataset_meta", "ds-1") in platform.call_log
