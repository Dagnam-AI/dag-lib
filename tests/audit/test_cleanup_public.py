"""A key from another account must not release anything, even when the owner made things public.

A project, a dataset or a model version can be public: another account's key is refused the delete
(not found) yet reads it back. A dataset the owner made public can also be linked into the other
account's own project. None of that shows the key belongs to the owner, so the walk must change
nothing local and mark nothing, and the owner's key must still finish it afterwards.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.audit._cleanup import FakeCleanup, as_cleanup_client
from tests.audit._recorded import recorded_state

from dagnam._core.exceptions import DatasetNotFoundError, ModelNotFoundError, ProjectNotFoundError
from dagnam.audit.cleanup import delete_unpublished, receipt_rows
from dagnam.audit.receipt_rows import exit_status
from dagnam.audit.secrets import SecretStore
from dagnam.audit.state import DELETED_STATE, load_state, save_state

NOT_FOUND = {
    "project": ProjectNotFoundError("proj-1"),
    "dataset": DatasetNotFoundError("ds-1"),
    "model": ModelNotFoundError("mv-1"),
}
PUBLIC_ID = {"project": "proj-1", "dataset": "ds-1", "model": "mv-1"}


@pytest.fixture
def owner() -> FakeCleanup:
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
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "deleted.json"
    }


def _exit(receipt: dict[str, object]) -> int:
    return exit_status(receipt_rows(receipt), receipt.get("audit_status"), verb="delete")


def _other_key_sees_public(kind: str) -> FakeCleanup:
    """Another account's key: it can read the one public resource but delete nothing."""
    other = FakeCleanup(**{kind: [PUBLIC_ID[kind]]})
    other.identity = "key-b"
    other.delete_errors[PUBLIC_ID[kind]] = NOT_FOUND[kind]
    if kind == "model":
        other.purge_errors["mv-1"] = NOT_FOUND[kind]
    return other


def _other_key_links_public_dataset() -> FakeCleanup:
    """Another account's key whose own project has the owner's public dataset linked into it."""
    other = FakeCleanup()
    other.identity = "key-b"
    other.other_projects = {"proj-b": ["ds-1"]}
    return other


def _assert_untouched_then_owner_finishes(
    local: Path, owner: FakeCleanup, other: FakeCleanup, *, partial: bool
) -> None:
    if partial:
        owner.undeletable = {"dep-1"}
        delete_unpublished(local, as_cleanup_client(owner), load_state(local))
        owner.undeletable = set()
    before = _files(local)

    receipt = delete_unpublished(local, as_cleanup_client(other), load_state(local))

    assert _exit(receipt) == 1
    (row,) = receipt_rows(receipt)
    assert row["code"] == "not_answered"
    assert _files(local) == before
    state = load_state(local)
    assert state.halted is None
    assert state.kept_ids == []
    assert SecretStore(local).load("w1/head_tune") == "dk-secret"

    again = delete_unpublished(local, as_cleanup_client(owner), load_state(local))

    assert _exit(again) == 0
    assert load_state(local).halted == DELETED_STATE
    assert owner.present["deployment"] == set()
    assert owner.present["dataset"] == set()


@pytest.mark.parametrize("partial", [False, True], ids=["fresh", "after a partial delete"])
@pytest.mark.parametrize("kind", ["project", "dataset", "model"])
def test_a_public_resource_does_not_prove_another_accounts_key(
    local: Path, owner: FakeCleanup, kind: str, partial: bool
) -> None:
    _assert_untouched_then_owner_finishes(
        local, owner, _other_key_sees_public(kind), partial=partial
    )


@pytest.mark.parametrize("partial", [False, True], ids=["fresh", "after a partial delete"])
def test_a_public_dataset_linked_into_another_accounts_project_proves_nothing(
    local: Path, owner: FakeCleanup, partial: bool
) -> None:
    """The other account's project uses the dataset: that is a keep, but never proof of the key."""
    _assert_untouched_then_owner_finishes(
        local, owner, _other_key_links_public_dataset(), partial=partial
    )


def test_a_dataset_found_in_use_by_an_unproven_walk_is_not_kept_for_good(
    local: Path, owner: FakeCleanup
) -> None:
    delete_unpublished(
        local, as_cleanup_client(_other_key_links_public_dataset()), load_state(local)
    )

    assert load_state(local).kept_ids == []  # so the owner's own walk still deletes it
    delete_unpublished(local, as_cleanup_client(owner), load_state(local))
    assert "ds-1" not in owner.present["dataset"]


def test_a_dataset_kept_for_another_project_does_not_prove_a_walk_either_way(
    local: Path, owner: FakeCleanup
) -> None:
    """The owner's own key is proven by its deletes; the same keep is still recorded for good."""
    owner.other_projects = {"proj-x": ["ds-1"]}

    receipt = delete_unpublished(local, as_cleanup_client(owner), load_state(local))

    assert _exit(receipt) == 0
    assert load_state(local).kept_ids == ["ds-1"]
