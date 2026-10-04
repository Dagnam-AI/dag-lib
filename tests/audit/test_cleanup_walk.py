"""The walks this client makes itself: ids it confirmed gone, datasets others use, weights a live endpoint serves."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.audit import _receipts as r
from tests.audit._cleanup import FakeCleanup, as_cleanup_client
from tests.audit._recorded import HEAD, recorded_state

from dagnam._core.exceptions import APIError, ModelError
from dagnam.audit.cleanup import cancel_audit, delete_audit, delete_unpublished, receipt_rows
from dagnam.audit.cleanup_kinds import IN_USE_ELSEWHERE, datasets_in_other_projects
from dagnam.audit.cleanup_walk import remember
from dagnam.audit.receipt_rows import exit_status
from dagnam.audit.secrets import SecretStore
from dagnam.audit.state import DELETED_STATE, load_state, save_state


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


def _statuses(receipt: dict[str, object]) -> dict[tuple[str, str], str]:
    return {(str(x["kind"]), str(x["id"])): str(x["status"]) for x in receipt_rows(receipt)}


def _exit(receipt: dict[str, object]) -> int:
    return exit_status(receipt_rows(receipt), receipt.get("audit_status"), verb="delete")


class TestIdsAnEarlierWalkConfirmedGone:
    def test_a_second_walk_that_finds_everything_gone_finishes_the_directory(
        self, local: Path, platform: FakeCleanup
    ) -> None:
        platform.undeletable = {"dep-1"}
        first = delete_unpublished(local, as_cleanup_client(platform), load_state(local))
        assert _exit(first) == 1
        assert load_state(local).halted is None
        assert "ds-1" in load_state(local).confirmed_gone  # a delete this key saw land

        platform.present = {kind: set() for kind in platform.present}  # all gone, the rest by hand
        second = delete_unpublished(local, as_cleanup_client(platform), load_state(local))

        assert _exit(second) == 0
        assert load_state(local).halted == DELETED_STATE
        assert all(s == "already_absent" for s in _statuses(second).values())

    def test_a_directory_that_never_saw_one_land_is_still_guarded_and_says_how_to_clear_it(
        self, local: Path
    ) -> None:
        receipt = delete_unpublished(local, as_cleanup_client(FakeCleanup()), load_state(local))

        (row,) = receipt_rows(receipt)
        assert "--already-deleted" in row["reason"]
        assert "irreversibly" not in row["reason"]  # the reason says how; the CLI warns
        assert load_state(local).halted is None

    def test_a_cancel_after_a_confirmed_delete_is_not_called_another_accounts(
        self, local: Path
    ) -> None:
        state = load_state(local)
        state.workloads["w1"][HEAD].run_status = "running"
        state.confirmed_gone = ["ds-1"]
        save_state(local, state)

        receipt = cancel_audit(local, as_cleanup_client(FakeCleanup()))

        assert {x["status"] for x in receipt_rows(receipt)} == {"already_absent"}
        assert load_state(local).workloads["w1"][HEAD].run_status != "running"

    def test_remember_records_only_what_was_deleted_and_the_final_keeps(self, local: Path) -> None:
        state = load_state(local)
        remember(
            state,
            [
                r.row("dataset", "ds-1", "deleted", "deleted"),
                r.row("dataset", "ds-2", "kept", IN_USE_ELSEWHERE),
                r.row("project", "proj-1", "kept", "project_held"),  # asked again, never final
                r.row("dataset", "ds-1", "deleted", "deleted"),
            ],
        )
        assert (state.confirmed_gone, state.kept_ids) == (["ds-1"], ["ds-2"])


class TestDatasetsAnotherProjectUses:
    def test_a_dataset_linked_into_a_project_this_directory_did_not_create_is_kept(
        self, local: Path, platform: FakeCleanup
    ) -> None:
        platform.other_projects = {"proj-9": ["ds-2"], "proj-1": ["ds-1"]}

        receipt = delete_unpublished(local, as_cleanup_client(platform), load_state(local))

        statuses = _statuses(receipt)
        assert statuses[("dataset", "ds-2")] == "kept"
        assert statuses[("dataset", "ds-1")] == "deleted"  # its own project's link is no use
        assert "ds-2" in platform.present["dataset"]
        assert ("delete_dataset", "ds-2") not in platform.call_log
        assert _exit(receipt) == 0
        assert load_state(local).halted == DELETED_STATE
        assert load_state(local).kept_ids == ["ds-2"]

    def test_a_kept_dataset_is_not_touched_by_the_next_walk_or_a_cancel(
        self, local: Path, platform: FakeCleanup
    ) -> None:
        state = load_state(local)
        state.kept_ids = ["ds-2"]
        state.workloads["w2"][next(iter(state.workloads["w2"]))].run_status = "running"
        save_state(local, state)

        delete_unpublished(local, as_cleanup_client(platform), load_state(local))

        assert not [c for c in platform.call_log if c[1] == "ds-2"]

    @pytest.mark.parametrize("failure", [APIError(500, "boom"), TypeError("not an object")])
    def test_links_that_cannot_be_read_block_the_dataset_instead_of_guessing(
        self, local: Path, platform: FakeCleanup, failure: Exception
    ) -> None:
        platform.project_reads_fail = failure
        receipt = delete_unpublished(local, as_cleanup_client(platform), load_state(local))

        rows = {(x["kind"], x["id"]): x for x in receipt_rows(receipt)}
        assert rows[("dataset", "ds-1")]["status"] == "blocked"
        assert "could not check" in rows[("dataset", "ds-1")]["reason"]
        assert ("delete_dataset", "ds-1") not in platform.call_log
        assert _exit(receipt) == 1

    def test_the_project_list_is_read_page_by_page_and_odd_entries_are_skipped(self) -> None:
        platform = FakeCleanup()
        platform.project_pages = {
            1: {"items": [{"id": "p1"}, "junk"], "pages": 3},
            2: {"items": [{"id": "p2"}], "pages": 3},
            3: {"items": [{"id": "p3"}], "pages": 3},
        }
        platform.project_datasets = {
            "p1": {"training": [{"id": "d-p1"}, "junk"], "note": "x"},
            "p2": {"training": [{"id": "d-p2"}]},
            "p3": {"training": [{"id": "d-p3"}]},
        }

        found = datasets_in_other_projects(as_cleanup_client(platform), "p2")

        assert found == {"d-p1", "d-p3"}  # p2 is this directory's own project

    def test_a_listing_without_a_page_count_or_items_is_one_empty_page(self) -> None:
        platform = FakeCleanup()
        platform.project_pages = {1: {"items": None}}
        assert datasets_in_other_projects(as_cleanup_client(platform), None) == set()

    def test_a_listing_that_is_not_an_object_raises(self) -> None:
        platform = FakeCleanup()
        platform.project_pages = {1: "nope"}
        with pytest.raises(TypeError):
            datasets_in_other_projects(as_cleanup_client(platform), None)

    def test_the_read_stops_at_the_page_bound(self, monkeypatch: pytest.MonkeyPatch) -> None:
        platform = FakeCleanup()
        platform.project_pages = {n: {"items": [], "pages": 10**6} for n in range(1, 10)}
        monkeypatch.setattr("dagnam.audit.cleanup_kinds.PROJECT_PAGES", 3)

        datasets_in_other_projects(as_cleanup_client(platform), None)

        assert [page for name, page in platform.call_log if name == "list_projects"] == [
            "1",
            "2",
            "3",
        ]


SERVED = json.dumps(
    {
        "detail": {
            "error": "weights_served",
            "status": "kept",
            "code": "weights_served",
            "message": "A live deployment serves this version; delete it first.",
        }
    }
)


class TestWeightsAnEndpointServes:
    def test_the_purge_refusal_is_kept_not_blocked_and_the_delete_finishes(
        self, local: Path, platform: FakeCleanup
    ) -> None:
        platform.purge_errors = {"mv-1": ModelError(SERVED)}

        receipt = delete_unpublished(local, as_cleanup_client(platform), load_state(local))

        rows = {(x["kind"], x["id"]): x for x in receipt_rows(receipt)}
        assert (
            rows[("model_version", "mv-1")]["status"],
            rows[("model_version", "mv-1")]["code"],
        ) == (
            "kept",
            "weights_served",
        )
        assert rows[("project", "proj-1")]["code"] == "project_held"  # the weights live in it
        assert "mv-1" in platform.present["model"]
        assert _exit(receipt) == 0
        assert load_state(local).halted == DELETED_STATE

    @pytest.mark.parametrize(
        "body",
        [
            json.dumps({"detail": {"error": "weights_served", "status": "blocked"}}),
            json.dumps({"detail": "a string"}),
            json.dumps(["not", "an", "object"]),
            "not json at all",
        ],
        ids=["other status", "string detail", "not an object", "not json"],
    )
    def test_any_other_conflict_stays_a_blocked_row(
        self, local: Path, platform: FakeCleanup, body: str
    ) -> None:
        platform.purge_errors = {"mv-1": ModelError(body)}

        receipt = delete_unpublished(local, as_cleanup_client(platform), load_state(local))

        assert _statuses(receipt)[("model_version", "mv-1")] == "blocked"
        assert _exit(receipt) == 1

    def test_a_refusal_with_no_words_of_its_own_is_still_named_by_its_code(
        self, local: Path, platform: FakeCleanup
    ) -> None:
        platform.purge_errors = {"mv-1": ModelError(json.dumps({"detail": {"status": "kept"}}))}
        row = next(
            x
            for x in receipt_rows(
                delete_unpublished(local, as_cleanup_client(platform), load_state(local))
            )
            if x["id"] == "mv-1"
        )
        assert (row["code"], row["reason"]) == ("weights_served", "weights_served")


def test_a_published_delete_keeps_what_the_platform_answered_for_out_of_the_direct_walk(
    local: Path, platform: FakeCleanup
) -> None:
    """A claim whose answer was lost leaves every id unclaimed; the platform's own rows say which it owns."""
    state = load_state(local)
    state.audit_id = "audit-1"
    state.claim_pending = True
    save_state(local, state)
    platform.server_receipt = r.designed(
        r.row("deployment", "dep-1", "deleted", "deleted"),
        r.row("dataset", "ds-1", "kept", "in_use_elsewhere"),
        r.row("dataset", "ds-2", "kept", "not_created_here"),
    )

    delete_audit(local, as_cleanup_client(platform))

    touched = {c for c in platform.call_log if c[0].startswith(("delete_", "purge_"))}
    assert ("delete_deployment", "dep-1") not in touched  # the platform deleted it
    assert ("delete_dataset", "ds-1") not in touched  # the platform kept it for its owner
    assert ("delete_dataset", "ds-2") in touched  # recorded, not created here: this directory's
