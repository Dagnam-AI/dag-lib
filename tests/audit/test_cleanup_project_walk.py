"""The walk of the owner's projects is stable, or it answers nothing and nothing is deleted."""

from __future__ import annotations

from collections.abc import Mapping
import itertools
from pathlib import Path

import pytest
from tests.audit._cleanup import FakeCleanup, as_cleanup_client
from tests.audit._recorded import recorded_state

from dagnam._types import JsonArray, JsonObject
from dagnam.audit.cleanup import delete_unpublished, receipt_rows
from dagnam.audit.cleanup_kinds import datasets_in_other_projects
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
def local(audit_dir: Path) -> Path:
    save_state(audit_dir, recorded_state())
    SecretStore(audit_dir).store("w1/head_tune", "dk-secret")
    return audit_dir


def _statuses(receipt: Mapping[str, object]) -> dict[tuple[str, str], str]:
    return {(str(x["kind"]), str(x["id"])): str(x["status"]) for x in receipt_rows(receipt)}


def _exit(receipt: Mapping[str, object]) -> int:
    return exit_status(receipt_rows(receipt), receipt.get("audit_status"), verb="delete")


def test_a_project_touched_during_the_walk_is_not_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """The platform lists by last update unless told otherwise: touching a project mid-walk
    moves it across the page boundary. Oldest-first listing is stable under updates."""
    platform = FakeCleanup()
    created = [f"p{i}" for i in range(150)]
    recent_first = list(reversed(created))
    touched = False
    asked: list[dict[str, str | int]] = []

    def list_projects(**params: str | int) -> JsonObject:
        nonlocal touched
        asked.append(params)
        stable = params.get("sort_by") == "created_at" and params.get("order") == "asc"
        order = created if stable else recent_first
        page, limit = int(params["page"]), int(params["limit"])
        items: JsonArray = [{"id": i} for i in order[(page - 1) * limit : page * limit]]
        if page == 1 and not touched:  # the oldest project is updated while the walk runs
            touched = True
            recent_first.remove("p0")
            recent_first.insert(0, "p0")
        return {"items": items, "pages": 2, "total": 150}

    monkeypatch.setattr(platform, "list_projects", list_projects)
    platform.project_datasets = {"p0": {"training": [{"id": "ds-oldest"}]}}

    found = datasets_in_other_projects(as_cleanup_client(platform), None)

    assert "ds-oldest" in found
    assert all(p.get("sort_by") == "created_at" and p.get("order") == "asc" for p in asked)


@pytest.mark.parametrize(
    "totals", [[3, 2, 2], [2, 2, 3]], ids=["a project went between pages", "one came after"]
)
def test_a_total_that_moves_during_the_walk_raises_so_the_dataset_is_kept(
    monkeypatch: pytest.MonkeyPatch, totals: list[int]
) -> None:
    """Two pages, then page 1 read again: a count that changed means one may have been missed."""
    platform = FakeCleanup()
    reads = iter(totals)

    def list_projects(**params: str | int) -> JsonObject:
        return {"items": [{"id": f"p{params['page']}"}], "pages": 2, "total": next(reads)}

    monkeypatch.setattr(platform, "list_projects", list_projects)

    with pytest.raises(TypeError, match="changed while they were being listed"):
        datasets_in_other_projects(as_cleanup_client(platform), None)


def test_an_unstable_walk_deletes_nothing(
    local: Path, platform: FakeCleanup, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads = itertools.count()  # every read finds one more project

    def list_projects(**_params: str | int) -> JsonObject:
        return {"items": [], "pages": 1, "total": next(reads)}

    monkeypatch.setattr(platform, "list_projects", list_projects)

    receipt = delete_unpublished(local, as_cleanup_client(platform), load_state(local))

    assert _statuses(receipt)[("dataset", "ds-1")] == "blocked"
    assert ("delete_dataset", "ds-1") not in platform.call_log
    assert _exit(receipt) == 1
