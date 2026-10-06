"""What a walk does with odd platform answers: nesting, empty words, and words of any length.

The real client over HTTP for the bodies a server can shape; the in-memory platform for the
receipt a long message ends up in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from tests.audit._cleanup import FakeCleanup, as_cleanup_client
from tests.audit._recorded import HEAD

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import APIError
from dagnam.audit.cleanup import delete_unpublished, receipt_rows
from dagnam.audit.cleanup_kinds import BY_KIND, NO_REASON, STILL_THERE, delete_one
from dagnam.audit.state import AuditState, StepState, load_state, save_state

if TYPE_CHECKING:
    from tests.typing_helpers import RequestsMocker

API = "https://api.test"
VERSION = f"{API}/api/v1/model-versions/mv-1"
CAP = 3000
"""More than the cap on platform words (2,048 characters) plus its truncation note."""


def _purge_row(requests_mock: RequestsMocker, **answer: object) -> dict[str, object]:
    client = DagnamClient(API, "dk-test-key-0000")
    client._sleep = lambda _s: None
    requests_mock.delete(VERSION, status_code=409, **answer)
    requests_mock.get(VERSION, json={"id": "mv-1"})  # it still reads back
    delete, get, absent = BY_KIND["model_version"]
    return delete_one(client, "model_version", delete, get, absent, "mv-1")


def test_a_deeply_nested_409_body_is_an_unparseable_body_and_the_version_blocked(
    requests_mock: RequestsMocker,
) -> None:
    row = _purge_row(requests_mock, text="[" * 200_000 + "]" * 200_000)

    assert row["status"] == "blocked"


def test_an_empty_409_body_on_a_version_that_still_reads_is_a_refusal_with_no_reason(
    requests_mock: RequestsMocker,
) -> None:
    row = _purge_row(requests_mock, text="")

    assert row["status"] == "blocked"
    assert row["reason"] == NO_REASON
    assert row["reason"] != STILL_THERE


def test_a_kept_versions_words_are_capped_in_its_row(requests_mock: RequestsMocker) -> None:
    kept = {"detail": {"status": "kept", "error": "weights_served", "message": "m" * 5_000_000}}

    row = _purge_row(requests_mock, json=kept)

    assert row["status"] == "kept"
    assert isinstance(row["reason"], str)
    assert len(row["reason"]) < CAP
    assert "truncated" in row["reason"]


def test_a_long_refusal_is_capped_in_the_receipt_file_and_the_walk_still_finishes(
    audit_dir: Path,
) -> None:
    state = AuditState(project_id="proj-1")
    state.workloads["w1"] = {HEAD: StepState(deployment_id="dep-1", dataset_id="ds-1")}
    save_state(audit_dir, state)
    platform = FakeCleanup(deployment=["dep-1"], dataset=["ds-1"], project=["proj-1"])
    platform.undeletable = {"dep-1"}  # an endpoint only the owner can read: the walk is proven
    platform.delete_errors = {"ds-1": APIError(409, "w" * 5_000_000)}

    receipt = delete_unpublished(audit_dir, as_cleanup_client(platform), load_state(audit_dir))

    row = next(r for r in receipt_rows(receipt) if r["id"] == "ds-1")
    assert (row["status"], row["code"]) == ("blocked", "not_removed")
    assert len(row["reason"]) < CAP
    assert len((audit_dir / "deleted.json").read_bytes()) < 10 * CAP
    assert json.loads((audit_dir / "deleted.json").read_text())["entries"]


def test_an_error_with_no_message_gets_a_reason_a_receipt_can_show(audit_dir: Path) -> None:
    state = AuditState(project_id="proj-1")
    state.workloads["w1"] = {HEAD: StepState(deployment_id="dep-1", dataset_id="ds-1")}
    save_state(audit_dir, state)
    platform = FakeCleanup(deployment=["dep-1"], dataset=["ds-1"], project=["proj-1"])
    platform.undeletable = {"dep-1"}
    platform.delete_errors = {"ds-1": APIError(409, "")}

    receipt = delete_unpublished(audit_dir, as_cleanup_client(platform), load_state(audit_dir))

    row = next(r for r in receipt_rows(receipt) if r["id"] == "ds-1")
    assert row["reason"] == NO_REASON
