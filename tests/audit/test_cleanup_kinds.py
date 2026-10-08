"""One id at a time: every answer the platform can give ends as a receipt row, never an exception."""

from __future__ import annotations

from typing import override

import pytest
from tests.audit._cleanup import FakeCleanup, as_cleanup_client

from dagnam._core.exceptions import APIError, DagnamError
from dagnam._types import JsonObject
from dagnam.audit.cleanup_kinds import (
    BY_KIND,
    KINDS,
    STILL_THERE,
    STOPPED,
    STOPS,
    delete_one,
)
from dagnam.audit.receipt_rows import ALREADY_STOPPED, NOT_REMOVED, PLATFORM_ONLY


def _delete(platform: FakeCleanup, kind: str, item_id: str) -> dict[str, object]:
    delete, get, absent = BY_KIND[kind]
    return delete_one(as_cleanup_client(platform), kind, delete, get, absent, item_id)


def test_the_kinds_are_walked_children_first() -> None:
    """A deployment before the version it serves, a job before the dataset it holds, the project last."""
    assert [kind for kind, *_ in KINDS] == [
        "deployment",
        "model_version",
        "training_job",
        "dataset",
        "project",
    ]


def test_a_delete_that_lands_is_deleted_and_one_with_nothing_there_is_already_absent() -> None:
    platform = FakeCleanup(dataset=["ds-1"])
    assert _delete(platform, "dataset", "ds-1") == {
        "kind": "dataset",
        "id": "ds-1",
        "status": "deleted",
    }
    assert _delete(platform, "dataset", "ds-1")["status"] == "already_absent"


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        # The platform's own wording for a conflict, without the status in front of it.
        (
            APIError(409, "Dataset is referenced by a training run"),
            "Dataset is referenced by a training run",
        ),
        # Anything else says what it was: a fault is told apart from a refusal by its reason.
        (APIError(500, "boom"), "API error 500: boom"),
        (
            APIError(0, "Request failed: connection reset"),
            "API error 0: Request failed: connection reset",
        ),
        (DagnamError("the key lacks the write scope"), "the key lacks the write scope"),
    ],
)
def test_whatever_the_delete_raises_is_the_rows_reason(error: DagnamError, reason: str) -> None:
    platform = FakeCleanup(dataset=["ds-1"])
    platform.dataset_error = error
    assert _delete(platform, "dataset", "ds-1") == {
        "kind": "dataset",
        "id": "ds-1",
        "status": "blocked",
        "code": NOT_REMOVED,
        "reason": reason,
    }


def test_a_refusal_for_an_id_that_is_in_fact_gone_is_already_absent() -> None:
    """The re-read decides: a 409 for something another process already removed is not a block."""
    platform = FakeCleanup()
    platform.dataset_error = APIError(409, "busy")
    assert _delete(platform, "dataset", "ds-1")["status"] == "already_absent"


def test_a_delete_nobody_could_confirm_is_blocked_not_assumed_done() -> None:
    platform = FakeCleanup(project=["proj-1"])
    platform.unreadable = {"proj-1": APIError(502, "bad gateway")}
    row = _delete(platform, "project", "proj-1")
    assert row["status"] == "blocked"
    assert row["reason"] == "the delete could not be confirmed: API error 502: bad gateway"


def test_a_refusal_keeps_its_own_reason_when_the_re_read_fails_too() -> None:
    platform = FakeCleanup(deployment=["dep-1"])
    platform.undeletable = {"dep-1"}
    platform.unreadable = {"dep-1": APIError(502, "bad gateway")}
    assert _delete(platform, "deployment", "dep-1")["reason"] == (
        "Cannot delete a deployment that is still deploying"
    )


def test_an_id_that_reads_back_after_its_delete_says_so() -> None:
    platform = FakeCleanup(dataset=["ds-1"])
    platform.sticky = {"ds-1"}
    assert _delete(platform, "dataset", "ds-1")["reason"] == STILL_THERE


def test_a_live_run_is_cancelled_and_a_live_endpoint_paused() -> None:
    platform = FakeCleanup(job=["job-1"], deployment=["dep-1"])
    client = as_cleanup_client(platform)
    assert STOPS["training_job"](client, "job-1") == {
        "kind": "training_job",
        "id": "job-1",
        "status": STOPPED,
    }
    assert STOPS["deployment"](client, "dep-1")["status"] == STOPPED
    assert platform.call_log == [("cancel_training_job", "job-1"), ("pause_deployment", "dep-1")]


class _OddClient(FakeCleanup):
    """A platform whose answers have the wrong shape: the failures a delete must not die of."""

    @override
    def get_project(self, project_id: str) -> JsonObject:
        raise TypeError("Expected JSON object, got str")

    @override
    def get_dataset_meta(self, dataset_id: str, version: str | None = None) -> JsonObject:
        raise KeyError("data")


def test_an_answer_of_the_wrong_shape_is_a_row_never_an_exception() -> None:
    platform = _OddClient(project=["proj-1"], dataset=["ds-1"])
    project = _delete(platform, "project", "proj-1")
    assert project["status"] == "blocked"
    assert (
        project["reason"]
        == "the delete could not be confirmed: TypeError('Expected JSON object, got str')"
    )
    dataset = _delete(platform, "dataset", "ds-1")
    assert dataset["reason"] == "the delete could not be confirmed: KeyError('data')"


@pytest.mark.parametrize(
    ("kind", "item", "method"),
    [
        ("training_job", "j1", "cancel_training_job"),
        ("deployment", "d1", "pause_deployment"),
    ],
)
def test_a_stop_that_fails_is_that_ids_row_and_never_a_raise(
    monkeypatch: pytest.MonkeyPatch, kind: str, item: str, method: str
) -> None:
    platform = FakeCleanup(job=["j1"], deployment=["d1"])
    platform.statuses.update({"j1": "running", "d1": "running"})  # still up: the stop failed

    def broken(_: str) -> JsonObject:
        raise APIError(500, "boom")

    monkeypatch.setattr(platform, method, broken)
    assert STOPS[kind](as_cleanup_client(platform), item) == {
        "kind": kind,
        "id": item,
        "status": "blocked",
        "code": NOT_REMOVED,
        "reason": "API error 500: boom",
    }


def test_a_stop_for_something_that_is_gone_is_already_absent() -> None:
    client = as_cleanup_client(FakeCleanup())
    assert STOPS["training_job"](client, "nope")["status"] == "already_absent"
    assert STOPS["deployment"](client, "nope")["status"] == "already_absent"


def test_a_pause_the_platform_refuses_says_why_and_a_wrong_shaped_answer_is_a_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refusing = FakeCleanup(deployment=["d2"])
    refusing.unpausable = {"d2"}
    assert STOPS["deployment"](as_cleanup_client(refusing), "d2")["reason"] == (
        "Invalid status transition from not_provisioned to paused"
    )

    def malformed(_: str) -> JsonObject:
        raise TypeError("not an object")

    platform = FakeCleanup(deployment=["d1"])
    platform.statuses["d1"] = "running"
    monkeypatch.setattr(platform, "pause_deployment", malformed)
    assert STOPS["deployment"](as_cleanup_client(platform), "d1")["reason"] == (
        "TypeError('not an object')"
    )


def test_a_run_that_already_ended_is_already_stopped_and_marks_nothing() -> None:
    platform = FakeCleanup(job=["j1"])
    platform.finished = {"j1"}
    row = STOPS["training_job"](as_cleanup_client(platform), "j1")
    assert row == {"kind": "training_job", "id": "j1", "status": STOPPED, "code": ALREADY_STOPPED}


def test_a_version_is_purged_through_its_own_route_never_through_its_entry() -> None:
    platform = FakeCleanup(model=["mv-1"])
    assert _delete(platform, "model_version", "mv-1")["status"] == "deleted"
    assert platform.call_log == [("purge_model_version", "mv-1")]  # its own row settles it
    assert _delete(platform, "model_version", "mv-1")["status"] == "already_absent"


def test_a_platform_without_the_purge_route_reports_the_version_platform_only() -> None:
    """The route's 404 looks like a missing version: the re-read tells them apart, and nothing is deleted."""
    platform = FakeCleanup(model=["mv-1"])
    platform.no_purge_route = True
    row = _delete(platform, "model_version", "mv-1")
    assert (row["status"], row["code"]) == ("blocked", PLATFORM_ONLY)
    assert "mv-1" in str(row["reason"])
    assert platform.present["model"] == {"mv-1"}


def test_a_405_from_the_purge_route_is_the_same_missing_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform = FakeCleanup(model=["mv-1"])

    def method_not_allowed(_: str) -> None:
        raise APIError(405, "Method Not Allowed")

    monkeypatch.setattr(platform, "purge_model_version", method_not_allowed)
    assert _delete(platform, "model_version", "mv-1")["code"] == PLATFORM_ONLY


def test_a_purge_the_platform_fails_is_that_versions_blocked_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform = FakeCleanup(model=["mv-1"])

    def boom(_: str) -> None:
        raise APIError(500, "storage down")

    monkeypatch.setattr(platform, "purge_model_version", boom)
    row = _delete(platform, "model_version", "mv-1")
    assert (row["status"], row["code"], row["reason"]) == (
        "blocked",
        NOT_REMOVED,
        "API error 500: storage down",
    )


def test_a_purge_that_failed_after_revoking_the_row_is_not_taken_for_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The platform's 502 `weights_not_removed`: the row reads revoked, the bytes may remain."""
    platform = FakeCleanup(model=["mv-1"])
    platform.purge_errors = {"mv-1": APIError(502, "weights_not_removed")}

    def revoked(_: str) -> JsonObject:
        return {"id": "mv-1", "status": "revoked"}

    monkeypatch.setattr(platform, "get_model_version", revoked)
    row = _delete(platform, "model_version", "mv-1")
    assert (row["status"], row["code"]) == ("blocked", NOT_REMOVED)
    assert "502" in str(row["reason"])


def test_a_version_a_deployment_serves_is_refused_by_the_platform_and_stays_blocked() -> None:
    platform = FakeCleanup(model=["mv-1"])
    platform.purge_errors = {"mv-1": APIError(409, "weights_served")}
    row = _delete(platform, "model_version", "mv-1")
    assert (row["status"], row["reason"]) == ("blocked", "weights_served")
    assert platform.present["model"] == {"mv-1"}


def test_a_purge_that_answers_no_row_is_confirmed_by_reading_the_version_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform = FakeCleanup(model=["mv-1"])
    monkeypatch.setattr(platform, "purge_model_version", lambda _: None)
    assert _delete(platform, "model_version", "mv-1")["status"] == "blocked"  # it still reads back


def test_a_run_stop_that_fails_in_an_unexpected_shape_is_that_ids_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform = FakeCleanup(job=["j1"])

    def malformed(_: str) -> JsonObject:
        raise TypeError("not an object")

    monkeypatch.setattr(platform, "cancel_training_job", malformed)
    row = STOPS["training_job"](as_cleanup_client(platform), "j1")
    assert (row["status"], row["code"], row["reason"]) == (
        "blocked",
        NOT_REMOVED,
        "TypeError('not an object')",
    )


def test_an_endpoint_the_owner_paused_already_is_already_stopped_not_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform = FakeCleanup(deployment=["d1"])
    platform.unpausable = {"d1"}
    platform.statuses = {"d1": "paused"}
    row = STOPS["deployment"](as_cleanup_client(platform), "d1")
    assert row == {"kind": "deployment", "id": "d1", "status": STOPPED, "code": ALREADY_STOPPED}

    platform.statuses = {"d1": "deploying"}  # one that will not pause and is not paused
    assert STOPS["deployment"](as_cleanup_client(platform), "d1")["status"] == "blocked"

    def unreadable(_: str) -> JsonObject:
        raise APIError(503, "down")

    monkeypatch.setattr(platform, "get_deployment", unreadable)
    assert STOPS["deployment"](as_cleanup_client(platform), "d1")["status"] == "blocked"
