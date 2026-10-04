"""What each receipt row means to this client, from its ``status`` and ``code`` alone."""

from __future__ import annotations

import pytest
from tests.audit import _receipts as r

from dagnam._types import JsonObject
from dagnam.audit.receipt_rows import (
    LEGACY_KEPT_PREFIXES,
    NOT_ANSWERED,
    NOT_REMOVED,
    PLATFORM_ONLY,
    Decision,
    Verdict,
    blocked,
    decide,
    exit_status,
)

DONE, KEPT, LEFT, UNKNOWN = Verdict.DONE, Verdict.KEPT, Verdict.LEFT, Verdict.UNKNOWN

# The contract's closed table, one case per code: (entry, verdict, whether a cancel marks with it).
TABLE: list[tuple[JsonObject, Verdict, bool]] = [
    (r.DELETED_ROW, DONE, True),
    (r.ALREADY_ABSENT_ROW, DONE, True),
    (r.STOPPED_ROW, DONE, True),
    (r.ALREADY_STOPPED_ROW, DONE, False),
    (r.PROJECT_NOT_OURS_ROW, KEPT, True),
    (r.PROJECT_SHARED_ROW, KEPT, True),
    (r.PROJECT_HELD_ROW, KEPT, True),
    (r.ENTRY_SHARED_ROW, KEPT, True),
    (r.WEIGHTS_SERVED_ROW, KEPT, True),
    (r.NOT_IN_PROJECT_ROW, KEPT, True),
    (r.NOT_CREATED_HERE_ROW, KEPT, True),
    (r.WEIGHTS_NOT_REMOVED_ROW, LEFT, True),
    (r.HAS_SERVED_ROW, LEFT, True),
    (r.REFUSED_ROW, LEFT, True),
    (r.FAILED_ROW, LEFT, True),
]


@pytest.mark.parametrize(
    ("entry", "expected", "marks"), TABLE, ids=[str(entry["code"]) for entry, _, _ in TABLE]
)
def test_every_code_of_the_contracts_table_is_decided(
    entry: JsonObject, expected: Verdict, marks: bool
) -> None:
    assert decide(entry) == Decision(expected, marks=marks)


def test_the_table_covers_every_code_the_contract_documents() -> None:
    assert [entry["code"] for entry, _, _ in TABLE] == [
        "deleted",
        "already_absent",
        "stopped",
        "already_stopped",
        "project_not_ours",
        "project_shared",
        "project_held",
        "entry_shared",
        "weights_served",
        "not_in_project",
        "not_created_here",
        "weights_not_removed",
        "has_served",
        "refused",
        "failed",
    ]


@pytest.mark.parametrize("status", ["deleted", "already_absent", "stopped"])
def test_a_plain_outcome_with_a_code_from_a_newer_platform_is_still_done(status: str) -> None:
    assert decide({"status": status, "code": "a_code_from_the_future"}) == Decision(DONE)


def test_a_kept_row_is_final_whatever_its_code() -> None:
    for code in (None, "a_code_from_the_future", "refused", "deleted"):
        assert decide({"kind": "project", "status": "kept", "code": code}) == Decision(KEPT)


def test_an_unknown_code_on_a_blocked_row_is_a_leftover_nothing_is_done_to() -> None:
    entry = r.row("deployment", "dep-1", "blocked", "needs_a_human", "ask somebody")
    assert decide(entry) == Decision(LEFT)


def test_a_code_never_turns_a_blocked_row_into_a_settled_one() -> None:
    for code in ("deleted", "already_absent", "stopped", "project_shared", "weights_served"):
        assert decide({"status": "blocked", "code": code}).verdict is LEFT


def test_a_status_this_client_does_not_know_is_shown_and_never_acted_on() -> None:
    for entry in ({"status": "quarantined", "code": "x"}, {"kind": "dataset"}, {"status": None}):
        assert decide(entry) == Decision(UNKNOWN)


class TestPlatformWithoutCodes:
    """A platform that predates ``code``: ``status`` plus the four reasons it used."""

    @pytest.mark.parametrize("prefix", LEGACY_KEPT_PREFIXES)
    def test_the_four_reasons_it_kept_things_on_purpose_with_are_kept(self, prefix: str) -> None:
        entry = r.legacy(r.row("project", "p1", "blocked", None, f"{prefix}; and then more"))
        assert decide(entry) == Decision(KEPT)

    def test_the_four_reasons_are_the_platforms_own_words(self) -> None:
        platform_words = (
            r.NOT_OURS,
            r.NOT_ONLY_OURS,
            r.ENTRY_SHARED,
            r.WEIGHTS_SERVED.split(";")[0],
        )
        assert platform_words == LEGACY_KEPT_PREFIXES

    def test_any_other_blocked_row_is_a_leftover_and_nothing_is_retried(self) -> None:
        for kind in ("deployment", "training_job", "dataset", "project", "model_version"):
            entry = r.legacy(r.row(kind, "x", "blocked", None, "Cannot delete: still deploying"))
            assert decide(entry) == Decision(LEFT)

    def test_a_run_it_could_not_cancel_because_it_had_ended_is_what_already_stopped_says(
        self,
    ) -> None:
        old = r.legacy(
            r.row("training_job", "j1", "blocked", None, "Cannot cancel job with status completed")
        )
        assert decide(old) == Decision(DONE, marks=False)
        assert exit_status([old], None, verb="cancel") == 0
        other_kind = {**old, "kind": "deployment"}  # only a run's words are read this way
        assert decide(other_kind) == Decision(LEFT)

    def test_an_endpoint_a_cancel_found_already_paused_is_what_already_stopped_says(self) -> None:
        old = r.legacy(
            r.row(
                "deployment",
                "d1",
                "blocked",
                None,
                "Invalid status transition from paused to paused",
            )
        )
        assert decide(old) == Decision(DONE, marks=False)
        assert exit_status([old], None, verb="cancel") == 0  # the second cancel of a finished audit
        assert decide({**old, "kind": "training_job"}) == Decision(LEFT)  # only an endpoint's words
        other = r.legacy(
            r.row("deployment", "d1", "blocked", None, "Invalid status transition from a to b")
        )
        assert decide(other) == Decision(LEFT)

    def test_a_blocked_row_with_no_reason_is_a_leftover(self) -> None:
        assert decide({"kind": "dataset", "status": "blocked"}) == Decision(LEFT)

    def test_the_plain_outcomes_read_the_same_without_a_code(self) -> None:
        for entry in (r.DELETED_ROW, r.ALREADY_ABSENT_ROW, r.STOPPED_ROW):
            assert decide(r.legacy(entry)) == Decision(DONE)


def test_a_row_this_client_writes_is_never_read_by_its_reason() -> None:
    """The reason of a local row can be the platform's words, which may start like a decision."""
    entry = blocked("deployment", "dep-1", f"{r.NOT_OURS}: said by the platform on a refusal")
    assert entry["code"] == NOT_REMOVED
    assert decide(entry) == Decision(LEFT)


def test_the_rows_this_client_writes_have_the_codes_the_contract_names() -> None:
    assert (NOT_ANSWERED, NOT_REMOVED, PLATFORM_ONLY) == (
        "not_answered",
        "not_removed",
        "platform_only",
    )
    for code in (NOT_ANSWERED, NOT_REMOVED, PLATFORM_ONLY):
        assert decide(blocked("audit", "a-1", "r", code)) == Decision(LEFT)


class TestExitStatus:
    """One function decides the exit status: 1 iff something of the audit's own is left."""

    def test_nothing_left_exits_zero_and_a_kept_row_never_fails(self) -> None:
        rows = [r.DELETED_ROW, r.PROJECT_SHARED_ROW, r.NOT_IN_PROJECT_ROW]
        assert exit_status(rows, "deleted", verb="delete") == 0

    @pytest.mark.parametrize(
        "entry", [r.REFUSED_ROW, r.HAS_SERVED_ROW, r.FAILED_ROW, r.WEIGHTS_NOT_REMOVED_ROW]
    )
    def test_a_blocked_row_exits_one_for_both_verbs(self, entry: JsonObject) -> None:
        assert exit_status([r.DELETED_ROW, entry], "halted", verb="delete") == 1
        assert exit_status([entry], "halted", verb="cancel") == 1

    def test_a_status_the_client_cannot_read_exits_one(self) -> None:
        assert (
            exit_status([{"kind": "dataset", "status": "quarantined"}], "deleted", verb="delete")
            == 1
        )

    def test_a_delete_the_platform_halted_exits_one_with_no_blocked_row_at_all(self) -> None:
        assert exit_status([r.DELETED_ROW], "halted", verb="delete") == 1

    def test_a_cancel_reports_halted_as_the_normal_state_of_a_cancelled_audit(self) -> None:
        assert exit_status([r.STOPPED_ROW, r.ALREADY_STOPPED_ROW], "halted", verb="cancel") == 0

    def test_a_platform_that_sends_no_audit_status_is_judged_by_its_rows_alone(self) -> None:
        assert exit_status([r.DELETED_ROW], None, verb="delete") == 0
        assert exit_status([r.legacy(r.REFUSED_ROW)], None, verb="delete") == 1
