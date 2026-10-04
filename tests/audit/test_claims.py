"""An unpublished run's ids are handed to the audit that publishes the directory."""

from __future__ import annotations

import pytest
from tests.audit._recorded import recorded_state

from dagnam._core.exceptions import APIError
from dagnam._types import JsonArray, JsonObject
from dagnam.audit.candidates import CandidateKind
from dagnam.audit.claims import BATCH, CLAIMABLE, ClaimError, claim_recorded
from dagnam.audit.state import AuditState


class _Client:
    def __init__(self, refuse: set[str] | None = None, failure: APIError | None = None) -> None:
        self.refuse = refuse or set()
        self.failure = failure
        self.asked: list[tuple[str, JsonArray]] = []

    def claim_audit_resources(self, audit_id: str, entries: JsonArray) -> JsonObject:
        self.asked.append((audit_id, entries))
        if self.failure is not None:
            raise self.failure
        results: JsonArray = [
            {
                "kind": e["kind"],
                "id": e["id"],
                "result": "refused" if e["id"] in self.refuse else "claimed",
                "code": "x",
            }
            for e in entries
            if isinstance(e, dict)
        ]
        return {"results": results}


def _state() -> AuditState:
    state = recorded_state()
    state.audit_id = "audit-1"
    state.claim_pending = True
    return state


def test_every_recorded_id_but_the_versions_is_asked_for_with_the_project_last() -> None:
    client = _Client()
    said: list[str] = []
    state = _state()

    claim_recorded(client, state, said.append)

    ((audit_id, entries),) = client.asked
    assert audit_id == "audit-1"
    kinds = [str(e["kind"]) for e in entries if isinstance(e, dict)]
    assert kinds == sorted(
        kinds, key=CLAIMABLE.index
    )  # datasets, runs, endpoints, then the project
    assert set(kinds) == {"dataset", "training_job", "deployment", "project"}
    assert set(CLAIMABLE) == set(kinds)
    assert said == []
    assert state.claim_pending is False
    assert state.unclaimed_ids == []


def test_a_refused_id_is_unclaimed_not_kept_and_its_runs_version_follows_it() -> None:
    state = _state()
    said: list[str] = []

    claim_recorded(_Client(refuse={"job-1", "ds-2"}), state, said.append)

    assert state.unclaimed_ids == ["ds-2", "job-1", "mv-1"]  # mv-1 is what job-1 pushed
    assert state.kept_ids == []  # a refusal is not a verdict that the id is not ours
    (line,) = said
    assert "3 resources" in line
    assert "handle them directly" in line


def test_a_refused_project_is_left_alone_and_not_recorded_for_deletion() -> None:
    state = _state()
    claim_recorded(_Client(refuse={"proj-1"}), state)
    assert state.unclaimed_ids == []


@pytest.mark.parametrize(
    "failure", [APIError(404, "not found"), APIError(500, "boom"), APIError(422, "no")]
)
def test_a_request_that_fails_as_a_whole_records_nothing_and_stays_pending(
    failure: APIError,
) -> None:
    state = _state()

    with pytest.raises(ClaimError):
        claim_recorded(_Client(failure=failure), state)

    assert (state.unclaimed_ids, state.kept_ids, state.claim_pending) == ([], [], True)


def test_an_answer_with_no_list_of_results_is_a_failure_not_a_claim() -> None:
    class Odd:
        def claim_audit_resources(self, audit_id: str, entries: JsonArray) -> JsonObject:
            return {"results": "none"}

    state = _state()
    with pytest.raises(ClaimError, match="no list of results"):
        claim_recorded(Odd(), state)
    assert state.claim_pending is True


def test_an_id_the_answer_leaves_out_is_not_claimed() -> None:
    class Short:
        def claim_audit_resources(self, audit_id: str, entries: JsonArray) -> JsonObject:
            return {"results": ["junk", {"kind": "dataset", "id": "ds-1", "result": "claimed"}]}

    state = _state()
    claim_recorded(Short(), state)
    assert "ds-2" in state.unclaimed_ids
    assert "ds-1" not in state.unclaimed_ids


def test_a_long_list_goes_in_batches_and_one_failing_batch_fails_the_claim() -> None:
    state = AuditState(audit_id="audit-1", claim_pending=True)
    for number in range(205):
        state.candidate(f"w{number}", CandidateKind.HEAD_TUNE).dataset_id = f"ds-{number}"
    client = _Client()
    claim_recorded(client, state)
    assert BATCH == 200  # the platform's limit for one request
    assert [len(entries) for _, entries in client.asked] == [200, 205 - 200]

    state.claim_pending = True
    with pytest.raises(ClaimError):
        claim_recorded(_Client(failure=APIError(500, "boom")), state)
    assert state.claim_pending is True


def test_nothing_recorded_but_a_project_asks_nothing() -> None:
    client = _Client()
    state = AuditState(audit_id="audit-1", project_id="proj-1", claim_pending=True)
    claim_recorded(client, state)
    assert client.asked == []
    assert state.claim_pending is False
    claim_recorded(client, AuditState(claim_pending=True))
    assert client.asked == []
