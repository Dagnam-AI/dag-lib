"""The teardown's reading of the platform's answers, through the real client over HTTP.

Every body here is the one the backend sends, byte for byte in shape. A test that hands the walk a
hand-built exception proves nothing about what the client turns a response into: the client has
already reduced a 409 to its message by the time anything else sees it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import ModelError, TeardownInProgressError, VersionKeptError
from dagnam.audit.candidates import CandidateKind
from dagnam.audit.claims import ClaimError, claim_recorded
from dagnam.audit.cleanup import ask_platform, delete_unpublished, receipt_rows
from dagnam.audit.receipt_rows import exit_status
from dagnam.audit.state import DELETED_STATE, AuditState, StepState, load_state, save_state

if TYPE_CHECKING:
    from tests.typing_helpers import RequestsMocker

API = "https://api.test"
VERSION = f"{API}/api/v1/model-versions/mv-1"
AUDITS = f"{API}/api/v1/audits"
SERVED = {
    "detail": {
        "error": "weights_served",
        "message": "A live deployment serves this version; delete it first.",
        "kind": "model_version",
        "id": "mv-1",
        "status": "kept",
        "code": "weights_served",
    }
}
"""The 409 of ``DELETE /model-versions/{id}`` for a version a live deployment serves."""


def _state(audit_dir: Path) -> AuditState:
    state = AuditState(project_id="proj-1")
    state.workloads["w1"] = {CandidateKind.HEAD_TUNE: StepState(model_version_id="mv-1")}
    save_state(audit_dir, state)
    return state


def _client() -> DagnamClient:
    client = DagnamClient(API, "dk-test-key-0000")
    client._sleep = lambda _s: None
    return client


def test_the_real_client_keeps_the_409s_body_on_the_error(requests_mock: RequestsMocker) -> None:
    requests_mock.delete(VERSION, json=SERVED, status_code=409)

    with pytest.raises(VersionKeptError) as exc:
        _client().purge_model_version("mv-1")

    assert exc.value.row == SERVED["detail"]
    assert "serves this version" in str(exc.value)


def test_a_version_a_live_endpoint_serves_is_kept_and_the_unpublished_delete_finishes(
    audit_dir: Path, requests_mock: RequestsMocker
) -> None:
    state = _state(audit_dir)
    requests_mock.delete(VERSION, json=SERVED, status_code=409)

    receipt = delete_unpublished(audit_dir, _client(), state)

    rows = {(r["kind"], r["id"]): r for r in receipt_rows(receipt)}
    kept = rows[("model_version", "mv-1")]
    assert (kept["status"], kept["code"]) == ("kept", "weights_served")
    assert "serves this version" in kept["reason"]
    assert rows[("project", "proj-1")]["code"] == "project_held"
    assert exit_status(receipt_rows(receipt), receipt.get("audit_status"), verb="delete") == 0
    assert load_state(audit_dir).halted == DELETED_STATE
    assert [r.method for r in requests_mock.request_history] == ["DELETE"]  # nothing else was asked


@pytest.mark.parametrize(
    "answer",
    [
        {"json": {"detail": {**SERVED["detail"], "status": "blocked"}}},
        {"json": {"detail": "a string"}},
        {"json": ["not", "an", "object"]},
        {"text": "not json at all"},
        {"json": {"error": "weights_served", "status": "kept"}},  # no ``detail`` wrapper
    ],
    ids=["other status", "string detail", "not an object", "not json", "no detail wrapper"],
)
def test_any_other_409_stays_a_refusal_and_the_version_blocked(
    audit_dir: Path, requests_mock: RequestsMocker, answer: dict[str, Any]
) -> None:
    state = _state(audit_dir)
    requests_mock.delete(VERSION, status_code=409, **answer)
    requests_mock.get(VERSION, json={"id": "mv-1", "status": "ready"})  # it still reads back

    receipt = delete_unpublished(audit_dir, _client(), state)

    rows = {(r["kind"], r["id"]): r for r in receipt_rows(receipt)}
    assert rows[("model_version", "mv-1")]["status"] == "blocked"
    assert exit_status(receipt_rows(receipt), receipt.get("audit_status"), verb="delete") == 1
    with pytest.raises(ModelError) as exc:
        _client().purge_model_version("mv-1")
    assert not isinstance(exc.value, VersionKeptError)


BACKEND_WALKS = {
    409: {
        "detail": "A delete or cancel of this audit is already running; try again.",
        "error": "teardown_in_progress",
    },
    503: {"detail": "Try again in a moment.", "error": "teardown_unavailable"},
}
"""The 409 and 503 the teardown lock answers, exactly (``TeardownRefusedError.response``), with
``Retry-After: 5``."""


@pytest.mark.parametrize("code", [409, 503])
def test_the_backends_own_walk_in_progress_answers_are_waited_out(
    requests_mock: RequestsMocker, code: int
) -> None:
    requests_mock.delete(
        f"{AUDITS}/a1",
        [
            {"json": BACKEND_WALKS[code], "status_code": code, "headers": {"Retry-After": "5"}},
            {"json": {"schema": "x", "entries": []}},
        ],
    )
    slept: list[float] = []

    answer = ask_platform(_client().delete_audit, "a1", sleep=slept.append)

    assert answer.receipt == {"schema": "x", "entries": []}
    assert slept == [5.0]


@pytest.mark.parametrize("code", [409, 503])
def test_the_backends_walk_answer_carries_its_words_on_the_typed_error(
    requests_mock: RequestsMocker, code: int
) -> None:
    requests_mock.delete(
        f"{AUDITS}/a1",
        json=BACKEND_WALKS[code],
        status_code=code,
        headers={"Retry-After": "5"},
    )

    with pytest.raises(TeardownInProgressError) as exc:
        _client().delete_audit("a1")

    assert str(BACKEND_WALKS[code]["detail"]) in str(exc.value)
    assert (exc.value.status_code, exc.value.retry_after_header) == (code, "5")


@pytest.mark.parametrize("code", [409, 503])
def test_a_platform_that_nests_the_marker_under_detail_is_waited_out_too(
    requests_mock: RequestsMocker, code: int
) -> None:
    body = {"detail": {"error": BACKEND_WALKS[code]["error"], "message": "wait"}}
    requests_mock.delete(
        f"{AUDITS}/a1",
        [
            {"json": body, "status_code": code, "headers": {"Retry-After": "7"}},
            {"json": {"schema": "x", "entries": []}},
        ],
    )
    slept: list[float] = []

    assert ask_platform(_client().delete_audit, "a1", sleep=slept.append).receipt is not None
    assert slept == [7.0]


def test_the_backends_idempotency_in_progress_409_is_waited_out_by_a_create(
    requests_mock: RequestsMocker,
) -> None:
    in_progress = {
        "detail": "A request with this Idempotency-Key is in progress",
        "error": "idempotency_in_progress",
    }
    requests_mock.post(
        AUDITS,
        [{"json": in_progress, "status_code": 409}, {"json": {"id": "a1"}, "status_code": 201}],
    )
    client = _client()
    client.resume_creates = True

    assert client.create_audit({"project_id": "p1"})["id"] == "a1"
    assert len(requests_mock.request_history) == 2


def test_a_claim_refused_as_in_use_elsewhere_is_kept_through_the_real_client(
    audit_dir: Path, requests_mock: RequestsMocker
) -> None:
    state = AuditState(project_id="proj-1", audit_id="a1", claim_pending=True)
    state.workloads["w1"] = {
        CandidateKind.HEAD_TUNE: StepState(dataset_id="ds-1", training_job_id="job-1")
    }
    requests_mock.post(
        f"{AUDITS}/a1/claims",
        json={
            "results": [
                {"kind": "dataset", "id": "ds-1", "result": "refused", "code": "in_use_elsewhere"},
                {
                    "kind": "training_job",
                    "id": "job-1",
                    "result": "refused",
                    "code": "not_claimable",
                },
                {"kind": "project", "id": "proj-1", "result": "claimed", "code": "claimed"},
            ]
        },
    )

    claim_recorded(_client(), state)

    assert state.kept_ids == ["ds-1"]
    assert state.unclaimed_ids == ["job-1"]
    assert state.claim_pending is False


def test_a_claim_that_fails_whole_through_the_real_client_is_a_claim_error(
    requests_mock: RequestsMocker,
) -> None:
    state = AuditState(project_id="proj-1", audit_id="a1", claim_pending=True)
    state.workloads["w1"] = {CandidateKind.HEAD_TUNE: StepState(dataset_id="ds-1")}
    requests_mock.post(f"{AUDITS}/a1/claims", json={"detail": "boom"}, status_code=500)

    with pytest.raises(ClaimError):
        claim_recorded(_client(), state)
    assert state.claim_pending is True


def test_the_client_identity_is_a_digest_of_host_and_key_and_never_either() -> None:
    a, b = DagnamClient(API, "dk-a"), DagnamClient(API, "dk-b")
    assert a.identity == DagnamClient(API + "/", "dk-a").identity
    assert a.identity != b.identity != DagnamClient("https://other", "dk-b").identity
    assert "dk-a" not in a.identity
    assert "api.test" not in a.identity


@pytest.mark.parametrize(
    "answer", [{"text": "<html>proxy</html>"}, {"json": ["not", "a", "project"]}]
)
def test_a_project_read_answered_with_a_non_project_raises_the_type_error_the_adoption_names(
    requests_mock: RequestsMocker, answer: dict[str, Any]
) -> None:
    """The real client's own reaction to a 200 that is not a project, which the adoption catches."""
    requests_mock.get(f"{API}/api/v1/projects/proj-1", **answer)
    with pytest.raises(TypeError):
        _client().get_project("proj-1")
