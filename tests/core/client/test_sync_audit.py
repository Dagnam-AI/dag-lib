"""Wire-level coverage for the sync workload-audit publish client mixin."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pytest

from dagnam._core.client import DagnamClient
from dagnam._core.client.audit import WALK_TIMEOUT
from dagnam._core.client.base import DEFAULT_TIMEOUT
from dagnam._core.exceptions import (
    APIError,
    AuthError,
    EndpointsServingError,
    TeardownInProgressError,
)
from dagnam._types import JsonArray, JsonObject

if TYPE_CHECKING:
    from tests.typing_helpers import RequestsMocker

API = "https://api.test"
AUDITS = f"{API}/api/v1/audits"
RECEIPT = {"schema": "dagnam.audit.deleted/1", "deleted_at": "2026-09-07T10:00:00Z", "entries": []}


def test_create_audit(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(AUDITS, json={"id": "a1"}, status_code=201)
    assert client.create_audit({"project_id": "p1"}) == {"id": "a1"}
    assert rmock.last_request.json() == {"project_id": "p1"}


def test_create_audit_candidate(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{AUDITS}/a1/candidates", json={"id": "c1"}, status_code=201)
    assert client.create_audit_candidate("a1", {"kind": "head_tune"}) == {"id": "c1"}
    assert rmock.last_request.json() == {"kind": "head_tune"}


def test_patch_audit_candidate(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.patch(f"{AUDITS}/a1/candidates/c1", json={"id": "c1", "status": "scored"})
    body: JsonObject = {"step": "replay_and_score", "status": "scored"}
    assert client.patch_audit_candidate("a1", "c1", body)["status"] == "scored"
    assert rmock.last_request.json() == body


def test_halt_audit(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{AUDITS}/a1/halt", json={"id": "a1", "status": "halted"})
    assert client.halt_audit("a1", "budget")["status"] == "halted"
    assert rmock.last_request.json() == {"reason": "budget"}


def test_claim_audit_resources_posts_the_ids_and_returns_the_answer(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    answer = {"entries": [{"kind": "dataset", "id": "ds-1", "status": "claimed"}]}
    rmock.post(f"{AUDITS}/a1/claims", json=answer)
    entries: list[JsonObject] = [{"kind": "dataset", "id": "ds-1"}]
    assert client.claim_audit_resources("a1", [*entries]) == answer
    assert rmock.last_request.json() == {"entries": entries}


@pytest.mark.parametrize(
    "body",
    [
        {"error": "teardown_in_progress", "message": "another walk"},
        {"detail": {"error": "teardown_in_progress", "message": "another walk"}},
    ],
    ids=["top level", "nested under detail"],
)
def test_a_409_saying_a_teardown_is_running_is_typed_with_its_retry_after(
    client: DagnamClient, rmock: RequestsMocker, body: JsonObject
) -> None:
    rmock.delete(f"{AUDITS}/a1", json=body, status_code=409, headers={"Retry-After": "7"})
    with pytest.raises(TeardownInProgressError) as exc:
        client.delete_audit("a1")
    assert exc.value.retry_after_header == "7"
    assert "another walk" in str(exc.value)


@pytest.mark.parametrize(
    "body",
    [
        {"error": "teardown_unavailable", "message": "try again"},
        {"detail": {"error": "teardown_unavailable", "message": "try again"}},
    ],
    ids=["top level", "nested under detail"],
)
def test_a_503_saying_the_lock_is_unavailable_is_waited_like_a_running_teardown(
    client: DagnamClient, rmock: RequestsMocker, body: JsonObject
) -> None:
    rmock.delete(f"{AUDITS}/a1", json=body, status_code=503, headers={"Retry-After": "11"})
    with pytest.raises(TeardownInProgressError) as exc:
        client.delete_audit("a1")
    assert (exc.value.status_code, exc.value.retry_after_header) == (503, "11")
    assert "try again" in str(exc.value)


def test_any_other_503_stays_a_plain_api_error(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.delete(f"{AUDITS}/a1", json={"detail": "overloaded"}, status_code=503)
    with pytest.raises(APIError) as exc:
        client.delete_audit("a1")
    assert not isinstance(exc.value, TeardownInProgressError)
    # The 409 marker on a 503 (or the reverse) is not the other's word.
    rmock.delete(f"{AUDITS}/a2", json={"error": "teardown_in_progress"}, status_code=503)
    with pytest.raises(APIError) as mixed:
        client.delete_audit("a2")
    assert not isinstance(mixed.value, TeardownInProgressError)


def test_any_other_409_stays_a_plain_api_error(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.delete(f"{AUDITS}/a1", json={"detail": "audit is halted"}, status_code=409)
    with pytest.raises(APIError) as exc:
        client.delete_audit("a1")
    assert not isinstance(exc.value, TeardownInProgressError)
    rmock.delete(f"{AUDITS}/a2", text="not json", status_code=409)
    with pytest.raises(APIError) as plain:
        client.delete_audit("a2")
    assert not isinstance(plain.value, TeardownInProgressError)


SERVING_SENTENCE = (
    "Nothing was deleted: this audit still has 2 serving endpoints: tickets-ft-small "
    "(3f2a0c1e-0000-4000-8000-000000000001), tickets-ft-base (9b1c0c1e-0000-4000-8000-000000000002). "
    "Deleting the audit would delete them, and any app calling them would start getting errors. "
    "Stop them first with `dagnam audit cancel <audit-dir>`, or Pause on each one's page under "
    "Deployments in the Studio, then delete again. "
    "To delete them anyway, send include_endpoints=true "
    "(`dagnam audit delete <audit-dir> --include-endpoints`, dagnam 0.18.0 or later)."
)
SERVING_ENDPOINTS: JsonArray = [
    {
        "id": "3f2a0c1e-0000-4000-8000-000000000001",
        "name": "tickets-ft-small",
        "status": "running",
        "last_request_at": "2026-10-07T14:03:11Z",
    },
    {
        "id": "9b1c0c1e-0000-4000-8000-000000000002",
        "name": "tickets-ft-base",
        "status": "deploying",
        "last_request_at": None,
    },
]
SERVING_BODY: JsonObject = {
    "detail": SERVING_SENTENCE,
    "error": "endpoints_serving",
    "endpoints": SERVING_ENDPOINTS,
}


@pytest.mark.parametrize(
    "body",
    [
        SERVING_BODY,
        {
            "detail": {
                "error": "endpoints_serving",
                "message": SERVING_SENTENCE,
                "endpoints": SERVING_ENDPOINTS,
            }
        },
    ],
    ids=["as the platform sends it", "nested under detail"],
)
def test_a_409_saying_endpoints_are_serving_is_typed_with_the_endpoints_it_names(
    client: DagnamClient, rmock: RequestsMocker, body: JsonObject
) -> None:
    rmock.delete(f"{AUDITS}/a1", json=body, status_code=409)
    with pytest.raises(EndpointsServingError) as exc:
        client.delete_audit("a1")
    assert isinstance(exc.value, APIError)  # existing `except APIError` handlers still catch it
    assert not isinstance(exc.value, TeardownInProgressError)  # and nothing waits on it
    assert exc.value.status_code == 409
    assert exc.value.message == SERVING_SENTENCE
    assert exc.value.endpoints == SERVING_ENDPOINTS
    assert len(rmock.request_history) == 1  # a refusal is never retried


def test_an_endpoints_list_that_is_not_a_list_of_objects_is_read_as_far_as_it_goes(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.delete(
        f"{AUDITS}/a1",
        json={"detail": "x", "error": "endpoints_serving", "endpoints": [SERVING_ENDPOINTS[0], 7]},
        status_code=409,
    )
    with pytest.raises(EndpointsServingError) as exc:
        client.delete_audit("a1")
    assert exc.value.endpoints == [SERVING_ENDPOINTS[0]]
    rmock.delete(
        f"{AUDITS}/a2", json={"error": "endpoints_serving", "endpoints": "none"}, status_code=409
    )
    with pytest.raises(EndpointsServingError) as bare:
        client.delete_audit("a2")
    assert (bare.value.endpoints, bare.value.message) == ([], "endpoints_serving")


def test_the_serving_marker_on_any_other_status_is_not_the_refusal(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.delete(f"{AUDITS}/a1", json=SERVING_BODY, status_code=503)
    with pytest.raises(APIError) as exc:
        client.delete_audit("a1")
    assert not isinstance(exc.value, EndpointsServingError)


def test_include_endpoints_is_sent_as_a_query_parameter_only_when_asked_for(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.delete(f"{AUDITS}/a1", json=RECEIPT)
    client.delete_audit("a1")
    assert rmock.last_request.qs == {}
    client.delete_audit("a1", include_endpoints=True)
    assert rmock.last_request.qs == {"include_endpoints": ["true"]}
    client.delete_audit("a1", include_endpoints=False)
    assert rmock.last_request.qs == {}


def test_cancel_and_delete_audit_return_the_receipt(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{AUDITS}/a1/cancel", json=RECEIPT)
    rmock.delete(f"{AUDITS}/a1", json=RECEIPT)
    assert client.cancel_audit("a1") == RECEIPT
    assert client.delete_audit("a1") == RECEIPT


def test_an_id_with_a_slash_cannot_escape_its_path(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{AUDITS}/a%2F..%2Fx/halt", json={"id": "a1"})
    client.halt_audit("a/../x", "error")
    assert rmock.last_request.path.lower() == "/api/v1/audits/a%2f..%2fx/halt"


def test_a_key_without_the_write_scope_is_a_uniform_404(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(AUDITS, status_code=404, json={"detail": "Audit not found"})
    with pytest.raises(APIError) as exc:
        client.create_audit({})
    assert exc.value.status_code == 404


def test_an_expired_key_is_an_auth_error(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.delete(f"{AUDITS}/a1", status_code=401, json={"detail": "nope"})
    with pytest.raises(AuthError):
        client.delete_audit("a1")


def test_create_audit_retries_a_blip_into_the_same_idempotency_key(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    """A read timeout after the server created the audit must not open a second one."""
    client._sleep = lambda _s: None
    rmock.post(
        AUDITS, [{"status_code": 503, "json": {}}, {"json": {"id": "a1"}, "status_code": 201}]
    )
    assert client.create_audit({"project_id": "p1"}) == {"id": "a1"}
    keys = [r.headers.get("Idempotency-Key") for r in rmock.request_history]
    assert len(keys) == 2
    assert keys[0] is not None
    assert keys[0] == keys[1]


def test_the_other_audit_routes_carry_no_idempotency_key(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(f"{AUDITS}/a1/candidates", json={"id": "c1"}, status_code=201)
    client.create_audit_candidate("a1", {"kind": "head_tune"})
    assert "Idempotency-Key" not in rmock.last_request.headers


def test_resume_and_read_an_audit(client: DagnamClient, rmock: RequestsMocker) -> None:
    """A run un-halts its audit once, up front; a read settles what a uniform 404 meant."""
    rmock.post(f"{AUDITS}/a1/resume", json={"id": "a1", "status": "running"})
    rmock.get(f"{AUDITS}/a1", json={"id": "a1", "status": "halted"})
    assert client.resume_audit("a1")["status"] == "running"
    assert client.get_audit("a1")["status"] == "halted"


def test_get_platform_build_reads_the_unversioned_health_route(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    """The route sits on the API base, beside the versioned ones, and is read as it arrives."""
    rmock.get(f"{API}/health/build", json={"revision": "abc", "version": "1", "contracts": "0.4.0"})
    assert client.get_platform_build() == {"revision": "abc", "version": "1", "contracts": "0.4.0"}
    assert rmock.last_request.path == "/health/build"


def test_get_platform_build_on_a_platform_without_the_route_is_a_404(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.get(f"{API}/health/build", status_code=404, json={"detail": "Not Found"})
    with pytest.raises(APIError) as exc:
        client.get_platform_build()
    assert exc.value.status_code == 404


@pytest.mark.parametrize(
    ("call", "method", "path"),
    [
        (lambda c: c.cancel_audit("a1"), "post", "/api/v1/audits/a1/cancel"),
        (lambda c: c.delete_audit("a1"), "delete", "/api/v1/audits/a1"),
    ],
)
def test_the_accounts_walk_is_sent_once_with_a_long_read_timeout(
    client: DagnamClient,
    rmock: RequestsMocker,
    call: Callable[[DagnamClient], JsonObject],
    method: str,
    path: str,
) -> None:
    """The walk is one synchronous request over every artifact: it is not retried into a second.

    A retry of a DELETE that timed out would start a second walk while the first may still be
    running; a failure is the caller's to handle, and asking again is what finishes it.
    """
    getattr(rmock, method)(f"{API}{path}", status_code=503, json={"detail": "down"})

    with pytest.raises(APIError) as exc:
        call(client)

    assert exc.value.status_code == 503
    assert rmock.call_count == 1
    assert rmock.last_request.timeout == WALK_TIMEOUT
    assert WALK_TIMEOUT[1] >= 120  # the walk can outlast an ordinary request's 30 seconds


def test_the_other_audit_routes_keep_the_ordinary_timeout_and_retries(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.get(f"{AUDITS}/a1", [{"status_code": 503}, {"json": {"id": "a1"}}])
    assert client.get_audit("a1") == {"id": "a1"}
    assert rmock.call_count == 2
    assert rmock.last_request.timeout == DEFAULT_TIMEOUT


def test_create_audit_with_a_nonce_replays_for_that_nonce_and_is_read_back(
    rmock: RequestsMocker,
) -> None:
    """The body of an audit is keyed with the directory's nonce, and a replay is read back."""
    created: JsonObject = {"id": "a1"}
    seen: list[str | None] = []

    def create(request: Any, context: Any) -> JsonObject:
        key = request.headers.get("Idempotency-Key")
        if key in seen:
            context.headers["Idempotency-Replayed"] = "true"
        seen.append(key)
        return created

    rmock.post(AUDITS, json=create)
    rmock.get(f"{AUDITS}/a1", json={"id": "a1", "status": "running"})
    first, second = DagnamClient(API, "k"), DagnamClient(API, "k")
    first.resume_creates = second.resume_creates = True

    first.create_audit({"project_id": "p1"}, resume_nonce="nonce-a")
    again = second.create_audit({"project_id": "p1"}, resume_nonce="nonce-a")
    other = second.create_audit({"project_id": "p1"}, resume_nonce="nonce-b")

    assert again == other == created
    assert seen[0] == seen[1] != seen[2]  # the same nonce is the same key, another nonce is not
    assert [r.path for r in rmock.request_history if r.method == "GET"] == ["/api/v1/audits/a1"]
