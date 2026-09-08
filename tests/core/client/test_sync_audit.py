"""Wire-level coverage for the sync workload-audit publish client mixin."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import APIError, AuthError
from dagnam._types import JsonObject

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
