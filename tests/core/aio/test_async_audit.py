"""Wire-level coverage for the async workload-audit publish client mixin."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import pytest

from dagnam._core.exceptions import APIError, AuthError
from dagnam._types import JsonObject

if TYPE_CHECKING:
    import respx

    from dagnam._core.aio import AsyncDagnamClient

pytestmark = pytest.mark.anyio

AUDITS = "/api/v1/audits"
RECEIPT = {"schema": "dagnam.audit.deleted/1", "deleted_at": "2026-09-07T10:00:00Z", "entries": []}


async def test_create_audit(client: AsyncDagnamClient, mock: respx.MockRouter) -> None:
    route = mock.post(AUDITS).mock(return_value=httpx.Response(201, json={"id": "a1"}))
    assert await client.create_audit({"project_id": "p1"}) == {"id": "a1"}
    assert json.loads(route.calls.last.request.read()) == {"project_id": "p1"}


async def test_create_audit_candidate(client: AsyncDagnamClient, mock: respx.MockRouter) -> None:
    mock.post(f"{AUDITS}/a1/candidates").mock(return_value=httpx.Response(201, json={"id": "c1"}))
    assert await client.create_audit_candidate("a1", {"kind": "sft_small"}) == {"id": "c1"}


async def test_patch_audit_candidate(client: AsyncDagnamClient, mock: respx.MockRouter) -> None:
    mock.patch(f"{AUDITS}/a1/candidates/c1").mock(
        return_value=httpx.Response(200, json={"id": "c1", "status": "scored"})
    )
    body: JsonObject = {"step": "replay_and_score", "status": "scored"}
    result = await client.patch_audit_candidate("a1", "c1", body)
    assert result["status"] == "scored"


async def test_halt_audit(client: AsyncDagnamClient, mock: respx.MockRouter) -> None:
    route = mock.post(f"{AUDITS}/a1/halt").mock(
        return_value=httpx.Response(200, json={"id": "a1", "status": "halted"})
    )
    assert (await client.halt_audit("a1", "cancelled"))["status"] == "halted"
    assert json.loads(route.calls.last.request.read()) == {"reason": "cancelled"}


async def test_cancel_and_delete_audit_return_the_receipt(
    client: AsyncDagnamClient, mock: respx.MockRouter
) -> None:
    mock.post(f"{AUDITS}/a1/cancel").mock(return_value=httpx.Response(200, json=RECEIPT))
    mock.delete(f"{AUDITS}/a1").mock(return_value=httpx.Response(200, json=RECEIPT))
    assert await client.cancel_audit("a1") == RECEIPT
    assert await client.delete_audit("a1") == RECEIPT


async def test_an_id_with_a_slash_cannot_escape_its_path(
    client: AsyncDagnamClient, mock: respx.MockRouter
) -> None:
    route = mock.post(f"{AUDITS}/a%2F..%2Fx/halt").mock(
        return_value=httpx.Response(200, json={"id": "a1"})
    )
    await client.halt_audit("a/../x", "error")
    assert route.calls.last.request.url.raw_path.endswith(b"/api/v1/audits/a%2F..%2Fx/halt")


async def test_a_key_without_the_write_scope_is_a_uniform_404(
    client: AsyncDagnamClient, mock: respx.MockRouter
) -> None:
    mock.post(AUDITS).mock(return_value=httpx.Response(404, json={"detail": "Audit not found"}))
    with pytest.raises(APIError) as exc:
        await client.create_audit({})
    assert exc.value.status_code == 404


async def test_an_expired_key_is_an_auth_error(
    client: AsyncDagnamClient, mock: respx.MockRouter
) -> None:
    mock.delete(f"{AUDITS}/a1").mock(return_value=httpx.Response(401, json={"detail": "nope"}))
    with pytest.raises(AuthError):
        await client.delete_audit("a1")
