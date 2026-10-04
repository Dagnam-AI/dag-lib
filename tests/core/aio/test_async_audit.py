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


async def test_create_audit(client: AsyncDagnamClient, mock: respx.MockRouter) -> None:
    route = mock.post(AUDITS).mock(return_value=httpx.Response(201, json={"id": "a1"}))
    assert await client.create_audit({"project_id": "p1"}) == {"id": "a1"}
    assert json.loads(route.calls.last.request.read()) == {"project_id": "p1"}


async def test_create_audit_candidate(client: AsyncDagnamClient, mock: respx.MockRouter) -> None:
    mock.post(f"{AUDITS}/a1/candidates").mock(return_value=httpx.Response(201, json={"id": "c1"}))
    assert await client.create_audit_candidate("a1", {"kind": "sft_small"}) == {"id": "c1"}


async def test_claim_audit_resources(client: AsyncDagnamClient, mock: respx.MockRouter) -> None:
    route = mock.post(f"{AUDITS}/a1/claims").mock(
        return_value=httpx.Response(200, json={"entries": []})
    )
    assert await client.claim_audit_resources("a1", [{"kind": "dataset", "id": "d1"}]) == {
        "entries": []
    }
    assert json.loads(route.calls.last.request.read()) == {
        "entries": [{"kind": "dataset", "id": "d1"}]
    }


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


async def test_the_async_client_does_not_tear_an_audit_down(client: AsyncDagnamClient) -> None:
    """A teardown is the platform's one walk plus what the CLI does with its receipt: sync only."""
    assert not hasattr(client, "cancel_audit")
    assert not hasattr(client, "delete_audit")


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
    mock.post(f"{AUDITS}/a1/halt").mock(return_value=httpx.Response(401, json={"detail": "nope"}))
    with pytest.raises(AuthError):
        await client.halt_audit("a1", "error")


async def test_create_audit_sends_an_idempotency_key(
    client: AsyncDagnamClient, mock: respx.MockRouter
) -> None:
    """The async twin keys the create the same way."""
    route = mock.post(AUDITS).mock(return_value=httpx.Response(201, json={"id": "a1"}))
    await client.create_audit({"project_id": "p1"})
    assert route.calls.last.request.headers.get("Idempotency-Key")


async def test_resume_and_read_an_audit(client: AsyncDagnamClient, mock: respx.MockRouter) -> None:
    mock.post(f"{AUDITS}/a1/resume").mock(
        return_value=httpx.Response(200, json={"id": "a1", "status": "running"})
    )
    mock.get(f"{AUDITS}/a1").mock(return_value=httpx.Response(200, json={"status": "halted"}))
    assert (await client.resume_audit("a1"))["status"] == "running"
    assert (await client.get_audit("a1"))["status"] == "halted"
