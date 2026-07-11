"""Wire-level coverage for dataset preview / update / delete / roles (async client).

Async mirror of ``tests/core/client/test_dataset_manage.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import parse_qsl

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.exceptions import DatasetNotFoundError

if TYPE_CHECKING:
    from tests.typing_helpers import RespxMockRouter

PREVIEW = "/api/v1/datasets/ds-1/preview"
DATASET = "/api/v1/datasets/ds-1"
ROLES = "/api/v1/datasets/ds-1/roles"

pytestmark = pytest.mark.anyio


# --------------------------------------------------------------------- preview


async def test_preview_dataset_returns_object_and_sends_rows(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    payload = {"samples": [{"a": 1}], "statistics": {"count": 1}}
    route = mock.get(PREVIEW).mock(return_value=httpx.Response(200, json=payload))
    result = await client.preview_dataset("ds-1", rows=5)
    assert result == payload
    assert route.calls[0].request.url.params["rows"] == "5"


async def test_preview_dataset_404_raises(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get(PREVIEW).mock(return_value=httpx.Response(404, json={"detail": "nope"}))
    with pytest.raises(DatasetNotFoundError):
        await client.preview_dataset("ds-1")


# ---------------------------------------------------------------------- update


async def test_update_dataset_sends_only_provided_fields(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.put(DATASET).mock(return_value=httpx.Response(200, json={"id": "ds-1"}))
    await client.update_dataset("ds-1", name="new", visibility="public")
    body = dict(parse_qsl(route.calls[0].request.content.decode()))
    assert body == {"name": "new", "visibility": "public"}


async def test_update_dataset_requires_a_field(client: AsyncDagnamClient) -> None:
    with pytest.raises(ValueError, match="at least one of"):
        await client.update_dataset("ds-1")


async def test_update_dataset_description_only(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.put(DATASET).mock(return_value=httpx.Response(200, json={"id": "ds-1"}))
    await client.update_dataset("ds-1", description="desc")
    assert dict(parse_qsl(route.calls[0].request.content.decode())) == {"description": "desc"}


# ---------------------------------------------------------------------- delete


async def test_delete_dataset_returns_none(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.delete(DATASET).mock(return_value=httpx.Response(204))
    assert await client.delete_dataset("ds-1") is None


# ----------------------------------------------------------------------- roles


async def test_update_dataset_roles_sends_json(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.patch(ROLES).mock(
        return_value=httpx.Response(200, json={"column_roles": {"a": "target"}})
    )
    await client.update_dataset_roles("ds-1", {"a": "target"}, task_type_hint="classification")
    assert route.calls[0].request.read() == (
        b'{"column_roles":{"a":"target"},"task_type_hint":"classification"}'
    )


async def test_update_dataset_roles_default_hint_is_null(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.patch(ROLES).mock(
        return_value=httpx.Response(200, json={"column_roles": {"a": "target"}})
    )
    await client.update_dataset_roles("ds-1", {"a": "target"})
    assert b'"task_type_hint":null' in route.calls[0].request.read()
