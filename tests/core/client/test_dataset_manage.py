"""Wire-level coverage for dataset preview / update / delete / roles (sync client).

Covers ``DatasetsClientMixin.preview_dataset/update_dataset/delete_dataset/
update_dataset_roles`` including every connect/timeout/not-found error branch.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl

import pytest
import requests

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import APIError, DatasetNotFoundError

if TYPE_CHECKING:
    from tests.typing_helpers import RequestsMocker

API = "https://api.test"
PREVIEW = f"{API}/api/v1/datasets/ds-1/preview"
DATASET = f"{API}/api/v1/datasets/ds-1"
ROLES = f"{API}/api/v1/datasets/ds-1/roles"

# (register-matcher, call) pairs exercising the four new methods uniformly. The
# register callable binds the right HTTP verb so the shared error-branch tests
# below stay verb-agnostic.
OPS: list[tuple[Callable[..., object], Callable[[DagnamClient], object]]] = [
    (lambda m, **kw: m.get(PREVIEW, **kw), lambda c: c.preview_dataset("ds-1")),
    (lambda m, **kw: m.put(DATASET, **kw), lambda c: c.update_dataset("ds-1", name="x")),
    (lambda m, **kw: m.delete(DATASET, **kw), lambda c: c.delete_dataset("ds-1")),
    (lambda m, **kw: m.patch(ROLES, **kw), lambda c: c.update_dataset_roles("ds-1", {"a": "t"})),
]


# --------------------------------------------------------------------- preview


def test_preview_dataset_returns_object_and_sends_rows(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    payload = {"samples": [{"a": 1}], "statistics": {"count": 1}}
    rmock.get(PREVIEW, json=payload)
    result = client.preview_dataset("ds-1", rows=5)
    assert result == payload
    assert rmock.last_request.qs == {"rows": ["5"]}


def test_preview_dataset_default_rows(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(PREVIEW, json={"samples": [], "statistics": {}})
    client.preview_dataset("ds-1")
    assert rmock.last_request.qs == {"rows": ["10"]}


# ---------------------------------------------------------------------- update


def test_update_dataset_sends_only_provided_fields(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.put(DATASET, json={"id": "ds-1", "name": "new"})
    result = client.update_dataset("ds-1", name="new", visibility="public")
    assert result == {"id": "ds-1", "name": "new"}
    body = dict(parse_qsl(rmock.last_request.text or ""))
    assert body == {"name": "new", "visibility": "public"}


def test_update_dataset_requires_a_field(client: DagnamClient) -> None:
    with pytest.raises(ValueError, match="at least one of"):
        client.update_dataset("ds-1")


def test_update_dataset_description_only(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.put(DATASET, json={"id": "ds-1"})
    client.update_dataset("ds-1", description="desc")
    assert dict(parse_qsl(rmock.last_request.text or "")) == {"description": "desc"}


# ---------------------------------------------------------------------- delete


def test_delete_dataset_returns_none(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.delete(DATASET, status_code=204)
    assert client.delete_dataset("ds-1") is None


# ----------------------------------------------------------------------- roles


def test_update_dataset_roles_sends_json(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.patch(ROLES, json={"column_roles": {"a": "target"}, "roles_confirmed": True})
    result = client.update_dataset_roles("ds-1", {"a": "target"}, task_type_hint="classification")
    assert result["roles_confirmed"] is True
    assert rmock.last_request.json() == {
        "column_roles": {"a": "target"},
        "task_type_hint": "classification",
    }


def test_update_dataset_roles_default_hint_is_null(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.patch(ROLES, json={"column_roles": {"a": "target"}})
    client.update_dataset_roles("ds-1", {"a": "target"})
    assert rmock.last_request.json()["task_type_hint"] is None


def test_update_dataset_roles_403_system_dataset_raises_apierror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.patch(ROLES, status_code=403, json={"detail": "system dataset"})
    with pytest.raises(APIError) as exc_info:
        client.update_dataset_roles("ds-1", {"a": "target"})
    assert exc_info.value.status_code == 403


def test_update_dataset_roles_422_invalid_role_raises_apierror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.patch(ROLES, status_code=422, json={"detail": "bad role"})
    with pytest.raises(APIError) as exc_info:
        client.update_dataset_roles("ds-1", {"a": "nope"})
    assert exc_info.value.status_code == 422


# ------------------------------------------------------------- shared error paths


@pytest.mark.parametrize(("register", "call"), OPS)
def test_404_raises_dataset_not_found(
    client: DagnamClient,
    rmock: RequestsMocker,
    register: Callable[..., object],
    call: Callable[[DagnamClient], object],
) -> None:
    register(rmock, status_code=404, json={"detail": "nope"})
    with pytest.raises(DatasetNotFoundError):
        call(client)


@pytest.mark.parametrize(("register", "call"), OPS)
def test_connection_error_wrapped(
    client: DagnamClient,
    rmock: RequestsMocker,
    register: Callable[..., object],
    call: Callable[[DagnamClient], object],
) -> None:
    # Transport errors are now mapped centrally in ``_request`` to
    # ``APIError(0, "Request failed: ...")``; retryable verbs exhaust the retry
    # budget first, so the sleep is stubbed to keep the test fast.
    client._sleep = lambda _s: None
    register(rmock, exc=requests.exceptions.ConnectionError("down"))
    with pytest.raises(APIError, match="Request failed"):
        call(client)


@pytest.mark.parametrize(("register", "call"), OPS)
def test_timeout_wrapped(
    client: DagnamClient,
    rmock: RequestsMocker,
    register: Callable[..., object],
    call: Callable[[DagnamClient], object],
) -> None:
    client._sleep = lambda _s: None
    register(rmock, exc=requests.exceptions.Timeout("slow"))
    with pytest.raises(APIError, match="Request failed"):
        call(client)
