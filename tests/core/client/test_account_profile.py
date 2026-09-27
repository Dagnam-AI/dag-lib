"""Wire-level coverage for the sync ``get_public_profile`` client method."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import APIError

if TYPE_CHECKING:
    from tests.typing_helpers import RequestsMocker

API = "https://api.test"


def test_get_public_profile(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/users/ada/profile", json={"display_name": "Ada"})
    result = client.get_public_profile("ada")
    assert result["display_name"] == "Ada"


def test_get_public_profile_quotes_username(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/users/a%2Fb/profile", json={"display_name": "x"})
    result = client.get_public_profile("a/b")
    assert result["display_name"] == "x"


def test_get_public_profile_404_raises_apierror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.get(f"{API}/api/v1/users/ghost/profile", status_code=404, json={"detail": "not found"})
    with pytest.raises(APIError):
        client.get_public_profile("ghost")
