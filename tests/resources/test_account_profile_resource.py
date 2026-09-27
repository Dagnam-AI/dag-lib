"""Unit tests for dagnam.account.get_public_profile."""

from __future__ import annotations

from unittest.mock import MagicMock

from dagnam import account
from dagnam._core.client import DagnamClient


def test_get_public_profile_delegates() -> None:
    payload = {"display_name": "Ada"}
    c = MagicMock(spec=DagnamClient, get_public_profile=MagicMock(return_value=payload))
    assert account.get_public_profile("ada", client=c) == payload
    c.get_public_profile.assert_called_once_with("ada")
