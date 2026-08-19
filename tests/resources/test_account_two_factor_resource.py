"""Unit tests for the public ``dagnam.account`` two-factor helpers.

These pin DELEGATION -- that each public function forwards to the client method
it names, with the caller's arguments intact. The wire contract is pinned in
tests/core/client/test_account_two_factor.py; duplicating it here would mean two
places to update and one of them going stale.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dagnam import account
from dagnam._core.client import DagnamClient

ENROLLMENT = {"secret": "S", "qr_code_uri": "otpauth://x", "backup_codes": ["1", "2"]}


@pytest.mark.parametrize("enabled", [True, False])
def test_two_factor_enabled_delegates(enabled: bool) -> None:
    c = MagicMock(spec=DagnamClient, two_factor_enabled=MagicMock(return_value=enabled))
    assert account.two_factor_enabled(client=c) is enabled
    c.two_factor_enabled.assert_called_once_with()


def test_enable_two_factor_delegates_and_returns_the_material() -> None:
    c = MagicMock(spec=DagnamClient, enable_two_factor=MagicMock(return_value=ENROLLMENT))
    assert account.enable_two_factor("Passw0rd!", client=c) == ENROLLMENT
    c.enable_two_factor.assert_called_once_with("Passw0rd!")


def test_verify_two_factor_delegates() -> None:
    payload = {"message": "Two-factor authentication enabled successfully"}
    c = MagicMock(spec=DagnamClient, verify_two_factor=MagicMock(return_value=payload))
    assert account.verify_two_factor("123456", client=c) == payload
    c.verify_two_factor.assert_called_once_with("123456")


def test_disable_two_factor_delegates() -> None:
    payload = {"message": "Two-factor authentication disabled"}
    c = MagicMock(spec=DagnamClient, disable_two_factor=MagicMock(return_value=payload))
    assert account.disable_two_factor("Passw0rd!", client=c) == payload
    c.disable_two_factor.assert_called_once_with("Passw0rd!")
