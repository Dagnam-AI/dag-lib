"""Unit tests for ``dagnam.account.register``, the bootstrap orchestrator.

Patches ``DagnamClient`` itself (not just its methods) so the test can assert
the constructor sees the unauthenticated client (empty API key), then the
session-token-authenticated client, in the right order - without ever hitting
the network.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from tests.typing_helpers import PytestMonkeyPatch

from dagnam import account
from dagnam._core.client import DagnamClient

KEY_OBJ = {
    "id": "k1",
    "name": "dagnam-cli",
    "key": "dgk_abcSECRET",
    "key_prefix": "dgk_abc",
    "permissions": ["read", "write"],
    "expires_at": None,
    "created_at": "2026-01-01T00:00:00",
}


def test_register_orchestrates_register_login_and_create_key(
    monkeypatch: PytestMonkeyPatch,
) -> None:
    call_order: list[str] = []
    constructed: list[tuple[str, str]] = []

    unauth = MagicMock(spec=DagnamClient)
    unauth.register.side_effect = lambda *_a, **_k: call_order.append("register")

    def _login(*_a: object, **_k: object) -> str:
        call_order.append("login")
        return "tok-123"

    unauth.login_for_bootstrap.side_effect = _login

    authed = MagicMock(spec=DagnamClient)

    def _create_key(*_a: object, **_k: object) -> object:
        call_order.append("create_key")
        return KEY_OBJ

    authed.create_api_key.side_effect = _create_key

    def _factory(url: str, api_key: str) -> MagicMock:
        constructed.append((url, api_key))
        return unauth if api_key == "" else authed

    monkeypatch.setattr("dagnam.resources.account.DagnamClient", _factory)

    result = account.register("a@b.c", "Secret123!", api_url="https://api.test")

    assert result == KEY_OBJ
    assert call_order == ["register", "login", "create_key"]
    assert constructed == [("https://api.test", ""), ("https://api.test", "tok-123")]
    unauth.register.assert_called_once_with("a@b.c", "Secret123!")
    unauth.login_for_bootstrap.assert_called_once_with("a@b.c", "Secret123!")
    authed.create_api_key.assert_called_once_with(
        name=account.DEFAULT_KEY_NAME, scopes=account.DEFAULT_SDK_SCOPES
    )


def test_register_resolves_default_api_url_when_not_overridden(
    monkeypatch: PytestMonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dagnam.resources.account.get_api_url", lambda override=None: "https://resolved.test"
    )
    constructed: list[tuple[str, str]] = []

    def _factory(url: str, api_key: str) -> MagicMock:
        constructed.append((url, api_key))
        client = MagicMock(spec=DagnamClient)
        client.login_for_bootstrap.return_value = "tok"
        client.create_api_key.return_value = KEY_OBJ
        return client

    monkeypatch.setattr("dagnam.resources.account.DagnamClient", _factory)

    result = account.register("a@b.c", "Secret123!")

    assert result == KEY_OBJ
    assert constructed[0][0] == "https://resolved.test"
    assert constructed[1][0] == "https://resolved.test"


def test_register_default_constants() -> None:
    assert account.DEFAULT_SDK_SCOPES == ("read", "write")
    assert account.DEFAULT_KEY_NAME == "dagnam-cli"
