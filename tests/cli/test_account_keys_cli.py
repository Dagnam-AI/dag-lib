"""CLI coverage for the top-level `dagnam keys` command group."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, StrCapture

SECRET_KEY = "dgk_abcdEFGH12345678SECRET"

CREATED = {
    "id": "key-1",
    "name": "ci-key",
    "key_prefix": "dgk_abcd",
    "permissions": ["read"],
    "usage_count": 0,
    "last_used_at": None,
    "expires_at": None,
    "created_at": "2026-01-01T00:00:00",
    "key": SECRET_KEY,
}

LISTED = [{k: v for k, v in CREATED.items() if k != "key"}]

# --------------------------------------------------------------------- keys create


def test_keys_create_prints_secret_exactly_once(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(create_api_key=mock.Mock(return_value=CREATED))
    with mock.patch("dagnam.account", fake):
        run_cli(["keys", "create", "--name", "ci-key"])
    fake.create_api_key.assert_called_once_with("ci-key", None, None)
    out = capsys.readouterr().out
    assert out.count(SECRET_KEY) == 1
    assert "Store this secret now" in out
    assert "ci-key" in out


def test_keys_create_forwards_scopes_and_expiry(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(create_api_key=mock.Mock(return_value=CREATED))
    with mock.patch("dagnam.account", fake):
        run_cli(
            [
                "keys",
                "create",
                "--name",
                "ci-key",
                "--scope",
                "read",
                "--scope",
                "write",
                "--expires-in-days",
                "30",
            ]
        )
    fake.create_api_key.assert_called_once_with("ci-key", ["read", "write"], 30)


def test_keys_create_json_includes_key(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(create_api_key=mock.Mock(return_value=CREATED))
    with mock.patch("dagnam.account", fake):
        run_cli(["keys", "create", "--name", "ci-key", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["key"] == SECRET_KEY


def test_keys_create_writes_output_file(run_cli: CliRunner, tmp_path: Path) -> None:
    fake = SimpleNamespace(create_api_key=mock.Mock(return_value=CREATED))
    out_path = tmp_path / "key.json"
    with mock.patch("dagnam.account", fake):
        run_cli(["keys", "create", "--name", "ci-key", "--output", str(out_path)])
    assert json.loads(out_path.read_text())["key"] == SECRET_KEY


def test_keys_create_empty_name_exits_1_no_http_call(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    fake = SimpleNamespace(create_api_key=mock.Mock(return_value=CREATED))
    with (
        mock.patch("dagnam.account", fake),
        pytest.raises(SystemExit) as exc_info,
    ):
        run_cli(["keys", "create", "--name", "   "])
    assert exc_info.value.code == 1
    assert "Key name cannot be empty" in capsys.readouterr().err
    fake.create_api_key.assert_not_called()


def test_keys_create_degrades_without_key_field(run_cli: CliRunner, capsys: StrCapture) -> None:
    payload = {"id": "key-1", "name": "ci-key"}
    fake = SimpleNamespace(create_api_key=mock.Mock(return_value=payload))
    with mock.patch("dagnam.account", fake):
        run_cli(["keys", "create", "--name", "ci-key"])
    out = capsys.readouterr().out
    assert "Store this secret now" not in out
    assert "ci-key" in out


def test_keys_create_secret_only_no_metadata_table(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(create_api_key=mock.Mock(return_value={"key": SECRET_KEY}))
    with mock.patch("dagnam.account", fake):
        run_cli(["keys", "create", "--name", "ci-key"])
    out = capsys.readouterr().out
    assert out.count(SECRET_KEY) == 1
    assert "No data returned." not in out


def test_keys_create_no_data_returned(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(create_api_key=mock.Mock(return_value={}))
    with mock.patch("dagnam.account", fake):
        run_cli(["keys", "create", "--name", "ci-key"])
    out = capsys.readouterr().out
    assert "No data returned." in out
    assert "Store this secret now" not in out


def test_keys_create_apierror_exits_1(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import QuotaExceededError

    fake = SimpleNamespace(
        create_api_key=mock.Mock(side_effect=QuotaExceededError("Plan limit reached"))
    )
    with mock.patch("dagnam.account", fake):
        assert run_cli(["keys", "create", "--name", "ci-key"]) == 1
    assert "Plan limit reached" in capsys.readouterr().err


# ---------------------------------------------------------------------- keys list


def test_keys_list_table(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(list_api_keys=mock.Mock(return_value=LISTED))
    with mock.patch("dagnam.account", fake):
        run_cli(["keys", "list"])
    fake.list_api_keys.assert_called_once_with()
    out = capsys.readouterr().out
    assert "ci-key" in out
    assert "key-1" in out
    assert SECRET_KEY not in out


def test_keys_list_empty_table(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(list_api_keys=mock.Mock(return_value=[]))
    with mock.patch("dagnam.account", fake):
        run_cli(["keys", "list"])
    assert "No API keys." in capsys.readouterr().out


def test_keys_list_json(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(list_api_keys=mock.Mock(return_value=LISTED))
    with mock.patch("dagnam.account", fake):
        run_cli(["keys", "list", "--json"])
    assert json.loads(capsys.readouterr().out) == LISTED


def test_keys_list_apierror_exits_1(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import APIError

    fake = SimpleNamespace(list_api_keys=mock.Mock(side_effect=APIError(401, "invalid key")))
    with mock.patch("dagnam.account", fake):
        assert run_cli(["keys", "list"]) == 1
    assert "invalid key" in capsys.readouterr().err


# -------------------------------------------------------------------- keys revoke


def test_keys_revoke_calls_client(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(revoke_api_key=mock.Mock(return_value=None))
    with mock.patch("dagnam.account", fake):
        run_cli(["keys", "revoke", "key-1"])
    fake.revoke_api_key.assert_called_once_with("key-1")
    assert "Revoked API key key-1." in capsys.readouterr().out


def test_keys_revoke_not_found_exits_1(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import APIError

    fake = SimpleNamespace(revoke_api_key=mock.Mock(side_effect=APIError(404, "API key not found")))
    with mock.patch("dagnam.account", fake):
        assert run_cli(["keys", "revoke", "ghost"]) == 1
    assert "API key not found" in capsys.readouterr().err


def test_keys_revoke_empty_id_exits_1_no_http_call(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(revoke_api_key=mock.Mock(return_value=None))
    with (
        mock.patch("dagnam.account", fake),
        pytest.raises(SystemExit) as exc_info,
    ):
        run_cli(["keys", "revoke", "   "])
    assert exc_info.value.code == 1
    assert "Key id cannot be empty" in capsys.readouterr().err
    fake.revoke_api_key.assert_not_called()
