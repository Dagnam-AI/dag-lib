"""CLI coverage for `dagnam profile show`."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, StrCapture


def test_profile_show_table(run_cli: CliRunner, capsys: StrCapture) -> None:
    payload = {
        "display_name": "Ada Lovelace",
        "bio": "Mathematician",
        "avatar_url": "/uploads/avatars/x.png",
        "role": "user",
        "join_date": "2020-01-01T00:00:00Z",
        "models": [
            {"name": "model-a", "stars_count": 3, "downloads_count": 10},
        ],
        "stats": {"models_published": 1, "stars_received": 3, "total_downloads": 10},
    }
    fake = SimpleNamespace(get_public_profile=mock.Mock(return_value=payload))
    with mock.patch("dagnam.account", fake):
        run_cli(["profile", "show", "ada"])
    fake.get_public_profile.assert_called_once_with("ada")
    out = capsys.readouterr().out
    assert "Ada Lovelace" in out
    assert "model-a" in out


def test_profile_show_table_no_models(run_cli: CliRunner, capsys: StrCapture) -> None:
    payload = {
        "display_name": "Ada Lovelace",
        "bio": None,
        "avatar_url": None,
        "role": "user",
        "join_date": "2020-01-01T00:00:00Z",
        "models": [],
        "stats": {"models_published": 0, "stars_received": 0, "total_downloads": 0},
    }
    fake = SimpleNamespace(get_public_profile=mock.Mock(return_value=payload))
    with mock.patch("dagnam.account", fake):
        run_cli(["profile", "show", "ada"])
    out = capsys.readouterr().out
    assert "Ada Lovelace" in out
    assert "Models: 0" in out


def test_profile_show_table_minimal_payload(run_cli: CliRunner, capsys: StrCapture) -> None:
    # No bio/avatar_url/join_date at all -> every optional-field branch is False.
    payload = {
        "display_name": "Anon",
        "role": "user",
        "models": [],
        "stats": {},
    }
    fake = SimpleNamespace(get_public_profile=mock.Mock(return_value=payload))
    with mock.patch("dagnam.account", fake):
        run_cli(["profile", "show", "anon"])
    out = capsys.readouterr().out
    assert "Anon" in out
    assert "Joined" not in out
    assert "Bio" not in out
    assert "Avatar" not in out


def test_profile_show_json(run_cli: CliRunner, capsys: StrCapture) -> None:
    payload = {"display_name": "Ada Lovelace"}
    fake = SimpleNamespace(get_public_profile=mock.Mock(return_value=payload))
    with mock.patch("dagnam.account", fake):
        run_cli(["profile", "show", "ada", "--json"])
    assert json.loads(capsys.readouterr().out) == payload


def test_profile_show_not_found_exits_1(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import APIError

    fake = SimpleNamespace(get_public_profile=mock.Mock(side_effect=APIError(404, "not found")))
    with mock.patch("dagnam.account", fake):
        assert run_cli(["profile", "show", "ghost"]) == 1
    assert "not found" in capsys.readouterr().err
