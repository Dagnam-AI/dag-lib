"""CLI coverage for `dagnam account profile` and `dagnam profile show`."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, StrCapture

# ---------------------------------------------------------------- profile get


def test_profile_get_table(run_cli: CliRunner, capsys: StrCapture) -> None:
    payload = {"id": "u1", "first_name": "Ada", "bio": "engineer"}
    fake = SimpleNamespace(get_profile=mock.Mock(return_value=payload))
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "profile", "get"])
    fake.get_profile.assert_called_once_with()
    out = capsys.readouterr().out
    assert "first_name" in out
    assert "Ada" in out
    assert "u1" not in out  # internal id is suppressed from the human table


def test_profile_get_empty_table(run_cli: CliRunner, capsys: StrCapture) -> None:
    # Only "id" is present, and _render_profile_table always filters it out,
    # so the rendered table has zero rows -> the "No data returned." branch.
    fake = SimpleNamespace(get_profile=mock.Mock(return_value={"id": "u1"}))
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "profile", "get"])
    assert "No data returned." in capsys.readouterr().out


def test_profile_get_json(run_cli: CliRunner, capsys: StrCapture) -> None:
    payload = {"first_name": "Ada"}
    fake = SimpleNamespace(get_profile=mock.Mock(return_value=payload))
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "profile", "get", "--json"])
    assert json.loads(capsys.readouterr().out) == payload


def test_profile_get_writes_output_file(run_cli: CliRunner, tmp_path: object) -> None:
    from pathlib import Path

    payload = {"first_name": "Ada"}
    fake = SimpleNamespace(get_profile=mock.Mock(return_value=payload))
    out_path = Path(str(tmp_path)) / "profile.json"
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "profile", "get", "--output", str(out_path)])
    assert json.loads(out_path.read_text()) == payload


# ---------------------------------------------------------------- profile set


def test_profile_set_parses_kv_pairs(run_cli: CliRunner) -> None:
    fake = SimpleNamespace(update_profile=mock.Mock(return_value={"bio": "new bio"}))
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "profile", "set", "bio=new bio", "location=Remote"])
    fake.update_profile.assert_called_once_with(bio="new bio", location="Remote")


def test_profile_set_rejects_pair_without_equals(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(update_profile=mock.Mock(return_value={}))
    with mock.patch("dagnam.account", fake), pytest.raises(SystemExit) as exc_info:
        run_cli(["account", "profile", "set", "bio"])
    assert exc_info.value.code == 1
    assert "Invalid KEY=VALUE" in capsys.readouterr().err
    fake.update_profile.assert_not_called()


def test_profile_set_rejects_reserved_field_name(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(update_profile=mock.Mock(return_value={}))
    with mock.patch("dagnam.account", fake), pytest.raises(SystemExit) as exc_info:
        run_cli(["account", "profile", "set", "client=nope"])
    assert exc_info.value.code == 1
    assert "Unsupported field name" in capsys.readouterr().err
    fake.update_profile.assert_not_called()


def test_profile_set_apierror_exits_1(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import APIError

    fake = SimpleNamespace(update_profile=mock.Mock(side_effect=APIError(422, "bad value")))
    with mock.patch("dagnam.account", fake):
        assert run_cli(["account", "profile", "set", "website=notaurl"]) == 1
    assert "bad value" in capsys.readouterr().err


# -------------------------------------------------------------- profile photo


def test_profile_photo_uploads_path(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: object
) -> None:
    from pathlib import Path

    fp = Path(str(tmp_path)) / "avatar.png"
    fp.write_bytes(b"\x89PNG\r\n")
    payload = {"profile_photo_url": "/uploads/avatars/x.png"}
    fake = SimpleNamespace(upload_profile_photo=mock.Mock(return_value=payload))
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "profile", "photo", str(fp)])
    fake.upload_profile_photo.assert_called_once_with(str(fp))
    assert "/uploads/avatars/x.png" in capsys.readouterr().out


def test_profile_photo_json(run_cli: CliRunner, capsys: StrCapture, tmp_path: object) -> None:
    from pathlib import Path

    fp = Path(str(tmp_path)) / "avatar.png"
    fp.write_bytes(b"\x89PNG\r\n")
    payload = {"profile_photo_url": "/uploads/avatars/x.png"}
    fake = SimpleNamespace(upload_profile_photo=mock.Mock(return_value=payload))
    with mock.patch("dagnam.account", fake):
        run_cli(["account", "profile", "photo", str(fp), "--json"])
    assert json.loads(capsys.readouterr().out) == payload


def test_profile_photo_missing_file_exits_1(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: object
) -> None:
    from pathlib import Path

    missing = Path(str(tmp_path)) / "nope.png"
    fake = SimpleNamespace(upload_profile_photo=mock.Mock())
    with mock.patch("dagnam.account", fake), pytest.raises(SystemExit) as exc_info:
        run_cli(["account", "profile", "photo", str(missing)])
    assert exc_info.value.code == 1
    assert "No such file" in capsys.readouterr().err
    fake.upload_profile_photo.assert_not_called()


# --------------------------------------------------------------- profile show


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
