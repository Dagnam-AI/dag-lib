"""Unit tests for dagnam.account profile / photo-upload / public-profile helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from dagnam import account
from dagnam._core.client import DagnamClient


def test_get_profile_delegates() -> None:
    payload = {"first_name": "Ada", "bio": "engineer"}
    c = MagicMock(spec=DagnamClient, get_profile=MagicMock(return_value=payload))
    assert account.get_profile(client=c) == payload
    c.get_profile.assert_called_once_with()


def test_update_profile_forwards_fields_as_patch() -> None:
    payload = {"bio": "new bio"}
    c = MagicMock(spec=DagnamClient, update_profile=MagicMock(return_value=payload))
    assert account.update_profile(client=c, bio="new bio") == payload
    c.update_profile.assert_called_once_with({"bio": "new bio"})


def test_update_profile_with_no_fields_sends_empty_patch() -> None:
    c = MagicMock(spec=DagnamClient, update_profile=MagicMock(return_value={}))
    account.update_profile(client=c)
    c.update_profile.assert_called_once_with({})


def test_upload_profile_photo_delegates(tmp_path: Path) -> None:
    payload = {"profile_photo_url": "/uploads/avatars/x.png"}
    c = MagicMock(spec=DagnamClient, upload_profile_photo=MagicMock(return_value=payload))
    fp = tmp_path / "avatar.png"
    assert account.upload_profile_photo(fp, client=c) == payload
    c.upload_profile_photo.assert_called_once_with(fp)


def test_get_public_profile_delegates() -> None:
    payload = {"display_name": "Ada"}
    c = MagicMock(spec=DagnamClient, get_public_profile=MagicMock(return_value=payload))
    assert account.get_public_profile("ada", client=c) == payload
    c.get_public_profile.assert_called_once_with("ada")
