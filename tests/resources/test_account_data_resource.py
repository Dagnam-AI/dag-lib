"""Unit tests for dagnam.account export_data / download_export / delete_account."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from dagnam import account
from dagnam._core.client import DagnamClient


def test_export_data_delegates() -> None:
    payload = {"export_id": "exp-1", "status": "pending"}
    c = MagicMock(spec=DagnamClient, export_data=MagicMock(return_value=payload))
    assert account.export_data(client=c) == payload
    c.export_data.assert_called_once_with()


def test_download_export_defaults_to_cwd() -> None:
    dest = Path("/tmp/dagnam_export_u1.zip")
    c = MagicMock(spec=DagnamClient, download_export=MagicMock(return_value=dest))
    assert account.download_export("exp-1", client=c) == dest
    c.download_export.assert_called_once_with("exp-1", Path.cwd())


def test_download_export_uses_explicit_out(tmp_path: Path) -> None:
    dest = tmp_path / "dagnam_export_u1.zip"
    c = MagicMock(spec=DagnamClient, download_export=MagicMock(return_value=dest))
    assert account.download_export("exp-1", out=tmp_path, client=c) == dest
    c.download_export.assert_called_once_with("exp-1", tmp_path)


def test_delete_account_delegates() -> None:
    payload = {"message": "Account deleted"}
    c = MagicMock(spec=DagnamClient, delete_account=MagicMock(return_value=payload))
    assert account.delete_account("S3cret-Password", client=c) == payload
    c.delete_account.assert_called_once_with("S3cret-Password")
