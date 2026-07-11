"""CLI coverage for `dagnam projects thumbnail` (upload/download)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, StrCapture


def test_thumbnail_upload_prints_url(run_cli: CliRunner, capsys: StrCapture) -> None:
    upload = mock.Mock(return_value={"thumbnail_url": "https://cdn.test/t.png"})
    with mock.patch("dagnam.upload_project_thumbnail", upload):
        run_cli(["projects", "thumbnail", "proj-1", "--set", "thumb.png"])
    upload.assert_called_once_with("proj-1", "thumb.png")
    assert "Thumbnail uploaded: https://cdn.test/t.png" in capsys.readouterr().out


def test_thumbnail_download_explicit_out(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path
) -> None:
    dest = tmp_path / "proj-1-thumbnail.png"
    download = mock.Mock(return_value=dest)
    with mock.patch("dagnam.download_project_thumbnail", download):
        run_cli(["projects", "thumbnail", "proj-1", "--out", str(tmp_path)])
    download.assert_called_once_with("proj-1", out=str(tmp_path))
    assert str(dest) in capsys.readouterr().out


def test_thumbnail_download_defaults_to_cwd(run_cli: CliRunner, capsys: StrCapture) -> None:
    download = mock.Mock(return_value=Path("./proj-1-thumbnail.png"))
    with mock.patch("dagnam.download_project_thumbnail", download):
        run_cli(["projects", "thumbnail", "proj-1"])
    download.assert_called_once_with("proj-1", out=".")
    assert "Saved thumbnail to" in capsys.readouterr().out
