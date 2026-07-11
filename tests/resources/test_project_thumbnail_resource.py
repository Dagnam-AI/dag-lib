"""Unit tests for dagnam.projects thumbnail resource helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from dagnam._core.client import DagnamClient
from dagnam.resources import projects


def test_upload_project_thumbnail_delegates() -> None:
    c = MagicMock(
        spec=DagnamClient,
        upload_project_thumbnail=MagicMock(return_value={"thumbnail_url": "u"}),
    )
    assert projects.upload_project_thumbnail("proj-1", "thumb.png", client=c) == {
        "thumbnail_url": "u"
    }
    c.upload_project_thumbnail.assert_called_once_with("proj-1", "thumb.png")


def test_download_project_thumbnail_defaults_to_cwd() -> None:
    dest = Path("./proj-1-thumbnail.png")
    c = MagicMock(spec=DagnamClient, download_project_thumbnail=MagicMock(return_value=dest))
    assert projects.download_project_thumbnail("proj-1", client=c) == dest
    c.download_project_thumbnail.assert_called_once_with("proj-1", ".")


def test_download_project_thumbnail_uses_explicit_out(tmp_path: Path) -> None:
    dest = tmp_path / "thumb.png"
    c = MagicMock(spec=DagnamClient, download_project_thumbnail=MagicMock(return_value=dest))
    assert projects.download_project_thumbnail("proj-1", out=tmp_path, client=c) == dest
    c.download_project_thumbnail.assert_called_once_with("proj-1", tmp_path)
