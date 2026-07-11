"""Wire-level coverage for project thumbnail upload/download (async client).

Async mirror of ``tests/core/client/test_project_thumbnail.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.exceptions import APIError, ProjectNotFoundError

if TYPE_CHECKING:
    from tests.typing_helpers import RespxMockRouter

THUMB = "/api/v1/projects/proj-1/thumbnail"

pytestmark = pytest.mark.anyio


def _image(tmp_path: Path) -> Path:
    path = tmp_path / "thumb.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    return path


# --------------------------------------------------------------------- upload


async def test_upload_thumbnail_returns_url(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.post(THUMB).mock(
        return_value=httpx.Response(200, json={"thumbnail_url": "https://cdn.test/t.png"})
    )
    result = await client.upload_project_thumbnail("proj-1", _image(tmp_path))
    assert result == {"thumbnail_url": "https://cdn.test/t.png"}


async def test_upload_thumbnail_missing_file_raises(
    client: AsyncDagnamClient, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError):
        await client.upload_project_thumbnail("proj-1", tmp_path / "nope.png")


async def test_upload_thumbnail_404_raises_project_not_found(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.post(THUMB).mock(return_value=httpx.Response(404, json={"detail": "missing"}))
    with pytest.raises(ProjectNotFoundError):
        await client.upload_project_thumbnail("proj-1", _image(tmp_path))


# ------------------------------------------------------------------- download


async def test_download_thumbnail_writes_file(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.get(THUMB).mock(
        return_value=httpx.Response(
            200,
            content=b"img-bytes",
            headers={"content-disposition": 'attachment; filename="cover.png"'},
        )
    )
    out = await client.download_project_thumbnail("proj-1", tmp_path)
    assert out.name == "cover.png"
    assert out.read_bytes() == b"img-bytes"


async def test_download_thumbnail_default_name_when_header_absent(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.get(THUMB).mock(return_value=httpx.Response(200, content=b"img-bytes"))
    out = await client.download_project_thumbnail("proj-1", tmp_path)
    assert out.name == "proj-1-thumbnail.png"


async def test_download_thumbnail_traversal_filename_confined(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.get(THUMB).mock(
        return_value=httpx.Response(
            200,
            content=b"img-bytes",
            headers={"content-disposition": 'attachment; filename="../../etc/passwd"'},
        )
    )
    out = await client.download_project_thumbnail("proj-1", tmp_path)
    assert out == tmp_path / "passwd"
    assert out.read_bytes() == b"img-bytes"


async def test_download_thumbnail_404_raises_project_not_found(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.get(THUMB).mock(return_value=httpx.Response(404, json={"detail": "missing"}))
    with pytest.raises(ProjectNotFoundError):
        await client.download_project_thumbnail("proj-1", tmp_path)


async def test_download_thumbnail_connect_error(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.get(THUMB).mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(APIError, match="Connection failed"):
        await client.download_project_thumbnail("proj-1", tmp_path)


async def test_download_thumbnail_timeout(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.get(THUMB).mock(side_effect=httpx.ConnectTimeout("slow"))
    with pytest.raises(APIError, match="Request timed out"):
        await client.download_project_thumbnail("proj-1", tmp_path)
