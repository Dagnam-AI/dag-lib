"""Wire-level coverage for project thumbnail upload/download (sync client).

Covers ``ProjectsClientMixin.upload_project_thumbnail/download_project_thumbnail``
including the traversal-safe filename contract and connect/timeout wrapping.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import requests

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import APIError, ProjectNotFoundError

if TYPE_CHECKING:
    from tests.typing_helpers import RequestsMocker

API = "https://api.test"
THUMB = f"{API}/api/v1/projects/proj-1/thumbnail"


def _image(tmp_path: Path) -> Path:
    path = tmp_path / "thumb.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    return path


# --------------------------------------------------------------------- upload


def test_upload_thumbnail_returns_url(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    rmock.post(THUMB, json={"thumbnail_url": "https://cdn.test/thumb.png"})
    result = client.upload_project_thumbnail("proj-1", _image(tmp_path))
    assert result == {"thumbnail_url": "https://cdn.test/thumb.png"}
    assert rmock.last_request.method == "POST"


def test_upload_thumbnail_missing_file_raises(client: DagnamClient, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        client.upload_project_thumbnail("proj-1", tmp_path / "nope.png")


def test_upload_thumbnail_404_raises_project_not_found(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    rmock.post(THUMB, status_code=404, json={"detail": "missing"})
    with pytest.raises(ProjectNotFoundError):
        client.upload_project_thumbnail("proj-1", _image(tmp_path))


def test_upload_thumbnail_403_raises_apierror(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    rmock.post(THUMB, status_code=403, json={"detail": "not owner"})
    with pytest.raises(APIError) as exc_info:
        client.upload_project_thumbnail("proj-1", _image(tmp_path))
    assert exc_info.value.status_code == 403


def test_upload_thumbnail_empty_body_raises_typeerror(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    rmock.post(THUMB, status_code=200)
    with pytest.raises(TypeError, match="Expected JSON object"):
        client.upload_project_thumbnail("proj-1", _image(tmp_path))


# ------------------------------------------------------------------- download


def test_download_thumbnail_writes_file(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    rmock.get(
        THUMB,
        content=b"img-bytes",
        headers={"Content-Disposition": 'attachment; filename="cover.png"'},
    )
    out = client.download_project_thumbnail("proj-1", tmp_path)
    assert out.name == "cover.png"
    assert out.parent == tmp_path
    assert out.read_bytes() == b"img-bytes"


def test_download_thumbnail_default_name_when_header_absent(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    rmock.get(THUMB, content=b"img-bytes")
    out = client.download_project_thumbnail("proj-1", tmp_path)
    assert out.name == "proj-1-thumbnail.png"


def test_download_thumbnail_traversal_filename_confined(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    rmock.get(
        THUMB,
        content=b"img-bytes",
        headers={"Content-Disposition": 'attachment; filename="../../etc/passwd"'},
    )
    out = client.download_project_thumbnail("proj-1", tmp_path)
    assert out == tmp_path / "passwd"
    assert out.read_bytes() == b"img-bytes"


def test_download_thumbnail_404_raises_project_not_found(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    rmock.get(THUMB, status_code=404, json={"detail": "missing"})
    with pytest.raises(ProjectNotFoundError):
        client.download_project_thumbnail("proj-1", tmp_path)


def test_download_thumbnail_connection_error(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    rmock.get(THUMB, exc=requests.exceptions.ConnectionError("down"))
    with pytest.raises(APIError, match="Connection failed"):
        client.download_project_thumbnail("proj-1", tmp_path)


def test_download_thumbnail_timeout(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    rmock.get(THUMB, exc=requests.exceptions.Timeout("slow"))
    with pytest.raises(APIError, match="Request timed out"):
        client.download_project_thumbnail("proj-1", tmp_path)
