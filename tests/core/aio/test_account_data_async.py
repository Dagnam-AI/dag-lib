"""Wire-level coverage for async data export + download + delete-account methods.

Async mirror of ``tests/core/client/test_account_data.py``: exercises
``AsyncAccountMixin.export_data/download_export/delete_account`` plus the
streaming-download connect/timeout error wrapping.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.exceptions import APIError, AuthError

if TYPE_CHECKING:
    from tests.typing_helpers import RespxMockRouter

EXPORT = "/api/v1/users/me/export"
EXPORT_DOWNLOAD = f"{EXPORT}/exp-1"
DELETE_ACCOUNT = "/api/v1/users/me"

pytestmark = pytest.mark.anyio


# --------------------------------------------------------------------- export_data


async def test_export_data_sends_post(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    payload = {
        "export_id": "exp-1",
        "status": "pending",
        "created_at": "2026-07-01T00:00:00Z",
        "expires_at": "2026-07-08T00:00:00Z",
    }
    route = mock.post(EXPORT).mock(return_value=httpx.Response(200, json=payload))
    result = await client.export_data()
    assert result == payload
    assert route.called


async def test_export_data_401_raises_autherror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.post(EXPORT).mock(return_value=httpx.Response(401))
    with pytest.raises(AuthError):
        await client.export_data()


# ----------------------------------------------------------------- download_export


async def test_download_export_writes_file(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.get(EXPORT_DOWNLOAD).mock(
        return_value=httpx.Response(
            200,
            content=b"zip-bytes",
            headers={"content-disposition": 'attachment; filename="dagnam_export_u1.zip"'},
        )
    )
    out = await client.download_export("exp-1", tmp_path)
    assert out.name == "dagnam_export_u1.zip"
    assert out.parent == tmp_path
    assert out.read_bytes() == b"zip-bytes"


async def test_download_export_default_name_when_header_absent(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.get(EXPORT_DOWNLOAD).mock(return_value=httpx.Response(200, content=b"zip-bytes"))
    out = await client.download_export("exp-1", tmp_path)
    assert out.name == "export.zip"


async def test_download_export_404_raises_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.get(EXPORT_DOWNLOAD).mock(
        return_value=httpx.Response(404, json={"detail": "not found or expired"})
    )
    with pytest.raises(APIError) as exc_info:
        await client.download_export("exp-1", tmp_path)
    assert exc_info.value.status_code == 404


async def test_download_export_401_raises_autherror(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.get(EXPORT_DOWNLOAD).mock(return_value=httpx.Response(401))
    with pytest.raises(AuthError):
        await client.download_export("exp-1", tmp_path)


async def test_download_export_connect_error(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.get(EXPORT_DOWNLOAD).mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(APIError, match="Connection failed"):
        await client.download_export("exp-1", tmp_path)


async def test_download_export_timeout(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.get(EXPORT_DOWNLOAD).mock(side_effect=httpx.ConnectTimeout("slow"))
    with pytest.raises(APIError, match="Request timed out"):
        await client.download_export("exp-1", tmp_path)


async def test_download_export_traversal_filename_lands_inside_dest_dir(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    """Async mirror of the mandatory traversal test."""
    mock.get(EXPORT_DOWNLOAD).mock(
        return_value=httpx.Response(
            200,
            content=b"zip-bytes",
            headers={"content-disposition": 'attachment; filename="../../etc/passwd"'},
        )
    )
    out = await client.download_export("exp-1", tmp_path)
    # Hostile filename is reduced to its basename and confined to dest_dir; the
    # containment assertions below prove no escape (the sync mirror additionally
    # probes /etc/passwd - avoided here to keep the async test off blocking I/O).
    assert out.parent == tmp_path
    assert out == tmp_path / "passwd"
    assert out.read_bytes() == b"zip-bytes"


# ------------------------------------------------------------------- delete_account


async def test_delete_account_sends_password_and_confirmation(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.delete(DELETE_ACCOUNT).mock(
        return_value=httpx.Response(200, json={"message": "Account deleted"})
    )
    result = await client.delete_account("S3cret-Password")
    assert result == {"message": "Account deleted"}
    assert route.calls[0].request.read() == (
        b'{"password":"S3cret-Password","confirmation":"DELETE MY ACCOUNT"}'
    )


async def test_delete_account_401_bad_password_raises_autherror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.delete(DELETE_ACCOUNT).mock(
        return_value=httpx.Response(401, json={"detail": "Incorrect password"})
    )
    with pytest.raises(AuthError):
        await client.delete_account("wrong-password")


async def test_delete_account_422_raises_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.delete(DELETE_ACCOUNT).mock(
        return_value=httpx.Response(422, json={"detail": "confirmation mismatch"})
    )
    with pytest.raises(APIError) as exc_info:
        await client.delete_account("S3cret-Password")
    assert exc_info.value.status_code == 422
