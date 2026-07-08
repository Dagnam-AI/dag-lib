"""Wire-level coverage for async profile get/update/photo-upload methods.

Async mirror of ``tests/core/client/test_account_profile.py``: exercises
``AsyncAccountMixin.get_profile/update_profile/upload_profile_photo`` (the
last streaming a multipart body via ``_request``, not ``_account_write``) and
``get_public_profile``, plus the shared error-mapping helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.exceptions import APIError, AuthError, QuotaExceededError, UploadError

if TYPE_CHECKING:
    from tests.typing_helpers import RespxMockRouter

PROFILE = "/api/v1/users/me/profile"
PHOTO = "/api/v1/users/me/profile/photo"

pytestmark = pytest.mark.anyio


async def test_get_profile(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get(PROFILE).mock(return_value=httpx.Response(200, json={"first_name": "Ada"}))
    result = await client.get_profile()
    assert result["first_name"] == "Ada"


async def test_update_profile_sends_patch_body(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    route = mock.put(PROFILE).mock(return_value=httpx.Response(200, json={"bio": "new bio"}))
    result = await client.update_profile({"bio": "new bio"})
    assert result["bio"] == "new bio"
    assert route.calls[0].request.read() == b'{"bio":"new bio"}'


async def test_get_public_profile(client: AsyncDagnamClient, mock: RespxMockRouter) -> None:
    mock.get("/api/v1/users/ada/profile").mock(
        return_value=httpx.Response(200, json={"display_name": "Ada"})
    )
    result = await client.get_public_profile("ada")
    assert result["display_name"] == "Ada"


async def test_get_public_profile_quotes_username(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/users/a%2Fb/profile").mock(
        return_value=httpx.Response(200, json={"display_name": "x"})
    )
    result = await client.get_public_profile("a/b")
    assert result["display_name"] == "x"


async def test_upload_profile_photo_streams_file(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    fp = tmp_path / "avatar.png"
    fp.write_bytes(b"\x89PNG\r\n")
    route = mock.post(PHOTO).mock(
        return_value=httpx.Response(200, json={"profile_photo_url": "/uploads/avatars/x.png"})
    )
    result = await client.upload_profile_photo(fp)
    assert result["profile_photo_url"] == "/uploads/avatars/x.png"
    sent = route.calls[0].request.read()
    assert b'name="file"' in sent
    assert b"avatar.png" in sent


async def test_upload_profile_photo_missing_file_raises(
    client: AsyncDagnamClient, tmp_path: Path
) -> None:
    missing = tmp_path / "nope.png"
    with pytest.raises(FileNotFoundError, match=r"nope\.png"):
        await client.upload_profile_photo(missing)


async def test_upload_profile_photo_400_raises_uploaderror(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    fp = tmp_path / "avatar.png"
    fp.write_bytes(b"x")
    mock.post(PHOTO).mock(return_value=httpx.Response(400, json={"detail": "Invalid image"}))
    with pytest.raises(UploadError):
        await client.upload_profile_photo(fp)


async def test_update_profile_401_raises_autherror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.put(PROFILE).mock(return_value=httpx.Response(401))
    with pytest.raises(AuthError):
        await client.update_profile({"bio": "x"})


async def test_update_profile_402_raises_quotaexceedederror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.put(PROFILE).mock(return_value=httpx.Response(402, json={"message": "Plan limit"}))
    with pytest.raises(QuotaExceededError):
        await client.update_profile({"bio": "x"})


async def test_update_profile_404_raises_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.put(PROFILE).mock(return_value=httpx.Response(404, json={"detail": "not found"}))
    with pytest.raises(APIError):
        await client.update_profile({"bio": "x"})


async def test_update_profile_409_raises_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.put(PROFILE).mock(return_value=httpx.Response(409, json={"detail": "conflict"}))
    with pytest.raises(APIError):
        await client.update_profile({"bio": "x"})


async def test_update_profile_422_raises_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.put(PROFILE).mock(return_value=httpx.Response(422, json={"detail": "bad field"}))
    with pytest.raises(APIError):
        await client.update_profile({"website": "not-a-url"})


async def test_get_public_profile_404_raises_apierror(
    client: AsyncDagnamClient, mock: RespxMockRouter
) -> None:
    mock.get("/api/v1/users/ghost/profile").mock(
        return_value=httpx.Response(404, json={"detail": "not found"})
    )
    with pytest.raises(APIError):
        await client.get_public_profile("ghost")


async def test_upload_profile_photo_401_raises_autherror(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    fp = tmp_path / "avatar.png"
    fp.write_bytes(b"x")
    mock.post(PHOTO).mock(return_value=httpx.Response(401))
    with pytest.raises(AuthError):
        await client.upload_profile_photo(fp)


async def test_upload_profile_photo_413_raises_quotaexceedederror(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    fp = tmp_path / "avatar.png"
    fp.write_bytes(b"x")
    mock.post(PHOTO).mock(return_value=httpx.Response(413, text="Storage quota exceeded"))
    with pytest.raises(QuotaExceededError):
        await client.upload_profile_photo(fp)
