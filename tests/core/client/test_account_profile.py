"""Wire-level coverage for profile get/update/photo-upload sync client methods.

Covers ``AccountClientMixin.get_profile/update_profile/upload_profile_photo``
(the last streaming a multipart body, not routed through ``_account_write``)
and ``get_public_profile``, plus the ``raise_for_generic``/``raise_for_upload``
error mapping these methods share with the rest of the account surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import requests

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import APIError, AuthError, QuotaExceededError, UploadError

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch, RequestsMocker

API = "https://api.test"
PROFILE = f"{API}/api/v1/users/me/profile"
PHOTO = f"{API}/api/v1/users/me/profile/photo"


def test_get_profile(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(PROFILE, json={"first_name": "Ada", "bio": "engineer"})
    result = client.get_profile()
    assert result["first_name"] == "Ada"
    assert rmock.last_request.headers["Authorization"] == "Bearer k"


def test_update_profile_sends_patch_body(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.put(PROFILE, json={"bio": "new bio"})
    result = client.update_profile({"bio": "new bio"})
    assert result["bio"] == "new bio"
    assert rmock.last_request.json() == {"bio": "new bio"}


def test_get_public_profile(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/users/ada/profile", json={"display_name": "Ada"})
    result = client.get_public_profile("ada")
    assert result["display_name"] == "Ada"


def test_get_public_profile_quotes_username(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/users/a%2Fb/profile", json={"display_name": "x"})
    result = client.get_public_profile("a/b")
    assert result["display_name"] == "x"


def test_upload_profile_photo_streams_file(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    f = tmp_path / "avatar.png"
    # ASCII-only content: last_request.text decodes the multipart body as UTF-8,
    # and only the wire framing (field name, filename) is under test here.
    f.write_bytes(b"fake-png-bytes")
    rmock.post(PHOTO, json={"profile_photo_url": "/uploads/avatars/x.png"})
    result = client.upload_profile_photo(f)
    assert result["profile_photo_url"] == "/uploads/avatars/x.png"
    assert rmock.last_request.headers["Authorization"] == "Bearer k"
    # Multipart body must carry the file under the backend's "file" field name.
    sent = rmock.last_request.text or ""
    assert 'name="file"' in sent
    assert "avatar.png" in sent


def test_upload_profile_photo_missing_file_raises(client: DagnamClient, tmp_path: Path) -> None:
    missing = tmp_path / "nope.png"
    with pytest.raises(FileNotFoundError, match=r"nope\.png"):
        client.upload_profile_photo(missing)


def test_upload_profile_photo_connectionerror(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    f = tmp_path / "avatar.png"
    f.write_bytes(b"x")

    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "post", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.upload_profile_photo(f)


def test_upload_profile_photo_timeout(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    f = tmp_path / "avatar.png"
    f.write_bytes(b"x")

    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "post", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.upload_profile_photo(f)


def test_upload_profile_photo_400_raises_uploaderror(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    f = tmp_path / "avatar.png"
    f.write_bytes(b"x")
    rmock.post(PHOTO, status_code=400, json={"detail": "Invalid image format"})
    with pytest.raises(UploadError):
        client.upload_profile_photo(f)


def test_update_profile_401_raises_autherror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.put(PROFILE, status_code=401, text="nope")
    with pytest.raises(AuthError):
        client.update_profile({"bio": "x"})


def test_update_profile_402_raises_quotaexceedederror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.put(PROFILE, status_code=402, json={"message": "Plan limit reached"})
    with pytest.raises(QuotaExceededError):
        client.update_profile({"bio": "x"})


def test_update_profile_404_raises_apierror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.put(PROFILE, status_code=404, json={"detail": "not found"})
    with pytest.raises(APIError):
        client.update_profile({"bio": "x"})


def test_update_profile_409_raises_apierror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.put(PROFILE, status_code=409, json={"detail": "conflict"})
    with pytest.raises(APIError):
        client.update_profile({"bio": "x"})


def test_update_profile_422_raises_apierror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.put(PROFILE, status_code=422, json={"detail": "bad field"})
    with pytest.raises(APIError):
        client.update_profile({"website": "not-a-url"})


def test_get_public_profile_404_raises_apierror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.get(f"{API}/api/v1/users/ghost/profile", status_code=404, json={"detail": "not found"})
    with pytest.raises(APIError):
        client.get_public_profile("ghost")


def test_upload_profile_photo_401_raises_autherror(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    f = tmp_path / "avatar.png"
    f.write_bytes(b"x")
    rmock.post(PHOTO, status_code=401, text="nope")
    with pytest.raises(AuthError):
        client.upload_profile_photo(f)


def test_upload_profile_photo_413_raises_quotaexceedederror(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    f = tmp_path / "avatar.png"
    f.write_bytes(b"x")
    rmock.post(PHOTO, status_code=413, text="Storage quota exceeded")
    with pytest.raises(QuotaExceededError):
        client.upload_profile_photo(f)
