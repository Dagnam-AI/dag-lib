"""Wire-level coverage for data export + download + delete-account sync methods.

Covers ``AccountClientMixin.export_data/download_export/delete_account``, plus
the mandatory traversal test proving a hostile ``Content-Disposition``
filename lands inside ``dest_dir`` rather than escaping it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import APIError, AuthError

if TYPE_CHECKING:
    from tests.typing_helpers import RequestsMocker

API = "https://api.test"
EXPORT = f"{API}/api/v1/users/me/export"
EXPORT_DOWNLOAD = f"{EXPORT}/exp-1"
DELETE_ACCOUNT = f"{API}/api/v1/users/me"


# --------------------------------------------------------------------- export_data


def test_export_data_sends_post(client: DagnamClient, rmock: RequestsMocker) -> None:
    payload = {
        "export_id": "exp-1",
        "status": "pending",
        "created_at": "2026-07-01T00:00:00Z",
        "expires_at": "2026-07-08T00:00:00Z",
    }
    rmock.post(EXPORT, json=payload)
    result = client.export_data()
    assert result == payload
    assert rmock.last_request.method == "POST"
    assert rmock.last_request.headers["Authorization"] == "Bearer k"


def test_export_data_401_raises_autherror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(EXPORT, status_code=401, text="nope")
    with pytest.raises(AuthError):
        client.export_data()


# ----------------------------------------------------------------- download_export


def test_download_export_writes_file(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    rmock.get(
        EXPORT_DOWNLOAD,
        content=b"zip-bytes",
        headers={"Content-Disposition": 'attachment; filename="dagnam_export_u1.zip"'},
    )
    out = client.download_export("exp-1", tmp_path)
    assert out.name == "dagnam_export_u1.zip"
    assert out.parent == tmp_path
    assert out.read_bytes() == b"zip-bytes"


def test_download_export_default_name_when_header_absent(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    rmock.get(EXPORT_DOWNLOAD, content=b"zip-bytes")
    out = client.download_export("exp-1", tmp_path)
    assert out.name == "export.zip"


def test_download_export_quotes_export_id(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    rmock.get(f"{EXPORT}/a%2Fb", content=b"x")
    client.download_export("a/b", tmp_path)


def test_download_export_401_raises_autherror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(EXPORT_DOWNLOAD, status_code=401, text="nope")
    with pytest.raises(AuthError):
        client.download_export("exp-1", "/tmp")


def test_download_export_404_raises_apierror(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    rmock.get(EXPORT_DOWNLOAD, status_code=404, json={"detail": "not found or expired"})
    with pytest.raises(APIError) as exc_info:
        client.download_export("exp-1", tmp_path)
    assert exc_info.value.status_code == 404


def test_download_export_traversal_filename_lands_inside_dest_dir(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    """Mandatory traversal test (plan acceptance criterion).

    A hostile Content-Disposition filename must be reduced to its basename and
    written inside ``dest_dir`` - never escape it - and /etc/passwd must never
    be touched.
    """
    rmock.get(
        EXPORT_DOWNLOAD,
        content=b"zip-bytes",
        headers={"Content-Disposition": 'attachment; filename="../../etc/passwd"'},
    )
    out = client.download_export("exp-1", tmp_path)
    assert out.parent == tmp_path
    assert out == tmp_path / "passwd"
    assert out.read_bytes() == b"zip-bytes"
    assert not Path("/etc/passwd").exists() or Path("/etc/passwd").read_bytes() != b"zip-bytes"


# ------------------------------------------------------------------- delete_account


def test_delete_account_sends_password_and_confirmation(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.delete(DELETE_ACCOUNT, json={"message": "Account deleted"})
    result = client.delete_account("S3cret-Password")
    assert result == {"message": "Account deleted"}
    body = rmock.last_request.json()
    assert body == {"password": "S3cret-Password", "confirmation": "DELETE MY ACCOUNT"}
    assert rmock.last_request.method == "DELETE"


def test_delete_account_401_bad_password_raises_autherror(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.delete(DELETE_ACCOUNT, status_code=401, json={"detail": "Incorrect password"})
    with pytest.raises(AuthError):
        client.delete_account("wrong-password")


def test_delete_account_422_raises_apierror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.delete(DELETE_ACCOUNT, status_code=422, json={"detail": "confirmation mismatch"})
    with pytest.raises(APIError) as exc_info:
        client.delete_account("S3cret-Password")
    assert exc_info.value.status_code == 422
