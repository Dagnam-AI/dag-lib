"""Wire-level coverage for the sync datasets client mixin."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import requests
import requests_mock as rm_module

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import (
    APIError,
    DatasetNotFoundError,
)

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch, RequestsMocker

API = "https://api.test"


# ---------------------------------------------------------------- datasets client


def test_list_datasets_with_search(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/datasets/browse", json=[{"id": "ds1"}])
    client.list_datasets(type="vision", search="cifar")
    qs = rmock.last_request.qs
    assert qs == {"type": ["vision"], "search": ["cifar"]}


def test_list_datasets_no_search(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/datasets/browse", json=[])
    client.list_datasets()
    qs = rmock.last_request.qs
    assert qs == {"type": ["all"]}


def test_list_datasets_connectionerror(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch
) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.list_datasets()


def test_list_datasets_timeout(client: DagnamClient, monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.list_datasets()


def test_get_dataset_meta_with_version(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/datasets/ds1/meta", json={"id": "ds1"})
    client.get_dataset_meta("ds1", version="v2")
    assert rmock.last_request.qs == {"version": ["v2"]}


def test_get_dataset_meta_connectionerror(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch
) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.get_dataset_meta("ds1")


def test_get_dataset_meta_timeout(client: DagnamClient, monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.get_dataset_meta("ds1")


def test_list_system_datasets(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/datasets/system", json=[{"id": "iris"}])
    assert client.list_system_datasets() == [{"id": "iris"}]


def test_list_system_datasets_connectionerror(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch
) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.list_system_datasets()


def test_list_system_datasets_timeout(client: DagnamClient, monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.list_system_datasets()


def test_get_system_dataset_meta(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/datasets/system/iris", json={"id": "iris"})
    client.get_system_dataset_meta("iris", version="2.0")
    assert rmock.last_request.qs == {"version": ["2.0"]}


def test_get_system_dataset_meta_no_version(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/datasets/system/iris", json={"id": "iris"})
    client.get_system_dataset_meta("iris")
    assert rmock.last_request.qs == {}


def test_get_system_dataset_meta_connectionerror(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch
) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.get_system_dataset_meta("ds1")


def test_get_system_dataset_meta_timeout(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch
) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.get_system_dataset_meta("ds1")


def test_download_system_dataset_writes_file(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    rmock.get(
        f"{API}/api/v1/datasets/system/iris/download",
        content=b"hello",
        headers={"Content-Disposition": 'attachment; filename="iris.csv"'},
    )
    out = client.download_system_dataset("iris", tmp_path)
    assert out.name == "iris.csv"
    assert out.read_bytes() == b"hello"


def test_upload_dataset_streams_file(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    f = tmp_path / "data.csv"
    f.write_text("a,b\n1,2")
    rmock.post(f"{API}/api/v1/datasets/upload", json={"id": "ds1"})
    result = client.upload_dataset(
        f,
        name="x",
        dataset_type="tabular",
        format="csv",
        description="desc",
        license="MIT",
    )
    assert result == {"id": "ds1"}


def test_upload_dataset_connectionerror(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    f = tmp_path / "data.csv"
    f.write_text("x")

    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "post", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.upload_dataset(f, name="x", dataset_type="t", format="csv")


def test_upload_dataset_timeout(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    f = tmp_path / "data.csv"
    f.write_text("x")

    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "post", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.upload_dataset(f, name="x", dataset_type="t", format="csv")


def test_upload_dataset_from_url(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/datasets/upload-url", json={"task_id": "t1"})
    result = client.upload_dataset_from_url(
        "https://example/data.csv",
        name="x",
        dataset_type="tabular",
        format="csv",
        description="d",
    )
    assert result == {"task_id": "t1"}
    body = rmock.last_request.json()
    assert body["url"] == "https://example/data.csv"
    assert body["description"] == "d"


def test_upload_dataset_from_url_connectionerror(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch
) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "post", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.upload_dataset_from_url("u", name="x", dataset_type="t", format="csv")


def test_upload_dataset_from_url_timeout(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch
) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "post", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.upload_dataset_from_url("u", name="x", dataset_type="t", format="csv")


def test_get_dataset_task_status(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/datasets/tasks/t1", json={"status": "done"})
    assert client.get_dataset_task_status("t1") == {"status": "done"}


def test_get_dataset_task_status_connectionerror(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch
) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.get_dataset_task_status("t1")


def test_get_dataset_task_status_timeout(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch
) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.get_dataset_task_status("t1")


def test_download_dataset_full(client: DagnamClient, rmock: RequestsMocker, tmp_path: Path) -> None:
    rmock.get(
        f"{API}/api/v1/datasets/ds1/download",
        content=b"abc",
        headers={"Content-Disposition": 'attachment; filename="ds.bin"'},
    )
    out = client.download_dataset("ds1", tmp_path)
    assert out.read_bytes() == b"abc"


def test_download_dataset_full_removes_stale_derived_part(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    # No filename arg → derived from Content-Disposition. A stale .part with the
    # derived name must be unlinked before the fresh full download (line 228).
    stale = tmp_path / "ds.bin.part"
    stale.write_bytes(b"stale")
    rmock.get(
        f"{API}/api/v1/datasets/ds1/download",
        content=b"fresh",
        headers={"Content-Disposition": 'attachment; filename="ds.bin"'},
    )
    out = client.download_dataset("ds1", tmp_path)
    assert out.read_bytes() == b"fresh"


def test_download_dataset_resume_disabled_without_partial(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    # resume=False with a filename but no pre-existing .part exercises the
    # false leg of the cleanup guard (branch 185->187).
    rmock.get(f"{API}/api/v1/datasets/ds1/download", content=b"fresh")
    out = client.download_dataset("ds1", tmp_path, filename="ds.bin", resume=False)
    assert out.read_bytes() == b"fresh"


def test_download_dataset_resume_206(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    # Pre-existing partial file
    part = tmp_path / "ds.bin.part"
    part.write_bytes(b"abc")
    rmock.get(
        f"{API}/api/v1/datasets/ds1/download",
        content=b"def",
        status_code=206,
    )
    out = client.download_dataset("ds1", tmp_path, filename="ds.bin", resume=True)
    assert out.read_bytes() == b"abcdef"


def test_download_dataset_resume_falls_back_to_full(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    part = tmp_path / "ds.bin.part"
    part.write_bytes(b"old")
    rmock.get(
        f"{API}/api/v1/datasets/ds1/download",
        content=b"new",
        status_code=200,
    )
    out = client.download_dataset("ds1", tmp_path, filename="ds.bin", resume=True)
    # Server ignored Range; we restart with fresh body.
    assert out.read_bytes() == b"new"


def test_download_dataset_resume_disabled_cleans_partial(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    part = tmp_path / "ds.bin.part"
    part.write_bytes(b"stale")
    rmock.get(
        f"{API}/api/v1/datasets/ds1/download",
        content=b"fresh",
    )
    out = client.download_dataset("ds1", tmp_path, filename="ds.bin", resume=False)
    assert out.read_bytes() == b"fresh"


def test_download_dataset_with_presigned_url(client: DagnamClient, tmp_path: Path) -> None:
    presigned = "https://presigned.test/blob"
    with rm_module.Mocker() as m:
        m.get(
            presigned,
            content=b"blob",
            headers={"Content-Disposition": 'attachment; filename="blob.bin"'},
        )
        out = client.download_dataset("ds1", tmp_path, download_url=presigned)
        assert out.name == "blob.bin"
        # Presigned path must not send auth headers
        assert "Authorization" not in m.last_request.headers


def test_download_dataset_connectionerror(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.download_dataset("ds1", tmp_path)


def test_download_dataset_timeout(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.download_dataset("ds1", tmp_path)


def test_download_dataset_404(client: DagnamClient, rmock: RequestsMocker, tmp_path: Path) -> None:
    rmock.get(f"{API}/api/v1/datasets/missing/download", status_code=404)
    with pytest.raises(DatasetNotFoundError):
        client.download_dataset("missing", tmp_path)


def test_download_system_dataset_connectionerror(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.download_system_dataset("iris", tmp_path)


def test_download_system_dataset_timeout(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.download_system_dataset("iris", tmp_path)
