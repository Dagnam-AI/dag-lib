"""Wire-level coverage for the sync hub client mixin."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import requests

from dagnam._core.client import DagnamClient
from dagnam._core.client.hub import safe_dest
from dagnam._core.exceptions import (
    APIError,
    HubError,
    HubModelNotFoundError,
    ResponseError,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tests.typing_helpers import PytestMonkeyPatch, RequestsMocker

API = "https://api.test"


# ---------------------------------------------------------------- hub client


def test_list_hub_models(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/hub/models", json={"items": []})
    assert client.list_hub_models(category="vision") == {"items": []}
    assert rmock.last_request.qs == {"category": ["vision"]}


def test_list_hub_models_repeats_list_filters(client: DagnamClient, rmock: RequestsMocker) -> None:
    """Each list item is its own key=value pair; the platform reads them as a list."""
    rmock.get(f"{API}/api/v1/hub/models", json={"items": []})
    client.list_hub_models(
        tags=["vision", "tiny"], framework=["pytorch", "flax"], task_type="classification"
    )
    assert rmock.last_request.qs == {
        "tags": ["vision", "tiny"],
        "framework": ["pytorch", "flax"],
        "task_type": ["classification"],
    }
    assert "tags=vision&tags=tiny" in rmock.last_request.url


def test_get_hub_model(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/hub/models/m1", json={"id": "m1"})
    assert client.get_hub_model("m1") == {"id": "m1"}


def test_get_hub_model_404(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/hub/models/missing", status_code=404)
    with pytest.raises(HubModelNotFoundError):
        client.get_hub_model("missing")


def test_create_hub_model(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/hub/models", json={"id": "m1"})
    assert client.create_hub_model({"name": "x"}) == {"id": "m1"}


def test_update_hub_model(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.put(f"{API}/api/v1/hub/models/m1", json={"id": "m1"})
    assert client.update_hub_model("m1", {"name": "y"}) == {"id": "m1"}


def test_delete_hub_model_empty_body(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.delete(f"{API}/api/v1/hub/models/m1", status_code=204, text="")
    assert client.delete_hub_model("m1") is None


def test_list_hub_model_files(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/hub/models/m1/files", json={"files": []})
    assert client.list_hub_model_files("m1") == {"files": []}


DOWNLOAD = f"{API}/api/v1/hub/models/m1/download"
CONTENT_F1 = f"{API}/api/v1/hub/models/m1/files/f1/content"
CONTENT_F2 = f"{API}/api/v1/hub/models/m1/files/f2/content"


def _links(*entries: tuple[str, str, str]) -> dict[str, object]:
    """The platform's download answer: one signed content link per file."""
    return {
        "files": [
            {"file_id": fid, "file_name": name, "size": 1, "url": url} for fid, name, url in entries
        ],
        "download_all_url": None,
    }


def test_download_hub_model_posts_with_file_id(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(DOWNLOAD, json=_links(("f1", "w.safetensors", f"{CONTENT_F1}?token=t1")))
    out = client.download_hub_model("m1", file_id="f1")
    assert out["download_all_url"] is None
    assert rmock.last_request.method == "POST"
    assert rmock.last_request.qs == {"file_id": ["f1"]}


def test_download_hub_model_posts_without_file_id(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.post(DOWNLOAD, json=_links())
    assert client.download_hub_model("m1") == {"files": [], "download_all_url": None}
    assert rmock.last_request.qs == {}


def test_download_hub_model_files_asks_for_a_fresh_link_per_file(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    """A link lasts minutes: each file's link is issued just before its fetch, not in one batch.

    The first fresh link is relative, the second absolute on the API's origin; both carry the key.
    """
    rmock.post(
        DOWNLOAD,
        [
            {
                "json": _links(
                    ("f1", "model.safetensors", f"{CONTENT_F1}?token=batch1"),
                    ("f2", "head.onnx", f"{CONTENT_F2}?token=batch2"),
                )
            },
            {
                "json": _links(
                    (
                        "f1",
                        "model.safetensors",
                        "/api/v1/hub/models/m1/files/f1/content?token=fresh1",
                    )
                )
            },
            {"json": _links(("f2", "head.onnx", f"{CONTENT_F2}?token=fresh2"))},
        ],
    )
    rmock.get(CONTENT_F1, content=b"weights")
    rmock.get(CONTENT_F2, content=b"onnx")
    saved = client.download_hub_model_files("m1", tmp_path)
    assert saved == [tmp_path / "model.safetensors", tmp_path / "head.onnx"]
    assert saved[0].read_bytes() == b"weights"
    assert saved[1].read_bytes() == b"onnx"
    calls = [(record.method, record.qs) for record in rmock.request_history]
    assert calls == [
        ("POST", {}),
        ("POST", {"file_id": ["f1"]}),
        ("GET", {"token": ["fresh1"]}),
        ("POST", {"file_id": ["f2"]}),
        ("GET", {"token": ["fresh2"]}),
    ]
    assert rmock.request_history[2].headers["Authorization"] == "Bearer k"
    assert rmock.request_history[4].headers["Authorization"] == "Bearer k"


@pytest.mark.parametrize("name", ["../x", "/etc/passwd", "a/b", "..", "."])
def test_safe_dest_refuses_a_path_that_leaves_dest_dir(tmp_path: Path, name: str) -> None:
    with pytest.raises(ResponseError, match="outside"):
        safe_dest(tmp_path, name)


def test_safe_dest_joins_a_bare_name(tmp_path: Path) -> None:
    assert safe_dest(tmp_path, "weights.safetensors") == tmp_path / "weights.safetensors"


def test_download_hub_model_files_foreign_absolute_link_is_fetched_without_api_key(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    """An absolute link on another host (object storage) never receives the key."""
    store = "https://bucket.example.com/m1/f1?X-Amz-Signature=abc"
    rmock.post(DOWNLOAD, json=_links(("f1", "model.safetensors", store)))
    rmock.get(store, content=b"weights")
    saved = client.download_hub_model_files("m1", tmp_path)
    assert saved == [tmp_path / "model.safetensors"]
    assert saved[0].read_bytes() == b"weights"
    assert "Authorization" not in rmock.last_request.headers


@pytest.mark.parametrize("status", [302, 307])
def test_download_hub_model_files_follows_redirect_without_api_key(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path, status: int
) -> None:
    """A content link that redirects to object storage is followed, and the key stays home."""
    presigned = "https://bucket.example.com/m1/f1?X-Amz-Signature=abc"
    rmock.post(DOWNLOAD, json=_links(("f1", "model.safetensors", f"{CONTENT_F1}?token=t1")))
    rmock.get(CONTENT_F1, status_code=status, headers={"Location": presigned})
    rmock.get(presigned, content=b"weights")
    saved = client.download_hub_model_files("m1", tmp_path, file_id="f1")
    assert saved == [tmp_path / "model.safetensors"]
    assert saved[0].read_bytes() == b"weights"
    assert rmock.request_history[0].qs == {"file_id": ["f1"]}
    assert "Authorization" not in rmock.last_request.headers


def test_download_hub_model_files_joins_a_relative_redirect_location(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    """A relative Location is resolved against the link that answered it, then fetched keyless."""
    rmock.post(DOWNLOAD, json=_links(("f1", "model.safetensors", f"{CONTENT_F1}?token=t1")))
    rmock.get(CONTENT_F1, status_code=302, headers={"Location": "/storage/m1/f1?sig=x"})
    rmock.get(f"{API}/storage/m1/f1", content=b"weights")
    saved = client.download_hub_model_files("m1", tmp_path, file_id="f1")
    assert saved[0].read_bytes() == b"weights"
    assert rmock.last_request.url == f"{API}/storage/m1/f1?sig=x"
    assert "Authorization" not in rmock.last_request.headers


@pytest.mark.parametrize(
    ("hostile", "landed"),
    [
        ("../../etc/passwd", "passwd"),
        ("/etc/passwd", "passwd"),
        ("dir/weights.safetensors", "weights.safetensors"),
        ("..", "f1"),
    ],
)
def test_download_hub_model_files_hostile_name_lands_inside_dest_dir(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path, hostile: str, landed: str
) -> None:
    """A traversal, an absolute path, a separator, or nothing at all: the file stays in dest_dir."""
    rmock.post(DOWNLOAD, json=_links(("f1", hostile, f"{CONTENT_F1}?token=t1")))
    rmock.get(CONTENT_F1, content=b"x")
    saved = client.download_hub_model_files("m1", tmp_path, file_id="f1")
    assert saved == [tmp_path / landed]
    assert saved[0].read_bytes() == b"x"


def test_download_hub_model_files_hostile_file_id_fallback_lands_inside_dest_dir(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    """When the name reduces to nothing the fallback is the file id, itself reduced to a basename."""
    rmock.post(DOWNLOAD, json=_links(("../../x", "..", f"{CONTENT_F1}?token=t1")))
    rmock.get(CONTENT_F1, content=b"x")
    saved = client.download_hub_model_files("m1", tmp_path, file_id="f1")
    assert saved == [tmp_path / "x"]
    assert not (tmp_path.parent.parent / "x").exists()


def test_download_hub_model_files_content_404_raises_not_found(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    rmock.post(DOWNLOAD, json=_links(("f1", "model.safetensors", f"{CONTENT_F1}?token=t1")))
    rmock.get(CONTENT_F1, status_code=404, text="File not found")
    with pytest.raises(HubModelNotFoundError):
        client.download_hub_model_files("m1", tmp_path)
    assert not (tmp_path / "model.safetensors").exists()


def test_download_hub_model_files_redirect_without_location_raises(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    rmock.post(DOWNLOAD, json=_links(("f1", "model.safetensors", f"{CONTENT_F1}?token=t1")))
    rmock.get(CONTENT_F1, status_code=302)
    with pytest.raises(APIError):
        client.download_hub_model_files("m1", tmp_path)


def test_download_hub_model_files_link_missing_raises(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    rmock.post(DOWNLOAD, json={"files": [{"file_id": "f1", "file_name": "w"}]})
    with pytest.raises(ResponseError, match="content link"):
        client.download_hub_model_files("m1", tmp_path)


def test_list_hub_model_versions(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/hub/models/m1/versions", json=[{"v": 1}])
    assert client.list_hub_model_versions("m1") == [{"v": 1}]


def test_create_hub_model_version(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/hub/models/m1/versions", json={"v": 1})
    assert client.create_hub_model_version("m1", {"v": 1}) == {"v": 1}


def test_star_and_unstar(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/hub/models/m1/star", json={"starred": True})
    rmock.delete(f"{API}/api/v1/hub/models/m1/star", json={"starred": False})
    assert client.star_hub_model("m1") == {"starred": True}
    assert client.unstar_hub_model("m1") == {"starred": False}


def test_fork_hub_model(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/hub/models/m1/fork", json={"new_id": "m2"})
    assert client.fork_hub_model("m1") == {"new_id": "m2"}


def test_list_hub_model_reviews(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/hub/models/m1/reviews", json={"items": []})
    client.list_hub_model_reviews("m1", page=2, limit=50)
    assert rmock.last_request.qs == {"page": ["2"], "limit": ["50"]}


def test_add_hub_model_review(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/hub/models/m1/reviews", json={"id": "r1"})
    assert client.add_hub_model_review("m1", {"rating": 5}) == {"id": "r1"}


def test_use_hub_model_in_studio(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/hub/models/m1/use-in-studio", json={"ok": True})
    assert client.use_hub_model_in_studio("m1") == {"ok": True}


def test_hub_categories_featured_trending_starred(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.get(f"{API}/api/v1/hub/categories", json=["a"])
    rmock.get(f"{API}/api/v1/hub/featured", json=["f"])
    rmock.get(f"{API}/api/v1/hub/trending", json=["t"])
    rmock.get(f"{API}/api/v1/hub/models/starred", json={"items": []})
    assert client.list_hub_categories() == ["a"]
    assert client.get_hub_featured() == ["f"]
    assert client.get_hub_trending(days=14) == ["t"]
    client.list_hub_starred(sort_by="name", page=3, limit=10)
    assert rmock.last_request.qs == {"sort_by": ["name"], "page": ["3"], "limit": ["10"]}


def test_hub_text_body_returned_when_not_json(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(
        f"{API}/api/v1/hub/categories", text="plain text", headers={"Content-Type": "text/plain"}
    )
    assert client.list_hub_categories() == "plain text"


def test_hub_500_raises_apierror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/hub/categories", status_code=500, text="boom")
    client._sleep = lambda _s: None  # 500 is transient on a GET → retried; don't sleep
    with pytest.raises(APIError):
        client.list_hub_categories()


def test_hub_get_retries_transient(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(
        f"{API}/api/v1/hub/models/m1",
        [{"status_code": 503}, {"status_code": 200, "json": {"id": "m1"}}],
    )
    client._sleep = lambda _s: None
    client._rng = lambda: 1.0
    assert client.get_hub_model("m1") == {"id": "m1"}
    assert rmock.call_count == 2


def test_hub_404_not_retried(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.get(f"{API}/api/v1/hub/models/missing", status_code=404)
    client._sleep = lambda _s: None
    with pytest.raises(HubModelNotFoundError):
        client.get_hub_model("missing")
    assert rmock.call_count == 1


def test_hub_400_raises_huberror(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/hub/models", status_code=400, text="bad")
    with pytest.raises(HubError):
        client.create_hub_model({})


def test_hub_connectionerror_wrapped(client: DagnamClient, rmock: RequestsMocker) -> None:
    client._sleep = lambda _s: None
    rmock.get(f"{API}/api/v1/hub/categories", exc=requests.ConnectionError("nope"))
    with pytest.raises(APIError, match="Request failed"):
        client.list_hub_categories()


def test_hub_timeout_wrapped(client: DagnamClient, rmock: RequestsMocker) -> None:
    client._sleep = lambda _s: None
    rmock.get(f"{API}/api/v1/hub/categories", exc=requests.Timeout("slow"))
    with pytest.raises(APIError, match="Request failed"):
        client.list_hub_categories()


# ---------------------------------------------------------------- file upload


def test_upload_model_file_streams_a_sized_multipart_body(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    """The file goes out as a streamed body with a Content-Length, never read whole into memory."""
    f = tmp_path / "weights.safetensors"
    f.write_bytes(b"\x00\x01\x02")
    rmock.post(
        f"{API}/api/v1/hub/models/m1/files",
        json={
            "id": "f1",
            "file_name": "weights.safetensors",
            "file_size": 3,
            "file_type": "safetensors",
            "checksum": "sha256:abc",
            "created_at": "2026-10-10T00:00:00Z",
        },
        status_code=201,
    )
    out = client.upload_model_file("m1", str(f))
    assert out["id"] == "f1"
    sent = rmock.last_request
    body = sent.body
    assert body is not None
    assert not isinstance(body, bytes), "the multipart body was built in memory"
    raw = b"".join(body)
    boundary = sent.headers["Content-Type"].removeprefix("multipart/form-data; boundary=")
    assert boundary
    assert boundary != sent.headers["Content-Type"]
    assert sent.headers["Content-Length"] == str(len(raw))
    assert "Transfer-Encoding" not in sent.headers
    assert sent.timeout == (10, 900)
    assert raw == (
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="weights.safetensors"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        + b"\x00\x01\x02"
        + f"\r\n--{boundary}--\r\n".encode()
    )


def test_upload_model_file_follows_a_symlink(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    """A linked file (a Hugging Face cache snapshot) uploads under the link's own name."""
    real = tmp_path / "blobs" / "abc123"
    real.parent.mkdir()
    real.write_bytes(b"\x01\x02")
    link = tmp_path / "weights.safetensors"
    link.symlink_to(real)
    rmock.post(f"{API}/api/v1/hub/models/m1/files", json={"id": "f1"}, status_code=201)
    assert client.upload_model_file("m1", str(link))["id"] == "f1"
    body = rmock.last_request.body
    assert body is not None
    assert not isinstance(body, bytes)
    raw = b"".join(body)
    assert b'filename="weights.safetensors"' in raw
    assert b"\r\n\r\n\x01\x02\r\n" in raw


def test_upload_model_file_raises_the_file_error_when_the_file_changes_mid_send(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    """requests wraps a body error as ConnectionError; the caller must see the file error."""
    f = tmp_path / "weights.safetensors"
    f.write_bytes(b"\x00" * 16)

    def _send(*_a: object, **kw: object) -> None:
        f.write_bytes(b"\x00")  # the file shrinks while the body is being read
        try:
            for _chunk in cast("Iterable[bytes]", kw["data"]):
                pass
        except OSError as exc:
            raise requests.ConnectionError("broken pipe") from exc
        raise AssertionError("the body did not notice the change")

    monkeypatch.setattr(requests, "post", _send)
    with pytest.raises(OSError, match="changed size"):
        client.upload_model_file("m1", str(f))


def test_upload_model_file_refuses_a_directory_before_any_request(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    with pytest.raises(OSError, match="not a regular file"):
        client.upload_model_file("m1", str(tmp_path))
    assert rmock.call_count == 0


def test_upload_model_file_404(client: DagnamClient, rmock: RequestsMocker, tmp_path: Path) -> None:
    f = tmp_path / "weights.bin"
    f.write_bytes(b"\x00")
    rmock.post(f"{API}/api/v1/hub/models/missing/files", status_code=404)
    with pytest.raises(HubModelNotFoundError):
        client.upload_model_file("missing", str(f))


def test_upload_model_file_connectionerror(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    f = tmp_path / "weights.bin"
    f.write_bytes(b"\x00")

    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(requests, "post", _boom)
    with pytest.raises(APIError, match="Connection failed"):
        client.upload_model_file("m1", str(f))


def test_upload_model_file_timeout(
    client: DagnamClient, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    f = tmp_path / "weights.bin"
    f.write_bytes(b"\x00")

    def _boom(*_a: object, **_kw: object) -> None:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "post", _boom)
    with pytest.raises(APIError, match="Request timed out"):
        client.upload_model_file("m1", str(f))


# ---------------------------------------------------------------- finalize


def test_finalize_hub_model(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/hub/models/m1/finalize", json={"id": "m1", "status": "published"})
    assert client.finalize_hub_model("m1")["status"] == "published"


def test_finalize_hub_model_404(client: DagnamClient, rmock: RequestsMocker) -> None:
    rmock.post(f"{API}/api/v1/hub/models/m1/finalize", status_code=404)
    with pytest.raises(HubModelNotFoundError):
        client.finalize_hub_model("m1")
