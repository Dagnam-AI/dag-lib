"""Synchronous hub client methods."""

from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path
from urllib.parse import urljoin
import uuid

from dagnam._core.client.base import (
    ALLOW_REDIRECTS,
    DEFAULT_TIMEOUT,
    APIError,
    BaseDagnamClient,
    is_redirect_response,
    requests,
    safe_download_basename,
)
from dagnam._core.client.common import (
    build_url,
    quote_path_segment,
    raise_for_hub,
    requests_query_params,
    response_json_object,
    response_json_value,
    same_origin,
)
from dagnam._core.exceptions import ResponseError
from dagnam._core.tar_stream import FileStream
from dagnam._types import (
    JsonArray,
    JsonObject,
    JsonValue,
    QueryParams,
    QueryValue,
    ensure_json_array,
    ensure_json_object,
)


def safe_dest(dest_dir: str | Path, name: str) -> Path:
    """``dest_dir / name``, refused unless the result stays inside ``dest_dir``.

    ``name`` is already a bare basename when the download calls this; it is the last line of
    defence for a server-supplied name, not the first.
    """
    root = Path(dest_dir)
    if (root / name).resolve().parent != root.resolve():
        raise ResponseError(0, f"Download refused: {name!r} would land outside {root}")
    return root / name


UPLOAD_TIMEOUT = (10, 900)
"""Connect and read timeouts, in seconds, for a hub file upload.

The read half is the wait for the platform's answer after the last byte, which for a
multi-gigabyte body can take far longer than the 30 s used for ordinary calls.
"""


class _MultipartFile:
    """One file as a sized ``multipart/form-data`` body, streamed from disk.

    ``requests`` reads a ``files=`` part whole into memory before sending, and a weight file
    is gigabytes. A body with ``__len__`` goes out with a ``Content-Length`` instead, read in
    1 MiB slices by :class:`FileStream`, so the platform can refuse an oversized upload
    before it reads a byte and the client never holds more than one slice. The stream is
    built from the resolved path, so a symlink uploads its target under the link's own name;
    ``failure`` holds the ``OSError`` that aborted the send, if the file changed meanwhile.
    """

    def __init__(self, path: Path) -> None:
        self.boundary = uuid.uuid4().hex
        name = path.name.replace('"', "%22").replace("\r", "%0D").replace("\n", "%0A")
        self._head = (
            f"--{self.boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        self._tail = f"\r\n--{self.boundary}--\r\n".encode()
        self._file = FileStream(os.path.realpath(path))

    @property
    def content_type(self) -> str:
        return f"multipart/form-data; boundary={self.boundary}"

    @property
    def failure(self) -> OSError | None:
        return self._file.failure

    def __len__(self) -> int:
        return len(self._head) + len(self._file) + len(self._tail)

    def __iter__(self) -> Iterator[bytes]:
        yield self._head
        yield from self._file
        yield self._tail


class HubClientMixin(BaseDagnamClient):
    """Hub resource methods for DagnamClient."""

    def _hub_request(
        self,
        method: str,
        path: str,
        *,
        model_id: str | None = None,
        params: QueryParams | None = None,
        json_body: JsonValue = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> JsonValue | str | None:
        url = f"{self.api_url}{path}"
        resp = self._request(
            method,
            url,
            raise_for=lambda r: raise_for_hub(r, model_id),
            params=requests_query_params(params),
            json=json_body,
            timeout=timeout,
            allow_redirects=ALLOW_REDIRECTS,
        )
        if not resp.content:
            return None
        try:
            return response_json_value(resp)
        except ResponseError:
            return resp.text

    def _hub_object(
        self,
        method: str,
        path: str,
        *,
        model_id: str | None = None,
        params: QueryParams | None = None,
        json_body: JsonValue = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> JsonObject:
        value = self._hub_request(
            method,
            path,
            model_id=model_id,
            params=params,
            json_body=json_body,
            timeout=timeout,
        )
        if isinstance(value, dict):
            return value
        raise TypeError(f"Expected JSON object, got {type(value).__name__}")

    def _hub_array(
        self,
        method: str,
        path: str,
        *,
        model_id: str | None = None,
        params: QueryParams | None = None,
        json_body: JsonValue = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> JsonArray:
        value = self._hub_request(
            method,
            path,
            model_id=model_id,
            params=params,
            json_body=json_body,
            timeout=timeout,
        )
        if isinstance(value, list):
            return value
        raise TypeError(f"Expected JSON array, got {type(value).__name__}")

    def list_hub_models(self, **filter_params: QueryValue) -> JsonObject:
        return self._hub_object("GET", "/api/v1/hub/models", params=filter_params)

    def get_hub_model(self, model_id: str) -> JsonObject:
        return self._hub_object(
            "GET", f"/api/v1/hub/models/{quote_path_segment(model_id)}", model_id=model_id
        )

    def create_hub_model(self, payload: JsonObject) -> JsonObject:
        return self._hub_object("POST", "/api/v1/hub/models", json_body=payload)

    def update_hub_model(self, model_id: str, payload: JsonObject) -> JsonObject:
        return self._hub_object(
            "PUT",
            f"/api/v1/hub/models/{quote_path_segment(model_id)}",
            model_id=model_id,
            json_body=payload,
        )

    def delete_hub_model(self, model_id: str) -> None:
        self._hub_request(
            "DELETE", f"/api/v1/hub/models/{quote_path_segment(model_id)}", model_id=model_id
        )

    def finalize_hub_model(self, model_id: str) -> JsonObject:
        """Flip a draft model live. ``POST /api/v1/hub/models/{model_id}/finalize``."""
        return self._hub_object(
            "POST",
            f"/api/v1/hub/models/{quote_path_segment(model_id)}/finalize",
            model_id=model_id,
        )

    def list_hub_model_files(self, model_id: str) -> JsonObject:
        return self._hub_object(
            "GET", f"/api/v1/hub/models/{quote_path_segment(model_id)}/files", model_id=model_id
        )

    def upload_model_file(self, model_id: str, file_path: str) -> JsonObject:
        """Upload a file to a hub model. ``POST /api/v1/hub/models/{model_id}/files``.

        Sends ``multipart/form-data`` with a single ``file`` part, streamed from disk with a
        ``Content-Length`` (see :class:`_MultipartFile`); a symlink is followed. The target
        must be a regular file and must not change while it is being sent; either raises
        ``OSError`` (before any request, or in place of the connection error the change
        caused). Files above about 500 MB are not supported yet: the platform refuses the
        request (``PayloadTooLargeError``) whatever the plan allows.
        """
        body = _MultipartFile(Path(file_path))
        url = f"{self.api_url}/api/v1/hub/models/{quote_path_segment(model_id)}/files"
        try:
            resp = requests.post(
                url,
                headers={**self._headers(), "Content-Type": body.content_type},
                data=body,
                timeout=UPLOAD_TIMEOUT,
                allow_redirects=ALLOW_REDIRECTS,
            )
        except requests.ConnectionError as exc:
            # requests wraps what the body raises as a connection error; the file is the cause.
            if body.failure is not None:
                raise body.failure from exc
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc
        raise_for_hub(resp, model_id)
        return response_json_object(resp)

    def download_hub_model(self, model_id: str, file_id: str | None = None) -> JsonObject:
        """Issue content links. ``POST /api/v1/hub/models/{model_id}/download[?file_id=]``.

        The platform answers ``{"files": [{"file_id", "file_name", "size", "url"}, ...],
        "download_all_url"}``, each ``url`` a short-lived signed link. To save the bytes use
        :meth:`download_hub_model_files`.
        """
        path = f"/api/v1/hub/models/{quote_path_segment(model_id)}/download"
        params: QueryParams | None = {"file_id": file_id} if file_id else None
        return self._hub_object("POST", path, model_id=model_id, params=params)

    def download_hub_model_files(
        self, model_id: str, dest_dir: str | Path, file_id: str | None = None
    ) -> list[Path]:
        """Save a model's files (one file with ``file_id``) under ``dest_dir``; returns the paths.

        A link lasts minutes and the files are fetched in turn, so each file's link is issued
        just before its fetch (one ``POST`` per file after the listing). Each is streamed to
        disk, bounded by ``max_download_bytes``. The API key is sent only to the API's own
        origin: a link on another host, or a redirect to object storage (302 or 307), is fetched
        WITHOUT it, since a presigned URL carries its own signature. The file name (and the
        file id it falls back to) is reduced to a bare basename and the final path is checked
        to lie inside ``dest_dir``, so a hostile name can never escape it.
        """
        saved: list[Path] = []
        for entry in ensure_json_array(self.download_hub_model(model_id, file_id).get("files")):
            item = ensure_json_object(entry)
            fid = str(item.get("file_id") or "")
            if file_id is None and fid:
                # The listing's links age while earlier files download: issue this one's now.
                for fresh in ensure_json_array(self.download_hub_model(model_id, fid).get("files")):
                    item = ensure_json_object(fresh)
            link = item.get("url")
            if not isinstance(link, str):
                raise ResponseError(0, "Download response names a file without a content link")
            fallback = safe_download_basename(fid, default="model-file")
            name = safe_download_basename(str(item.get("file_name") or ""), default=fallback)
            if not link.startswith(("http://", "https://")):
                link = build_url(self.api_url, link)
            fetch = (
                self._get_stream if same_origin(self.api_url, link) else self._get_stream_no_auth
            )
            resp = fetch(link)
            if is_redirect_response(resp):
                location = urljoin(link, resp.headers["Location"])
                resp.close()
                resp = self._get_stream_no_auth(location)
            raise_for_hub(resp, model_id)
            saved.append(self._stream_response_to_file(resp, safe_dest(dest_dir, name)))
        return saved

    def list_hub_model_versions(self, model_id: str) -> JsonArray:
        return self._hub_array(
            "GET", f"/api/v1/hub/models/{quote_path_segment(model_id)}/versions", model_id=model_id
        )

    def create_hub_model_version(self, model_id: str, payload: JsonObject) -> JsonObject:
        return self._hub_object(
            "POST",
            f"/api/v1/hub/models/{quote_path_segment(model_id)}/versions",
            model_id=model_id,
            json_body=payload,
        )

    def star_hub_model(self, model_id: str) -> JsonObject:
        return self._hub_object(
            "POST", f"/api/v1/hub/models/{quote_path_segment(model_id)}/star", model_id=model_id
        )

    def unstar_hub_model(self, model_id: str) -> JsonObject:
        return self._hub_object(
            "DELETE", f"/api/v1/hub/models/{quote_path_segment(model_id)}/star", model_id=model_id
        )

    def fork_hub_model(self, model_id: str) -> JsonObject:
        return self._hub_object(
            "POST", f"/api/v1/hub/models/{quote_path_segment(model_id)}/fork", model_id=model_id
        )

    def list_hub_model_reviews(self, model_id: str, page: int = 1, limit: int = 20) -> JsonObject:
        return self._hub_object(
            "GET",
            f"/api/v1/hub/models/{quote_path_segment(model_id)}/reviews",
            model_id=model_id,
            params={"page": page, "limit": limit},
        )

    def add_hub_model_review(self, model_id: str, payload: JsonObject) -> JsonObject:
        return self._hub_object(
            "POST",
            f"/api/v1/hub/models/{quote_path_segment(model_id)}/reviews",
            model_id=model_id,
            json_body=payload,
        )

    def use_hub_model_in_studio(self, model_id: str) -> JsonObject:
        return self._hub_object(
            "POST",
            f"/api/v1/hub/models/{quote_path_segment(model_id)}/use-in-studio",
            model_id=model_id,
        )

    def list_hub_categories(self) -> JsonArray | str | None:
        value = self._hub_request("GET", "/api/v1/hub/categories")
        if isinstance(value, list | str) or value is None:
            return value
        raise TypeError(f"Expected JSON array, got {type(value).__name__}")

    def get_hub_featured(self) -> JsonArray:
        return self._hub_array("GET", "/api/v1/hub/featured")

    def get_hub_trending(self, days: int = 7) -> JsonArray:
        return self._hub_array("GET", "/api/v1/hub/trending", params={"days": days})

    def list_hub_starred(
        self, sort_by: str = "date_starred", page: int = 1, limit: int = 20
    ) -> JsonObject:
        return self._hub_object(
            "GET",
            "/api/v1/hub/models/starred",
            params={"sort_by": sort_by, "page": page, "limit": limit},
        )
