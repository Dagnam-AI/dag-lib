"""Async hub client methods."""

from __future__ import annotations

from pathlib import Path

import httpx

from dagnam._core.aio.base import BaseAsyncDagnamClient
from dagnam._core.client.base import safe_download_basename, scrub_secret_params
from dagnam._core.client.common import (
    build_url,
    quote_path_segment,
    raise_for_hub,
    response_json_value,
    same_origin,
)
from dagnam._core.client.hub import safe_dest
from dagnam._core.exceptions import APIError, ResponseError
from dagnam._types import (
    JsonArray,
    JsonObject,
    JsonValue,
    QueryParams,
    QueryValue,
    ensure_json_array,
    ensure_json_object,
)


class AsyncHubMixin(BaseAsyncDagnamClient):
    """Async Hub resource methods."""

    async def _hub_req(
        self,
        method: str,
        path: str,
        *,
        model_id: str | None = None,
        params: QueryParams | None = None,
        json_body: JsonValue = None,
    ) -> JsonValue | str | None:
        resp = await self._request(
            method,
            path,
            params=params,
            json=json_body,
            raise_for=lambda r: raise_for_hub(r, model_id),
        )
        if not resp.content:
            return None
        try:
            return response_json_value(resp)
        except ResponseError:
            return resp.text

    async def list_hub_models(self, **filter_params: QueryValue) -> JsonObject:
        return ensure_json_object(
            await self._hub_req("GET", "/api/v1/hub/models", params=filter_params)
        )

    async def get_hub_model(self, model_id: str) -> JsonObject:
        return ensure_json_object(
            await self._hub_req(
                "GET", f"/api/v1/hub/models/{quote_path_segment(model_id)}", model_id=model_id
            )
        )

    async def create_hub_model(self, payload: JsonObject) -> JsonObject:
        return ensure_json_object(
            await self._hub_req("POST", "/api/v1/hub/models", json_body=payload)
        )

    async def update_hub_model(self, model_id: str, payload: JsonObject) -> JsonObject:
        return ensure_json_object(
            await self._hub_req(
                "PUT",
                f"/api/v1/hub/models/{quote_path_segment(model_id)}",
                model_id=model_id,
                json_body=payload,
            )
        )

    async def delete_hub_model(self, model_id: str) -> None:
        await self._hub_req(
            "DELETE", f"/api/v1/hub/models/{quote_path_segment(model_id)}", model_id=model_id
        )

    async def finalize_hub_model(self, model_id: str) -> JsonObject:
        """Flip a draft model live. ``POST /api/v1/hub/models/{model_id}/finalize``."""
        return ensure_json_object(
            await self._hub_req(
                "POST",
                f"/api/v1/hub/models/{quote_path_segment(model_id)}/finalize",
                model_id=model_id,
            )
        )

    async def list_hub_model_files(self, model_id: str) -> JsonObject:
        return ensure_json_object(
            await self._hub_req(
                "GET", f"/api/v1/hub/models/{quote_path_segment(model_id)}/files", model_id=model_id
            )
        )

    async def upload_model_file(self, model_id: str, file_path: str) -> JsonObject:
        """Upload a file to a hub model. ``POST /api/v1/hub/models/{model_id}/files``.

        Sends ``multipart/form-data`` with a single ``file`` part; ``httpx`` sets
        the boundary Content-Type itself, so only the bearer auth header is sent.
        """
        path = Path(file_path)
        with path.open("rb") as fh:
            resp = await self._request(
                "POST",
                f"/api/v1/hub/models/{quote_path_segment(model_id)}/files",
                files={"file": (path.name, fh)},
            )
        raise_for_hub(resp, model_id)
        return ensure_json_object(response_json_value(resp))

    async def download_hub_model(self, model_id: str, file_id: str | None = None) -> JsonObject:
        """Issue content links. ``POST /api/v1/hub/models/{model_id}/download[?file_id=]``."""
        params: QueryParams | None = {"file_id": file_id} if file_id else None
        return ensure_json_object(
            await self._hub_req(
                "POST",
                f"/api/v1/hub/models/{quote_path_segment(model_id)}/download",
                model_id=model_id,
                params=params,
            )
        )

    async def download_hub_model_files(
        self, model_id: str, dest_dir: str | Path, file_id: str | None = None
    ) -> list[Path]:
        """Save a model's files (one file with ``file_id``) under ``dest_dir``; returns the paths.

        Async mirror of the sync ``download_hub_model_files``: each file's link is issued just
        before its fetch (a link lasts minutes) and streamed to disk, bounded by
        ``max_download_bytes``; the API key goes only to the API's own origin (a link on
        another host, or a 302/307 to object storage, is fetched without it); the file name and
        its file-id fallback are reduced to a bare basename and the path is checked to lie
        inside ``dest_dir``.
        """
        links = await self.download_hub_model(model_id, file_id)
        saved: list[Path] = []
        for entry in ensure_json_array(links.get("files")):
            item = ensure_json_object(entry)
            fid = str(item.get("file_id") or "")
            if file_id is None and fid:
                # The listing's links age while earlier files download: issue this one's now.
                fresh_links = await self.download_hub_model(model_id, fid)
                for fresh in ensure_json_array(fresh_links.get("files")):
                    item = ensure_json_object(fresh)
            link = item.get("url")
            if not isinstance(link, str):
                raise ResponseError(0, "Download response names a file without a content link")
            fallback = safe_download_basename(fid, default="model-file")
            name = safe_download_basename(str(item.get("file_name") or ""), default=fallback)
            if not link.startswith(("http://", "https://")):
                link = build_url(self.api_url, link)
            saved.append(await self._save_hub_link(link, safe_dest(dest_dir, name), model_id))
        return saved

    async def _save_hub_link(self, url: str, dest: Path, model_id: str) -> Path:
        """Stream one content link to ``dest``; the key only to the API's origin, never to a redirect."""
        headers = self._headers() if same_origin(self.api_url, url) else None
        try:
            async with self._client.stream("GET", url, headers=headers) as resp:
                location = resp.headers.get("location") if resp.is_redirect else None
                if location is None:
                    if not resp.is_success:
                        await resp.aread()
                        raise_for_hub(resp, model_id)
                    await self._stream_response_to_file(resp, dest)
                    return dest
                location = str(resp.url.join(location))  # a relative Location is joined
            async with self._client.stream("GET", location) as resp:
                if not resp.is_success:
                    await resp.aread()
                    raise_for_hub(resp, model_id)
                await self._stream_response_to_file(resp, dest)
        except httpx.ConnectError as exc:
            raise APIError(0, f"Connection failed: {scrub_secret_params(str(exc))}") from exc
        except httpx.TimeoutException as exc:
            raise APIError(0, f"Request timed out: {scrub_secret_params(str(exc))}") from exc
        return dest

    async def list_hub_model_versions(self, model_id: str) -> JsonArray:
        return ensure_json_array(
            await self._hub_req(
                "GET",
                f"/api/v1/hub/models/{quote_path_segment(model_id)}/versions",
                model_id=model_id,
            )
        )

    async def create_hub_model_version(self, model_id: str, payload: JsonObject) -> JsonObject:
        return ensure_json_object(
            await self._hub_req(
                "POST",
                f"/api/v1/hub/models/{quote_path_segment(model_id)}/versions",
                model_id=model_id,
                json_body=payload,
            )
        )

    async def star_hub_model(self, model_id: str) -> JsonObject:
        return ensure_json_object(
            await self._hub_req(
                "POST", f"/api/v1/hub/models/{quote_path_segment(model_id)}/star", model_id=model_id
            )
        )

    async def unstar_hub_model(self, model_id: str) -> JsonObject:
        return ensure_json_object(
            await self._hub_req(
                "DELETE",
                f"/api/v1/hub/models/{quote_path_segment(model_id)}/star",
                model_id=model_id,
            )
        )

    async def fork_hub_model(self, model_id: str) -> JsonObject:
        return ensure_json_object(
            await self._hub_req(
                "POST", f"/api/v1/hub/models/{quote_path_segment(model_id)}/fork", model_id=model_id
            )
        )

    async def list_hub_model_reviews(
        self, model_id: str, page: int = 1, limit: int = 20
    ) -> JsonObject:
        return ensure_json_object(
            await self._hub_req(
                "GET",
                f"/api/v1/hub/models/{quote_path_segment(model_id)}/reviews",
                model_id=model_id,
                params={"page": page, "limit": limit},
            )
        )

    async def add_hub_model_review(self, model_id: str, payload: JsonObject) -> JsonObject:
        return ensure_json_object(
            await self._hub_req(
                "POST",
                f"/api/v1/hub/models/{quote_path_segment(model_id)}/reviews",
                model_id=model_id,
                json_body=payload,
            )
        )

    async def use_hub_model_in_studio(self, model_id: str) -> JsonObject:
        return ensure_json_object(
            await self._hub_req(
                "POST",
                f"/api/v1/hub/models/{quote_path_segment(model_id)}/use-in-studio",
                model_id=model_id,
            )
        )

    async def list_hub_categories(self) -> JsonArray | str | None:
        value = await self._hub_req("GET", "/api/v1/hub/categories")
        if isinstance(value, list):
            return ensure_json_array(value)
        if isinstance(value, str) or value is None:
            return value
        raise TypeError(f"Expected JSON array, got {type(value).__name__}")

    async def get_hub_featured(self) -> JsonArray:
        return ensure_json_array(await self._hub_req("GET", "/api/v1/hub/featured"))

    async def get_hub_trending(self, days: int = 7) -> JsonArray:
        return ensure_json_array(
            await self._hub_req("GET", "/api/v1/hub/trending", params={"days": days})
        )

    async def list_hub_starred(
        self, sort_by: str = "date_starred", page: int = 1, limit: int = 20
    ) -> JsonObject:
        return ensure_json_object(
            await self._hub_req(
                "GET",
                "/api/v1/hub/models/starred",
                params={"sort_by": sort_by, "page": page, "limit": limit},
            )
        )
