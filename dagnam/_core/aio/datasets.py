"""Async datasets client methods."""

from __future__ import annotations

from pathlib import Path

import httpx

from dagnam._core.aio.base import BaseAsyncDagnamClient, parse_content_disposition_filename
from dagnam._core.client.common import (
    quote_path_segment,
    raise_for_dataset,
    raise_for_task,
    raise_for_upload,
    response_json_array,
    response_json_object,
)
from dagnam._core.exceptions import APIError
from dagnam._types import JsonObject, QueryParams, ensure_json_object


class AsyncDatasetsMixin(BaseAsyncDagnamClient):
    """Async Datasets resource methods."""

    async def list_datasets(self, type: str = "all", search: str | None = None) -> list[JsonObject]:
        params: QueryParams = {"type": type}
        if search:
            params["search"] = search
        resp = await self._request(
            "GET",
            "/api/v1/datasets/browse",
            params=params,
            raise_for=lambda r: raise_for_dataset(r, "browse"),
        )
        return [item for item in response_json_array(resp) if isinstance(item, dict)]

    async def get_dataset_meta(self, dataset_id: str) -> JsonObject:
        resp = await self._request(
            "GET",
            f"/api/v1/datasets/{quote_path_segment(dataset_id)}/meta",
            raise_for=lambda r: raise_for_dataset(r, dataset_id),
        )
        return response_json_object(resp)

    async def list_system_datasets(self) -> list[JsonObject]:
        resp = await self._request(
            "GET", "/api/v1/datasets/system", raise_for=lambda r: raise_for_dataset(r, "system")
        )
        return [item for item in response_json_array(resp) if isinstance(item, dict)]

    async def get_system_dataset_meta(self, dataset_id: str) -> JsonObject:
        resp = await self._request(
            "GET",
            f"/api/v1/datasets/system/{quote_path_segment(dataset_id)}",
            raise_for=lambda r: raise_for_dataset(r, dataset_id),
        )
        return response_json_object(resp)

    async def _download_to_dir(self, path: str, output_dir: Path, dataset_id: str) -> Path:
        """Stream a dataset download to disk chunk by chunk.

        The whole body is never buffered in memory (``resp.content``) — large
        datasets would OOM. Chunks are written as they arrive; the destination
        filename is taken from the ``Content-Disposition`` header, which is
        available as soon as the response headers land.
        """
        url = f"{self.api_url}{path}"
        try:
            async with self._client.stream("GET", url, headers=self._headers()) as resp:
                if not resp.is_success:
                    await resp.aread()  # populate the body for the error message
                    raise_for_dataset(resp, dataset_id)
                filename = parse_content_disposition_filename(
                    resp.headers.get("content-disposition")
                )
                dest = Path(output_dir) / filename
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as fh:
                    async for chunk in resp.aiter_bytes():
                        fh.write(chunk)
                return dest
        except httpx.ConnectError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

    async def download_dataset(self, dataset_id: str, output_dir: Path) -> Path:
        return await self._download_to_dir(
            f"/api/v1/datasets/{quote_path_segment(dataset_id)}/download",
            output_dir,
            dataset_id,
        )

    async def download_system_dataset(self, dataset_id: str, output_dir: Path) -> Path:
        return await self._download_to_dir(
            f"/api/v1/datasets/system/{quote_path_segment(dataset_id)}/download",
            output_dir,
            dataset_id,
        )

    async def upload_dataset(
        self,
        file_path: str | Path,
        name: str,
        dataset_type: str,
        format: str,
        description: str | None = None,
        visibility: str = "private",
        license: str | None = None,
    ) -> JsonObject:
        fields: dict[str, str] = {
            "name": name,
            "type": dataset_type,
            "format": format,
            "visibility": visibility,
        }
        if description:
            fields["description"] = description
        if license:
            fields["license"] = license

        fp = Path(file_path)
        with open(fp, "rb") as fh:
            files = {"file": (fp.name, fh, "application/octet-stream")}
            resp = await self._request(
                "POST",
                "/api/v1/datasets/upload",
                data=fields,
                files=files,
                timeout=None,
            )
        raise_for_upload(resp)
        return resp.json()

    async def upload_dataset_from_url(
        self,
        url: str,
        name: str,
        dataset_type: str,
        format: str,
        description: str | None = None,
        visibility: str = "private",
    ) -> JsonObject:
        body: JsonObject = {
            "url": url,
            "name": name,
            "type": dataset_type,
            "format": format,
            "visibility": visibility,
        }
        if description:
            body["description"] = description
        resp = await self._request(
            "POST", "/api/v1/datasets/upload-url", json=body, raise_for=raise_for_upload
        )
        return response_json_object(resp)

    async def get_dataset_task_status(self, task_id: str) -> JsonObject:
        resp = await self._request(
            "GET",
            f"/api/v1/datasets/tasks/{quote_path_segment(task_id)}",
            raise_for=lambda r: raise_for_task(r, task_id),
        )
        return response_json_object(resp)

    async def preview_dataset(self, dataset_id: str, rows: int = 10) -> JsonObject:
        """Preview a dataset's samples and statistics.

        ``GET /api/v1/datasets/{dataset_id}/preview?rows=N`` — see the sync
        mirror in ``dagnam._core.client.datasets``.
        """
        params: QueryParams = {"rows": rows}
        resp = await self._request(
            "GET",
            f"/api/v1/datasets/{quote_path_segment(dataset_id)}/preview",
            params=params,
            raise_for=lambda r: raise_for_dataset(r, dataset_id),
        )
        return response_json_object(resp)

    async def update_dataset(
        self,
        dataset_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        visibility: str | None = None,
    ) -> JsonObject:
        """Update a dataset's mutable fields. ``PUT /api/v1/datasets/{dataset_id}``.

        Sends only the provided fields as multipart form data. At least one field
        is required; all three omitted raises ``ValueError`` before any request.
        """
        if name is None and description is None and visibility is None:
            raise ValueError(
                "update_dataset requires at least one of name, description, or visibility"
            )
        fields: dict[str, str] = {}
        if name is not None:
            fields["name"] = name
        if description is not None:
            fields["description"] = description
        if visibility is not None:
            fields["visibility"] = visibility
        resp = await self._request(
            "PUT",
            f"/api/v1/datasets/{quote_path_segment(dataset_id)}",
            data=fields,
            raise_for=lambda r: raise_for_dataset(r, dataset_id),
        )
        return response_json_object(resp)

    async def delete_dataset(self, dataset_id: str) -> None:
        """Delete a dataset. ``DELETE /api/v1/datasets/{dataset_id}`` (204 No Content)."""
        await self._request(
            "DELETE",
            f"/api/v1/datasets/{quote_path_segment(dataset_id)}",
            raise_for=lambda r: raise_for_dataset(r, dataset_id),
        )

    async def update_dataset_roles(
        self,
        dataset_id: str,
        column_roles: dict[str, str],
        task_type_hint: str | None = None,
    ) -> JsonObject:
        """Set a dataset's column roles. ``PATCH /api/v1/datasets/{dataset_id}/roles``."""
        body: JsonObject = {
            "column_roles": ensure_json_object(column_roles),
            "task_type_hint": task_type_hint,
        }
        resp = await self._request(
            "PATCH",
            f"/api/v1/datasets/{quote_path_segment(dataset_id)}/roles",
            json=body,
            raise_for=lambda r: raise_for_dataset(r, dataset_id),
        )
        return response_json_object(resp)
