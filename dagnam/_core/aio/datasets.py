"""Async datasets client methods."""

from __future__ import annotations

from pathlib import Path

from dagnam._types import JsonObject, QueryParams
from dagnam._core.aio.base import BaseAsyncDagnamClient, parse_content_disposition_filename
from dagnam._core.client.common import (
    quote_path_segment,
    raise_for_dataset,
    raise_for_task,
    raise_for_upload,
    response_json_array,
    response_json_object,
)


class AsyncDatasetsMixin(BaseAsyncDagnamClient):
    """Async Datasets resource methods."""

    async def list_datasets(self, type: str = "all", search: str | None = None) -> list[JsonObject]:
        params: QueryParams = {"type": type}
        if search:
            params["search"] = search
        resp = await self._request("GET", "/api/v1/datasets/browse", params=params)
        raise_for_dataset(resp, "browse")
        return [item for item in response_json_array(resp) if isinstance(item, dict)]

    async def get_dataset_meta(self, dataset_id: str) -> JsonObject:
        resp = await self._request("GET", f"/api/v1/datasets/{quote_path_segment(dataset_id)}/meta")
        raise_for_dataset(resp, dataset_id)
        return response_json_object(resp)

    async def list_system_datasets(self) -> list[JsonObject]:
        resp = await self._request("GET", "/api/v1/datasets/system")
        raise_for_dataset(resp, "system")
        return [item for item in response_json_array(resp) if isinstance(item, dict)]

    async def get_system_dataset_meta(self, dataset_id: str) -> JsonObject:
        resp = await self._request(
            "GET", f"/api/v1/datasets/system/{quote_path_segment(dataset_id)}"
        )
        raise_for_dataset(resp, dataset_id)
        return response_json_object(resp)

    async def download_dataset(self, dataset_id: str, output_dir: Path) -> Path:
        resp = await self._request(
            "GET", f"/api/v1/datasets/{quote_path_segment(dataset_id)}/download"
        )
        raise_for_dataset(resp, dataset_id)
        filename = parse_content_disposition_filename(resp.headers.get("content-disposition"))
        dest = Path(output_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return dest

    async def download_system_dataset(self, dataset_id: str, output_dir: Path) -> Path:
        resp = await self._request(
            "GET", f"/api/v1/datasets/system/{quote_path_segment(dataset_id)}/download"
        )
        raise_for_dataset(resp, dataset_id)
        filename = parse_content_disposition_filename(resp.headers.get("content-disposition"))
        dest = Path(output_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return dest

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
        resp = await self._request("POST", "/api/v1/datasets/upload-url", json=body)
        raise_for_upload(resp)
        return response_json_object(resp)

    async def get_dataset_task_status(self, task_id: str) -> JsonObject:
        resp = await self._request("GET", f"/api/v1/datasets/tasks/{quote_path_segment(task_id)}")
        raise_for_task(resp, task_id)
        return response_json_object(resp)
