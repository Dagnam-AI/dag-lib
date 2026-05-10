"""Async datasets client methods."""

from __future__ import annotations

from pathlib import Path

from dagnam._core.aio.base import _parse_cd
from dagnam._core.client.common import raise_for_dataset, raise_for_task, raise_for_upload


class AsyncDatasetsMixin:
    """Async Datasets resource methods."""

    async def list_datasets(self, type: str = "all", search: str | None = None) -> list[dict]:
        params: dict[str, str] = {"type": type}
        if search:
            params["search"] = search
        resp = await self._request("GET", "/api/v1/datasets/browse", params=params)
        raise_for_dataset(resp, "browse")
        return resp.json()

    async def get_dataset_meta(self, dataset_id: str) -> dict:
        resp = await self._request("GET", f"/api/v1/datasets/{dataset_id}/meta")
        raise_for_dataset(resp, dataset_id)
        return resp.json()

    async def list_system_datasets(self) -> list[dict]:
        resp = await self._request("GET", "/api/v1/datasets/system")
        raise_for_dataset(resp, "system")
        return resp.json()

    async def get_system_dataset_meta(self, dataset_id: str) -> dict:
        resp = await self._request("GET", f"/api/v1/datasets/system/{dataset_id}")
        raise_for_dataset(resp, dataset_id)
        return resp.json()

    async def download_dataset(self, dataset_id: str, output_dir: Path) -> Path:
        resp = await self._request("GET", f"/api/v1/datasets/{dataset_id}/download")
        raise_for_dataset(resp, dataset_id)
        filename = _parse_cd(resp.headers.get("content-disposition"))
        dest = Path(output_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return dest

    async def download_system_dataset(self, dataset_id: str, output_dir: Path) -> Path:
        resp = await self._request("GET", f"/api/v1/datasets/system/{dataset_id}/download")
        raise_for_dataset(resp, dataset_id)
        filename = _parse_cd(resp.headers.get("content-disposition"))
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
    ) -> dict:
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
    ) -> dict:
        body: dict[str, str] = {
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
        return resp.json()

    async def get_dataset_task_status(self, task_id: str) -> dict:
        resp = await self._request("GET", f"/api/v1/datasets/tasks/{task_id}")
        raise_for_task(resp, task_id)
        return resp.json()
