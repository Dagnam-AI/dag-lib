"""Async hub client methods."""

from __future__ import annotations

from typing import Any

from dagnam._core.client.common import raise_for_hub


class AsyncHubMixin:
    """Async Hub resource methods."""

    async def _hub_req(
        self,
        method: str,
        path: str,
        *,
        model_id: str | None = None,
        params: dict | None = None,
        json_body: Any = None,
    ) -> dict | list | None:
        resp = await self._request(method, path, params=params, json=json_body)
        raise_for_hub(resp, model_id)
        if not resp.content:
            return None
        try:
            return resp.json()
        except ValueError:
            return resp.text

    async def list_hub_models(self, **filter_params: Any) -> dict:
        return await self._hub_req("GET", "/api/v1/hub/models", params=filter_params)

    async def get_hub_model(self, model_id: str) -> dict:
        return await self._hub_req("GET", f"/api/v1/hub/models/{model_id}", model_id=model_id)

    async def create_hub_model(self, payload: dict) -> dict:
        return await self._hub_req("POST", "/api/v1/hub/models", json_body=payload)

    async def update_hub_model(self, model_id: str, payload: dict) -> dict:
        return await self._hub_req(
            "PUT", f"/api/v1/hub/models/{model_id}", model_id=model_id, json_body=payload
        )

    async def delete_hub_model(self, model_id: str) -> None:
        await self._hub_req("DELETE", f"/api/v1/hub/models/{model_id}", model_id=model_id)

    async def list_hub_model_files(self, model_id: str) -> dict:
        return await self._hub_req("GET", f"/api/v1/hub/models/{model_id}/files", model_id=model_id)

    async def download_hub_model(self, model_id: str, file_id: str | None = None) -> dict:
        params = {"file_id": file_id} if file_id else None
        return await self._hub_req(
            "GET", f"/api/v1/hub/models/{model_id}/download", model_id=model_id, params=params
        )

    async def list_hub_model_versions(self, model_id: str) -> list:
        return await self._hub_req(
            "GET", f"/api/v1/hub/models/{model_id}/versions", model_id=model_id
        )

    async def create_hub_model_version(self, model_id: str, payload: dict) -> dict:
        return await self._hub_req(
            "POST", f"/api/v1/hub/models/{model_id}/versions", model_id=model_id, json_body=payload
        )

    async def star_hub_model(self, model_id: str) -> dict:
        return await self._hub_req("POST", f"/api/v1/hub/models/{model_id}/star", model_id=model_id)

    async def unstar_hub_model(self, model_id: str) -> dict:
        return await self._hub_req(
            "DELETE", f"/api/v1/hub/models/{model_id}/star", model_id=model_id
        )

    async def fork_hub_model(self, model_id: str) -> dict:
        return await self._hub_req("POST", f"/api/v1/hub/models/{model_id}/fork", model_id=model_id)

    async def list_hub_model_reviews(self, model_id: str, page: int = 1, limit: int = 20) -> dict:
        return await self._hub_req(
            "GET",
            f"/api/v1/hub/models/{model_id}/reviews",
            model_id=model_id,
            params={"page": page, "limit": limit},
        )

    async def add_hub_model_review(self, model_id: str, payload: dict) -> dict:
        return await self._hub_req(
            "POST", f"/api/v1/hub/models/{model_id}/reviews", model_id=model_id, json_body=payload
        )

    async def use_hub_model_in_studio(self, model_id: str) -> dict:
        return await self._hub_req(
            "POST", f"/api/v1/hub/models/{model_id}/use-in-studio", model_id=model_id
        )

    async def list_hub_categories(self) -> list:
        return await self._hub_req("GET", "/api/v1/hub/categories")

    async def get_hub_featured(self) -> list:
        return await self._hub_req("GET", "/api/v1/hub/featured")

    async def get_hub_trending(self, days: int = 7) -> list:
        return await self._hub_req("GET", "/api/v1/hub/trending", params={"days": days})

    async def list_hub_starred(
        self, sort_by: str = "date_starred", page: int = 1, limit: int = 20
    ) -> dict:
        return await self._hub_req(
            "GET", "/api/v1/hub/starred", params={"sort_by": sort_by, "page": page, "limit": limit}
        )
