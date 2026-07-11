"""Async hub client methods."""

from __future__ import annotations

from pathlib import Path

from dagnam._core.aio.base import BaseAsyncDagnamClient
from dagnam._core.client.common import quote_path_segment, raise_for_hub, response_json_value
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
        resp = await self._request(method, path, params=params, json=json_body)
        raise_for_hub(resp, model_id)
        if not resp.content:
            return None
        try:
            return response_json_value(resp)
        except ValueError:
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
        params: QueryParams | None = {"file_id": file_id} if file_id else None
        return ensure_json_object(
            await self._hub_req(
                "GET",
                f"/api/v1/hub/models/{quote_path_segment(model_id)}/download",
                model_id=model_id,
                params=params,
            )
        )

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
                "/api/v1/hub/starred",
                params={"sort_by": sort_by, "page": page, "limit": limit},
            )
        )
