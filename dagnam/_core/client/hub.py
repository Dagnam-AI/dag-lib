"""Synchronous hub client methods."""

from __future__ import annotations

from pathlib import Path

from dagnam._core.client.base import (
    ALLOW_REDIRECTS,
    DEFAULT_TIMEOUT,
    APIError,
    BaseDagnamClient,
    requests,
)
from dagnam._core.client.common import quote_path_segment, requests_query_params
from dagnam._types import JsonArray, JsonObject, JsonValue, QueryParams, QueryValue


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
        from dagnam._core.client.common import raise_for_hub, response_json_value

        url = f"{self.api_url}{path}"
        try:
            resp = requests.request(
                method,
                url,
                headers=self._headers(),
                params=requests_query_params(params),
                json=json_body,
                timeout=timeout,
                allow_redirects=ALLOW_REDIRECTS,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        raise_for_hub(resp, model_id)

        if not resp.content:
            return None
        try:
            return response_json_value(resp)
        except ValueError:
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

    def list_hub_model_files(self, model_id: str) -> JsonObject:
        return self._hub_object(
            "GET", f"/api/v1/hub/models/{quote_path_segment(model_id)}/files", model_id=model_id
        )

    def upload_model_file(self, model_id: str, file_path: str) -> JsonObject:
        """Upload a file to a hub model. ``POST /api/v1/hub/models/{model_id}/files``.

        Sends ``multipart/form-data`` with a single ``file`` part. The multipart
        boundary Content-Type is set by ``requests`` itself, so only the bearer
        auth header is supplied (a manual Content-Type would corrupt the body).
        """
        from dagnam._core.client.common import raise_for_hub, response_json_object

        path = Path(file_path)
        url = f"{self.api_url}/api/v1/hub/models/{quote_path_segment(model_id)}/files"
        try:
            with path.open("rb") as fh:
                resp = requests.post(
                    url,
                    headers=self._headers(),
                    files={"file": (path.name, fh)},
                    timeout=DEFAULT_TIMEOUT,
                    allow_redirects=ALLOW_REDIRECTS,
                )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc
        raise_for_hub(resp, model_id)
        return response_json_object(resp)

    def download_hub_model(self, model_id: str, file_id: str | None = None) -> JsonObject:
        path = f"/api/v1/hub/models/{quote_path_segment(model_id)}/download"
        params: QueryParams | None = {"file_id": file_id} if file_id else None
        return self._hub_object("GET", path, model_id=model_id, params=params)

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
            "GET", "/api/v1/hub/starred", params={"sort_by": sort_by, "page": page, "limit": limit}
        )
