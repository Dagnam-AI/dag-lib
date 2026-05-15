"""Synchronous hub client methods."""

from __future__ import annotations

from typing import Any

from dagnam._core.client.base import _ALLOW_REDIRECTS, _TIMEOUT, APIError, requests
from dagnam._core.client.common import quote_path_segment


class HubClientMixin:
    """Hub resource methods for DagnamClient."""

    def _hub_request(
        self,
        method: str,
        path: str,
        *,
        model_id: str | None = None,
        params: dict | None = None,
        json_body: Any = None,
        timeout: int = _TIMEOUT,
    ) -> dict | list | None:
        from dagnam._core.client.common import raise_for_hub

        url = f"{self.api_url}{path}"
        try:
            resp = requests.request(
                method,
                url,
                headers=self._headers(),
                params=params,
                json=json_body,
                timeout=timeout,
                allow_redirects=_ALLOW_REDIRECTS,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        raise_for_hub(resp, model_id)

        if not resp.content:
            return None
        try:
            return resp.json()
        except ValueError:
            return resp.text

    def list_hub_models(self, **filter_params) -> dict:
        return self._hub_request("GET", "/api/v1/hub/models", params=filter_params)

    def get_hub_model(self, model_id: str) -> dict:
        return self._hub_request(
            "GET", f"/api/v1/hub/models/{quote_path_segment(model_id)}", model_id=model_id
        )

    def create_hub_model(self, payload: dict) -> dict:
        return self._hub_request("POST", "/api/v1/hub/models", json_body=payload)

    def update_hub_model(self, model_id: str, payload: dict) -> dict:
        return self._hub_request(
            "PUT",
            f"/api/v1/hub/models/{quote_path_segment(model_id)}",
            model_id=model_id,
            json_body=payload,
        )

    def delete_hub_model(self, model_id: str) -> None:
        self._hub_request(
            "DELETE", f"/api/v1/hub/models/{quote_path_segment(model_id)}", model_id=model_id
        )

    def list_hub_model_files(self, model_id: str) -> dict:
        return self._hub_request(
            "GET", f"/api/v1/hub/models/{quote_path_segment(model_id)}/files", model_id=model_id
        )

    def download_hub_model(self, model_id: str, file_id: str | None = None) -> dict:
        path = f"/api/v1/hub/models/{quote_path_segment(model_id)}/download"
        params = {"file_id": file_id} if file_id else None
        return self._hub_request("GET", path, model_id=model_id, params=params)

    def list_hub_model_versions(self, model_id: str) -> list:
        return self._hub_request(
            "GET", f"/api/v1/hub/models/{quote_path_segment(model_id)}/versions", model_id=model_id
        )

    def create_hub_model_version(self, model_id: str, payload: dict) -> dict:
        return self._hub_request(
            "POST",
            f"/api/v1/hub/models/{quote_path_segment(model_id)}/versions",
            model_id=model_id,
            json_body=payload,
        )

    def star_hub_model(self, model_id: str) -> dict:
        return self._hub_request(
            "POST", f"/api/v1/hub/models/{quote_path_segment(model_id)}/star", model_id=model_id
        )

    def unstar_hub_model(self, model_id: str) -> dict:
        return self._hub_request(
            "DELETE", f"/api/v1/hub/models/{quote_path_segment(model_id)}/star", model_id=model_id
        )

    def fork_hub_model(self, model_id: str) -> dict:
        return self._hub_request(
            "POST", f"/api/v1/hub/models/{quote_path_segment(model_id)}/fork", model_id=model_id
        )

    def list_hub_model_reviews(self, model_id: str, page: int = 1, limit: int = 20) -> dict:
        return self._hub_request(
            "GET",
            f"/api/v1/hub/models/{quote_path_segment(model_id)}/reviews",
            model_id=model_id,
            params={"page": page, "limit": limit},
        )

    def add_hub_model_review(self, model_id: str, payload: dict) -> dict:
        return self._hub_request(
            "POST",
            f"/api/v1/hub/models/{quote_path_segment(model_id)}/reviews",
            model_id=model_id,
            json_body=payload,
        )

    def use_hub_model_in_studio(self, model_id: str) -> dict:
        return self._hub_request(
            "POST",
            f"/api/v1/hub/models/{quote_path_segment(model_id)}/use-in-studio",
            model_id=model_id,
        )

    def list_hub_categories(self) -> list:
        return self._hub_request("GET", "/api/v1/hub/categories")

    def get_hub_featured(self) -> list:
        return self._hub_request("GET", "/api/v1/hub/featured")

    def get_hub_trending(self, days: int = 7) -> list:
        return self._hub_request("GET", "/api/v1/hub/trending", params={"days": days})

    def list_hub_starred(
        self, sort_by: str = "date_starred", page: int = 1, limit: int = 20
    ) -> dict:
        return self._hub_request(
            "GET", "/api/v1/hub/starred", params={"sort_by": sort_by, "page": page, "limit": limit}
        )
