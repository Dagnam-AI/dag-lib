"""Synchronous inference client methods."""

from __future__ import annotations

from dagnam._types import JsonArray, JsonObject

from dagnam._core.client.base import (
    ALLOW_REDIRECTS,
    APIError,
    BaseDagnamClient,
    DEFAULT_TIMEOUT,
    requests,
)
from dagnam._core.client.common import quote_path_segment


class InferenceClientMixin(BaseDagnamClient):
    """Inference resource methods for DagnamClient."""

    def predict(self, deployment_id: str, inputs: JsonObject, timeout: int = DEFAULT_TIMEOUT) -> JsonObject:
        """POST /api/v1/inference/{deployment_id}/predict"""
        deployment_path = quote_path_segment(deployment_id)
        url = f"{self.api_url}/api/v1/inference/{deployment_path}/predict"
        headers = self._headers()
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=inputs,
                timeout=timeout,
                allow_redirects=ALLOW_REDIRECTS,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc
        self._raise_for_deployment(resp, deployment_id)
        return resp.json()

    def predict_batch(self, deployment_id: str, inputs: JsonArray, timeout: int = DEFAULT_TIMEOUT) -> JsonArray:
        """POST /api/v1/inference/{deployment_id}/predict/batch"""
        deployment_path = quote_path_segment(deployment_id)
        url = f"{self.api_url}/api/v1/inference/{deployment_path}/predict/batch"
        headers = self._headers()
        try:
            resp = requests.post(
                url,
                headers=headers,
                json={"inputs": inputs},
                timeout=timeout,
                allow_redirects=ALLOW_REDIRECTS,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc
        self._raise_for_deployment(resp, deployment_id)
        return resp.json()

    def deployment_health(self, deployment_id: str) -> JsonObject:
        """GET /api/v1/inference/{deployment_id}/health"""
        deployment_path = quote_path_segment(deployment_id)
        url = f"{self.api_url}/api/v1/inference/{deployment_path}/health"
        headers = self._headers()
        try:
            resp = requests.get(
                url,
                headers=headers,
                timeout=DEFAULT_TIMEOUT,
                allow_redirects=ALLOW_REDIRECTS,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc
        self._raise_for_deployment(resp, deployment_id)
        return resp.json()
