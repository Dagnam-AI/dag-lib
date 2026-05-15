"""Synchronous inference client methods."""

from __future__ import annotations

from dagnam._core.client.base import _ALLOW_REDIRECTS, _TIMEOUT, APIError, requests
from dagnam._core.client.common import quote_path_segment


class InferenceClientMixin:
    """Inference resource methods for DagnamClient."""

    def predict(self, deployment_id: str, inputs: dict, timeout: int = _TIMEOUT) -> dict:
        """POST /api/v1/inference/{deployment_id}/predict"""
        deployment_path = quote_path_segment(deployment_id)
        url = f"{self.api_url}/api/v1/inference/{deployment_path}/predict"
        headers = {**self._headers(), "X-API-Key": self.api_key}
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=inputs,
                timeout=timeout,
                allow_redirects=_ALLOW_REDIRECTS,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc
        self._raise_for_deployment(resp, deployment_id)
        return resp.json()

    def predict_batch(self, deployment_id: str, inputs: list, timeout: int = _TIMEOUT) -> list:
        """POST /api/v1/inference/{deployment_id}/predict/batch"""
        deployment_path = quote_path_segment(deployment_id)
        url = f"{self.api_url}/api/v1/inference/{deployment_path}/predict/batch"
        headers = {**self._headers(), "X-API-Key": self.api_key}
        try:
            resp = requests.post(
                url,
                headers=headers,
                json={"inputs": inputs},
                timeout=timeout,
                allow_redirects=_ALLOW_REDIRECTS,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc
        self._raise_for_deployment(resp, deployment_id)
        return resp.json()

    def deployment_health(self, deployment_id: str) -> dict:
        """GET /api/v1/inference/{deployment_id}/health"""
        deployment_path = quote_path_segment(deployment_id)
        url = f"{self.api_url}/api/v1/inference/{deployment_path}/health"
        headers = {**self._headers(), "X-API-Key": self.api_key}
        try:
            resp = requests.get(
                url,
                headers=headers,
                timeout=_TIMEOUT,
                allow_redirects=_ALLOW_REDIRECTS,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc
        self._raise_for_deployment(resp, deployment_id)
        return resp.json()
