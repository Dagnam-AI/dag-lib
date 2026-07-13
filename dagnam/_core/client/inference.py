"""Synchronous inference client methods."""

from __future__ import annotations

import json

from dagnam._core.client.base import (
    ALLOW_REDIRECTS,
    DEFAULT_TIMEOUT,
    SSE_READ_TIMEOUT,
    STREAM_CONNECT_TIMEOUT,
    APIError,
    BaseDagnamClient,
    requests,
)
from dagnam._core.client.common import quote_path_segment, stream_query_params
from dagnam._types import JsonArray, JsonObject


class InferenceClientMixin(BaseDagnamClient):
    """Inference resource methods for DagnamClient."""

    def predict(
        self, deployment_id: str, inputs: JsonObject, timeout: int = DEFAULT_TIMEOUT
    ) -> JsonObject:
        """POST /api/v1/inference/{deployment_id}/predict"""
        deployment_path = quote_path_segment(deployment_id)
        url = f"{self.api_url}/api/v1/inference/{deployment_path}/predict"
        resp = self._request(
            "POST",
            url,
            raise_for=lambda r: self._raise_for_deployment(r, deployment_id),
            json=inputs,
            timeout=timeout,
            allow_redirects=ALLOW_REDIRECTS,
        )
        return resp.json()

    def predict_batch(
        self, deployment_id: str, inputs: JsonArray, timeout: int = DEFAULT_TIMEOUT
    ) -> JsonArray:
        """POST /api/v1/inference/{deployment_id}/predict/batch"""
        deployment_path = quote_path_segment(deployment_id)
        url = f"{self.api_url}/api/v1/inference/{deployment_path}/predict/batch"
        resp = self._request(
            "POST",
            url,
            raise_for=lambda r: self._raise_for_deployment(r, deployment_id),
            json={"inputs": inputs},
            timeout=timeout,
            allow_redirects=ALLOW_REDIRECTS,
        )
        return resp.json()

    def mint_inference_stream_token(self, deployment_id: str) -> str:
        """Mint a short-lived stream-access token for one deployment's inference SSE.

        POST /api/v1/inference/{deployment_id}/stream-access-token
        """
        deployment_path = quote_path_segment(deployment_id)
        url = f"{self.api_url}/api/v1/inference/{deployment_path}/stream-access-token"
        resp = self._request(
            "POST",
            url,
            raise_for=lambda r: self._raise_for_deployment(r, deployment_id),
            allow_redirects=ALLOW_REDIRECTS,
        )
        return str(resp.json()["token"])

    def open_inference_stream(self, deployment_id: str, inputs: JsonObject) -> requests.Response:
        """Open a single-shot streaming-predict SSE connection.

        GET /api/v1/inference/{deployment_id}/predict/stream?token=...&input=...
        Auth: a per-connection scoped stream token in the query string (minted
        via the header-authenticated endpoint above); the long-lived API key
        never appears in a URL.
        """
        token = self.mint_inference_stream_token(deployment_id)
        deployment_path = quote_path_segment(deployment_id)
        url = f"{self.api_url}/api/v1/inference/{deployment_path}/predict/stream"
        params = {**stream_query_params(token), "input": json.dumps(inputs)}
        try:
            resp = requests.get(
                url,
                params=params,
                headers={"Accept": "text/event-stream"},
                stream=True,
                timeout=(STREAM_CONNECT_TIMEOUT, SSE_READ_TIMEOUT),
                allow_redirects=ALLOW_REDIRECTS,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc
        self._raise_for_deployment(resp, deployment_id)
        return resp

    def deployment_health(self, deployment_id: str) -> JsonObject:
        """GET /api/v1/inference/{deployment_id}/health"""
        deployment_path = quote_path_segment(deployment_id)
        url = f"{self.api_url}/api/v1/inference/{deployment_path}/health"
        resp = self._request(
            "GET",
            url,
            raise_for=lambda r: self._raise_for_deployment(r, deployment_id),
            allow_redirects=ALLOW_REDIRECTS,
        )
        return resp.json()

    def schema(self, deployment_id: str, timeout: int = DEFAULT_TIMEOUT) -> JsonObject:
        """Return a deployment's inference input/output schema.

        ``GET /api/v1/inference/{deployment_id}/schema`` →
        ``{input_schema, output_schema, examples?}``.
        """
        deployment_path = quote_path_segment(deployment_id)
        url = f"{self.api_url}/api/v1/inference/{deployment_path}/schema"
        resp = self._request(
            "GET",
            url,
            raise_for=lambda r: self._raise_for_deployment(r, deployment_id),
            timeout=timeout,
            allow_redirects=ALLOW_REDIRECTS,
        )
        return resp.json()
