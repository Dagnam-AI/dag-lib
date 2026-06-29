"""Async inference client methods."""

from __future__ import annotations

from dagnam._core.aio.base import BaseAsyncDagnamClient
from dagnam._core.client.common import quote_path_segment, raise_for_deployment
from dagnam._types import JsonArray, JsonObject


class AsyncInferenceMixin(BaseAsyncDagnamClient):
    """Async Inference resource methods."""

    async def predict(
        self, deployment_id: str, inputs: JsonObject, timeout: int | None = None
    ) -> JsonObject:
        resp = await self._request(
            "POST",
            f"/api/v1/inference/{quote_path_segment(deployment_id)}/predict",
            json=inputs,
            headers=self._headers(),
            timeout=timeout,
        )
        raise_for_deployment(resp, deployment_id)
        return resp.json()

    async def predict_batch(
        self, deployment_id: str, inputs: JsonArray, timeout: int | None = None
    ) -> JsonArray:
        resp = await self._request(
            "POST",
            f"/api/v1/inference/{quote_path_segment(deployment_id)}/predict/batch",
            json={"inputs": inputs},
            headers=self._headers(),
            timeout=timeout,
        )
        raise_for_deployment(resp, deployment_id)
        return resp.json()

    async def deployment_health(self, deployment_id: str) -> JsonObject:
        resp = await self._request(
            "GET",
            f"/api/v1/inference/{quote_path_segment(deployment_id)}/health",
            headers=self._headers(),
        )
        raise_for_deployment(resp, deployment_id)
        return resp.json()

    async def schema(self, deployment_id: str, timeout: int | None = None) -> JsonObject:
        """Return a deployment's inference schema. ``GET /api/v1/inference/{id}/schema``."""
        resp = await self._request(
            "GET",
            f"/api/v1/inference/{quote_path_segment(deployment_id)}/schema",
            headers=self._headers(),
            timeout=timeout,
        )
        raise_for_deployment(resp, deployment_id)
        return resp.json()
