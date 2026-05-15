"""Async inference client methods."""

from __future__ import annotations

from dagnam._core.client.common import quote_path_segment, raise_for_deployment


class AsyncInferenceMixin:
    """Async Inference resource methods."""

    async def predict(self, deployment_id: str, inputs: dict, timeout: int | None = None) -> dict:
        resp = await self._request(
            "POST",
            f"/api/v1/inference/{quote_path_segment(deployment_id)}/predict",
            json=inputs,
            headers=self._inference_headers(),
            timeout=timeout,
        )
        raise_for_deployment(resp, deployment_id)
        return resp.json()

    async def predict_batch(
        self, deployment_id: str, inputs: list, timeout: int | None = None
    ) -> list:
        resp = await self._request(
            "POST",
            f"/api/v1/inference/{quote_path_segment(deployment_id)}/predict/batch",
            json={"inputs": inputs},
            headers=self._inference_headers(),
            timeout=timeout,
        )
        raise_for_deployment(resp, deployment_id)
        return resp.json()

    async def deployment_health(self, deployment_id: str) -> dict:
        resp = await self._request(
            "GET",
            f"/api/v1/inference/{quote_path_segment(deployment_id)}/health",
            headers=self._inference_headers(),
        )
        raise_for_deployment(resp, deployment_id)
        return resp.json()
