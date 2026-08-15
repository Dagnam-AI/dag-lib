"""Async foundation fine-tuning client methods.

Async mirror of :mod:`dagnam._core.client.foundation`, including its
pass-through rule: a recipe's hyperparameter JSON Schema is returned exactly as
served, because it is the single source both the API validates against and a
caller renders from. The 404 mapping is literally shared -- both mixins call
``raise_for_foundation_run`` -- so the two surfaces cannot drift on it.
"""

from __future__ import annotations

from dagnam._core.aio.base import BaseAsyncDagnamClient
from dagnam._core.client.common import quote_path_segment, response_json_value
from dagnam._core.client.foundation import raise_for_foundation_run
from dagnam._types import (
    JsonArray,
    JsonObject,
    JsonValue,
    QueryParams,
    ensure_json_array,
    ensure_json_object,
)

_CATALOG_PATH = "/api/v1/foundation-catalog"
_RECIPES_PATH = "/api/v1/training/recipes"
_RUNS_PATH = "/api/v1/training/foundation-runs"


class AsyncFoundationMixin(BaseAsyncDagnamClient):
    """Async foundation fine-tuning resource methods."""

    async def _foundation_req(
        self,
        method: str,
        path: str,
        *,
        run_id: str | None = None,
        params: QueryParams | None = None,
        json_body: JsonValue = None,
        idempotent: bool = False,
    ) -> JsonValue:
        """Issue an authenticated foundation request and decode its JSON body."""
        resp = await self._request(
            method,
            path,
            params=params,
            json=json_body,
            raise_for=lambda r: raise_for_foundation_run(r, run_id),
            idempotent=idempotent,
        )
        return response_json_value(resp)

    async def list_foundation_catalog(self, *, page: int = 1, limit: int = 20) -> JsonArray:
        """One bounded page of the curated base models."""
        return ensure_json_array(
            await self._foundation_req("GET", _CATALOG_PATH, params={"page": page, "limit": limit})
        )

    async def list_training_recipes(self) -> JsonArray:
        """Every shipped training recipe, serialized and untouched."""
        return ensure_json_array(await self._foundation_req("GET", _RECIPES_PATH))

    async def create_foundation_run(self, payload: JsonObject) -> JsonObject:
        """Submit a fine-tuning run, with an ``Idempotency-Key`` so it cannot double-spend."""
        return ensure_json_object(
            await self._foundation_req("POST", _RUNS_PATH, json_body=payload, idempotent=True)
        )

    async def get_foundation_run(self, run_id: str) -> JsonObject:
        """Read one run's frozen specification and its job's live status."""
        return ensure_json_object(
            await self._foundation_req(
                "GET", f"{_RUNS_PATH}/{quote_path_segment(run_id)}", run_id=run_id
            )
        )
