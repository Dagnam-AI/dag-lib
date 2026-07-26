"""Synchronous deployments client methods."""

from __future__ import annotations

from dagnam._core.client.base import (
    ALLOW_REDIRECTS,
    DEFAULT_TIMEOUT,
    SSE_READ_TIMEOUT,
    STREAM_CONNECT_TIMEOUT,
    APIError,
    BaseDagnamClient,
    requests,
)
from dagnam._core.client.common import (
    quote_path_segment,
    raise_for_deployment,
    requests_query_params,
    response_json_value,
    stream_query_params,
)
from dagnam._core.exceptions import ResponseError
from dagnam._types import JsonArray, JsonObject, JsonValue, QueryParams, ensure_json_array


class DeploymentsClientMixin(BaseDagnamClient):
    """Deployments resource methods for DagnamClient."""

    def _deployment_request(
        self,
        method: str,
        path: str,
        *,
        deployment_id: str | None = None,
        params: QueryParams | None = None,
        json_body: JsonValue = None,
        timeout: int = DEFAULT_TIMEOUT,
        idempotent: bool = False,
    ) -> JsonValue | str | None:
        """Issue an authenticated request against a deployment route.

        Maps transport errors to ``APIError(0, …)``, translates status
        codes through :func:`_common.raise_for_deployment`, and decodes
        JSON on success.  Returns ``None`` for empty bodies (e.g. 204).

        ``idempotent=True`` mints an ``Idempotency-Key`` so a transient failure
        on a create POST retries into a server-side replay instead of a
        duplicate deployment.
        """
        url = f"{self.api_url}{path}"
        resp = self._request(
            method,
            url,
            raise_for=lambda r: raise_for_deployment(r, deployment_id or "deployment"),
            params=requests_query_params(params),
            json=json_body,
            timeout=timeout,
            allow_redirects=ALLOW_REDIRECTS,
            idempotent=idempotent,
        )
        if not resp.content:
            return None
        try:
            return response_json_value(resp)
        except ResponseError:
            return resp.text

    def _deployment_object(
        self,
        method: str,
        path: str,
        *,
        deployment_id: str | None = None,
        params: QueryParams | None = None,
        json_body: JsonValue = None,
        timeout: int = DEFAULT_TIMEOUT,
        idempotent: bool = False,
    ) -> JsonObject:
        value = self._deployment_request(
            method,
            path,
            deployment_id=deployment_id,
            params=params,
            json_body=json_body,
            timeout=timeout,
            idempotent=idempotent,
        )
        if isinstance(value, dict):
            return value
        raise TypeError(f"Expected JSON object, got {type(value).__name__}")

    def list_deployments(
        self,
        *,
        page: int = 1,
        limit: int = 20,
        status_filter: str | None = None,
        platform: str | None = None,
        project_id: str | None = None,
        search: str | None = None,
    ) -> JsonObject | str | None:
        """GET /api/v1/deployments"""
        params: dict[str, str | int] = {"page": page, "limit": limit}
        if status_filter is not None:
            params["status"] = status_filter
        if platform is not None:
            params["platform"] = platform
        if project_id is not None:
            params["project_id"] = project_id
        if search is not None:
            params["search"] = search
        value = self._deployment_request("GET", "/api/v1/deployments", params=params)
        if isinstance(value, dict | str) or value is None:
            return value
        raise TypeError(f"Expected JSON object, got {type(value).__name__}")

    def get_deployment(self, deployment_id: str) -> JsonObject:
        """GET /api/v1/deployments/{id}"""
        return self._deployment_object(
            "GET",
            f"/api/v1/deployments/{quote_path_segment(deployment_id)}",
            deployment_id=deployment_id,
        )

    def create_deployment(self, payload: JsonObject) -> JsonObject:
        """POST /api/v1/deployments"""
        return self._deployment_object(
            "POST", "/api/v1/deployments", json_body=payload, idempotent=True
        )

    def estimate_cost(self, payload: JsonObject) -> JsonObject:
        """Estimate deployment cost. ``POST /api/v1/deployments/estimate-cost``."""
        return self._deployment_object(
            "POST", "/api/v1/deployments/estimate-cost", json_body=payload
        )

    def validate_deployment(self, payload: JsonObject) -> JsonObject:
        """Validate a deployment config without creating it. ``POST /api/v1/deployments/validate``."""
        return self._deployment_object("POST", "/api/v1/deployments/validate", json_body=payload)

    def list_deployment_platforms(self) -> JsonArray:
        """List serving platforms and capabilities. ``GET /api/v1/deployments-platforms``.

        Note the hyphenated path: this is a sibling of ``/deployments`` on the
        backend router, not a ``/deployments/platforms`` child route.
        """
        return ensure_json_array(self._deployment_request("GET", "/api/v1/deployments-platforms"))

    def update_deployment(self, deployment_id: str, payload: JsonObject) -> JsonObject:
        """PUT /api/v1/deployments/{id}"""
        return self._deployment_object(
            "PUT",
            f"/api/v1/deployments/{quote_path_segment(deployment_id)}",
            deployment_id=deployment_id,
            json_body=payload,
        )

    def delete_deployment(self, deployment_id: str) -> JsonObject | None:
        """DELETE /api/v1/deployments/{id}"""
        value = self._deployment_request(
            "DELETE",
            f"/api/v1/deployments/{quote_path_segment(deployment_id)}",
            deployment_id=deployment_id,
        )
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        raise TypeError(f"Expected JSON object, got {type(value).__name__}")

    def pause_deployment(self, deployment_id: str) -> JsonObject:
        return self._deployment_object(
            "POST",
            f"/api/v1/deployments/{quote_path_segment(deployment_id)}/pause",
            deployment_id=deployment_id,
        )

    def resume_deployment(self, deployment_id: str) -> JsonObject:
        return self._deployment_object(
            "POST",
            f"/api/v1/deployments/{quote_path_segment(deployment_id)}/resume",
            deployment_id=deployment_id,
        )

    def scale_deployment(self, deployment_id: str, num_instances: int) -> JsonObject:
        return self._deployment_object(
            "PUT",
            f"/api/v1/deployments/{quote_path_segment(deployment_id)}/scale",
            deployment_id=deployment_id,
            params={"num_instances": num_instances},
        )

    def rollback_deployment(self, deployment_id: str, checkpoint_id: str) -> JsonObject:
        return self._deployment_object(
            "POST",
            f"/api/v1/deployments/{quote_path_segment(deployment_id)}/rollback",
            deployment_id=deployment_id,
            params={"checkpoint_id": checkpoint_id},
        )

    def retry_deployment(self, deployment_id: str) -> JsonObject:
        """Retry a failed or stuck deployment. ``POST /api/v1/deployments/{id}/retry``.

        Distinct from :meth:`rollback_deployment` (which redeploys a prior
        checkpoint) and :meth:`resume_deployment` (which un-pauses): retry
        re-queues the existing deployment task with no body.
        """
        return self._deployment_object(
            "POST",
            f"/api/v1/deployments/{quote_path_segment(deployment_id)}/retry",
            deployment_id=deployment_id,
        )

    def get_deployment_metrics(self, deployment_id: str, time_range: str = "24h") -> JsonObject:
        return self._deployment_object(
            "GET",
            f"/api/v1/deployments/{quote_path_segment(deployment_id)}/metrics",
            deployment_id=deployment_id,
            params={"time_range": time_range},
        )

    def collect_deployment_metrics(
        self, deployment_id: str, backfill_minutes: int = 60
    ) -> JsonObject:
        """Trigger an immediate metrics collection (with first-time backfill).

        POST /api/v1/deployments/{id}/metrics/collect
        """
        return self._deployment_object(
            "POST",
            f"/api/v1/deployments/{quote_path_segment(deployment_id)}/metrics/collect",
            deployment_id=deployment_id,
            params={"backfill_minutes": backfill_minutes},
        )

    def get_deployment_logs(
        self,
        deployment_id: str,
        *,
        level: str | None = None,
        search: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        page: int = 1,
        limit: int = 100,
    ) -> JsonObject:
        params: dict[str, str | int] = {"page": page, "limit": limit}
        if level is not None:
            params["level"] = level
        if search is not None:
            params["search"] = search
        if start_time is not None:
            params["start_time"] = start_time
        if end_time is not None:
            params["end_time"] = end_time
        return self._deployment_object(
            "GET",
            f"/api/v1/deployments/{quote_path_segment(deployment_id)}/logs",
            deployment_id=deployment_id,
            params=params,
        )

    def get_deployment_health_full(self, deployment_id: str) -> JsonObject:
        """GET /api/v1/deployments/{id}/health — platform-side health row.

        Distinct from :meth:`deployment_health` which hits the *inference*
        endpoint.  This returns the deployment's own health_status column.
        """
        return self._deployment_object(
            "GET",
            f"/api/v1/deployments/{quote_path_segment(deployment_id)}/health",
            deployment_id=deployment_id,
        )

    def mint_deployment_stream_token(self, deployment_id: str) -> str:
        """Mint a short-lived stream-access token for one deployment's SSE stream."""
        body = self._deployment_object(
            "POST",
            f"/api/v1/deployments/{quote_path_segment(deployment_id)}/stream-access-token",
            deployment_id=deployment_id,
        )
        return str(body["token"])

    def open_deployment_stream(
        self, deployment_id: str, last_event_id: str | None = None
    ) -> requests.Response:
        """Open an SSE stream for a deployment.

        GET /api/v1/deployments/{id}/stream?token=...
        """
        token = self.mint_deployment_stream_token(deployment_id)
        url = f"{self.api_url}/api/v1/deployments/{quote_path_segment(deployment_id)}/stream"
        params = stream_query_params(token)
        headers = {"Accept": "text/event-stream"}
        if last_event_id:
            headers["Last-Event-ID"] = last_event_id
        try:
            resp = requests.get(
                url,
                params=params,
                headers=headers,
                stream=True,
                timeout=(STREAM_CONNECT_TIMEOUT, SSE_READ_TIMEOUT),
                allow_redirects=ALLOW_REDIRECTS,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        raise_for_deployment(resp, deployment_id)
        return resp
