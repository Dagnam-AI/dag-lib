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
    response_json_object,
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
        idempotency_key: str | None = None,
    ) -> JsonValue | str | None:
        """Issue an authenticated request against a deployment route.

        Maps transport errors to ``APIError(0, …)``, translates status
        codes through :func:`_common.raise_for_deployment`, and decodes
        JSON on success.  Returns ``None`` for empty bodies (e.g. 204).

        ``idempotent=True`` mints an ``Idempotency-Key`` (unless an explicit
        ``idempotency_key`` is given) so a transient failure on a create POST
        retries into a server-side replay instead of a duplicate deployment.
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
            idempotency_key=idempotency_key,
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
        idempotency_key: str | None = None,
    ) -> JsonObject:
        value = self._deployment_request(
            method,
            path,
            deployment_id=deployment_id,
            params=params,
            json_body=json_body,
            timeout=timeout,
            idempotent=idempotent,
            idempotency_key=idempotency_key,
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

    def _post_created_deployment(
        self, path: str, body: JsonObject, *, idempotency_key: str | None = None
    ) -> JsonObject:
        """POST a create route; return the deployment with a usable one-time key.

        The server's idempotency cache never stores the key, so a replayed create
        (``Idempotency-Replayed: true``) comes back with ``api_key: null``. The
        caller never saw the original key, so it is re-issued through
        :meth:`rotate_deployment_key` and merged into the result. That rotation
        revokes the earlier key: a caller who replays with its own
        ``idempotency_key`` after storing the first response's key must store
        the new one.
        """
        resp = self._request(
            "POST",
            f"{self.api_url}{path}",
            raise_for=lambda r: raise_for_deployment(r, "deployment"),
            json=body,
            timeout=DEFAULT_TIMEOUT,
            allow_redirects=ALLOW_REDIRECTS,
            idempotent=True,
            idempotency_key=idempotency_key,
        )
        created = response_json_object(resp)
        if resp.headers.get("Idempotency-Replayed") == "true" and created.get("api_key") is None:
            created.update(self.rotate_deployment_key(str(created["id"])))
        return created

    def create_deployment(self, payload: JsonObject) -> JsonObject:
        """POST /api/v1/deployments"""
        return self._post_created_deployment("/api/v1/deployments", payload)

    def deploy_model_version(
        self,
        model_version_id: str,
        *,
        name: str | None = None,
        project_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> JsonObject:
        """POST /api/v1/deployments/from-model-version: deploy a registry model version.

        Creates the deployment and its first serving revision in one call and
        returns the deployment with its one-time ``api_key``. The revision
        activates asynchronously; poll :meth:`get_deployment_revisions`. The
        server names the deployment after the model version and uses the model
        entry's project unless ``name`` / ``project_id`` are given. An
        ``Idempotency-Key`` is minted when none is given, so a retried request
        replays instead of creating a second deployment. A replay rotates the
        key (see :meth:`_post_created_deployment`), so reusing an explicit
        ``idempotency_key`` revokes the key an earlier call returned.
        """
        body: JsonObject = {"model_version_id": model_version_id}
        if name is not None:
            body["name"] = name
        if project_id is not None:
            body["project_id"] = project_id
        return self._post_created_deployment(
            "/api/v1/deployments/from-model-version", body, idempotency_key=idempotency_key
        )

    def rotate_deployment_key(self, deployment_id: str) -> JsonObject:
        """POST /api/v1/deployments/{id}/rotate-key: replace the deployment's key.

        Returns ``{key_prefix, api_key}``; the new key is shown only here and
        the previous one stops working at once.
        """
        return self._deployment_object(
            "POST",
            f"/api/v1/deployments/{quote_path_segment(deployment_id)}/rotate-key",
            deployment_id=deployment_id,
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

    def get_deployment_metrics(self, deployment_id: str, time_range: str = "24h") -> JsonObject:
        return self._deployment_object(
            "GET",
            f"/api/v1/deployments/{quote_path_segment(deployment_id)}/metrics",
            deployment_id=deployment_id,
            params={"time_range": time_range},
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

    def get_deployment_revisions(
        self, deployment_id: str, *, page: int = 1, limit: int = 50
    ) -> JsonArray:
        """GET /api/v1/deployments/{id}/revisions — revision history, newest first."""
        return ensure_json_array(
            self._deployment_request(
                "GET",
                f"/api/v1/deployments/{quote_path_segment(deployment_id)}/revisions",
                deployment_id=deployment_id,
                params={"page": page, "limit": limit},
            )
        )

    def create_deployment_revision(
        self,
        deployment_id: str,
        payload: JsonObject,
        *,
        idempotency_key: str | None = None,
    ) -> JsonObject:
        """POST /api/v1/deployments/{id}/revisions — roll out a new model version.

        The route requires an ``Idempotency-Key``; one is minted when the
        caller gives none, so a retried request replays instead of creating a
        second revision. Returns the revision as created (its activation is
        asynchronous — poll :meth:`get_deployment_revisions`).
        """
        return self._deployment_object(
            "POST",
            f"/api/v1/deployments/{quote_path_segment(deployment_id)}/revisions",
            deployment_id=deployment_id,
            json_body=payload,
            idempotent=True,
            idempotency_key=idempotency_key,
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
