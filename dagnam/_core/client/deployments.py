"""Synchronous deployments client methods."""

from __future__ import annotations

from typing import Any

from dagnam._core.client.base import _TIMEOUT, APIError, requests


class DeploymentsClientMixin:
    """Deployments resource methods for DagnamClient."""

    def _deployment_request(
        self,
        method: str,
        path: str,
        *,
        deployment_id: str | None = None,
        params: dict | None = None,
        json_body: Any = None,
        timeout: int = _TIMEOUT,
    ) -> dict | list | None:
        """Issue an authenticated request against a deployment route.

        Maps transport errors to ``APIError(0, …)``, translates status
        codes through :func:`_common.raise_for_deployment`, and decodes
        JSON on success.  Returns ``None`` for empty bodies (e.g. 204).
        """
        from dagnam._core.client.common import raise_for_deployment

        url = f"{self.api_url}{path}"
        try:
            resp = requests.request(
                method,
                url,
                headers=self._headers(),
                params=params,
                json=json_body,
                timeout=timeout,
            )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        raise_for_deployment(resp, deployment_id or "deployment")

        if not resp.content:
            return None
        try:
            return resp.json()
        except ValueError:
            return resp.text

    def list_deployments(
        self,
        *,
        page: int = 1,
        limit: int = 20,
        status_filter: str | None = None,
        platform: str | None = None,
        project_id: str | None = None,
        search: str | None = None,
    ) -> dict:
        """GET /api/v1/deployments"""
        params: dict[str, Any] = {"page": page, "limit": limit}
        if status_filter is not None:
            params["status"] = status_filter
        if platform is not None:
            params["platform"] = platform
        if project_id is not None:
            params["project_id"] = project_id
        if search is not None:
            params["search"] = search
        return self._deployment_request("GET", "/api/v1/deployments", params=params)

    def get_deployment(self, deployment_id: str) -> dict:
        """GET /api/v1/deployments/{id}"""
        return self._deployment_request(
            "GET",
            f"/api/v1/deployments/{deployment_id}",
            deployment_id=deployment_id,
        )

    def create_deployment(self, payload: dict) -> dict:
        """POST /api/v1/deployments"""
        return self._deployment_request("POST", "/api/v1/deployments", json_body=payload)

    def update_deployment(self, deployment_id: str, payload: dict) -> dict:
        """PUT /api/v1/deployments/{id}"""
        return self._deployment_request(
            "PUT",
            f"/api/v1/deployments/{deployment_id}",
            deployment_id=deployment_id,
            json_body=payload,
        )

    def delete_deployment(self, deployment_id: str) -> dict | None:
        """DELETE /api/v1/deployments/{id}"""
        return self._deployment_request(
            "DELETE",
            f"/api/v1/deployments/{deployment_id}",
            deployment_id=deployment_id,
        )

    def pause_deployment(self, deployment_id: str) -> dict:
        return self._deployment_request(
            "POST",
            f"/api/v1/deployments/{deployment_id}/pause",
            deployment_id=deployment_id,
        )

    def resume_deployment(self, deployment_id: str) -> dict:
        return self._deployment_request(
            "POST",
            f"/api/v1/deployments/{deployment_id}/resume",
            deployment_id=deployment_id,
        )

    def scale_deployment(self, deployment_id: str, num_instances: int) -> dict:
        return self._deployment_request(
            "PUT",
            f"/api/v1/deployments/{deployment_id}/scale",
            deployment_id=deployment_id,
            params={"num_instances": num_instances},
        )

    def rollback_deployment(self, deployment_id: str, checkpoint_path: str) -> dict:
        return self._deployment_request(
            "POST",
            f"/api/v1/deployments/{deployment_id}/rollback",
            deployment_id=deployment_id,
            params={"checkpoint_path": checkpoint_path},
        )

    def get_deployment_metrics(self, deployment_id: str, time_range: str = "24h") -> dict:
        return self._deployment_request(
            "GET",
            f"/api/v1/deployments/{deployment_id}/metrics",
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
    ) -> dict:
        params: dict[str, Any] = {"page": page, "limit": limit}
        if level is not None:
            params["level"] = level
        if search is not None:
            params["search"] = search
        if start_time is not None:
            params["start_time"] = start_time
        if end_time is not None:
            params["end_time"] = end_time
        return self._deployment_request(
            "GET",
            f"/api/v1/deployments/{deployment_id}/logs",
            deployment_id=deployment_id,
            params=params,
        )

    def get_deployment_health_full(self, deployment_id: str) -> dict:
        """GET /api/v1/deployments/{id}/health — platform-side health row.

        Distinct from :meth:`deployment_health` which hits the *inference*
        endpoint.  This returns the deployment's own health_status column.
        """
        return self._deployment_request(
            "GET",
            f"/api/v1/deployments/{deployment_id}/health",
            deployment_id=deployment_id,
        )

    def open_deployment_stream(
        self, deployment_id: str, last_event_id: str | None = None
    ) -> requests.Response:
        """Open an SSE stream for a deployment (``?api_key=`` auth).

        GET /api/v1/deployments/{id}/stream?api_key=...
        """
        from dagnam._core.client.common import raise_for_deployment

        url = f"{self.api_url}/api/v1/deployments/{deployment_id}/stream"
        params = {"api_key": self.api_key}
        headers = {"Accept": "text/event-stream"}
        if last_event_id:
            headers["Last-Event-ID"] = last_event_id
        try:
            resp = requests.get(url, params=params, headers=headers, stream=True, timeout=_TIMEOUT)
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc

        raise_for_deployment(resp, deployment_id)
        return resp
