"""Unit + wire tests for dagnam.deployments."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from dagnam import deployments
from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import (
    APIError,
    AuthError,
    DeploymentNotFoundError,
    DeploymentStateError,
    DeploymentValidationError,
)
from dagnam._core.lro import LongRunningOperation


def _mock_response(status: int, body=None) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    resp.ok = status < 400
    resp.text = str(body) if body is not None else ""
    resp.content = b"x" if body is not None else b""
    resp.json.return_value = body if body is not None else {}
    return resp


# ---------------------------------------------------------------------------
# Delegation tests — functions call through to DagnamClient correctly
# ---------------------------------------------------------------------------


class TestReadDelegation:
    def test_list_passes_filters(self):
        client = MagicMock(spec=DagnamClient)
        client.list_deployments.return_value = {"items": [], "total": 0}
        deployments.list(
            page=2,
            limit=50,
            status="running",
            platform="fastapi",
            project_id="p1",
            search="foo",
            client=client,
        )
        client.list_deployments.assert_called_once_with(
            page=2,
            limit=50,
            status_filter="running",
            platform="fastapi",
            project_id="p1",
            search="foo",
        )

    def test_get_delegates(self):
        client = MagicMock(spec=DagnamClient)
        client.get_deployment.return_value = {"id": "dep-1", "status": "running"}
        out = deployments.get("dep-1", client=client)
        client.get_deployment.assert_called_once_with("dep-1")
        assert out["status"] == "running"

    def test_logs_forwards_all_filters(self):
        client = MagicMock(spec=DagnamClient)
        client.get_deployment_logs.return_value = {"items": []}
        deployments.logs(
            "dep-1",
            level="error",
            search="boom",
            start_time="2026-01-01",
            end_time="2026-01-02",
            page=3,
            limit=25,
            client=client,
        )
        client.get_deployment_logs.assert_called_once_with(
            "dep-1",
            level="error",
            search="boom",
            start_time="2026-01-01",
            end_time="2026-01-02",
            page=3,
            limit=25,
        )


class TestLifecycleLRO:
    def test_create_returns_lro_with_initial_payload(self):
        client = MagicMock(spec=DagnamClient)
        client.create_deployment.return_value = {"id": "dep-1", "status": "deploying"}
        op = deployments.create(
            name="d",
            project_id="p",
            checkpoint_path="/ckpt",
            platform="fastapi",
            deployment_type="text",
            instance_type="t3.medium",
            client=client,
        )
        assert isinstance(op, LongRunningOperation)
        assert op.initial()["id"] == "dep-1"
        # Body built correctly (omit optional Nones)
        sent = client.create_deployment.call_args.args[0]
        assert sent["name"] == "d"
        assert sent["platform"] == "fastapi"
        assert "min_instances" not in sent
        assert "region" not in sent

    def test_create_includes_optional_fields_when_set(self):
        client = MagicMock(spec=DagnamClient)
        client.create_deployment.return_value = {"id": "dep-1", "status": "deploying"}
        deployments.create(
            name="d",
            project_id="p",
            checkpoint_path="/ckpt",
            platform="fastapi",
            deployment_type="text",
            instance_type="t3.medium",
            auto_scaling_enabled=True,
            min_instances=1,
            max_instances=5,
            region="us-east-1",
            config={"k": "v"},
            client=client,
        )
        sent = client.create_deployment.call_args.args[0]
        assert sent["auto_scaling_enabled"] is True
        assert sent["min_instances"] == 1
        assert sent["max_instances"] == 5
        assert sent["region"] == "us-east-1"
        assert sent["config"] == {"k": "v"}

    def test_scale_returns_lro_and_passes_count(self):
        client = MagicMock(spec=DagnamClient)
        client.scale_deployment.return_value = {"id": "dep-1", "status": "running"}
        client.get_deployment.return_value = {"id": "dep-1", "status": "running"}
        op = deployments.scale("dep-1", num_instances=4, client=client)
        client.scale_deployment.assert_called_once_with("dep-1", num_instances=4)
        assert isinstance(op, LongRunningOperation)
        # wait returns immediately because status is already running
        op.wait(timeout=5).result()

    def test_rollback_returns_lro(self):
        client = MagicMock(spec=DagnamClient)
        client.rollback_deployment.return_value = {"id": "dep-1", "status": "deploying"}
        op = deployments.rollback("dep-1", "/ckpts/v2.pt", client=client)
        client.rollback_deployment.assert_called_once_with("dep-1", checkpoint_path="/ckpts/v2.pt")
        assert isinstance(op, LongRunningOperation)

    def test_pause_success_state_is_paused(self):
        client = MagicMock(spec=DagnamClient)
        client.pause_deployment.return_value = {"id": "dep-1", "status": "paused"}
        client.get_deployment.return_value = {"id": "dep-1", "status": "paused"}
        op = deployments.pause("dep-1", client=client)
        op.wait(timeout=5).result()

    def test_update_is_synchronous(self):
        client = MagicMock(spec=DagnamClient)
        client.update_deployment.return_value = {"id": "dep-1", "num_instances": 2}
        out = deployments.update("dep-1", num_instances=2, name="renamed", client=client)
        payload = client.update_deployment.call_args.args[1]
        assert payload == {"num_instances": 2, "name": "renamed"}
        assert out["num_instances"] == 2


# ---------------------------------------------------------------------------
# Wire-level error mapping on DagnamClient methods
# ---------------------------------------------------------------------------


class TestClientErrorMapping:
    def _client(self) -> DagnamClient:
        return DagnamClient("https://x", "key")

    def test_get_maps_404(self):
        with patch("dagnam._core.client.base.requests.request", return_value=_mock_response(404)):
            with pytest.raises(DeploymentNotFoundError):
                self._client().get_deployment("missing")

    def test_create_maps_401(self):
        with patch("dagnam._core.client.base.requests.request", return_value=_mock_response(401)):
            with pytest.raises(AuthError):
                self._client().create_deployment({"name": "x"})

    def test_create_maps_422(self):
        with (
            patch(
                "dagnam._core.client.base.requests.request",
                return_value=_mock_response(422, "bad fields"),
            ),
            pytest.raises(DeploymentValidationError),
        ):
            self._client().create_deployment({"name": "x"})

    def test_scale_maps_409_to_state_error(self):
        with (
            patch(
                "dagnam._core.client.base.requests.request",
                return_value=_mock_response(409, "not running"),
            ),
            pytest.raises(DeploymentStateError),
        ):
            self._client().scale_deployment("dep-1", num_instances=3)

    def test_connection_error_wrapped(self):
        with (
            patch(
                "dagnam._core.client.base.requests.request",
                side_effect=requests.ConnectionError("boom"),
            ),
            pytest.raises(APIError),
        ):
            self._client().get_deployment("dep-1")

    def test_list_success_returns_dict(self):
        body = {"items": [{"id": "dep-1"}], "total": 1, "page": 1}
        with patch(
            "dagnam._core.client.base.requests.request", return_value=_mock_response(200, body)
        ):
            out = self._client().list_deployments(page=1, limit=20)
            assert out["items"][0]["id"] == "dep-1"


# ---------------------------------------------------------------------------
# End-to-end LRO via the public surface
# ---------------------------------------------------------------------------


class TestEndToEndLRO:
    def test_create_then_wait_polls_get_until_running(self):
        client = MagicMock(spec=DagnamClient)
        client.create_deployment.return_value = {"id": "dep-1", "status": "deploying"}
        # First get call still deploying, second call running
        client.get_deployment.side_effect = [
            {"id": "dep-1", "status": "deploying"},
            {"id": "dep-1", "status": "running", "endpoint_url": "https://e"},
        ]
        op = deployments.create(
            name="d",
            project_id="p",
            checkpoint_path="/ckpt",
            platform="fastapi",
            deployment_type="text",
            instance_type="t3.medium",
            client=client,
        )
        # Tight poll intervals to keep the test fast
        op._poll_min = 0.001
        op._poll_max = 0.001
        dep = op.wait(timeout=5).result()
        assert dep["status"] == "running"
        assert dep["endpoint_url"] == "https://e"
