"""Unit + wire tests for dagnam.deployments."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest
import requests
from tests.typing_helpers import JsonObject

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

if TYPE_CHECKING:
    from tests.typing_helpers import RequestsMocker


# ---------------------------------------------------------------------------
# Delegation tests — functions call through to DagnamClient correctly
# ---------------------------------------------------------------------------


class TestReadDelegation:
    def test_list_passes_filters(self) -> None:
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

    def test_get_delegates(self) -> None:
        client = MagicMock(spec=DagnamClient)
        client.get_deployment.return_value = {"id": "dep-1", "status": "running"}
        out = deployments.get("dep-1", client=client)
        client.get_deployment.assert_called_once_with("dep-1")
        assert out["status"] == "running"

    def test_logs_forwards_all_filters(self) -> None:
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

    def test_revisions_delegates(self) -> None:
        client = MagicMock(spec=DagnamClient)
        client.get_deployment_revisions.return_value = [{"id": "rev1", "is_active": True}]
        result = deployments.revisions("dep-1", page=2, limit=5, client=client)
        assert result == [{"id": "rev1", "is_active": True}]
        client.get_deployment_revisions.assert_called_once_with("dep-1", page=2, limit=5)


class TestCreateRevision:
    def test_omits_capacity_policy_and_nulls_unless_given(self) -> None:
        client = MagicMock(spec=DagnamClient)
        client.create_deployment_revision.return_value = {"id": "rev2", "is_active": False}
        result = deployments.create_revision("dep-1", model_version_id="mv1", client=client)
        assert result == {"id": "rev2", "is_active": False}
        client.create_deployment_revision.assert_called_once_with(
            "dep-1",
            {
                "model_version_id": "mv1",
                "capacity_mode": "serverless",
            },
            idempotency_key=None,
        )

    def test_passes_every_field_through(self) -> None:
        client = MagicMock(spec=DagnamClient)
        client.create_deployment_revision.return_value = {"id": "rev3"}
        deployments.create_revision(
            "dep-1",
            model_version_id="mv1",
            capacity_mode="dedicated",
            capacity_policy={"min_replicas": 1, "max_replicas": 3},
            region="eu-west-1",
            engine_override="vllm",
            idempotency_key="key-9",
            client=client,
        )
        client.create_deployment_revision.assert_called_once_with(
            "dep-1",
            {
                "model_version_id": "mv1",
                "capacity_mode": "dedicated",
                "capacity_policy": {"min_replicas": 1, "max_replicas": 3},
                "region": "eu-west-1",
                "engine_override": "vllm",
            },
            idempotency_key="key-9",
        )


class TestDeployModelVersion:
    def test_delegates_and_returns_the_deployment_with_its_key(self) -> None:
        client = MagicMock(spec=DagnamClient)
        client.deploy_model_version.return_value = {"id": "dep-1", "api_key": "dep-key"}
        result = deployments.deploy_model_version("mv1", client=client)
        assert result == {"id": "dep-1", "api_key": "dep-key"}
        client.deploy_model_version.assert_called_once_with("mv1", name=None, project_id=None)

    def test_passes_name_and_project(self) -> None:
        client = MagicMock(spec=DagnamClient)
        client.deploy_model_version.return_value = {"id": "dep-1"}
        deployments.deploy_model_version("mv1", name="bot", project_id="p1", client=client)
        client.deploy_model_version.assert_called_once_with("mv1", name="bot", project_id="p1")


class TestLifecycleLRO:
    def test_create_returns_lro_with_initial_payload(self) -> None:
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
        initial = op.initial()
        assert initial is not None
        assert initial["id"] == "dep-1"
        # Body built correctly (omit optional Nones)
        sent = client.create_deployment.call_args.args[0]
        assert sent["name"] == "d"
        assert sent["platform"] == "fastapi"
        assert "min_instances" not in sent
        assert "region" not in sent

    def test_create_includes_optional_fields_when_set(self) -> None:
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

    def test_pause_success_state_is_paused(self) -> None:
        client = MagicMock(spec=DagnamClient)
        client.pause_deployment.return_value = {"id": "dep-1", "status": "paused"}
        client.get_deployment.return_value = {"id": "dep-1", "status": "paused"}
        op = deployments.pause("dep-1", client=client)
        op.wait(timeout=5).result()

    def test_update_renames(self) -> None:
        client = MagicMock(spec=DagnamClient)
        client.update_deployment.return_value = {"id": "dep-1", "name": "renamed"}
        out = deployments.update("dep-1", name="renamed", client=client)
        client.update_deployment.assert_called_once_with("dep-1", {"name": "renamed"})
        assert out["name"] == "renamed"


# ---------------------------------------------------------------------------
# Wire-level error mapping on DagnamClient methods
# ---------------------------------------------------------------------------


class TestClientErrorMapping:
    def _client(self) -> DagnamClient:
        client = DagnamClient("https://x", "key")
        client._sleep = lambda _s: None  # retryable verbs don't sleep on transient errors
        return client

    def test_get_maps_404(self, requests_mock: RequestsMocker) -> None:
        requests_mock.get("https://x/api/v1/deployments/missing", status_code=404)
        with pytest.raises(DeploymentNotFoundError):
            self._client().get_deployment("missing")

    def test_create_maps_401(self, requests_mock: RequestsMocker) -> None:
        requests_mock.post("https://x/api/v1/deployments", status_code=401)
        with pytest.raises(AuthError):
            self._client().create_deployment({"name": "x"})

    def test_create_maps_422(self, requests_mock: RequestsMocker) -> None:
        requests_mock.post("https://x/api/v1/deployments", status_code=422, text="bad fields")
        with pytest.raises(DeploymentValidationError):
            self._client().create_deployment({"name": "x"})

    def test_connectionerror_wrapped(self, requests_mock: RequestsMocker) -> None:
        requests_mock.get(
            "https://x/api/v1/deployments/dep-1", exc=requests.ConnectionError("boom")
        )
        with pytest.raises(APIError):
            self._client().get_deployment("dep-1")

    def test_list_success_returns_dict(self, requests_mock: RequestsMocker) -> None:
        body = {"items": [{"id": "dep-1"}], "total": 1, "page": 1}
        requests_mock.get("https://x/api/v1/deployments", json=body)
        out = cast("JsonObject", self._client().list_deployments(page=1, limit=20))
        items = cast("list[JsonObject]", out["items"])
        assert items[0]["id"] == "dep-1"


# ---------------------------------------------------------------------------
# End-to-end LRO via the public surface
# ---------------------------------------------------------------------------


class TestEndToEndLRO:
    def test_create_then_wait_polls_get_until_running(self) -> None:
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
        op.configure_polling(0.001, 0.001)
        dep = op.wait(timeout=5).result()
        assert dep["status"] == "running"
        assert dep["endpoint_url"] == "https://e"


# ---------------------------------------------------------------------------
# Planning delegation — validate / platforms
# ---------------------------------------------------------------------------


class TestPlanningDelegation:
    def test_validate_minimal_payload(self) -> None:
        client = MagicMock(spec=DagnamClient)
        client.validate_deployment.return_value = {"valid": True, "errors": []}
        out = deployments.validate(
            name="d",
            project_id="p1",
            checkpoint_path="/c.pt",
            platform="fastapi",
            deployment_type="text",
            instance_type="cpu.small",
            client=client,
        )
        assert out["valid"] is True
        sent = client.validate_deployment.call_args.args[0]
        assert sent["project_id"] == "p1"
        assert "min_instances" not in sent
        assert "config" not in sent

    def test_validate_includes_optional_fields(self) -> None:
        client = MagicMock(spec=DagnamClient)
        client.validate_deployment.return_value = {
            "valid": False,
            "errors": [{"field": "name", "message": "bad", "code": "x"}],
        }
        deployments.validate(
            name="d",
            project_id="p1",
            checkpoint_path="/c.pt",
            platform="fastapi",
            deployment_type="text",
            instance_type="cpu.small",
            num_instances=2,
            auto_scaling_enabled=True,
            min_instances=1,
            max_instances=5,
            region="us-east-1",
            config={"k": "v"},
            client=client,
        )
        sent = client.validate_deployment.call_args.args[0]
        assert sent["min_instances"] == 1
        assert sent["max_instances"] == 5
        assert sent["region"] == "us-east-1"
        assert sent["config"] == {"k": "v"}

    def test_platforms_delegates(self) -> None:
        client = MagicMock(spec=DagnamClient)
        client.list_deployment_platforms.return_value = [{"platform": "fastapi"}]
        out = deployments.platforms(client=client)
        first = out[0]
        assert isinstance(first, dict)
        assert first["platform"] == "fastapi"


def test_predict_stream_is_inference_stream_alias(monkeypatch) -> None:
    from dagnam.resources import deployments as deployments_mod

    calls = {}

    def fake_stream(
        deployment_id, inputs, *, include_heartbeats=False, client=None, api_key=None, api_url=None
    ):
        calls["args"] = (deployment_id, inputs, include_heartbeats, client)
        return iter(())

    monkeypatch.setattr(deployments_mod, "inference_stream", fake_stream)
    result = deployments_mod.predict_stream("dep1", {"text": "x"}, client="C")  # pyright: ignore[reportArgumentType]
    assert list(result) == []
    assert calls["args"] == ("dep1", {"text": "x"}, False, "C")


@pytest.mark.parametrize(
    "name",
    ["collect_metrics", "create_from_training_job", "estimate_cost", "retry", "rollback", "scale"],
)
def test_removed_functions_are_gone(name: str) -> None:
    assert not hasattr(deployments, name)
    assert name not in deployments.__all__


class TestSetWarm:
    @pytest.mark.parametrize("warm", [True, False])
    def test_delegates_with_the_flag(self, warm: bool) -> None:
        client = MagicMock(spec=DagnamClient)
        client.set_deployment_warm.return_value = {"warm": warm}
        result = deployments.set_warm("dep-1", warm, client=client)
        assert result == {"warm": warm}
        client.set_deployment_warm.assert_called_once_with("dep-1", warm)

    def test_is_exported_at_the_top_level(self) -> None:
        import dagnam

        assert dagnam.set_warm is deployments.set_warm

    def test_patches_the_capacity_route_over_the_wire(self, requests_mock: RequestsMocker) -> None:
        requests_mock.patch(
            "https://api.test/api/v1/deployments/dep-1/capacity", json={"warm": True}
        )
        result = deployments.set_warm("dep-1", True, api_key="k", api_url="https://api.test")
        assert result == {"warm": True}
        assert requests_mock.last_request.json() == {"warm": True}

    def test_a_deployment_that_is_not_serving_yet_is_a_state_error(
        self, requests_mock: RequestsMocker
    ) -> None:
        requests_mock.patch("https://api.test/api/v1/deployments/dep-1/capacity", status_code=409)
        with pytest.raises(DeploymentStateError):
            deployments.set_warm("dep-1", True, api_key="k", api_url="https://api.test")

    def test_someone_elses_deployment_is_not_found(self, requests_mock: RequestsMocker) -> None:
        requests_mock.patch("https://api.test/api/v1/deployments/dep-1/capacity", status_code=404)
        with pytest.raises(DeploymentNotFoundError):
            deployments.set_warm("dep-1", False, api_key="k", api_url="https://api.test")
