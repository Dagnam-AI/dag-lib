"""Unit tests for dagnam.inference."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from dagnam import inference, inference_batch, deployment_health
from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import (
    APIError,
    AuthError,
    DeploymentNotFoundError,
)


def _mock_response(status: int, body=None, ok: bool | None = None) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    resp.ok = (status < 400) if ok is None else ok
    resp.text = str(body) if body is not None else ""
    resp.json.return_value = body if body is not None else {}
    return resp


class TestInferenceDelegation:
    def test_inference_uses_provided_client(self):
        client = MagicMock(spec=DagnamClient)
        client.predict.return_value = {"label": "cat"}
        result = inference("dep_1", {"x": 1}, client=client)
        client.predict.assert_called_once_with("dep_1", {"x": 1}, timeout=30)
        assert result == {"label": "cat"}

    def test_batch_uses_provided_client(self):
        client = MagicMock(spec=DagnamClient)
        client.predict_batch.return_value = [{"y": 1}, {"y": 2}]
        result = inference_batch("dep_1", [{"x": 1}, {"x": 2}], client=client)
        client.predict_batch.assert_called_once_with(
            "dep_1", [{"x": 1}, {"x": 2}], timeout=30
        )
        assert result == [{"y": 1}, {"y": 2}]

    def test_health_uses_provided_client(self):
        client = MagicMock(spec=DagnamClient)
        client.deployment_health.return_value = {"status": "healthy"}
        result = deployment_health("dep_1", client=client)
        client.deployment_health.assert_called_once_with("dep_1")
        assert result == {"status": "healthy"}


class TestAuthResolution:
    def test_inference_builds_client_from_explicit_creds(self):
        with patch("dagnam._core._resolver.DagnamClient") as MockClient:
            instance = MockClient.return_value
            instance.predict.return_value = {"ok": True}
            inference(
                "dep_1",
                {"x": 1},
                api_key="secret",
                api_url="https://example.test",
            )
            MockClient.assert_called_once_with("https://example.test", "secret")


class TestClientErrorMapping:
    def test_predict_maps_401(self):
        client = DagnamClient("https://x", "key")
        with patch("dagnam._core.client.requests.post", return_value=_mock_response(401)):
            with pytest.raises(AuthError):
                client.predict("dep_1", {"x": 1})

    def test_predict_maps_404(self):
        client = DagnamClient("https://x", "key")
        with patch("dagnam._core.client.requests.post", return_value=_mock_response(404)):
            with pytest.raises(DeploymentNotFoundError):
                client.predict("dep_404", {"x": 1})

    def test_predict_maps_500(self):
        client = DagnamClient("https://x", "key")
        resp = _mock_response(500, body="boom")
        with patch("dagnam._core.client.requests.post", return_value=resp):
            with pytest.raises(APIError) as exc:
                client.predict("dep_1", {"x": 1})
            assert exc.value.status_code == 500

    def test_health_returns_json(self):
        client = DagnamClient("https://x", "key")
        resp = _mock_response(200, body={"status": "healthy"})
        with patch("dagnam._core.client.requests.get", return_value=resp):
            assert client.deployment_health("dep_1") == {"status": "healthy"}
