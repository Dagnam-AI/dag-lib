"""Unit tests for dagnam.inference."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import requests_mock as rm_module

from dagnam import deployment_health, inference, inference_batch, inference_schema
from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import (
    APIError,
    AuthError,
    DeploymentNotFoundError,
)

if TYPE_CHECKING:
    from tests.typing_helpers import RequestsMocker


@pytest.fixture
def rmock():
    with rm_module.Mocker() as m:
        yield m


class TestInferenceDelegation:
    def test_inference_uses_provided_client(self) -> None:
        client = MagicMock(spec=DagnamClient)
        client.predict.return_value = {"label": "cat"}
        result = inference("dep_1", {"x": 1}, client=client)
        client.predict.assert_called_once_with("dep_1", {"x": 1}, timeout=30)
        assert result == {"label": "cat"}

    def test_batch_uses_provided_client(self) -> None:
        client = MagicMock(spec=DagnamClient)
        client.predict_batch.return_value = [{"y": 1}, {"y": 2}]
        result = inference_batch("dep_1", [{"x": 1}, {"x": 2}], client=client)
        client.predict_batch.assert_called_once_with("dep_1", [{"x": 1}, {"x": 2}], timeout=30)
        assert result == [{"y": 1}, {"y": 2}]

    def test_health_uses_provided_client(self) -> None:
        client = MagicMock(spec=DagnamClient)
        client.deployment_health.return_value = {"status": "healthy"}
        result = deployment_health("dep_1", client=client)
        client.deployment_health.assert_called_once_with("dep_1")
        assert result == {"status": "healthy"}

    def test_schema_uses_provided_client(self) -> None:
        client = MagicMock(spec=DagnamClient)
        client.schema.return_value = {"input_schema": {}, "output_schema": {}}
        result = inference_schema("dep_1", client=client)
        client.schema.assert_called_once_with("dep_1")
        assert "input_schema" in result


class TestAuthResolution:
    def test_inference_builds_client_from_explicit_creds(self) -> None:
        with patch("dagnam._core.resolver.DagnamClient") as MockClient:
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
    def test_predict_maps_401(self, rmock: RequestsMocker) -> None:
        rmock.post("https://x/api/v1/inference/dep_1/predict", status_code=401)
        with pytest.raises(AuthError):
            DagnamClient("https://x", "key").predict("dep_1", {"x": 1})

    def test_predict_maps_404(self, rmock: RequestsMocker) -> None:
        rmock.post("https://x/api/v1/inference/dep_404/predict", status_code=404)
        with pytest.raises(DeploymentNotFoundError):
            DagnamClient("https://x", "key").predict("dep_404", {"x": 1})

    def test_predict_maps_500(self, rmock: RequestsMocker) -> None:
        # POST is not retried, so a single 500 surfaces directly.
        rmock.post("https://x/api/v1/inference/dep_1/predict", status_code=500, text="boom")
        with pytest.raises(APIError) as exc:
            DagnamClient("https://x", "key").predict("dep_1", {"x": 1})
        assert exc.value.status_code == 500
        assert rmock.call_count == 1

    def test_health_returns_json(self, rmock: RequestsMocker) -> None:
        rmock.get("https://x/api/v1/inference/dep_1/health", json={"status": "healthy"})
        assert DagnamClient("https://x", "key").deployment_health("dep_1") == {"status": "healthy"}

    def test_schema_returns_json(self, rmock: RequestsMocker) -> None:
        rmock.get(
            "https://x/api/v1/inference/dep_1/schema",
            json={"input_schema": {"type": "object"}, "output_schema": {}},
        )
        out = DagnamClient("https://x", "key").schema("dep_1")
        assert out["input_schema"] == {"type": "object"}

    def test_schema_maps_404(self, rmock: RequestsMocker) -> None:
        rmock.get("https://x/api/v1/inference/dep_404/schema", status_code=404)
        with pytest.raises(DeploymentNotFoundError):
            DagnamClient("https://x", "key").schema("dep_404")


def test_inference_stream_delegates_to_iter_sse_once(monkeypatch) -> None:
    from dagnam._core.sse import SSEEvent
    from dagnam.resources import inference as inference_mod

    captured = {}

    def fake_iter(open_stream, *, terminal_events, include_heartbeats, resource_label):
        captured["terminal_events"] = terminal_events
        captured["include_heartbeats"] = include_heartbeats
        captured["resource_label"] = resource_label
        captured["open_stream_callable"] = callable(open_stream)
        yield SSEEvent(event="token", data={"token": "hi"})
        yield SSEEvent(event="complete", data={})

    monkeypatch.setattr(inference_mod, "iter_sse_once", fake_iter)

    class _Client:
        def open_inference_stream(self, deployment_id, inputs):
            return object()

    events = list(
        inference_mod.inference_stream("dep_x", {"text": "hi"}, client=_Client())  # pyright: ignore[reportArgumentType]
    )
    assert [ev.event for ev in events] == ["token", "complete"]
    assert "complete" in captured["terminal_events"]
    assert "error" in captured["terminal_events"]
    assert captured["include_heartbeats"] is False
    assert captured["open_stream_callable"] is True
    assert "dep_x" in captured["resource_label"]


def test_inference_stream_wire_end_to_end(rmock) -> None:
    """Real client + mocked HTTP: token minted per connection, never the API key."""
    from dagnam._core.client import DagnamClient
    from dagnam.resources import inference as inference_mod

    client = DagnamClient("https://api.test", "k")
    rmock.post(
        "https://api.test/api/v1/inference/dep1/stream-access-token",
        json={"token": "stream-t-1"},
    )
    stream_route = rmock.get(
        "https://api.test/api/v1/inference/dep1/predict/stream",
        text=('event: token\ndata: {"token": "he"}\n\nevent: complete\ndata: {"done": true}\n\n'),
        headers={"Content-Type": "text/event-stream"},
    )
    events = list(inference_mod.inference_stream("dep1", {"text": "hi"}, client=client))
    assert [ev.event for ev in events] == ["token", "complete"]
    assert stream_route.request_history[0].qs["token"] == ["stream-t-1"]
