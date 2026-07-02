"""stream_training must delegate to the shared reconnect engine."""

from __future__ import annotations

import time

import pytest
import requests_mock as rm_module

from dagnam._core.client import DagnamClient
from dagnam.resources import training as training_mod


@pytest.fixture
def rmock():
    with rm_module.Mocker() as m:
        yield m


def test_stream_training_delegates_to_iter_with_reconnect(monkeypatch):
    captured = {}

    def fake_iter(open_stream, *, terminal_events, include_heartbeats, **kwargs):
        captured["terminal_events"] = terminal_events
        captured["include_heartbeats"] = include_heartbeats
        captured["open_stream_callable"] = callable(open_stream)
        from dagnam._core.sse import SSEEvent

        yield SSEEvent(event="metric", data={"loss": 1.0}, id="1")
        yield SSEEvent(event="complete", data={}, id="2")

    monkeypatch.setattr(training_mod, "iter_with_reconnect", fake_iter)

    class _Client:
        def open_training_stream(self, job_id, last_event_id=None):
            return object()

    # _Client is a minimal duck-typed stand-in for DagnamClient.
    events = list(
        training_mod.stream_training("job_x", client=_Client())  # pyright: ignore[reportArgumentType]
    )

    assert captured["open_stream_callable"] is True
    assert "complete" in captured["terminal_events"]
    assert [event.event for event in events] == ["metric", "complete"]
    assert training_mod.TrainingEvent is not None


def test_stream_training_remints_stream_token_on_reconnect(rmock, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    client = DagnamClient("https://api.test", "k")
    mint_route = rmock.post(
        "https://api.test/api/v1/training/jobs/job_x/stream-access-token",
        [
            {"json": {"token": "stream-t-1"}},
            {"json": {"token": "stream-t-2"}},
        ],
    )
    stream_route = rmock.get(
        "https://api.test/api/v1/streaming/training-jobs/job_x/stream",
        [
            {
                "text": 'event: progress\ndata: {"loss": 1}\nid: evt-1\n\n',
                "headers": {"Content-Type": "text/event-stream"},
            },
            {
                "text": "event: stream_end\ndata: {}\nid: evt-2\n\n",
                "headers": {"Content-Type": "text/event-stream"},
            },
        ],
    )

    events = list(training_mod.stream_training("job_x", max_reconnects=2, client=client))

    assert [event.event for event in events] == ["progress", "stream_end"]
    assert mint_route.call_count == 2
    assert stream_route.call_count == 2
    assert stream_route.request_history[0].qs == {"token": ["stream-t-1"]}
    assert stream_route.request_history[1].qs == {"token": ["stream-t-2"]}
    assert "api_key" not in stream_route.request_history[0].qs
    assert "api_key" not in stream_route.request_history[1].qs
