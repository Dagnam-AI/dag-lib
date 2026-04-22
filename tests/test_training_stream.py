"""Unit tests for dagnam.training (SSE event streaming)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import StreamError
from dagnam.services.training import TrainingEvent, _parse_event, stream_training


def _sse(event: str, data: str, id: str | None = None, retry: str | None = None):
    return SimpleNamespace(event=event, data=data, id=id, retry=retry)


class TestParseEvent:
    def test_decodes_json_payload(self):
        ev = _parse_event(_sse("metric", '{"loss": 0.5}', id="1"))
        assert ev.event == "metric"
        assert ev.data == {"loss": 0.5}
        assert ev.id == "1"

    def test_non_json_falls_back_to_string(self):
        ev = _parse_event(_sse("log", "plain text"))
        assert ev.data == "plain text"

    def test_empty_data_becomes_empty_dict(self):
        ev = _parse_event(_sse("heartbeat", ""))
        assert ev.data == {}


class _FakeSSE:
    def __init__(self, events):
        self._events = events

    def events(self):
        for e in self._events:
            yield e


class TestStreamTraining:
    def test_yields_events_until_terminal(self):
        client = MagicMock(spec=DagnamClient)
        client.open_training_stream.return_value = MagicMock()
        events = [
            _sse("metric", '{"epoch": 1, "loss": 0.5}', id="1"),
            _sse("metric", '{"epoch": 2, "loss": 0.3}', id="2"),
            _sse("complete", '{"status": "done"}', id="3"),
            _sse("metric", '{"never": "seen"}', id="4"),
        ]
        with patch("sseclient.SSEClient", return_value=_FakeSSE(events)):
            out = list(stream_training("job_1", client=client))
        assert [e.event for e in out] == ["metric", "metric", "complete"]
        assert out[0].data == {"epoch": 1, "loss": 0.5}

    def test_skips_heartbeats_by_default(self):
        client = MagicMock(spec=DagnamClient)
        client.open_training_stream.return_value = MagicMock()
        events = [
            _sse("heartbeat", "{}", id="1"),
            _sse("metric", '{"loss": 0.1}', id="2"),
            _sse("stream_end", "{}", id="3"),
        ]
        with patch("sseclient.SSEClient", return_value=_FakeSSE(events)):
            out = list(stream_training("job_1", client=client))
        assert [e.event for e in out] == ["metric", "stream_end"]

    def test_include_heartbeats(self):
        client = MagicMock(spec=DagnamClient)
        client.open_training_stream.return_value = MagicMock()
        events = [
            _sse("heartbeat", "{}", id="1"),
            _sse("complete", "{}", id="2"),
        ]
        with patch("sseclient.SSEClient", return_value=_FakeSSE(events)):
            out = list(
                stream_training("job_1", client=client, include_heartbeats=True)
            )
        assert [e.event for e in out] == ["heartbeat", "complete"]

    def test_reconnects_with_last_event_id(self):
        client = MagicMock(spec=DagnamClient)
        client.open_training_stream.return_value = MagicMock()

        # First batch drops without a terminal; second batch finishes.
        batches = [
            [_sse("metric", '{"loss": 0.9}', id="e1")],
            [
                _sse("metric", '{"loss": 0.5}', id="e2"),
                _sse("complete", "{}", id="e3"),
            ],
        ]
        fake_iter = iter(batches)

        def fake_sse_client(response):
            return _FakeSSE(next(fake_iter))

        with (
            patch("sseclient.SSEClient", side_effect=fake_sse_client),
            patch("dagnam.services.training.time.sleep"),
        ):
            out = list(stream_training("job_1", client=client))

        # Reconnect call should include last_event_id=e1
        assert client.open_training_stream.call_count == 2
        first_call = client.open_training_stream.call_args_list[0]
        second_call = client.open_training_stream.call_args_list[1]
        assert first_call.kwargs.get("last_event_id") is None
        assert second_call.kwargs.get("last_event_id") == "e1"
        assert [e.event for e in out] == ["metric", "metric", "complete"]

    def test_reconnect_exhaustion_raises(self):
        client = MagicMock(spec=DagnamClient)
        client.open_training_stream.return_value = MagicMock()
        with (
            patch("sseclient.SSEClient", return_value=_FakeSSE([])),
            patch("dagnam.services.training.time.sleep"),
        ):
            with pytest.raises(StreamError):
                list(stream_training("job_1", client=client, max_reconnects=2))
