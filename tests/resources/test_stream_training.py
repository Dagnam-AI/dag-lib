"""stream_training must delegate to the shared reconnect engine."""

from __future__ import annotations

from dagnam.resources import training as training_mod


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

    events = list(training_mod.stream_training("job_x", client=_Client()))

    assert captured["open_stream_callable"] is True
    assert "complete" in captured["terminal_events"]
    assert [event.event for event in events] == ["metric", "complete"]
    assert training_mod.TrainingEvent is not None
