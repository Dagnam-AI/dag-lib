"""iter_with_reconnect must tolerate many non-consecutive drops."""

from __future__ import annotations

import pytest

from dagnam._core import sse as sse_mod
from dagnam._core.exceptions import StreamError


class _FakeRaw:
    def __init__(self, event: str, data: str = "{}", id: str | None = None) -> None:
        self.event = event
        self.data = data
        self.id = id
        self.retry = None


class _FakeSSEClient:
    def __init__(self, response):
        self._events = response

    def events(self):
        yield from self._events


def _install_fake_sseclient(monkeypatch, scripts):
    calls = {"n": 0}

    class _Mod:
        @staticmethod
        def SSEClient(response):  # noqa: N802
            return _FakeSSEClient(response)

    monkeypatch.setattr(sse_mod, "import_module", lambda name: _Mod)

    def open_stream(cursor):
        idx = calls["n"]
        calls["n"] += 1
        script = scripts[min(idx, len(scripts) - 1)]

        def _gen():
            for item in script:
                if isinstance(item, type) and issubclass(item, Exception):
                    raise item("boom")
                yield item

        return _gen()

    return open_stream


def test_attempts_reset_after_each_successful_event(monkeypatch):
    import requests

    drop = requests.exceptions.ConnectionError
    scripts = [[_FakeRaw("metric"), drop] for _ in range(8)]
    scripts.append([_FakeRaw("complete")])

    events = list(
        sse_mod.iter_with_reconnect(
            _install_fake_sseclient(monkeypatch, scripts),
            terminal_events=sse_mod.TERMINAL_TRAINING_EVENTS,
            backoff_base=0.0,
            max_reconnects=5,
        )
    )

    assert [event.event for event in events] == ["metric"] * 8 + ["complete"]


def test_gives_up_after_consecutive_failures_with_no_progress(monkeypatch):
    import requests

    scripts = [[requests.exceptions.ConnectionError] for _ in range(20)]

    with pytest.raises(StreamError):
        list(
            sse_mod.iter_with_reconnect(
                _install_fake_sseclient(monkeypatch, scripts),
                terminal_events=sse_mod.TERMINAL_TRAINING_EVENTS,
                backoff_base=0.0,
                max_reconnects=5,
            )
        )
