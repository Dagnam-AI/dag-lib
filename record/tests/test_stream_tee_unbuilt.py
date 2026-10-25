"""A proxy that fails to build reports nothing when it is thrown away; raw-response streams count once."""

from __future__ import annotations

import gc
from typing import Any, List

from _mock_server import Reply
from _openai_bodies import CHAT_PATH, MODEL, chat_text_events
import pytest

from dagnam_record._stream import AsyncTeeStream, TeeStream, deliver_pending

MESSAGES: List[Any] = [{"role": "user", "content": "Hi"}]


class Told:
    """A watcher that notes every call."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def item(self, value: Any) -> None:
        """Item."""
        self.calls.append("item")

    def done(self) -> None:
        """Done."""
        self.calls.append("done")

    def error(self, exc: BaseException) -> None:
        """Error."""
        self.calls.append("error")

    def cancelled(self) -> None:
        """Cancelled."""
        self.calls.append("cancelled")


def guard(step: Any, *arguments: Any) -> bool:
    step(*arguments)
    return True


@pytest.mark.parametrize("proxy", [TeeStream, AsyncTeeStream])
def test_a_proxy_that_raises_in_its_constructor_reports_nothing_when_collected(
    proxy: Any,
) -> None:
    told = Told()
    with pytest.raises((TypeError, AttributeError)):
        proxy(object(), guard, told)  # not iterable
    gc.collect()
    assert told.calls == []


@pytest.mark.parametrize("proxy", [TeeStream, AsyncTeeStream])
def test_a_proxy_that_was_built_still_reports_when_collected(proxy: Any) -> None:
    class Stream:
        def __iter__(self) -> Any:
            return iter([1])

        def __aiter__(self) -> Any:
            return self

        async def __anext__(self) -> Any:
            raise StopAsyncIteration

    told = Told()
    tee = proxy(Stream(), guard, told)
    del tee
    gc.collect()
    deliver_pending()
    assert told.calls == ["done"]


def test_an_openai_raw_response_stream_is_counted_once_as_an_error_not_as_a_partial(
    rig: Any,
) -> None:
    rig.server.on(CHAT_PATH, Reply(events=chat_text_events()))
    with rig.client() as client:
        raw = client.chat.completions.with_raw_response.create(
            model=MODEL, messages=MESSAGES, stream=True
        )
        assert raw.status_code == 200
        assert len(list(raw.parse())) == 5  # the call itself is unaffected
        del raw
        gc.collect()
    assert rig.raw() == []
    assert rig.writer.counts["dropped_error"] == 1
    assert rig.writer.counts["dropped_partial"] == 0
