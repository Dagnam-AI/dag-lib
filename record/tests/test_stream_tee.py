"""The tee proxies, against scripted fake streams (the real SDK streams are in test_openai_stream)."""

from __future__ import annotations

import asyncio
import gc
from types import TracebackType
from typing import Any, Callable, List, Optional, Type

import pytest

from dagnam_record._stream import AsyncTeeStream, TeeStream, deliver_pending

DONE = object()  # a script entry: the iterator ends here


class BoomError(Exception):
    """An error raised by the stream."""


class Fake:
    """A stream that plays a script: values to yield, ``DONE`` to stop, exceptions to raise."""

    response = "the http response"

    def __init__(self, script: List[Any]) -> None:
        self.script = script
        self.entered = 0
        self.exited: List[Any] = []
        self.closed = 0
        self.suppress = True  # what __exit__ answers: a context manager that swallows errors

    def __iter__(self) -> Fake:
        return self

    def __next__(self) -> Any:
        step = self.script.pop(0)
        if step is DONE:
            raise StopIteration
        if isinstance(step, BaseException):
            raise step
        return step

    def __enter__(self) -> Fake:
        self.entered += 1
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        self.exited.append(exc_type)
        return self.suppress

    def close(self) -> str:
        """Close."""
        self.closed += 1
        return "closed"


class AsyncFake:
    """The async twin of ``Fake``."""

    response = "the http response"

    def __init__(self, script: List[Any]) -> None:
        self.script = script
        self.entered = 0
        self.exited: List[Any] = []
        self.closed = 0

    def __aiter__(self) -> AsyncFake:
        return self

    async def __anext__(self) -> Any:
        step = self.script.pop(0)
        if step is DONE:
            raise StopAsyncIteration
        if isinstance(step, BaseException):
            raise step
        return step

    async def __aenter__(self) -> AsyncFake:
        self.entered += 1
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        self.exited.append(exc_type)
        return True

    async def close(self) -> str:
        """Close."""
        self.closed += 1
        return "closed"


class Watcher:
    """Records what the watcher was told; ``fail_on`` names a hook that should raise."""

    def __init__(self, fail_on: str = "") -> None:
        self.events: List[Any] = []
        self.fail_on = fail_on
        self.counted: List[str] = []

    def guard(self, step: Callable[..., None], *arguments: Any) -> bool:
        """Run a step like ``Recorder.fail_open``, remembering what it raised."""
        try:
            step(*arguments)
            return True
        except Exception as exc:
            self.counted.append(str(exc))
            return False

    def item(self, value: Any) -> None:
        """An item was seen."""
        if self.fail_on == "item":
            raise RuntimeError("item hook broke")
        self.events.append(("item", value))

    def done(self) -> None:
        """The stream ended or was let go."""
        self.events.append("done")

    def error(self, exc: BaseException) -> None:
        """The stream failed."""
        self.events.append(("error", type(exc).__name__))

    def cancelled(self) -> None:
        """The consumer or the transport was cancelled."""
        self.events.append("cancelled")


def tee(script: List[Any], watcher: Watcher) -> Any:
    return TeeStream(Fake(script), watcher.guard, watcher)


def atee(script: List[Any], watcher: Watcher) -> Any:
    return AsyncTeeStream(AsyncFake(script), watcher.guard, watcher)


def test_items_pass_in_order_and_the_end_is_reported_once() -> None:
    watcher = Watcher()
    stream = tee([1, 2, DONE, DONE], watcher)
    assert list(stream) == [1, 2]
    with pytest.raises(StopIteration):
        next(stream)
    assert watcher.events == [("item", 1), ("item", 2), "done"]


def test_an_error_is_reported_once_and_propagates_unchanged() -> None:
    watcher = Watcher()
    stream = tee([1, BoomError("x"), 2, DONE], watcher)
    assert next(stream) == 1
    with pytest.raises(BoomError):
        next(stream)
    assert next(stream) == 2  # the script goes on; nothing more is recorded
    with pytest.raises(StopIteration):
        next(stream)
    assert watcher.events == [("item", 1), ("error", "BoomError")]


def test_an_error_after_the_end_is_not_a_second_record() -> None:
    watcher = Watcher()
    stream = tee([DONE, BoomError("late")], watcher)
    assert list(stream) == []
    with pytest.raises(BoomError):
        next(stream)
    assert watcher.events == ["done"]


def test_a_broken_watcher_never_reaches_the_caller_and_the_stream_is_not_recorded() -> None:
    watcher = Watcher(fail_on="item")
    stream = tee([1, 2, 3, DONE], watcher)
    assert list(stream) == [1, 2, 3]
    assert watcher.events == []  # no item, no done
    assert watcher.counted == ["item hook broke"]  # counted once, not once per item


def test_closing_reports_the_end_once_and_a_later_end_adds_nothing() -> None:
    watcher = Watcher()
    stream = tee([1, 2, DONE], watcher)
    assert next(stream) == 1
    assert stream.close() == "closed"
    assert stream.close() == "closed"
    assert list(stream) == [2]  # the script still plays; the recorder has let go
    assert watcher.events == [("item", 1), "done"]
    assert stream._stream.closed == 2


def test_a_close_that_raises_still_reports_the_end() -> None:
    watcher = Watcher()
    stream = tee([1], watcher)

    def refuse() -> None:
        raise OSError("socket gone")

    stream._stream.close = refuse
    with pytest.raises(OSError, match="socket gone"):
        stream.close()
    assert watcher.events == ["done"]


def test_leaving_a_with_block_reports_the_end_and_returns_what_the_stream_answered() -> None:
    watcher = Watcher()
    stream = tee([1, 2, DONE], watcher)
    with stream as entered:
        assert entered is stream
        assert next(stream) == 1
        assert stream.response == "the http response"
    assert stream._stream.entered == 1
    assert stream._stream.exited == [None]
    assert watcher.events == [("item", 1), "done"]
    assert stream.__exit__(BoomError, BoomError("in the body"), None) is True  # still suppresses
    assert stream._stream.exited == [None, BoomError]
    assert watcher.events == [("item", 1), "done"]  # once only


def test_an_interrupt_in_the_stream_propagates_and_is_cancelled_not_failed() -> None:
    watcher = Watcher()
    stream = tee([1, KeyboardInterrupt(), 2, DONE], watcher)
    assert next(stream) == 1
    with pytest.raises(KeyboardInterrupt):
        next(stream)
    assert next(stream) == 2  # consistent: still readable, nothing more is reported
    assert list(stream) == []
    assert watcher.events == [("item", 1), "cancelled"]


def test_an_interrupt_after_the_end_is_not_a_second_report() -> None:
    watcher = Watcher()
    stream = tee([DONE, KeyboardInterrupt()], watcher)
    assert list(stream) == []
    with pytest.raises(KeyboardInterrupt):
        next(stream)
    assert watcher.events == ["done"]


def test_an_interrupt_in_the_consumer_body_still_closes_and_reports_the_end() -> None:
    watcher = Watcher()
    stream = tee([1, 2], watcher)
    stream._stream.suppress = False

    def body() -> None:
        with stream:
            next(stream)
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        body()
    assert stream._stream.exited == [KeyboardInterrupt]
    assert watcher.events == [("item", 1), "done"]


def test_letting_go_of_a_stream_reports_the_end_once() -> None:
    watcher = Watcher()
    stream = tee([1, 2], watcher)
    next(stream)
    del stream
    gc.collect()
    assert watcher.events == [("item", 1)]  # letting go is reported at the next safe point
    deliver_pending()
    assert watcher.events == [("item", 1), "done"]
    finished = Watcher()
    done = tee([DONE], finished)
    assert list(done) == []
    del done
    gc.collect()
    assert finished.events == ["done"]  # not twice


def test_the_proxy_is_its_own_iterator_and_forwards_the_rest() -> None:
    stream = tee([1, DONE], Watcher())
    assert iter(stream) is stream
    assert stream.response == "the http response"


def test_the_proxy_reports_the_class_of_the_stream_it_wraps() -> None:
    stream = tee([], Watcher())
    assert isinstance(stream, Fake)
    assert stream.__class__ is Fake
    with pytest.raises(TypeError, match="cannot change its class"):
        stream.__class__ = Fake


def test_an_unbuilt_proxy_raises_attribute_error_instead_of_recursing() -> None:
    blank = object.__new__(TeeStream)
    assert not hasattr(blank, "anything")
    del blank  # and letting it go is quiet


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_async_items_pass_in_order_and_the_end_is_reported_once() -> None:
    watcher = Watcher()
    stream = atee([1, 2, DONE, DONE], watcher)

    async def go() -> List[Any]:
        got = [item async for item in stream]
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()
        return got

    assert run(go()) == [1, 2]
    assert watcher.events == [("item", 1), ("item", 2), "done"]


def test_async_errors_and_broken_watchers_behave_like_the_sync_ones() -> None:
    watcher = Watcher()
    stream = atee([1, BoomError("x"), DONE], watcher)

    async def go() -> None:
        assert await stream.__anext__() == 1
        with pytest.raises(BoomError):
            await stream.__anext__()

    run(go())
    assert watcher.events == [("item", 1), ("error", "BoomError")]
    broken = Watcher(fail_on="item")

    async def drain() -> List[Any]:
        return [item async for item in atee([1, 2, DONE], broken)]

    assert run(drain()) == [1, 2]
    assert broken.events == []


def test_a_cancelled_async_read_propagates_and_is_cancelled_not_failed() -> None:
    watcher = Watcher()
    stream = atee([1, asyncio.CancelledError(), 2, DONE], watcher)

    async def go() -> None:
        assert await stream.__anext__() == 1
        with pytest.raises(asyncio.CancelledError):
            await stream.__anext__()
        assert await stream.__anext__() == 2

    run(go())
    assert watcher.events == [("item", 1), "cancelled"]


def test_the_async_proxy_is_its_own_iterator_and_context_manager() -> None:
    watcher = Watcher()
    stream = atee([1, DONE], watcher)

    async def go() -> str:
        assert stream.__aiter__() is stream
        async with stream as entered:
            assert entered is stream
            assert await stream.__anext__() == 1
        return str(stream.response)

    assert run(go()) == "the http response"
    assert stream._stream.entered == 1
    assert stream._stream.exited == [None]
    assert isinstance(stream, AsyncFake)
    assert watcher.events == [("item", 1), "done"]


def test_the_async_exit_returns_the_streams_answer_so_suppression_still_works() -> None:
    stream = atee([DONE], Watcher())

    async def go() -> Any:
        return await stream.__aexit__(BoomError, BoomError("body"), None)

    assert run(go()) is True
    assert stream._stream.exited == [BoomError]


def test_async_close_reports_the_end_once_even_when_it_raises() -> None:
    watcher = Watcher()
    stream = atee([1, DONE], watcher)

    async def go() -> None:
        assert await stream.__anext__() == 1
        assert await stream.close() == "closed"
        assert await stream.close() == "closed"

    run(go())
    assert watcher.events == [("item", 1), "done"]

    async def refuse() -> None:
        raise OSError("socket gone")

    broken_watcher = Watcher()
    broken = atee([], broken_watcher)
    broken._stream.close = refuse

    async def close_it() -> None:
        with pytest.raises(OSError, match="socket gone"):
            await broken.close()

    run(close_it())
    assert broken_watcher.events == ["done"]


def test_letting_go_of_an_async_stream_reports_the_end_once() -> None:
    watcher = Watcher()
    stream = atee([1, 2], watcher)
    run(stream.__anext__())
    del stream
    gc.collect()
    assert watcher.events == [("item", 1)]
    deliver_pending()
    assert watcher.events == [("item", 1), "done"]
