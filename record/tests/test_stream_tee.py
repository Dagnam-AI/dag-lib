"""The tee proxies, against scripted fake streams (the real SDK streams are in test_openai_stream)."""

from __future__ import annotations

import asyncio
from types import TracebackType
from typing import Any, Callable, List, Optional, Type

import pytest

from dagnam_record._stream import AsyncTeeStream, TeeStream

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

    def __iter__(self) -> Fake:
        return self

    def __next__(self) -> Any:
        step = self.script.pop(0)
        if step is DONE:
            raise StopIteration
        if isinstance(step, Exception):
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
    ) -> str:
        self.exited.append(exc_type)
        return "suppress?"

    def close(self) -> str:
        """Close."""
        return "closed"


class AsyncFake:
    """The async twin of ``Fake``."""

    response = "the http response"

    def __init__(self, script: List[Any]) -> None:
        self.script = script
        self.entered = 0
        self.exited: List[Any] = []

    def __aiter__(self) -> AsyncFake:
        return self

    async def __anext__(self) -> Any:
        step = self.script.pop(0)
        if step is DONE:
            raise StopAsyncIteration
        if isinstance(step, Exception):
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
    ) -> str:
        self.exited.append(exc_type)
        return "suppress?"


class Hooks:
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
        """The stream ended."""
        self.events.append("done")

    def error(self, exc: BaseException) -> None:
        """The stream failed."""
        self.events.append(("error", type(exc).__name__))


def tee(script: List[Any], hooks: Hooks) -> Any:
    return TeeStream(Fake(script), hooks.guard, hooks.item, hooks.done, hooks.error)


def atee(script: List[Any], hooks: Hooks) -> Any:
    return AsyncTeeStream(AsyncFake(script), hooks.guard, hooks.item, hooks.done, hooks.error)


def test_items_pass_in_order_and_the_end_is_reported_once() -> None:
    hooks = Hooks()
    stream = tee([1, 2, DONE, DONE], hooks)
    assert list(stream) == [1, 2]
    with pytest.raises(StopIteration):
        next(stream)
    assert hooks.events == [("item", 1), ("item", 2), "done"]


def test_an_error_is_reported_once_and_propagates_unchanged() -> None:
    hooks = Hooks()
    stream = tee([1, BoomError("x"), 2, DONE], hooks)
    assert next(stream) == 1
    with pytest.raises(BoomError):
        next(stream)
    assert next(stream) == 2  # the script goes on; nothing more is recorded
    with pytest.raises(StopIteration):
        next(stream)
    assert hooks.events == [("item", 1), ("error", "BoomError")]


def test_an_error_after_the_end_is_not_a_second_record() -> None:
    hooks = Hooks()
    stream = tee([DONE, BoomError("late")], hooks)
    assert list(stream) == []
    with pytest.raises(BoomError):
        next(stream)
    assert hooks.events == ["done"]


def test_a_broken_watcher_never_reaches_the_caller_and_the_stream_is_not_recorded() -> None:
    hooks = Hooks(fail_on="item")
    stream = tee([1, 2, 3, DONE], hooks)
    assert list(stream) == [1, 2, 3]
    assert hooks.events == []  # no item, no done
    assert hooks.counted == ["item hook broke"]  # counted once, not once per item


def test_an_abandoned_stream_records_nothing() -> None:
    hooks = Hooks()
    stream = tee([1, 2, 3], hooks)
    assert next(stream) == 1
    assert stream.close() == "closed"
    assert hooks.events == [("item", 1)]


def test_the_proxy_is_its_own_iterator_and_context_manager_and_forwards_the_rest() -> None:
    hooks = Hooks()
    stream = tee([1, DONE], hooks)
    assert iter(stream) is stream
    with stream as entered:
        assert entered is stream
        assert stream.response == "the http response"
    assert stream._stream.entered == 1
    assert stream._stream.exited == [None]


def test_exit_forwards_the_exception_and_the_answer() -> None:
    hooks = Hooks()
    stream = tee([], hooks)
    boom = BoomError("in the body")
    assert stream.__exit__(BoomError, boom, None) == "suppress?"
    assert stream._stream.exited == [BoomError]


def test_the_proxy_reports_the_class_of_the_stream_it_wraps() -> None:
    stream = tee([], Hooks())
    assert isinstance(stream, Fake)
    assert stream.__class__ is Fake
    with pytest.raises(TypeError, match="cannot change its class"):
        stream.__class__ = Fake


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_async_items_pass_in_order_and_the_end_is_reported_once() -> None:
    hooks = Hooks()
    stream = atee([1, 2, DONE, DONE], hooks)

    async def go() -> List[Any]:
        got = [item async for item in stream]
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()
        return got

    assert run(go()) == [1, 2]
    assert hooks.events == [("item", 1), ("item", 2), "done"]


def test_async_errors_and_broken_watchers_behave_like_the_sync_ones() -> None:
    hooks = Hooks()
    stream = atee([1, BoomError("x"), DONE], hooks)

    async def go() -> None:
        assert await stream.__anext__() == 1
        with pytest.raises(BoomError):
            await stream.__anext__()

    run(go())
    assert hooks.events == [("item", 1), ("error", "BoomError")]
    broken = Hooks(fail_on="item")

    async def drain() -> List[Any]:
        return [item async for item in atee([1, 2, DONE], broken)]

    assert run(drain()) == [1, 2]
    assert broken.events == []


def test_the_async_proxy_is_its_own_iterator_and_context_manager() -> None:
    stream = atee([DONE], Hooks())

    async def go() -> str:
        assert stream.__aiter__() is stream
        async with stream as entered:
            assert entered is stream
        return str(stream.response)

    assert run(go()) == "the http response"
    assert stream._stream.entered == 1
    assert stream._stream.exited == [None]
    assert isinstance(stream, AsyncFake)


def test_an_unbuilt_proxy_raises_attribute_error_instead_of_recursing() -> None:
    blank = object.__new__(TeeStream)
    assert not hasattr(blank, "anything")
