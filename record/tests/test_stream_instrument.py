"""``instrument``: watch an SDK stream object in place, against scripted fake streams.

The fakes are built like the SDKs' own streams: the object keeps a generator in ``_iterator``,
``__iter__`` / ``__next__`` read it, ``close()`` closes the response and ``__exit__`` calls
``self.close()``. The real streams are in test_anthropic_stream.
"""

from __future__ import annotations

import asyncio
import gc
from types import TracebackType
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Type
import weakref

import pytest
from typing_extensions import override

from dagnam_record._stream import instrument

DONE = object()


class BoomError(Exception):
    """An error raised by the stream."""


class Watcher:
    """Records what it is told."""

    def __init__(self) -> None:
        self.events: List[Any] = []

    def item(self, value: Any) -> None:
        """Item."""
        self.events.append(("item", value))

    def done(self) -> None:
        """Done."""
        self.events.append(("done",))

    def error(self, exc: BaseException) -> None:
        """Error."""
        self.events.append(("error", type(exc).__name__))

    def cancelled(self) -> None:
        """Cancelled."""
        self.events.append(("cancelled",))


def guard(step: Any, *arguments: Any) -> bool:
    step(*arguments)
    return True


def script(steps: List[Any]) -> Iterator[Any]:
    for step in steps:
        if isinstance(step, BaseException):
            raise step
        yield step


class Fake:
    """A synchronous SDK-style stream."""

    def __init__(self, steps: List[Any]) -> None:
        self.closed = 0
        self.close_fails = False
        self.response = "the http response"
        self._iterator = script(steps)

    def __next__(self) -> Any:
        return self._iterator.__next__()

    def __iter__(self) -> Iterator[Any]:
        for item in self._iterator:
            yield item

    def __enter__(self) -> Fake:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.close()

    def close(self) -> str:
        """Close."""
        self.closed += 1
        if self.close_fails:
            raise BoomError("close")
        return "closed"


async def ascript(steps: List[Any]) -> AsyncIterator[Any]:
    for step in steps:
        if isinstance(step, BaseException):
            raise step
        yield step


class AsyncFake:
    """An asynchronous SDK-style stream."""

    def __init__(self, steps: List[Any]) -> None:
        self.closed = 0
        self._iterator = ascript(steps)

    async def __anext__(self) -> Any:
        return await self._iterator.__anext__()

    async def __aiter__(self) -> AsyncIterator[Any]:
        async for item in self._iterator:
            yield item

    async def __aenter__(self) -> AsyncFake:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> str:
        """Close."""
        self.closed += 1
        return "closed"


def watched(steps: List[Any]) -> Any:
    watcher = Watcher()
    stream = instrument(Fake(steps), guard, watcher, sync=True)
    return stream, watcher


def test_the_same_object_comes_back_with_its_type_and_attributes_intact() -> None:
    fake = Fake([1])
    watcher = Watcher()
    stream = instrument(fake, guard, watcher, sync=True)
    assert stream is fake
    assert type(stream) is Fake
    assert stream.response == "the http response"
    with stream as same:
        assert same is stream


def test_items_pass_in_order_and_the_end_is_reported_once() -> None:
    stream, watcher = watched([1, 2, 3])
    assert list(stream) == [1, 2, 3]
    assert list(stream) == []
    assert watcher.events == [("item", 1), ("item", 2), ("item", 3), ("done",)]


def test_next_and_iter_share_one_position_and_a_second_iter_is_harmless() -> None:
    stream, watcher = watched([1, 2, 3, 4])
    assert next(stream) == 1
    first = iter(stream)
    assert next(first) == 2
    second = iter(stream)
    assert next(second) == 3
    assert list(first) == [4]
    assert list(second) == []
    assert [event for event in watcher.events if event[0] == "done"] == [("done",)]
    assert [event[1] for event in watcher.events if event[0] == "item"] == [1, 2, 3, 4]


def test_an_error_passes_on_unchanged_and_is_reported_once() -> None:
    stream, watcher = watched([1, BoomError("x"), 2])
    with pytest.raises(BoomError):
        list(stream)
    assert list(stream) == []  # the generator is finished; nothing more is reported
    stream.close()
    assert watcher.events == [("item", 1), ("error", "BoomError")]


def test_an_interrupt_passes_on_and_is_reported_as_a_cancel() -> None:
    stream, watcher = watched([1, KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        list(stream)
    stream.close()
    assert watcher.events == [("item", 1), ("cancelled",)]


def test_close_returns_what_the_sdk_returns_and_reports_what_was_seen_once() -> None:
    stream, watcher = watched([1, 2])
    next(stream)
    assert stream.close() == "closed"
    assert stream.close() == "closed"
    assert stream.closed == 2
    assert watcher.events == [("item", 1), ("done",)]


def test_leaving_a_with_block_reports_through_the_sdks_own_exit() -> None:
    stream, watcher = watched([1, 2])
    with stream:
        next(stream)
    assert stream.closed == 1
    assert watcher.events == [("item", 1), ("done",)]


def test_close_that_raises_still_reports_and_raises() -> None:
    fake = Fake([1])
    fake.close_fails = True
    watcher = Watcher()
    stream = instrument(fake, guard, watcher, sync=True)
    with pytest.raises(BoomError):
        stream.close()
    assert watcher.events == [("done",)]


def test_letting_go_of_the_stream_reports_it_once_and_does_not_keep_it_alive() -> None:
    stream, watcher = watched([1, 2])
    next(stream)
    ref = weakref.ref(stream)
    del stream
    gc.collect()
    assert ref() is None  # the report hook holds nothing that holds the stream
    assert watcher.events == [("item", 1), ("done",)]


def test_a_finished_stream_is_not_reported_again_when_let_go() -> None:
    stream, watcher = watched([1])
    list(stream)
    del stream
    gc.collect()
    assert watcher.events == [("item", 1), ("done",)]


def test_a_failing_watcher_stops_watching_and_never_reaches_the_caller() -> None:
    class Broken(Watcher):
        @override
        def item(self, value: Any) -> None:
            """Fail."""
            raise RuntimeError("bug")

    seen: List[str] = []

    def counting(step: Any, *arguments: Any) -> bool:
        try:
            step(*arguments)
        except RuntimeError:
            seen.append("counted")
            return False
        return True

    watcher = Broken()
    stream = instrument(Fake([1, 2]), counting, watcher, sync=True)
    assert list(stream) == [1, 2]
    assert seen == ["counted"]
    assert watcher.events == []  # `done` is not called for a stream that stopped being watched


def test_async_items_close_and_letting_go() -> None:
    watcher = Watcher()
    fake = AsyncFake([1, 2, 3])

    async def main() -> None:
        stream = instrument(fake, guard, watcher, sync=False)
        assert stream is fake
        assert [item async for item in stream] == [1, 2, 3]
        assert await stream.close() == "closed"
        other = instrument(AsyncFake([1, 2]), guard, watcher, sync=False)
        async with other as same:
            await same.__anext__()
        assert other.closed == 1

    asyncio.run(main())
    assert watcher.events == [
        ("item", 1),
        ("item", 2),
        ("item", 3),
        ("done",),
        ("item", 1),
        ("done",),
    ]


def test_an_async_cancel_and_error_are_reported() -> None:
    watcher = Watcher()

    async def main() -> None:
        failing = instrument(AsyncFake([BoomError("x")]), guard, watcher, sync=False)
        with pytest.raises(BoomError):
            await failing.__anext__()
        stalled = instrument(AsyncFake([asyncio.CancelledError()]), guard, watcher, sync=False)
        with pytest.raises(asyncio.CancelledError):
            await stalled.__anext__()

    asyncio.run(main())
    assert watcher.events == [("error", "BoomError"), ("cancelled",)]


class NoIterator:
    """An SDK whose stream builds its generator only when first iterated."""

    def close(self) -> None:
        """Close."""

    def __iter__(self) -> Iterator[Any]:
        return iter([])


class ReadOnlyIterator:
    """``_iterator`` is a property without a setter."""

    _hidden = iter([1])

    @property
    def _iterator(self) -> Iterator[Any]:
        """The iterator."""
        return self._hidden

    def close(self) -> None:
        """Close."""


class Slotted:
    """No instance dict, so no instance ``close``."""

    __slots__ = ("__weakref__", "_iterator")

    def __init__(self) -> None:
        self._iterator = iter([1])

    def close(self) -> None:
        """Close."""


class NotWeak:
    """An instance dict but no weak references."""

    __slots__ = ("__dict__",)  # has an instance dict but cannot be weakly referenced

    def __init__(self) -> None:
        self._iterator = iter([1])

    def close(self) -> None:
        """Close."""


class NotAnIterator:
    """``_iterator`` is not an iterator."""

    def __init__(self) -> None:
        self._iterator = 5

    def close(self) -> None:
        """Close."""


class NoClose:
    """No ``close()``."""

    def __init__(self) -> None:
        self._iterator = iter([1])


def untouched(stream: Any) -> Dict[str, Any]:
    return dict(vars(stream)) if hasattr(stream, "__dict__") else {}


@pytest.mark.parametrize(
    "factory",
    [NoIterator, ReadOnlyIterator, Slotted, NotWeak, NotAnIterator, NoClose],
)
def test_a_stream_that_is_not_built_as_expected_is_refused_and_left_as_it_was(
    factory: Any,
) -> None:
    stream = factory()
    before = untouched(stream)
    iterator = getattr(stream, "_iterator", None)
    with pytest.raises((AttributeError, TypeError)):
        instrument(stream, guard, Watcher(), sync=True)
    assert untouched(stream) == before
    assert getattr(stream, "_iterator", None) is iterator


def test_an_async_stream_needs_an_async_iterator_and_a_sync_one_a_sync_iterator() -> None:
    with pytest.raises(TypeError):
        instrument(AsyncFake([1]), guard, Watcher(), sync=True)
    with pytest.raises(TypeError):
        instrument(Fake([1]), guard, Watcher(), sync=False)
