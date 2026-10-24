"""Tee proxies: hand a stream to the caller untouched while a recorder watches it go by.

Policy (one place, shared by every provider patch):

* every item is passed on, in order, exactly as the SDK produced it;
* a stream read to its end is recorded once, from what the watcher assembled;
* a stream that raises is recorded once as a failed call and the error propagates unchanged;
* a stream the caller abandons (``close()``, leaving a ``with`` block, ``break``, or just letting
  go of it) records nothing: a partial answer is no training row, and an abandoned stream is not
  counted as a drop either;
* a watcher that fails never reaches the caller: the failure is counted through ``guard`` and
  that stream is simply not recorded.

The proxy forwards every other attribute to the stream it wraps, reports the wrapped stream's
class for ``isinstance`` checks and returns itself from ``with``, so a caller cannot tell it from
the SDK's own stream.
"""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Iterator, Optional, Type

Guard = Callable[..., bool]
"""``Recorder.fail_open``: runs a step, counts an error, and says whether the step ran cleanly."""


def get(obj: Any, name: str) -> Any:
    """``obj[name]`` for a dict, ``obj.name`` for an SDK object, else ``None``."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


class _Tap:
    """The bookkeeping both proxies share: what to call, and that it is called only once."""

    def __init__(
        self,
        stream: Any,
        guard: Guard,
        item: Callable[[Any], None],
        done: Callable[[], None],
        error: Callable[[BaseException], None],
    ) -> None:
        self._stream = stream
        self._guard = guard
        self._item = item
        self._done = done
        self._error = error
        self._over = False  # finished or failed: nothing more is recorded
        self._broken = False  # a watcher failed: this stream is not recorded

    def _seen(self, value: Any) -> None:
        if not self._over and not self._broken:
            self._broken = not self._guard(self._item, value)

    def _finish(self) -> None:
        if not self._over:
            self._over = True
            if not self._broken:
                self._guard(self._done)

    def _fail(self, exc: BaseException) -> None:
        if not self._over:
            self._over = True
            self._guard(self._error, exc)

    def __getattr__(self, name: str) -> Any:
        if name == "_stream":  # not built yet (a copy, say): do not look for it forever
            raise AttributeError(name)
        return getattr(self._stream, name)

    def _wrapped_class(self) -> Type[Any]:
        return self._stream.__class__

    def _refuse_new_class(self, value: Type[Any], /) -> None:
        raise TypeError("a tee stream cannot change its class")

    # ``isinstance(proxy, openai.Stream)`` must stay true, as it is for the SDK's own stream.
    # The type checker is not shown this on purpose: it cannot express a property standing in
    # for ``object.__class__``, and nothing in this package reads the attribute.
    if TYPE_CHECKING:
        pass
    else:
        __class__ = property(_wrapped_class, _refuse_new_class)


class TeeStream(_Tap):
    """A synchronous stream, watched."""

    def __init__(
        self,
        stream: Any,
        guard: Guard,
        item: Callable[[Any], None],
        done: Callable[[], None],
        error: Callable[[BaseException], None],
    ) -> None:
        super().__init__(stream, guard, item, done, error)
        self._iterator: Iterator[Any] = iter(stream)

    def __iter__(self) -> TeeStream:
        return self

    def __next__(self) -> Any:
        try:
            value = next(self._iterator)
        except StopIteration:
            self._finish()
            raise
        except BaseException as exc:
            self._fail(exc)
            raise
        self._seen(value)
        return value

    def __enter__(self) -> TeeStream:
        self._stream.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Any:
        return self._stream.__exit__(exc_type, exc, traceback)


class AsyncTeeStream(_Tap):
    """An asynchronous stream, watched."""

    def __init__(
        self,
        stream: Any,
        guard: Guard,
        item: Callable[[Any], None],
        done: Callable[[], None],
        error: Callable[[BaseException], None],
    ) -> None:
        super().__init__(stream, guard, item, done, error)
        self._iterator: AsyncIterator[Any] = stream.__aiter__()

    def __aiter__(self) -> AsyncTeeStream:
        return self

    async def __anext__(self) -> Any:
        try:
            value = await self._iterator.__anext__()
        except StopAsyncIteration:
            self._finish()
            raise
        except BaseException as exc:
            self._fail(exc)
            raise
        self._seen(value)
        return value

    async def __aenter__(self) -> AsyncTeeStream:
        await self._stream.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Any:
        return await self._stream.__aexit__(exc_type, exc, traceback)
