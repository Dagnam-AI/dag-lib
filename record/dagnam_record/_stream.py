"""Tee proxies: hand a stream to the caller untouched while a recorder watches it go by.

Policy (one place, shared by every provider patch). A *watcher* is told about the stream with
four calls (``item``, ``done``, ``error``, ``cancelled``), each run through ``guard`` so that a
failing watcher never reaches the caller:

* every item is passed on, in order, exactly as the SDK produced it;
* ``done()`` is called exactly once when the recorder has seen all it will see of a stream:
  the stream ran out, the caller closed it (``close()``, leaving a ``with`` block, ``aclose``),
  or the caller let go of it. The watcher decides whether what it saw was a complete answer
  (record it) or a partial one (count it, write nothing);
* an exception raised by the stream is passed on unchanged and ``error(exc)`` is called once;
* an interrupt or cancellation (``KeyboardInterrupt``, ``asyncio.CancelledError``, anything that
  is not an ``Exception``) is passed on unchanged and ``cancelled()`` is called once: it is not a
  failed call;
* once any of these has happened nothing more is reported for that stream;
* a watcher that fails stops watching: the failure is counted through ``guard``, ``done`` is not
  called and the stream is simply not recorded.

Two ways to watch a stream. ``TeeStream`` / ``AsyncTeeStream`` are proxies: they forward every
other attribute to the stream they wrap, report the wrapped stream's class for ``isinstance``
checks and return themselves from ``with``. A proxy cannot satisfy an SDK that checks the exact
type of the object (the Anthropic ``Stream`` does), so ``instrument`` watches an SDK stream
IN PLACE instead: the caller keeps the SDK's own object, whose iterator and ``close`` are
replaced on the instance. Both report through the same ``_Watch``.
"""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Iterator, Optional, Type
import weakref

Guard = Callable[..., bool]
"""``Recorder.fail_open``: runs a step, counts an error, and says whether the step ran cleanly."""


def get(obj: Any, name: str) -> Any:
    """``obj[name]`` for a dict, ``obj.name`` for an SDK object, else ``None``."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


class _Watch:
    """What to call, and that it is called only once (state only: it never holds a stream)."""

    def __init__(self, guard: Guard, watcher: Any) -> None:
        self._guard = guard
        self._watcher = watcher
        self._over = False  # reported already: nothing more is recorded
        self._broken = False  # a watcher call failed: this stream is not recorded

    def watch_item(self, value: Any) -> None:
        """One item went by."""
        if not self._over and not self._broken:
            self._broken = not self._guard(self._watcher.item, value)

    def watch_end(self) -> None:
        """The recorder is done with the stream (it ended, was closed or let go)."""
        if not self._over:
            self._over = True
            if not self._broken:
                self._guard(self._watcher.done)

    def watch_error(self, exc: BaseException) -> None:
        """The stream raised an ``Exception``."""
        if not self._over:
            self._over = True
            self._guard(self._watcher.error, exc)

    def watch_cancel(self) -> None:
        """The stream was interrupted or cancelled."""
        if not self._over:
            self._over = True
            self._guard(self._watcher.cancelled)

    def watch_next(self, iterator: Iterator[Any]) -> Any:
        """The next item of ``iterator``, reported."""
        try:
            value = next(iterator)
        except StopIteration:
            self.watch_end()
            raise
        except Exception as exc:
            self.watch_error(exc)
            raise
        except BaseException:
            self.watch_cancel()
            raise
        self.watch_item(value)
        return value

    async def watch_anext(self, iterator: AsyncIterator[Any]) -> Any:
        """The next item of ``iterator``, reported."""
        try:
            value = await iterator.__anext__()
        except StopAsyncIteration:
            self.watch_end()
            raise
        except Exception as exc:
            self.watch_error(exc)
            raise
        except BaseException:
            self.watch_cancel()
            raise
        self.watch_item(value)
        return value


class _Tap(_Watch):
    """The proxies' base: a ``_Watch`` over the stream they wrap."""

    def __init__(self, stream: Any, guard: Guard, watcher: Any) -> None:
        super().__init__(guard, watcher)
        self._stream = stream
        self._armed = False  # set last: a proxy that failed to build reports nothing

    def __del__(self) -> None:
        if self.__dict__.get("_armed"):
            self.watch_end()

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

    def __init__(self, stream: Any, guard: Guard, watcher: Any) -> None:
        super().__init__(stream, guard, watcher)
        self._iterator: Iterator[Any] = iter(stream)
        self._armed = True

    def __iter__(self) -> TeeStream:
        return self

    def __next__(self) -> Any:
        return self.watch_next(self._iterator)

    def close(self) -> Any:
        """Close the stream; what the recorder saw so far is reported either way."""
        try:
            return self._stream.close()
        finally:
            self.watch_end()

    def __enter__(self) -> TeeStream:
        self._stream.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Any:
        try:
            return self._stream.__exit__(exc_type, exc, traceback)
        finally:
            self.watch_end()


class AsyncTeeStream(_Tap):
    """An asynchronous stream, watched."""

    def __init__(self, stream: Any, guard: Guard, watcher: Any) -> None:
        super().__init__(stream, guard, watcher)
        self._iterator: AsyncIterator[Any] = stream.__aiter__()
        self._armed = True

    def __aiter__(self) -> AsyncTeeStream:
        return self

    async def __anext__(self) -> Any:
        return await self.watch_anext(self._iterator)

    async def close(self) -> Any:
        """Close the stream; what the recorder saw so far is reported either way."""
        try:
            return await self._stream.close()
        finally:
            self.watch_end()

    async def __aenter__(self) -> AsyncTeeStream:
        await self._stream.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Any:
        try:
            return await self._stream.__aexit__(exc_type, exc, traceback)
        finally:
            self.watch_end()


class _Source:
    """Stands where the SDK keeps its iterator (``stream._iterator``): reads it, reports what it sees."""

    def __init__(self, watch: _Watch, iterator: Iterator[Any]) -> None:
        self._watch = watch
        self._iterator = iterator

    def __iter__(self) -> _Source:
        return self

    def __next__(self) -> Any:
        return self._watch.watch_next(self._iterator)


class _AsyncSource:
    """The async twin of ``_Source``."""

    def __init__(self, watch: _Watch, iterator: AsyncIterator[Any]) -> None:
        self._watch = watch
        self._iterator = iterator

    def __aiter__(self) -> _AsyncSource:
        return self

    async def __anext__(self) -> Any:
        return await self._watch.watch_anext(self._iterator)


def _closer(close: Callable[..., Any], watch: _Watch, *, sync: bool) -> Callable[..., Any]:
    """``close`` that also reports what was seen, whatever ``close`` does."""
    if sync:

        def closer(*args: Any, **kwargs: Any) -> Any:
            try:
                return close(*args, **kwargs)
            finally:
                watch.watch_end()

        return closer

    async def aclose(*args: Any, **kwargs: Any) -> Any:
        try:
            return await close(*args, **kwargs)
        finally:
            watch.watch_end()

    return aclose


def instrument(stream: Any, guard: Guard, watcher: Any, *, sync: bool) -> Any:
    """Watch an SDK stream in place and return that same object.

    Reads of its iterator go through ``_Source`` (so ``next()``, ``iter()``, ``with`` and the
    async forms all behave natively), ``close`` is wrapped on the instance (the SDKs' ``__exit__``
    calls ``self.close()``), and letting go of the stream reports it through a ``weakref.finalize``
    hook that holds the watch state and never the stream. Raises ``AttributeError`` or
    ``TypeError``, with the stream left exactly as it was, when it is not built the way the SDKs
    build theirs (no ``_iterator`` or no ``close``, a read-only ``_iterator``, no instance
    ``close`` slot, no weak references).
    """
    iterator = stream._iterator
    if not callable(getattr(iterator, "__next__" if sync else "__anext__", None)):
        raise TypeError("the stream's iterator is not of the expected kind")
    close = stream.close
    if not callable(close):
        raise TypeError("the stream has no close()")
    watch = _Watch(guard, watcher)
    stream._iterator = (_Source if sync else _AsyncSource)(watch, iterator)
    try:
        stream.close = _closer(close, watch, sync=sync)
        weakref.finalize(stream, watch.watch_end).atexit = False
    except Exception:
        stream._iterator = iterator
        try:
            del stream.close
        except AttributeError:  # it was never set
            pass
        raise
    return stream
