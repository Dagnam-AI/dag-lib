"""Watch a provider's stream while handing it to the caller untouched.

Policy (one place, shared by every provider patch). A *watcher* is told about the stream with
four calls (``item``, ``done``, ``error``, ``cancelled``), each run through ``guard`` so that a
failing watcher never reaches the caller:

* every item is passed on, in order, exactly as the SDK produced it;
* an exception raised by the stream is passed on unchanged and ``error(exc)`` is called once;
* an interrupt or cancellation (``KeyboardInterrupt``, ``asyncio.CancelledError``, anything that
  is not an ``Exception``) is passed on unchanged and ``cancelled()`` is called once: it is not a
  failed call;
* ``done()`` is called exactly once when the recorder has seen all it will see of a stream, and
  the watcher decides whether what it saw was a complete answer (record it) or a partial one
  (count it, write nothing). That is: when the stream runs out, when the caller closes it
  (``close()``, leaving a ``with`` block, ``aclose``), and when the caller lets go of it. Letting
  go is delivered LATER, never inside the collector (see below). At the latest it is delivered
  when the process exits (``finish_live``), so a stream that is still alive then, in a reference
  cycle or held by a global, is reported too: finished ones are recorded, unfinished ones counted;
* once any of these has happened nothing more is reported for that stream, whichever thread or
  hook gets there first (the report is claimed atomically);
* a watcher that fails stops watching: the failure is counted through ``guard``, ``done`` is not
  called and the stream is simply not recorded.

Letting go. A stream can die at any allocation, and the collector runs the hook that learns of it
on whatever thread triggered it, possibly inside a writer call that holds the writer's
non-reentrant locks. Reporting from there could deadlock, so the hook (a ``weakref.finalize`` for
an SDK stream, ``__del__`` for a proxy) only puts the stream's watch state on a ``SimpleQueue``,
which is safe to use from any such context. ``deliver_pending`` reports what is queued and is
called only from places that hold no writer lock: the start of each wrapped call, the writer's
loop and ``close()``, ``uninstall()`` and the exit hook.

The exit hook ``finish_live`` walks a registry of every watch that has not reported yet (strong
references to the small watch state, never to a stream, so nothing is kept alive; a stream that
died and waits in the queue is in it too). Each ``install`` registers it with ``atexit`` again,
after the writer the recorder was built on, so it runs first and the writer's close then writes
what it reported.

Two ways to watch a stream. ``TeeStream`` / ``AsyncTeeStream`` are proxies: they forward every
other attribute to the stream they wrap, report the wrapped stream's class for ``isinstance``
checks and return themselves from ``with``. A proxy cannot satisfy an SDK that checks the exact
type of the object (the Anthropic ``Stream`` does), so ``instrument`` watches an SDK stream
IN PLACE instead: the caller keeps the SDK's own object, whose iterator and ``close`` are
replaced on the instance. Both report through the same ``_Watch``. Two threads reading one stream
at once is misuse and is not guarded.
"""

from __future__ import annotations

import atexit
import os
from queue import Empty, SimpleQueue
from types import TracebackType
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Iterator, List, Optional, Set, Type
import weakref

Guard = Callable[..., bool]
"""``Recorder.fail_open``: runs a step, counts an error, and says whether the step ran cleanly."""


def get(obj: Any, name: str) -> Any:
    """``obj[name]`` for a dict, ``obj.name`` for an SDK object, else ``None``."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


class _Registry:
    """Every watch that has not reported, and those whose stream died and wait to be reported."""

    def __init__(self) -> None:
        self.live: Set[_Watch] = set()
        self.pending: SimpleQueue[_Watch] = SimpleQueue()


_REGISTRY = _Registry()


class _Watch:
    """What to call, and that it is called only once (state only: it never holds a stream)."""

    def __init__(self, guard: Guard, watcher: Any) -> None:
        self._guard = guard
        self._watcher = watcher
        self._open: List[None] = [None]  # one token: whoever takes it makes the report
        self._broken = False  # a watcher call failed: this stream is not recorded

    def _claim(self) -> bool:
        try:
            self._open.pop()  # atomic: of two threads racing to report, one gets the token
        except IndexError:
            return False
        _REGISTRY.live.discard(self)
        return True

    def arm(self) -> None:
        """The stream is built: report it at exit if nothing else has by then."""
        _REGISTRY.live.add(self)

    def disarm(self) -> None:
        """Never report this stream (it could not be given back after a failed setup)."""
        self._claim()

    def expire(self) -> None:
        """The stream died. Lock-free, so it is safe wherever the collector runs it."""
        if self._open:
            _REGISTRY.pending.put(self)

    def watch_item(self, value: Any) -> None:
        """One item went by."""
        if self._open and not self._broken:
            self._broken = not self._guard(self._watcher.item, value)

    def watch_end(self) -> None:
        """The recorder is done with the stream (it ended, was closed or let go)."""
        if self._claim() and not self._broken:
            self._guard(self._watcher.done)

    def watch_error(self, exc: BaseException) -> None:
        """The stream raised an ``Exception``."""
        if self._claim():
            self._guard(self._watcher.error, exc)

    def watch_cancel(self) -> None:
        """The stream was interrupted or cancelled."""
        if self._claim():
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


def _end(watch: _Watch) -> None:
    try:
        watch.watch_end()
    except Exception:  # a guard never raises; this is for a watcher object that is not ours
        pass


def deliver_pending() -> None:
    """Report the streams that died since the last call. Only from places that hold no writer lock."""
    while True:
        try:
            watch = _REGISTRY.pending.get_nowait()
        except Empty:
            return
        _end(watch)


def finish_live() -> None:
    """The exit hook: report every stream that has not reported (those that died are among them)."""
    for watch in list(_REGISTRY.live):
        _end(watch)


def register_exit() -> None:
    """(Re)register ``finish_live`` so that it is the next exit hook to run.

    Hooks run last in, first out, so this puts it before the close of every writer made so far.
    """
    atexit.unregister(finish_live)
    atexit.register(finish_live)


def live_count() -> int:
    """How many watches have not reported yet."""
    return len(_REGISTRY.live)


def forget_live() -> None:
    """Drop every registered watch (a forked child must not report its parent's streams)."""
    _REGISTRY.live = set()
    _REGISTRY.pending = SimpleQueue()


getattr(os, "register_at_fork", lambda **hooks: None)(after_in_child=forget_live)


class _Tap:
    """The proxies' base: a watch over the stream they wrap."""

    def __init__(self, stream: Any, guard: Guard, watcher: Any) -> None:
        self._watch = _Watch(guard, watcher)
        self._stream = stream
        self._armed = False  # set last: a proxy that failed to build reports nothing

    def _arm(self) -> None:
        self._armed = True
        self._watch.arm()

    def __del__(self) -> None:
        if self.__dict__.get("_armed"):
            try:
                self._watch.expire()  # reported at the next safe point, never from here
            except Exception:  # the interpreter is shutting down and its globals are gone
                pass

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
        self._arm()

    def __iter__(self) -> TeeStream:
        return self

    def __next__(self) -> Any:
        return self._watch.watch_next(self._iterator)

    def close(self) -> Any:
        """Close the stream; what the recorder saw so far is reported either way."""
        try:
            return self._stream.close()
        finally:
            self._watch.watch_end()

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
            self._watch.watch_end()


class AsyncTeeStream(_Tap):
    """An asynchronous stream, watched."""

    def __init__(self, stream: Any, guard: Guard, watcher: Any) -> None:
        super().__init__(stream, guard, watcher)
        self._iterator: AsyncIterator[Any] = stream.__aiter__()
        self._arm()

    def __aiter__(self) -> AsyncTeeStream:
        return self

    async def __anext__(self) -> Any:
        return await self._watch.watch_anext(self._iterator)

    async def close(self) -> Any:
        """Close the stream; what the recorder saw so far is reported either way."""
        try:
            return await self._stream.close()
        finally:
            self._watch.watch_end()

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
            self._watch.watch_end()


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
    calls ``self.close()``), and letting go of the stream is reported through a
    ``weakref.finalize`` hook that holds the watch state and never the stream (see the module
    docstring for when). Raises ``AttributeError`` or ``TypeError``, with the stream left exactly
    as it was, when it is not built the way the SDKs build theirs (no ``_iterator`` or no
    ``close``, a read-only ``_iterator``, no instance ``close`` slot, no weak references). If the
    old iterator cannot be given back after such a failure the watch is disarmed, so the reader
    left inside the stream passes items on and reports nothing.
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
        weakref.finalize(stream, watch.expire).atexit = False
    except Exception:
        try:
            stream._iterator = iterator
            try:
                del stream.close
            except AttributeError:  # it was never set
                pass
        except Exception:  # cannot be undone: the reader stays in the stream and must be silent
            watch.disarm()
        raise
    watch.arm()
    return stream
