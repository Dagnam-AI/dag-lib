"""What every SDK patch shares: wrap a method, tell the recorder how the call went, undo it.

A provider module (``_openai``, ``_anthropic``) lists the SDK methods to wrap and gives its
``Patcher`` three things: how to turn the call's keyword arguments and its result into plain JSON
bodies, and how to fold the events of a stream into the body a non-stream call would return (an
*assembler* with ``add(event)``, ``complete``, ``status`` and ``result()``).

The wrappers never change the call: the SDK's own result (or exception) is handed back as it
is, a stream is teed (see ``_stream``), and whatever the recorder does is fail-open and counted.
An ``Exception`` from the SDK is a failed call (a line with the error's status); anything else
(``KeyboardInterrupt``, ``CancelledError``) is not a failure, it is counted as ``dropped_partial``
and nothing is written.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from dagnam_record._envelope import MAX_LINE_BYTES
from dagnam_record._recorder import Recorder, recording_disabled
from dagnam_record._stream import (
    AsyncTeeStream,
    TeeStream,
    deliver_pending,
    instrument,
    register_exit,
)

LOGGER = logging.getLogger("dagnam_record")
Body = Dict[str, Any]
Build = Callable[[Callable[..., Any]], Callable[..., Any]]
Plan = Tuple[Any, str, Build]
"""``(class, method name, build)``: ``build(the SDK's function)`` makes the wrapper."""
MARKER = "__dagnam_record__"
MAX_STREAM_CHARS = MAX_LINE_BYTES
"""More text than this in one stream cannot fit a line, so the stream is let go at this point."""


def import_failed(name: str, exc: Exception) -> None:
    """Note that the SDK module ``name`` could not be imported, so what it holds is not patched.

    A missing module (``ImportError``) is the usual case and is silent. A half-installed or broken
    package fails with any exception (``AttributeError``, ``RuntimeError``, ...): that must not
    escape from ``install``, but it is logged, since the program's calls are then not recorded.
    """
    if not isinstance(exc, ImportError):
        LOGGER.warning(
            "dagnam_record: cannot import %s (%s); it is not recorded", name, type(exc).__name__
        )


def error_status(exc: BaseException) -> Optional[int]:
    """The HTTP status of a failed call, or ``None`` (recorded as "no response").

    Only an error status counts: an SDK raises a status error for a stream whose connection
    answered 200 and then sent an error event, and a 200 there would read as a good row.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool) and 400 <= status <= 599:
        return status
    return None


class Watcher:
    """Tells the recorder how a stream went (see ``_stream`` for when each call is made)."""

    def __init__(
        self, patcher: Patcher, recorder: Recorder, endpoint: str, request: Body, started: float
    ) -> None:
        self._patcher = patcher
        self._recorder = recorder
        self._endpoint = endpoint
        self._request = request
        self._started = started
        self._assembler = patcher.assembler(endpoint)
        self._oversized = False

    def item(self, value: Any) -> None:
        """One event of the stream; past ``MAX_STREAM_CHARS`` of text the stream is let go."""
        if self._oversized:
            return
        self._assembler.add(value)
        if self._assembler.size > MAX_STREAM_CHARS:
            self._oversized = True  # the line would be dropped anyway: keep no more of it
            self._assembler = self._patcher.assembler(self._endpoint)  # an empty one frees the text

    def done(self) -> None:
        """The recorder is done with the stream: write it if it finished, else count it."""
        if self._oversized:
            self._recorder.writer.count("dropped_large")
        elif self._assembler.complete:
            self._recorder.emit(
                self._patcher.provider,
                self._endpoint,
                self._request,
                self._assembler.result(),
                status=self._assembler.status,
                started=self._started,
            )
        else:
            self._recorder.writer.count("dropped_partial")

    def error(self, exc: BaseException) -> None:
        """The stream raised. After its end marker the answer is complete and is recorded as such."""
        if self._oversized or self._assembler.complete:
            self.done()
        else:
            self._patcher.record_error(
                self._recorder, self._endpoint, self._request, exc, self._started
            )

    def cancelled(self) -> None:
        """The stream was interrupted or cancelled: a complete answer is still kept."""
        if self._oversized or self._assembler.complete:
            self.done()
        else:
            self._patcher.cancelled(self._recorder)


class Patcher:
    """The state and wrappers of one provider's patch."""

    def __init__(
        self,
        provider: str,
        request_body: Callable[[Dict[str, Any]], Body],
        response_body: Callable[[Any], Body],
        assembler: Callable[[str], Any],
        status_of: Callable[[BaseException], Optional[int]] = error_status,
        *,
        in_place: bool = False,
    ) -> None:
        self.provider = provider
        self.request_body = request_body
        self.response_body = response_body
        self.assembler = assembler
        self.status_of = status_of
        self.in_place = in_place  # watch the SDK's own stream object, not a proxy of it
        self.recorder: Optional[Recorder] = None  # None: the wrappers pass straight through
        # (class, method name, the SDK's function, our wrapper)
        self._patched: List[Tuple[Any, str, Any, Any]] = []

    def snapshot(self, recorder: Recorder, kwargs: Dict[str, Any]) -> Optional[Body]:
        """The request as plain JSON now, or ``None`` (already counted) if it cannot be taken."""
        taken: List[Body] = []
        return (
            taken[0]
            if recorder.fail_open(lambda: taken.append(self.request_body(kwargs)))
            else None
        )

    def record_ok(
        self, recorder: Recorder, endpoint: str, request: Body, result: Any, started: float
    ) -> None:
        """A finished call that returned ``result``."""
        recorder.emit(
            self.provider,
            endpoint,
            request,
            self.response_body(result),
            status=200,
            started=started,
        )

    def record_error(
        self,
        recorder: Recorder,
        endpoint: str,
        request: Optional[Body],
        exc: BaseException,
        started: float,
    ) -> None:
        """A failed call: its HTTP status when there was one (else "no response") and the error's class."""
        if request is not None:
            recorder.emit(
                self.provider,
                endpoint,
                request,
                {"error": {"type": type(exc).__name__}},
                status=self.status_of(exc),
                started=started,
            )

    @staticmethod
    def cancelled(recorder: Recorder) -> None:
        """A call that was cancelled or interrupted: counted, never written."""
        recorder.fail_open(recorder.writer.count, "dropped_partial")

    def tee(
        self,
        recorder: Recorder,
        endpoint: str,
        request: Body,
        stream: Any,
        started: float,
        *,
        sync: bool,
    ) -> Any:
        """``stream`` watched by the recorder; ``stream`` itself if it cannot be watched.

        A provider whose SDK checks the exact type of its streams is watched in place (the same
        object comes back); the others get a proxy. A stream that cannot be watched is counted
        once as ``dropped_error`` and handed back untouched.
        """
        try:
            watcher = Watcher(self, recorder, endpoint, request, started)
            if self.in_place:
                return instrument(stream, recorder.fail_open, watcher, sync=sync)
            return (TeeStream if sync else AsyncTeeStream)(stream, recorder.fail_open, watcher)
        except Exception:  # a result that cannot be teed is handed back as it is
            recorder.writer.count("dropped_error")
            return stream

    def _after(
        self,
        recorder: Recorder,
        endpoint: str,
        request: Optional[Body],
        streaming: bool,
        result: Any,
        started: float,
        *,
        sync: bool,
    ) -> Any:
        """What the wrapper returns: the result itself, or the result teed when it is a stream."""
        if request is None:
            return result
        if not streaming:
            recorder.fail_open(self.record_ok, recorder, endpoint, request, result, started)
            return result
        return self.tee(recorder, endpoint, request, result, started, sync=sync)

    def wrap(
        self,
        original: Callable[..., Any],
        endpoint: str,
        *,
        is_async: bool,
        stream_flag: bool = True,
    ) -> Callable[..., Any]:
        """A wrapper for a method that makes one request and returns its result.

        With ``stream_flag`` a truthy ``stream`` argument means the result is a stream to tee.
        """
        patcher = self

        def finish(
            recorder: Recorder,
            request: Optional[Body],
            kwargs: Dict[str, Any],
            result: Any,
            started: float,
        ) -> Any:
            teed = stream_flag and bool(kwargs.get("stream"))
            return patcher._after(
                recorder, endpoint, request, teed, result, started, sync=not is_async
            )

        def failed(
            recorder: Recorder, request: Optional[Body], exc: Exception, started: float
        ) -> None:
            recorder.fail_open(patcher.record_error, recorder, endpoint, request, exc, started)

        if is_async:

            @functools.wraps(original)
            async def call_async(resource: Any, *args: Any, **kwargs: Any) -> Any:
                recorder = patcher.recorder
                if recorder is None or recorder.paused():
                    return await original(resource, *args, **kwargs)
                deliver_pending()  # a safe point: no writer lock is held here
                request = patcher.snapshot(recorder, kwargs)
                started = time.perf_counter()
                try:
                    result = await original(resource, *args, **kwargs)
                except Exception as exc:
                    failed(recorder, request, exc, started)
                    raise
                except BaseException:
                    patcher.cancelled(recorder)
                    raise
                return finish(recorder, request, kwargs, result, started)

            return call_async

        @functools.wraps(original)
        def call_sync(resource: Any, *args: Any, **kwargs: Any) -> Any:
            recorder = patcher.recorder
            if recorder is None or recorder.paused():
                return original(resource, *args, **kwargs)
            deliver_pending()
            request = patcher.snapshot(recorder, kwargs)
            started = time.perf_counter()
            try:
                result = original(resource, *args, **kwargs)
            except Exception as exc:
                failed(recorder, request, exc, started)
                raise
            except BaseException:
                patcher.cancelled(recorder)
                raise
            return finish(recorder, request, kwargs, result, started)

        return call_sync

    def patch(self, plan: Iterable[Plan]) -> bool:
        """Wrap each method of the plan once; True when at least one of them exists."""
        found = False
        for cls, method, build in plan:
            original = getattr(cls, method, None)
            if original is None:
                continue
            found = True
            if not getattr(original, MARKER, False):
                wrapper = build(original)
                setattr(wrapper, MARKER, True)
                setattr(cls, method, wrapper)
                self._patched.append((cls, method, original, wrapper))
        return found

    def install(self, recorder: Recorder, plan: Iterable[Plan]) -> bool:
        """Record through ``recorder``; False under the kill switch or when nothing is there to patch."""
        if recording_disabled():
            return False
        patched = self.patch(plan)
        self.recorder = recorder if patched else None
        if patched:
            register_exit()  # after any writer made so far: it must run before that writer closes
        return patched

    def uninstall(self) -> None:
        """Put the SDK's own methods back and stop recording.

        A method that another library has wrapped since is left as it is: putting the SDK's
        function back would remove that library's patch. Our wrapper under it is inert once the
        recorder is gone (it passes every call straight through).
        """
        deliver_pending()
        self.recorder = None
        while self._patched:
            cls, method, original, wrapper = self._patched.pop()
            if cls.__dict__.get(method) is wrapper:
                setattr(cls, method, original)
