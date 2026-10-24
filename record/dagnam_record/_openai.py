"""Record ``openai`` chat completions and Responses calls (sync, async and streaming).

``install`` wraps ``create`` and ``parse`` on the SDK's resource classes; it imports ``openai``
lazily and only there (a missing SDK, or a version without one of the resources, is simply not
patched), so ``import dagnam_record`` never needs the SDK. The wrapper records the call's JSON
body parameters (taken before the call, so later changes to the caller's data do not alter the
record) and the response body, and nothing else: no header, key or organisation id is read.

The ``stream()`` helpers call the wrapped ``create`` and so are recorded through it. Not
recorded in v1 (the call itself is unaffected): ``with_raw_response`` and
``with_streaming_response`` (their result is not a model; counted as ``dropped_error``).

An interrupted or cancelled call is not a failed call: it is counted as ``dropped_partial`` and
nothing is written.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from dagnam_record._openai_assemble import ChatAssembler, ResponsesAssembler
from dagnam_record._openai_body import Body, request_body, response_body
from dagnam_record._recorder import Recorder, recording_disabled
from dagnam_record._stream import AsyncTeeStream, TeeStream

PROVIDER = "openai"
_METHODS = ("create", "parse")


class _Active:
    """The recorder the wrappers report to; ``None`` after ``uninstall`` (they pass straight through)."""

    recorder: Optional[Recorder] = None


_ACTIVE = _Active()
_PATCHED: List[Tuple[Any, str, Any]] = []  # (class, method name, the SDK's own function)


def _snapshot(recorder: Recorder, kwargs: Dict[str, Any]) -> Optional[Body]:
    """The request as plain JSON now, or ``None`` (already counted) if it cannot be taken."""
    taken: List[Body] = []
    return taken[0] if recorder.fail_open(lambda: taken.append(request_body(kwargs))) else None


def _record_ok(
    recorder: Recorder, endpoint: str, request: Body, result: Any, started: float
) -> None:
    recorder.emit(PROVIDER, endpoint, request, response_body(result), status=200, started=started)


def _record_error(
    recorder: Recorder, endpoint: str, request: Optional[Body], exc: BaseException, started: float
) -> None:
    """A failed call: its HTTP status when the SDK had one (else "no response") and the error's class."""
    if request is not None:
        status = getattr(exc, "status_code", None)
        recorder.emit(
            PROVIDER,
            endpoint,
            request,
            {"error": {"type": type(exc).__name__}},
            status=status if isinstance(status, int) and not isinstance(status, bool) else None,
            started=started,
        )


def _cancelled(recorder: Recorder) -> None:
    """A call that was cancelled or interrupted: counted, never written."""
    recorder.fail_open(recorder.writer.count, "dropped_partial")


class _Watcher:
    """Tells the recorder how a stream went (see ``_stream`` for when each call is made)."""

    def __init__(
        self,
        recorder: Recorder,
        endpoint: str,
        request: Body,
        started: float,
        assembler: Union[ChatAssembler, ResponsesAssembler],
    ) -> None:
        self._recorder = recorder
        self._endpoint = endpoint
        self._request = request
        self._started = started
        self._assembler = assembler

    def item(self, value: Any) -> None:
        """One event of the stream."""
        self._assembler.add(value)

    def done(self) -> None:
        """The recorder is done with the stream: write it if it finished, else count it."""
        if self._assembler.complete:
            self._recorder.emit(
                PROVIDER,
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
        if self._assembler.complete:
            self.done()
        else:
            _record_error(self._recorder, self._endpoint, self._request, exc, self._started)

    def cancelled(self) -> None:
        """The stream was interrupted or cancelled: a complete answer is still kept."""
        if self._assembler.complete:
            self.done()
        else:
            _cancelled(self._recorder)


def _after(
    recorder: Recorder,
    endpoint: str,
    request: Optional[Body],
    stream: Any,
    result: Any,
    started: float,
    *,
    sync: bool,
) -> Any:
    """What the wrapper returns: the result itself, or the result teed when it is a stream."""
    if request is None:
        return result
    if not stream:
        recorder.fail_open(_record_ok, recorder, endpoint, request, result, started)
        return result
    try:
        assembler = ChatAssembler() if endpoint == "chat.completions" else ResponsesAssembler()
        watcher = _Watcher(recorder, endpoint, request, started, assembler)
        return (TeeStream if sync else AsyncTeeStream)(result, recorder.fail_open, watcher)
    except Exception:  # a result that cannot be teed is handed back as it is
        recorder.writer.count("dropped_error")
        return result


def _wrap_sync(original: Callable[..., Any], endpoint: str) -> Callable[..., Any]:
    @functools.wraps(original)
    def create(self: Any, *args: Any, **kwargs: Any) -> Any:
        recorder = _ACTIVE.recorder
        if recorder is None:
            return original(self, *args, **kwargs)
        request = _snapshot(recorder, kwargs)
        started = time.perf_counter()
        try:
            result = original(self, *args, **kwargs)
        except Exception as exc:
            recorder.fail_open(_record_error, recorder, endpoint, request, exc, started)
            raise
        except BaseException:
            _cancelled(recorder)
            raise
        return _after(recorder, endpoint, request, kwargs.get("stream"), result, started, sync=True)

    return create


def _wrap_async(original: Callable[..., Any], endpoint: str) -> Callable[..., Any]:
    @functools.wraps(original)
    async def create(self: Any, *args: Any, **kwargs: Any) -> Any:
        recorder = _ACTIVE.recorder
        if recorder is None:
            return await original(self, *args, **kwargs)
        request = _snapshot(recorder, kwargs)
        started = time.perf_counter()
        try:
            result = await original(self, *args, **kwargs)
        except Exception as exc:
            recorder.fail_open(_record_error, recorder, endpoint, request, exc, started)
            raise
        except BaseException:
            _cancelled(recorder)
            raise
        return _after(
            recorder, endpoint, request, kwargs.get("stream"), result, started, sync=False
        )

    return create


def _targets() -> List[Tuple[Any, str, bool]]:
    """The SDK classes this version has: ``(class, endpoint, is_async)``."""
    found: List[Tuple[Any, str, bool]] = []
    # Lazy, runtime-optional imports: ``import dagnam_record`` must not need the SDK.
    try:
        import openai.resources.chat.completions as chat
    except ImportError:
        pass
    else:
        found += [
            (getattr(chat, "Completions", None), "chat.completions", False),
            (getattr(chat, "AsyncCompletions", None), "chat.completions", True),
        ]
    try:
        import openai.resources.beta as beta
    except ImportError:
        pass
    else:  # older SDKs kept ``parse`` in a separate beta class; newer ones have no such module
        legacy = getattr(getattr(beta, "chat", None), "completions", None)
        found += [
            (getattr(legacy, "Completions", None), "chat.completions", False),
            (getattr(legacy, "AsyncCompletions", None), "chat.completions", True),
        ]
    try:
        import openai.resources.responses as responses
    except ImportError:
        pass
    else:
        found += [
            (getattr(responses, "Responses", None), "responses", False),
            (getattr(responses, "AsyncResponses", None), "responses", True),
        ]
    return found


def install(recorder: Recorder) -> bool:
    """Record through ``recorder``; True when at least one method is (now) patched.

    Safe to call again: a method is wrapped once, and the newest recorder is used. Returns False
    under the kill switch and when the SDK (or all its known resources) is not there.
    """
    if recording_disabled():
        return False
    patched = False
    for cls, endpoint, is_async in _targets():
        for method in _METHODS:
            original = getattr(cls, method, None)
            if original is None:
                continue
            patched = True
            if not getattr(original, "__dagnam_record__", False):
                wrapper: Any = (_wrap_async if is_async else _wrap_sync)(original, endpoint)
                wrapper.__dagnam_record__ = True
                setattr(cls, method, wrapper)
                _PATCHED.append((cls, method, original))
    _ACTIVE.recorder = recorder if patched else None
    return patched


def uninstall() -> None:
    """Put the SDK's own methods back and stop recording (used by tests and ``dagnam_record.uninstall``)."""
    _ACTIVE.recorder = None
    while _PATCHED:
        cls, method, original = _PATCHED.pop()
        setattr(cls, method, original)
