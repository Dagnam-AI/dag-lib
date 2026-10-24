"""Record ``anthropic`` Messages calls (sync, async, streaming and the ``stream()`` helper).

``install`` wraps ``create``, ``parse`` and ``stream`` on the SDK's ``Messages`` and
``AsyncMessages`` classes, in the main namespace and in ``beta``. The Bedrock and Vertex clients
use those same classes for ``messages``, but their ``beta`` resources are separate classes that
copy the first-party functions when the SDK is imported, so those two are patched explicitly.
It imports ``anthropic`` lazily and only there (a missing SDK, or a version without one of the
resources, is simply not patched), so ``import dagnam_record`` never needs the SDK. The wrapper
records the call's JSON body parameters (taken before the call) and the response body, and
nothing else: no header or key is read, and the tokens a request body can carry (an MCP
server's ``authorization_token``) are replaced.

A streaming call returns the SDK's OWN stream object (the SDK checks the exact type of its
streams, so a proxy would fail ``isinstance``): the stream is watched in place, see
``_stream.instrument``. ``stream()`` does not go through ``create``: it returns a manager whose
``__enter__`` opens the request. The ``stream`` wrapper only attaches the request snapshot to the
manager; the manager classes' ``__enter__`` (``__aenter__``) is patched to watch the raw event
stream that the SDK's ``MessageStream`` then reads, so the helper is recorded once, however it is
left (exhausted, ``get_final_message()``, ``close()``, leaving the ``with`` block). Not recorded
in v1 (the call itself is unaffected): ``with_raw_response`` and ``with_streaming_response``
(their result is not a model; counted as ``dropped_error``).

Streams follow the shared policy in ``_stream``: a finished answer (``message_stop`` seen) is
recorded once however the stream ends, a cut one is counted as ``dropped_partial``.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, List, Optional, Tuple

from dagnam_record._anthropic_assemble import ERROR_STATUS, MessageAssembler
from dagnam_record._anthropic_body import Body, request_body, response_body
from dagnam_record._patch import Patcher, Plan, error_status, import_failed
from dagnam_record._recorder import Recorder
from dagnam_record._stream import deliver_pending

PROVIDER = "anthropic"
ENDPOINT = "messages"
CALL = "_dagnam_record_call"  # set on a stream manager: what its ``__enter__`` should record
RAW = "_raw_stream"  # the attribute of the SDK's message stream that holds the event stream


def _status(exc: BaseException) -> Optional[int]:
    """The call's HTTP status, else the one an error event's type stands for (the stream said 200)."""
    status = error_status(exc)
    if status is None:
        body = getattr(exc, "body", None)
        error = body.get("error") if isinstance(body, dict) else None
        kind = error.get("type") if isinstance(error, dict) else None
        status = ERROR_STATUS.get(kind) if isinstance(kind, str) else None
    return status


_PATCHER = Patcher(
    PROVIDER, request_body, response_body, lambda _: MessageAssembler(), _status, in_place=True
)


def _wrap_stream(original: Callable[..., Any]) -> Callable[..., Any]:
    """``stream(...)``: hand the manager back as the SDK built it, with the request noted on it."""

    @functools.wraps(original)
    def stream(resource: Any, *args: Any, **kwargs: Any) -> Any:
        recorder = _PATCHER.recorder
        if recorder is not None:
            deliver_pending()  # a safe point: no writer lock is held here
        request = _PATCHER.snapshot(recorder, kwargs) if recorder is not None else None
        manager = original(resource, *args, **kwargs)
        if recorder is not None and request is not None:
            recorder.fail_open(setattr, manager, CALL, (ENDPOINT, request))
        return manager

    return stream


def _tee(
    recorder: Recorder, call: Tuple[str, Body], stream: Any, started: float, *, sync: bool
) -> Any:
    """Watch the SDK's event stream inside its message stream, in place; the message stream is the result."""
    endpoint, request = call
    # an SDK whose message stream is built differently has no such attribute (or an event stream
    # that cannot be watched): ``tee`` counts that once as a drop and the call runs untouched
    _PATCHER.tee(recorder, endpoint, request, getattr(stream, RAW, None), started, sync=sync)
    return stream


def _wrap_enter(original: Callable[..., Any], *, is_async: bool) -> Callable[..., Any]:
    """``__enter__`` / ``__aenter__`` of a stream manager: the moment the request is made."""

    def begin(manager: Any) -> Tuple[Optional[Recorder], Optional[Tuple[str, Body]]]:
        recorder = _PATCHER.recorder
        call = getattr(manager, CALL, None)
        return recorder, call

    def failed(recorder: Recorder, call: Tuple[str, Body], exc: Exception, started: float) -> None:
        recorder.fail_open(_PATCHER.record_error, recorder, call[0], call[1], exc, started)

    if is_async:

        @functools.wraps(original)
        async def enter_async(manager: Any) -> Any:
            recorder, call = begin(manager)
            if recorder is None or call is None:
                return await original(manager)
            started = time.perf_counter()
            try:
                stream = await original(manager)
            except Exception as exc:
                failed(recorder, call, exc, started)
                raise
            except BaseException:
                _PATCHER.cancelled(recorder)
                raise
            return _tee(recorder, call, stream, started, sync=False)

        return enter_async

    @functools.wraps(original)
    def enter(manager: Any) -> Any:
        recorder, call = begin(manager)
        if recorder is None or call is None:
            return original(manager)
        started = time.perf_counter()
        try:
            stream = original(manager)
        except Exception as exc:
            failed(recorder, call, exc, started)
            raise
        except BaseException:
            _PATCHER.cancelled(recorder)
            raise
        return _tee(recorder, call, stream, started, sync=True)

    return enter


def _resources() -> List[Tuple[Any, bool]]:
    """The SDK resource classes this version has: ``(class, is_async)``."""
    found: List[Tuple[Any, bool]] = []
    # Lazy, runtime-optional imports: ``import dagnam_record`` must not need the SDK.
    try:
        import anthropic.resources.messages as plain
    except Exception as exc:  # absent, or a broken install
        import_failed("anthropic.resources.messages", exc)
    else:
        found += [
            (getattr(plain, "Messages", None), False),
            (getattr(plain, "AsyncMessages", None), True),
        ]
    try:
        import anthropic.resources.beta.messages as beta
    except Exception as exc:  # absent, or a broken install
        import_failed("anthropic.resources.beta.messages", exc)
    else:
        found += [
            (getattr(beta, "Messages", None), False),
            (getattr(beta, "AsyncMessages", None), True),
        ]
    return found + _cloud_resources()


def _cloud_resources() -> List[Tuple[Any, bool]]:
    """The Bedrock and Vertex beta resources.

    They copy the first-party functions when the SDK is imported, so patching the first-party
    classes later does not reach them.
    """
    found: List[Tuple[Any, bool]] = []
    try:
        import anthropic.lib.bedrock._beta_messages as bedrock
    except Exception as exc:  # absent, or a broken install
        import_failed("anthropic.lib.bedrock._beta_messages", exc)
    else:
        found += [
            (getattr(bedrock, "Messages", None), False),
            (getattr(bedrock, "AsyncMessages", None), True),
        ]
    try:
        import anthropic.lib.vertex._beta_messages as vertex
    except Exception as exc:  # absent, or a broken install
        import_failed("anthropic.lib.vertex._beta_messages", exc)
    else:
        found += [
            (getattr(vertex, "Messages", None), False),
            (getattr(vertex, "AsyncMessages", None), True),
        ]
    return found


def _managers() -> List[Tuple[Any, str, bool]]:
    """The stream manager classes this version has: ``(class, enter method, is_async)``."""
    found: List[Tuple[Any, str, bool]] = []
    try:
        import anthropic.lib.streaming as streaming
    except Exception as exc:  # absent, or a broken install
        import_failed("anthropic.lib.streaming", exc)
        return found
    for name, is_async in (
        ("MessageStreamManager", False),
        ("AsyncMessageStreamManager", True),
        ("BetaMessageStreamManager", False),
        ("BetaAsyncMessageStreamManager", True),
    ):
        found.append(
            (getattr(streaming, name, None), "__aenter__" if is_async else "__enter__", is_async)
        )
    return found


def _resource_plan() -> List[Plan]:
    plan: List[Plan] = []
    for cls, is_async in _resources():
        plan += [
            (cls, "create", lambda f, a=is_async: _PATCHER.wrap(f, ENDPOINT, is_async=a)),
            # ``parse`` has a ``stream`` argument that the SDK does not honour: never a stream here
            (
                cls,
                "parse",
                lambda f, a=is_async: _PATCHER.wrap(f, ENDPOINT, is_async=a, stream_flag=False),
            ),
            (cls, "stream", _wrap_stream),
        ]
    return plan


def _manager_plan() -> List[Plan]:
    return [
        (cls, method, lambda f, a=is_async: _wrap_enter(f, is_async=a))
        for cls, method, is_async in _managers()
    ]


def install(recorder: Recorder) -> bool:
    """Record through ``recorder``; True when at least one method is (now) patched.

    Safe to call again: a method is wrapped once, and the newest recorder is used. Returns False
    under the kill switch and when the SDK (or all its known resources) is not there.
    """
    patched = _PATCHER.install(recorder, _resource_plan())
    if patched:
        _PATCHER.patch(_manager_plan())
    return patched


def uninstall() -> None:
    """Put the SDK's own methods back and stop recording (used by tests and ``dagnam_record.uninstall``)."""
    _PATCHER.uninstall()
