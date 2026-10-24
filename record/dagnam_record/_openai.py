"""Record ``openai`` chat completions and Responses calls (sync, async and streaming).

``install`` wraps ``create`` on the SDK's resource classes; it imports ``openai`` lazily and
only there (a missing SDK, or a version without one of the resources, is simply not patched), so
``import dagnam_record`` never needs the SDK. The wrapper records the call's JSON body
parameters and the response body and nothing else: no header, key or organisation id is read.

Not recorded in v1 (the call itself is unaffected): ``with_raw_response`` / ``with_streaming_response``
(their result is not a model; counted as ``dropped_error``), ``parse`` helpers and ``stream()``.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from dagnam_record._envelope import omit_media
from dagnam_record._recorder import (
    Recorder,
    media_omitted,
    recording_disabled,
    to_plain,
)
from dagnam_record._stream import AsyncTeeStream, TeeStream, get

PROVIDER = "openai"
_TRANSPORT = frozenset({"extra_headers", "extra_query", "extra_body", "timeout", "api_key"})
_TERMINAL = ("response.completed", "response.incomplete", "response.failed")
_AUDIO_PARTS = frozenset({"output_audio", "audio"})

Body = Dict[str, Any]


class _Active:
    """The recorder the wrappers report to; ``None`` after ``uninstall`` (they pass straight through)."""

    recorder: Optional[Recorder] = None


_ACTIVE = _Active()
_PATCHED: List[Tuple[Any, Any]] = []  # (class, the SDK's own create), for uninstall


def _request_body(kwargs: Dict[str, Any]) -> Body:
    """The call's JSON body parameters: its keyword arguments minus the transport ones."""
    kept = {key: value for key, value in kwargs.items() if key not in _TRANSPORT}
    body = to_plain(kept)
    _omit_output_media(body.get("input"))
    return body


def _replace_payload(container: Any, key: str) -> None:
    if isinstance(container, dict) and isinstance(container.get(key), str):
        container[key] = media_omitted(container[key])


def _omit_output_media(items: Any) -> None:
    """Swap base64 outputs (generated images, audio) in Responses-style items for their size."""
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict):
            if item.get("type") == "image_generation_call":
                _replace_payload(item, "result")
            parts = item.get("content")
            for part in parts if isinstance(parts, list) else []:
                if isinstance(part, dict) and part.get("type") in _AUDIO_PARTS:
                    _replace_payload(part, "data")


def _response_body(result: Any) -> Body:
    """A response as plain JSON with its media payloads (base64 audio, images) replaced by their size."""
    body = to_plain(result)
    if not isinstance(body, dict):
        raise ValueError("the response is not a model (a raw or streaming-response call)")
    for choice in body.get("choices") or []:
        message = choice.get("message") if isinstance(choice, dict) else None
        _replace_payload(message.get("audio") if isinstance(message, dict) else None, "data")
    _omit_output_media(body.get("output"))
    return omit_media(body)


def _record_ok(
    recorder: Recorder, endpoint: str, kwargs: Body, result: Any, started: float
) -> None:
    response = _response_body(result)
    recorder.emit(PROVIDER, endpoint, _request_body(kwargs), response, status=200, started=started)


def _record_error(
    recorder: Recorder,
    endpoint: str,
    kwargs: Body,
    exc: BaseException,
    started: float,
) -> None:
    """A failed call: its HTTP status when the SDK had one (else "no response") and the error's class."""
    status = getattr(exc, "status_code", None)
    recorder.emit(
        PROVIDER,
        endpoint,
        _request_body(kwargs),
        {"error": {"type": type(exc).__name__}},
        status=status if isinstance(status, int) and not isinstance(status, bool) else None,
        started=started,
    )


class ChatAssembler:
    """Fold ``chat.completion.chunk`` events into the message a non-stream call returns."""

    status = 200

    def __init__(self) -> None:
        self._head: Body = {}
        self._choices: Dict[int, Body] = {}
        self._usage: Any = None

    def add(self, chunk: Any) -> None:
        """Take one chunk."""
        for key in ("id", "model", "created", "system_fingerprint", "service_tier"):
            if key not in self._head and get(chunk, key) is not None:
                self._head[key] = get(chunk, key)
        if get(chunk, "usage") is not None:
            self._usage = get(chunk, "usage")
        for choice in get(chunk, "choices") or []:
            self._choice(choice)

    def _choice(self, choice: Any) -> None:
        index = get(choice, "index") or 0
        slot = self._choices.setdefault(
            index, {"role": None, "content": [], "refusal": [], "calls": {}, "finish": None}
        )
        if get(choice, "finish_reason"):
            slot["finish"] = get(choice, "finish_reason")
        delta = get(choice, "delta")
        if get(delta, "role"):
            slot["role"] = get(delta, "role")
        for key in ("content", "refusal"):
            if isinstance(get(delta, key), str):
                slot[key].append(get(delta, key))
        for call in get(delta, "tool_calls") or []:
            self._call(slot["calls"], get(call, "index") or 0, call)
        if get(delta, "function_call") is not None:
            self._call(slot["calls"], -1, {"function": get(delta, "function_call")})
        if get(delta, "audio") is not None:
            audio = slot.setdefault("audio", {"id": None, "transcript": [], "bytes": 0})
            audio["id"] = get(get(delta, "audio"), "id") or audio["id"]
            audio["transcript"].append(get(get(delta, "audio"), "transcript") or "")
            audio["bytes"] += len(get(get(delta, "audio"), "data") or "")

    @staticmethod
    def _call(calls: Dict[int, Body], index: int, delta: Any) -> None:
        call = calls.setdefault(index, {"id": None, "name": "", "arguments": ""})
        call["id"] = get(delta, "id") or call["id"]
        function = get(delta, "function")
        call["name"] += get(function, "name") or ""
        call["arguments"] += get(function, "arguments") or ""

    def result(self) -> Body:
        """The assembled response body."""
        body = dict(self._head, object="chat.completion")
        body["choices"] = [
            self._message(index, self._choices[index]) for index in sorted(self._choices)
        ]
        if self._usage is not None:
            body["usage"] = to_plain(self._usage)
        return body

    @staticmethod
    def _message(index: int, slot: Body) -> Body:
        message: Body = {
            "role": slot["role"] or "assistant",
            "content": "".join(slot["content"]) if slot["content"] else None,
        }
        if slot["refusal"]:
            message["refusal"] = "".join(slot["refusal"])
        calls = [slot["calls"][key] for key in sorted(slot["calls"]) if key >= 0]
        if calls:
            message["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"]},
                }
                for call in calls
            ]
        if -1 in slot["calls"]:
            message["function_call"] = {
                "name": slot["calls"][-1]["name"],
                "arguments": slot["calls"][-1]["arguments"],
            }
        if "audio" in slot:
            audio = slot["audio"]
            message["audio"] = {
                "id": audio["id"],
                "transcript": "".join(audio["transcript"]),
                "data": {"type": "media_omitted", "bytes": audio["bytes"]},
            }
        return {"index": index, "message": message, "finish_reason": slot["finish"]}


class ResponsesAssembler:
    """Keep the terminal ``response.*`` event of a Responses stream: it carries the whole response."""

    def __init__(self) -> None:
        self._kind: Optional[str] = None
        self._response: Any = None

    @property
    def status(self) -> Optional[int]:
        """200, or ``None`` (no usable response) when the stream ended in ``response.failed``."""
        return None if self._kind == "response.failed" else 200

    def add(self, event: Any) -> None:
        """Take one event."""
        if get(event, "type") in _TERMINAL:
            self._kind, self._response = get(event, "type"), get(event, "response")

    def result(self) -> Body:
        """The final response body; an error when the stream ended without a terminal event."""
        if self._kind is None:
            raise ValueError("the stream ended without a terminal event")
        if self._kind == "response.failed":
            return {"error": {"type": self._kind}}
        return _response_body(self._response)


def _tee_hooks(
    recorder: Recorder, endpoint: str, kwargs: Body, started: float
) -> Tuple[Callable[[Any], None], Callable[[], None], Callable[[BaseException], None]]:
    assembler = ChatAssembler() if endpoint == "chat.completions" else ResponsesAssembler()

    def done() -> None:
        recorder.emit(
            PROVIDER,
            endpoint,
            _request_body(kwargs),
            assembler.result(),
            status=assembler.status,
            started=started,
        )

    def error(exc: BaseException) -> None:
        _record_error(recorder, endpoint, kwargs, exc, started)

    return assembler.add, done, error


def _after(
    recorder: Recorder, endpoint: str, kwargs: Body, result: Any, started: float, *, sync: bool
) -> Any:
    """What the wrapper returns: the result itself, or the result teed when it is a stream."""
    if not kwargs.get("stream"):
        recorder.fail_open(_record_ok, recorder, endpoint, kwargs, result, started)
        return result
    try:
        item, done, error = _tee_hooks(recorder, endpoint, kwargs, started)
        tee = TeeStream if sync else AsyncTeeStream
        return tee(result, recorder.fail_open, item, done, error)
    except Exception:  # a result that cannot be teed is handed back as it is
        recorder.writer.count("dropped_error")
        return result


def _wrap_sync(original: Callable[..., Any], endpoint: str) -> Callable[..., Any]:
    @functools.wraps(original)
    def create(self: Any, *args: Any, **kwargs: Any) -> Any:
        recorder = _ACTIVE.recorder
        if recorder is None:
            return original(self, *args, **kwargs)
        started = time.perf_counter()
        try:
            result = original(self, *args, **kwargs)
        except BaseException as exc:
            recorder.fail_open(_record_error, recorder, endpoint, kwargs, exc, started)
            raise
        return _after(recorder, endpoint, kwargs, result, started, sync=True)

    return create


def _wrap_async(original: Callable[..., Any], endpoint: str) -> Callable[..., Any]:
    @functools.wraps(original)
    async def create(self: Any, *args: Any, **kwargs: Any) -> Any:
        recorder = _ACTIVE.recorder
        if recorder is None:
            return await original(self, *args, **kwargs)
        started = time.perf_counter()
        try:
            result = await original(self, *args, **kwargs)
        except BaseException as exc:
            recorder.fail_open(_record_error, recorder, endpoint, kwargs, exc, started)
            raise
        return _after(recorder, endpoint, kwargs, result, started, sync=False)

    return create


def _targets() -> List[Tuple[Any, str, bool]]:
    """The SDK classes this version has: ``(class, endpoint, is_async)``."""
    found: List[Tuple[Any, str, bool]] = []
    # A lazy, runtime-optional import: ``import dagnam_record`` must not need the SDK.
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
    """Record through ``recorder``; True when at least one ``create`` is (now) patched.

    Safe to call again: a method is wrapped once, and the newest recorder is used. Returns False
    under the kill switch and when the SDK (or all its known resources) is not there.
    """
    if recording_disabled():
        return False
    patched = False
    for cls, endpoint, is_async in _targets():
        original = getattr(cls, "create", None)
        if original is None:
            continue
        patched = True
        if not getattr(original, "__dagnam_record__", False):
            wrapper: Any = (_wrap_async if is_async else _wrap_sync)(original, endpoint)
            wrapper.__dagnam_record__ = True
            cls.create = wrapper
            _PATCHED.append((cls, original))
    _ACTIVE.recorder = recorder if patched else None
    return patched


def uninstall() -> None:
    """Put the SDK's own methods back and stop recording (used by tests and ``dagnam_record.uninstall``)."""
    _ACTIVE.recorder = None
    while _PATCHED:
        cls, original = _PATCHED.pop()
        cls.create = original
