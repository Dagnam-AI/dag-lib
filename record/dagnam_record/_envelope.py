"""Trace envelope v1: build one JSON line for a recorded LLM call.

Headers and keys never appear because callers pass only request and response bodies. Media
parts are replaced by their size, so a line stays valid JSON and the call still reads as
multimodal. ``norm`` is the provider-neutral fallback a reader uses when raw is absent; its
rules must equal what the reader derives from raw: system segments (empty ones too) joined with a
newline, text parts joined with nothing, OpenAI-shaped tool calls for every provider, and an
assistant turn of the prompt that made tool calls is the JSON of those calls (``_render_calls``),
not its text. Three things norm cannot say, pinned as EXPECTED_DIFFERENCES in the audit tests:
that a reply was only reasoning, that a ``<think>`` block opened a later text part (the reader
cleans a reply part by part, and norm holds the parts joined), and whether a reply that is JSON of
a message came from one plain string (the reader reads that as its content) or from a list part.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

ENVELOPE_VERSION = 1
NORM_VERSION = 1
PROVIDERS = ("openai", "anthropic", "other")
ENDPOINTS = ("chat.completions", "responses", "messages", "custom")
MAX_LINE_BYTES = 1_000_000
JS_SAFE = 2**53 - 1
"""The largest integer a JavaScript reader holds exactly: latency and token counts stop here."""
NO_RESPONSE_STATUS = 599
"""The status of a call that got no HTTP response at all (a connection error or a timeout).

A line must carry an HTTP status from 100 to 599, so ``build_line(status=None)`` writes this
one: 599 is the top of that range and no ordinary response uses it. A reader treats it as "no
response", and skips it like any other status that is not 200.
"""
_MAX_INT_BITS = 4096

_OMITTED = "media_omitted"
_MEDIA_TYPES = frozenset(
    {
        "image_url",
        "input_image",
        "image",
        "input_audio",
        "audio",
        "file",
        "input_file",
        "document",
        "video",
    }
)
_SYSTEM_ROLES = frozenset({"system", "developer"})
_CALL_ITEMS = frozenset({"function_call", "custom_tool_call"})
_OUTPUT_ITEMS = frozenset({"function_call_output", "custom_tool_call_output"})
_NAME_PATHS = ("function.name", "custom.name", "name")
_ARGUMENT_PATHS = ("function.arguments", "custom.input", "args", "arguments", "input")
_RESULT_PATHS = ("content", "output", "result", "response")

Body = Dict[str, Any]
Norm = Dict[str, Any]


class LineTooLargeError(ValueError):
    """The finished line is longer than the envelope contract allows."""


def _dict(value: object) -> Body:
    return value if isinstance(value, dict) else {}


def _list(value: object) -> List[Any]:
    return value if isinstance(value, list) else []


def _count(value: object) -> int:
    """A token count: a non-negative integer, at most ``JS_SAFE``; anything else is 0."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return min(value, JS_SAFE)
    return 0


def _text(content: object) -> str:
    """A string content as is; a list of parts as its text parts (and tool results) joined."""
    if isinstance(content, str):
        return content
    texts: List[str] = []
    for part in _list(content):
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            texts.append(part["text"])
        elif isinstance(part, dict) and part.get("type") == "tool_result":
            texts.append(_text(part.get("content")))
    return "".join(texts)


def _is_media(part: object) -> bool:
    return isinstance(part, dict) and part.get("type") in _MEDIA_TYPES


def _scrub(value: Any, omit: bool) -> Any:
    """Copy ``value`` (tuples become lists), refusing absurd integers.

    When ``omit`` is set each media part (a list item) is replaced by its size.
    """
    if isinstance(value, dict):
        return {key: _scrub(item, omit) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [
            {"type": _OMITTED, "bytes": len(json.dumps(item, ensure_ascii=False).encode("utf-8"))}
            if omit and _is_media(item)
            else _scrub(item, omit)
            for item in value
        ]
    if isinstance(value, int) and value.bit_length() > _MAX_INT_BITS:
        raise ValueError("an integer in the body is too large for a JSON reader")
    return value


def omit_media(value: Any) -> Any:
    """A copy of ``value`` with each media part (a list item) replaced by its size."""
    return _scrub(value, omit=True)


def _has_media(value: object) -> bool:
    if isinstance(value, dict):
        return any(_has_media(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(
            _is_media(item)
            or (isinstance(item, dict) and item.get("type") == _OMITTED)
            or _has_media(item)
            for item in value
        )
    return False


def _call(name: object, arguments: object) -> Optional[Body]:
    """An OpenAI-shaped tool call, or ``None`` when the call has no name."""
    if not isinstance(name, str):
        return None
    if arguments is None:
        arguments = ""
    elif not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


def _calls(candidates: List[Optional[Body]]) -> List[Body]:
    return [call for call in candidates if call is not None]


def _first(value: object, *paths: str) -> object:
    """The first value that is not ``None`` at any dotted path (a missing step is ``None``)."""
    for path in paths:
        found: object = value
        for key in path.split("."):
            found = found.get(key) if isinstance(found, dict) else None
        if found is not None:
            return found
    return None


def _render_calls(calls: List[Body]) -> str:
    """Tool calls in a prompt as one text, the way the audit's reader reads an assistant turn.

    One call is ``{"arguments": ..., "name": ...}`` with sorted keys; several are the list of
    them. Arguments that are JSON text of an object become that object; any other value, text
    that is not an object included, is kept as it is, and none at all is ``{}``.
    """
    objects: List[Body] = []
    for call in calls:
        arguments = _first(call, *_ARGUMENT_PATHS)
        parsed = arguments
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except (ValueError, RecursionError):
                pass
        if isinstance(parsed, dict):
            arguments = parsed
        objects.append(
            {
                "arguments": {} if arguments is None else arguments,
                "name": _first(call, *_NAME_PATHS),
            }
        )
    return json.dumps(objects[0] if len(objects) == 1 else objects, sort_keys=True)


def _given(value: object) -> List[str]:
    """A system text the request gave, as the one segment it is; none given is no segment."""
    return [] if value is None else [_text(value)]


def _assistant_text(turn: Body) -> str:
    """What an assistant turn of a prompt says: the calls it made, else its text."""
    calls = [item for item in _list(turn.get("tool_calls")) if isinstance(item, dict)]
    if isinstance(turn.get("function_call"), dict):
        calls.append(turn["function_call"])
    calls += [
        part
        for part in _list(turn.get("content"))
        if isinstance(part, dict) and part.get("type") == "tool_use"
    ]
    return _render_calls(calls) if calls else _text(turn.get("content"))


def _model(request: Body, response: Body) -> str:
    for source in (response, request):
        model = source.get("model")
        if isinstance(model, str) and model:
            return model
    return "unknown"


def _norm(
    request: Body,
    response: Body,
    *,
    system: List[str],
    messages: List[Body],
    reply: str,
    tool_calls: List[Body],
    tokens: Tuple[int, int, int],
) -> Norm:
    return {
        "model": _model(request, response),
        "system": "\n".join(system) or None,
        "messages": messages,
        "response": reply,
        "tool_calls": tool_calls,
        "prompt_tokens": tokens[0],
        "completion_tokens": tokens[1],
        "cached_prompt_tokens": tokens[2],
        "has_media": _has_media(request),
    }


def _split(turns: List[Any]) -> Tuple[List[str], List[Body]]:
    """Split role-carrying chat turns into system texts and the other ``{role, content}`` turns."""
    system: List[str] = []
    messages: List[Body] = []
    for turn in turns:
        role = _dict(turn).get("role")
        if not isinstance(role, str):
            continue
        if role in _SYSTEM_ROLES:
            system.append(_text(turn.get("content")))
        else:
            content = _assistant_text(turn) if role == "assistant" else _text(turn.get("content"))
            messages.append({"role": role, "content": content})
    return system, messages


def norm_openai_chat(request: Body, response: Body) -> Norm:
    """Chat Completions: ``messages`` in, ``choices[0].message`` out."""
    system, messages = _split(_list(request.get("messages")))
    choices = _list(response.get("choices"))
    message = _dict(_dict(choices[0] if choices else None).get("message"))
    declared = [item for item in _list(message.get("tool_calls")) if isinstance(item, dict)]
    if isinstance(message.get("function_call"), dict):  # the call API before ``tool_calls``
        declared.append(message["function_call"])
    calls = _calls(
        [_call(_first(item, *_NAME_PATHS), _first(item, *_ARGUMENT_PATHS)) for item in declared]
    )
    usage = _dict(response.get("usage"))
    cached = _dict(usage.get("prompt_tokens_details")).get("cached_tokens")
    return _norm(
        request,
        response,
        system=system,
        messages=messages,
        reply=_text(message.get("content")),
        tool_calls=calls,
        tokens=(
            _count(usage.get("prompt_tokens")),
            _count(usage.get("completion_tokens")),
            _count(cached),
        ),
    )


def norm_openai_responses(request: Body, response: Body) -> Norm:
    """Responses API: ``instructions`` and ``input`` in, ``output`` items out."""
    items = request.get("input")
    turns: List[Any] = [{"role": "user", "content": items}] if isinstance(items, str) else []
    for item in _list(items):
        kind = _dict(item).get("type")
        if kind in _OUTPUT_ITEMS:
            turns.append({"role": "tool", "content": _text(_first(item, *_RESULT_PATHS))})
        elif kind in _CALL_ITEMS and "role" not in item:  # a call the assistant made earlier
            turns.append({"role": "assistant", "tool_calls": [item]})
        else:
            turns.append(item)
    extra, messages = _split(turns)
    output = [item for item in _list(response.get("output")) if isinstance(item, dict)]
    usage = _dict(response.get("usage"))
    return _norm(
        request,
        response,
        system=[*_given(request.get("instructions")), *extra],
        messages=messages,
        reply="".join(
            _text(item.get("content")) for item in output if item.get("type") == "message"
        ),
        tool_calls=_calls(
            [
                _call(item.get("name"), _first(item, "arguments", "input"))
                for item in output
                if item.get("type") in _CALL_ITEMS
            ]
        ),
        tokens=(
            _count(usage.get("input_tokens")),
            _count(usage.get("output_tokens")),
            _count(_dict(usage.get("input_tokens_details")).get("cached_tokens")),
        ),
    )


def norm_anthropic(request: Body, response: Body) -> Norm:
    """Messages API: top-level ``system`` and ``messages`` in, content blocks out."""
    _, messages = _split(_list(request.get("messages")))
    blocks = [block for block in _list(response.get("content")) if isinstance(block, dict)]
    usage = _dict(response.get("usage"))
    read = _count(usage.get("cache_read_input_tokens"))
    created = _count(usage.get("cache_creation_input_tokens"))
    return _norm(
        request,
        response,
        system=_given(request.get("system")),
        messages=messages,
        reply="".join(_text([block]) for block in blocks if block.get("type") == "text"),
        tool_calls=_calls(
            [
                _call(block.get("name"), block.get("input", {}))
                for block in blocks
                if block.get("type") == "tool_use"
            ]
        ),
        tokens=(
            min(_count(usage.get("input_tokens")) + read + created, JS_SAFE),
            _count(usage.get("output_tokens")),
            read,
        ),
    )


_NORMS: Dict[Tuple[str, str], Callable[[Body, Body], Norm]] = {
    ("openai", "chat.completions"): norm_openai_chat,
    ("openai", "responses"): norm_openai_responses,
    ("anthropic", "messages"): norm_anthropic,
}


def _timestamp(now: object) -> str:
    if now is None:
        now = time.time()
    if isinstance(now, bool) or not isinstance(now, (int, float)):
        raise ValueError("now must be a number of seconds")
    stamp = datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="milliseconds")
    return stamp.replace("+00:00", "Z")


def _latency(value: object) -> float:
    """Milliseconds as a finite float in ``[0, JS_SAFE]``; a negative value is clamped to 0.0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("latency_ms must be a number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("latency_ms must be finite")
    return float(min(value, JS_SAFE)) if value > 0 else 0.0


def _status(value: object) -> int:
    """An HTTP status; ``None`` (no response at all) is recorded as ``NO_RESPONSE_STATUS``."""
    if value is None:
        return NO_RESPONSE_STATUS
    if isinstance(value, bool) or not isinstance(value, int) or not 100 <= value <= 599:
        raise ValueError("status must be an HTTP status from 100 to 599, or None")
    return value


def _build(
    provider: Any,
    endpoint: Any,
    request: Any,
    response: Any,
    latency_ms: Any,
    status: Any,
    trace_id: Any,
    no_raw: bool,
    now: Any,
) -> str:
    if provider not in PROVIDERS or endpoint not in ENDPOINTS:
        raise ValueError(f"unknown provider or endpoint: {provider!r}, {endpoint!r}")
    if not isinstance(request, dict) or not isinstance(response, dict):
        raise ValueError("request and response must be objects")
    norm = _NORMS.get((provider, endpoint))
    if no_raw and norm is None:
        raise ValueError("no_raw needs a provider and endpoint with a known norm")
    trace = uuid.uuid4().hex if trace_id is None else trace_id
    if not isinstance(trace, str) or not trace:
        raise ValueError("trace_id must be a non-empty string")
    body: Body = {
        "v": ENVELOPE_VERSION,
        "trace_id": trace,
        "ts": _timestamp(now),
        "provider": provider,
        "endpoint": endpoint,
        "latency_ms": _latency(latency_ms),
        "status": _status(status),
    }
    request, response = _scrub(request, omit=True), _scrub(response, omit=False)
    if not no_raw:
        body["request"], body["response"] = request, response
    if norm is not None:
        body["norm"], body["norm_v"] = norm(request, response), NORM_VERSION
    line = json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if len(line.encode("utf-8")) > MAX_LINE_BYTES:  # a lone surrogate also raises here
        raise LineTooLargeError(f"line exceeds {MAX_LINE_BYTES} bytes")
    return line


def build_line(
    *,
    provider: str,
    endpoint: str,
    request: Body,
    response: Body,
    latency_ms: float,
    status: Optional[int] = 200,
    trace_id: Optional[str] = None,
    no_raw: bool = False,
    now: Optional[float] = None,
) -> str:
    """One envelope v1 JSON line (no newline) that the contract's validator accepts.

    ``norm`` is added for the provider shapes this module knows; ``no_raw`` drops the bodies and
    so needs one of them. Media parts of the request are replaced by their size. ``latency_ms``
    is clamped to ``[0, JS_SAFE]``, token counts are capped at ``JS_SAFE``, and ``status=None``
    (no HTTP response) is recorded as ``NO_RESPONSE_STATUS``.

    Raises ``ValueError`` and nothing else, whatever the input: bad provider, endpoint, status,
    latency, trace id or time; a body that is not a dict or holds something JSON cannot carry (a
    datetime, bytes, an SDK object, a cycle, NaN, a lone surrogate, a huge integer); and
    ``LineTooLargeError`` (a subclass) over 1 MB. Nothing is converted on the caller's behalf: a body
    must already be plain JSON data. A call site must still catch ``Exception``, to be fail-open
    against its own bugs.
    """
    try:
        return _build(
            provider, endpoint, request, response, latency_ms, status, trace_id, no_raw, now
        )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"cannot build a line: {type(exc).__name__}: {exc}") from exc
