"""Trace envelope v1: build one JSON line for a recorded LLM call.

Headers and keys never appear because callers pass only request and response bodies. Media
parts are replaced by their size, so a line stays valid JSON and the call still reads as
multimodal. ``norm`` is the provider-neutral fallback a reader uses when raw is absent; its
rules (system joined with a newline, text parts joined with nothing, OpenAI-shaped tool calls
for every provider) must equal what the reader derives from raw.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

ENVELOPE_VERSION = 1
NORM_VERSION = 1
PROVIDERS = ("openai", "anthropic", "other")
ENDPOINTS = ("chat.completions", "responses", "messages", "custom")

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
    }
)
_SYSTEM_ROLES = frozenset({"system", "developer"})

Body = Dict[str, Any]
Norm = Dict[str, Any]


def _dict(value: object) -> Body:
    return value if isinstance(value, dict) else {}


def _list(value: object) -> List[Any]:
    return value if isinstance(value, list) else []


def _count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


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


def _omit_media(value: Any) -> Any:
    """A copy of ``value`` with each media part (a list item) replaced by its size."""
    if isinstance(value, dict):
        return {key: _omit_media(item) for key, item in value.items()}
    if isinstance(value, list):
        return [
            {"type": _OMITTED, "bytes": len(json.dumps(item, ensure_ascii=False).encode("utf-8"))}
            if _is_media(item)
            else _omit_media(item)
            for item in value
        ]
    return value


def _has_media(value: object) -> bool:
    if isinstance(value, dict):
        return any(_has_media(item) for item in value.values())
    if isinstance(value, list):
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
        "system": "\n".join(text for text in system if text) or None,
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
        content = _text(turn.get("content"))
        if role in _SYSTEM_ROLES:
            system.append(content)
        else:
            messages.append({"role": role, "content": content})
    return system, messages


def norm_openai_chat(request: Body, response: Body) -> Norm:
    """Chat Completions: ``messages`` in, ``choices[0].message`` out."""
    system, messages = _split(_list(request.get("messages")))
    choices = _list(response.get("choices"))
    message = _dict(_dict(choices[0] if choices else None).get("message"))
    calls = _calls(
        [
            _call(
                _dict(item.get("function")).get("name"),
                _dict(item.get("function")).get("arguments"),
            )
            for item in _list(message.get("tool_calls"))
            if isinstance(item, dict)
        ]
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
        if _dict(item).get("type") == "function_call_output":
            turns.append({"role": "tool", "content": _text(item.get("output"))})
        else:
            turns.append(item)
    extra, messages = _split(turns)
    output = [item for item in _list(response.get("output")) if isinstance(item, dict)]
    usage = _dict(response.get("usage"))
    return _norm(
        request,
        response,
        system=[_text(request.get("instructions")), *extra],
        messages=messages,
        reply="".join(
            _text(item.get("content")) for item in output if item.get("type") == "message"
        ),
        tool_calls=_calls(
            [
                _call(item.get("name"), item.get("arguments"))
                for item in output
                if item.get("type") == "function_call"
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
        system=[_text(request.get("system"))],
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
            _count(usage.get("input_tokens")) + read + created,
            _count(usage.get("output_tokens")),
            read,
        ),
    )


_NORMS: Dict[Tuple[str, str], Callable[[Body, Body], Norm]] = {
    ("openai", "chat.completions"): norm_openai_chat,
    ("openai", "responses"): norm_openai_responses,
    ("anthropic", "messages"): norm_anthropic,
}


def _timestamp(now: float) -> str:
    stamp = datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="milliseconds")
    return stamp.replace("+00:00", "Z")


def build_line(
    *,
    provider: str,
    endpoint: str,
    request: Body,
    response: Body,
    latency_ms: float,
    status: int = 200,
    trace_id: Optional[str] = None,
    no_raw: bool = False,
    now: Optional[float] = None,
) -> str:
    """One envelope v1 JSON line (no newline). Raises ``ValueError`` when it cannot be built.

    ``norm`` is added for the provider shapes this module knows. ``no_raw`` drops the bodies
    and so needs one of them.
    """
    if provider not in PROVIDERS or endpoint not in ENDPOINTS:
        raise ValueError(f"unknown provider or endpoint: {provider!r}, {endpoint!r}")
    norm = _NORMS.get((provider, endpoint))
    if no_raw and norm is None:
        raise ValueError("no_raw needs a provider and endpoint with a known norm")
    request = _omit_media(request)
    body: Body = {
        "v": ENVELOPE_VERSION,
        "trace_id": trace_id or uuid.uuid4().hex,
        "ts": _timestamp(time.time() if now is None else now),
        "provider": provider,
        "endpoint": endpoint,
        "latency_ms": latency_ms,
        "status": status,
    }
    if not no_raw:
        body["request"], body["response"] = request, response
    if norm is not None:
        body["norm"], body["norm_v"] = norm(request, response), NORM_VERSION
    line = json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    line.encode("utf-8")  # a lone surrogate raises here, not in the writer thread
    return line
