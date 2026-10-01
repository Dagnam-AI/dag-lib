"""Chat content in every provider's shape -> the text, turns and tool calls a record keeps.

The readers meet one conversation in many spellings: OpenAI Chat messages,
Responses API items (``instructions`` + ``input``, ``function_call`` and
``function_call_output`` items, an ``output`` list), Anthropic content blocks
(``text``, ``tool_use``, ``tool_result``, ``thinking``), Gemini parts
(``functionCall``, ``functionResponse``, ``thought``), LangChain's messages
once the reader has flattened them, and OTel GenAI text parts. Every reader
goes through this module, so a shape learned here is read from every source.

Two rules hold for all of them:
- Reasoning is never an answer: ``thinking`` / ``redacted_thinking`` /
  ``reasoning`` blocks, Gemini ``thought: true`` parts and a leading inline
  ``<think>...</think>`` are dropped from replies (their tokens stay priced).
- A tool call is judged and trained as ``{"arguments": ..., "name": ...}``,
  and a reply with several calls as the ordered list of them (P4, N1, N3).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import json
import re
from typing import Any

from dagnam.audit.readers.base import MalformedRowError, get, text
from dagnam.audit.record import Message

_SYSTEM_ROLES = frozenset({"system", "developer"})
_ASSISTANT_ROLES = frozenset({"assistant", "model"})
_REASONING_TYPES = frozenset({"thinking", "redacted_thinking", "reasoning"})
_TOOL_RESULT_TYPES = frozenset({"tool_result", "tool-result", "tool_call_response"})
_CALL_TYPES = frozenset({"tool_use", "function_call", "tool-call", "tool_call"})
_REPLY_KEYS = ("content", "parts", "tool_calls")
_MEDIA_TYPES = frozenset(
    {
        "image",
        "image_url",
        "input_image",
        "audio",
        "input_audio",
        "file",
        "input_file",
        "document",
        "video",
    }
)
_MEDIA_KEYS = ("inlineData", "inline_data", "fileData", "file_data")
"""Gemini carries media as a part holding one of these, with no ``type``."""
_LEADING_THINK = re.compile(r"\A\s*<think>.*?</think>\s*", re.DOTALL)
TOOL_CALL_NOTE = (
    "This workload answers with tool calls: the replacement returns the call as"
    ' {"name", "arguments"} JSON (an ordered list of them when the teacher made several)'
    " in message.content, not in tool_calls."
)
"""What a tool-calling workload's replacement answers with: every surface renders this sentence
verbatim (the scan's warning, the report's switch, the CLI, and the website's Markdown)."""


def _function_call(part: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The tool call a content part or Responses item makes, if it makes one."""
    if part.get("type") in _CALL_TYPES:
        return part
    call = get(part, "functionCall", "function_call")
    return call if isinstance(call, Mapping) else None


def _part_text(part: object) -> str:
    """The text one content part contributes: never reasoning, and a tool result's output."""
    if not isinstance(part, Mapping):
        return text(part)
    kind = part.get("type")
    if part.get("thought") is True or kind in _REASONING_TYPES:
        return ""
    if kind in _TOOL_RESULT_TYPES or kind == "function_call_output":
        return content_text(get(part, "content", "output", "result", "response"))
    if kind == "message":  # a Responses output item wrapping its own parts
        return content_text(part.get("content"))
    response = get(part, "functionResponse.response", "function_response.response")
    if response is not None:
        return text(response)
    return text(part.get("text") if "text" in part else part.get("content") if kind else None)


def content_text(content: object) -> str:
    """Flatten a chat ``content`` value (a string or a list of typed parts) to plain text."""
    if isinstance(content, list):
        return "".join(_part_text(part) for part in content)
    return text(content)


def reply_text(content: object) -> str:
    """:func:`content_text` of a reply, without the reasoning a model may inline before it."""
    return _LEADING_THINK.sub("", content_text(content))


def tool_calls(value: object) -> tuple[dict[str, Any], ...]:
    """A response's tool calls as a tuple of JSON objects (empty when absent)."""
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(call, dict) for call in value):
        raise MalformedRowError(f"tool_calls is not a list of objects: {value!r}")
    return tuple(value)


def _block_calls(content: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(content, list):
        return ()
    calls = (_function_call(part) for part in content if isinstance(part, Mapping))
    return tuple(dict(call) for call in calls if call is not None)


def is_reply(value: Mapping[str, Any]) -> bool:
    """Whether a mapping is a reply :func:`reply_of` unwraps, not an answer that is JSON itself.

    A reply is a tool-call item; a Responses object, whose ``output`` holds
    typed items; or a message -- a ``role`` or ``type`` (a LangChain
    ``AIMessage`` dump has only ``type: "ai"``) beside the ``content``,
    ``parts`` or ``tool_calls`` that carry it. An answer such as
    ``{"type": "refund", "amount": 12}`` or ``{"name": "Bob", "role": "admin"}``
    carries none of them: those keys are the model's own fields (B2-1, RR-3).
    """
    output = value.get("output")
    if value.get("type") in _CALL_TYPES or (
        isinstance(output, list)
        and bool(output)
        and all(isinstance(i, Mapping) and "type" in i for i in output)
    ):
        return True
    return ("role" in value or "type" in value) and any(key in value for key in _REPLY_KEYS)


def reply_of(value: object) -> tuple[str, tuple[dict[str, Any], ...]]:
    """``(text, tool calls)`` of an assistant reply in any provider's shape.

    ``value`` may be a string, a message object (``content`` beside
    ``tool_calls``, or content blocks holding ``tool_use`` / ``functionCall``),
    a Responses object (its ``output`` item list), one Responses item, or a
    list of typed blocks or items.
    """
    if isinstance(value, Mapping):
        if isinstance(value.get("output"), list):  # a Responses object
            value = value["output"]
        elif value.get("type") in _CALL_TYPES:  # one Responses output item
            value = [value]
        else:
            content = value.get("content", value.get("parts"))  # OTel GenAI: ``parts``
            calls = (*tool_calls(value.get("tool_calls")), *_block_calls(content))
            return reply_text(content), calls
    return reply_text(value), _block_calls(value)


def _call_object(call: Mapping[str, Any]) -> dict[str, Any]:
    """One tool call as ``{"arguments": ..., "name": ...}``, whatever the export's spelling.

    OpenAI ``function.name`` / ``function.arguments`` (a JSON string),
    LangChain ``name`` / ``args``, a bare ``arguments``, an Anthropic
    ``tool_use`` block's ``input``, a Gemini ``functionCall``'s ``args``.
    Arguments that are not a JSON object stay what they are.
    """
    arguments = get(call, "function.arguments", "args", "arguments", "input")
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except ValueError:
            parsed = None
        arguments = parsed if isinstance(parsed, dict) else arguments
    return {
        "arguments": {} if arguments is None else arguments,
        "name": get(call, "function.name", "name"),
    }


def effective_response(response: str, calls: Sequence[Mapping[str, Any]]) -> str:
    """The text a reply is judged and trained on: its tool calls, else its text.

    One call is the sorted-keys object ``{"arguments": ..., "name": ...}``;
    several are the ordered list of them (N3), so a reply that tags a ticket
    with a product *and* a team is judged on both. The name is part of it
    because a router or handoff step decides by *which* tool it calls, usually
    with ``{}`` arguments: judged on the arguments alone it has one output, and
    a student that always answers ``{}`` would score as agreeing (P4).
    """
    if not calls:
        return response
    objects = [_call_object(call) for call in calls]
    return json.dumps(objects[0] if len(objects) == 1 else objects, sort_keys=True)


def _turns(items: list[Any]) -> Iterator[tuple[str, str]]:
    """``(role, content)`` for each chat message or Responses input item, in order.

    An assistant turn keeps its text and its tool calls, rendered the way the
    student will answer them (:func:`effective_response`), so a later step's
    tool result answers a call the student can see (N5). A Responses
    ``function_call`` item is such a turn; its ``function_call_output`` is a
    ``tool`` turn; a ``reasoning`` item is dropped.
    """
    for item in items:
        if not isinstance(item, Mapping):
            raise MalformedRowError(f"not a chat message: {item!r}")
        kind = item.get("type")
        if kind in _REASONING_TYPES:
            continue
        if kind == "function_call_output":
            yield "tool", _part_text(item)
        elif _function_call(item) is not None and "role" not in item:
            yield "assistant", effective_response("", (item,))
        elif "role" not in item:
            raise MalformedRowError(f"not a chat message: {item!r}")
        else:
            role = text(item["role"])
            content = item.get("content", item.get("parts"))
            if role in _ASSISTANT_ROLES:
                reply, calls = reply_of({"content": content, "tool_calls": item.get("tool_calls")})
                yield "assistant", effective_response(reply, calls)
            else:
                yield role, content_text(content)


def split_prompt(prompt: object) -> tuple[str | None, tuple[Message, ...]]:
    """Split a chat prompt into ``(system, turns)``.

    ``prompt`` may be a list of ``{role, content}`` messages or Responses
    items; a mapping holding such a list under ``messages`` (or Responses
    ``input``) with the Anthropic ``system`` or the Responses ``instructions``
    beside it; or a bare string (one user turn). System turns are joined into
    ``system``; assistant turns (Gemini's ``model``) are kept as context, with
    the tool calls they made.
    """
    system: list[str] = []
    if isinstance(prompt, Mapping):
        for key in ("system", "instructions"):
            if prompt.get(key) is not None:
                system.append(content_text(prompt[key]))
        prompt = prompt.get("messages", prompt.get("input", prompt))
    if not isinstance(prompt, list):
        return ("\n".join(system) or None), (Message("user", text(prompt)),)
    turns: list[Message] = []
    for role, content in _turns(prompt):
        if role in _SYSTEM_ROLES:
            system.append(content)
        else:
            turns.append(Message("assistant" if role in _ASSISTANT_ROLES else role, content))
    if all(turn.role == "assistant" for turn in turns):
        raise MalformedRowError("prompt has no user turn")
    return ("\n".join(system) or None), tuple(turns)


def has_media(value: object) -> bool:
    """Whether a prompt carries an image, audio, file or document part anywhere in it.

    Such a part adds nothing to the text a student is trained on, so a
    workload whose answers depend on it cannot be learned from the text (N9).
    """
    if isinstance(value, list):
        return any(has_media(item) for item in value)
    if not isinstance(value, Mapping):
        return False
    if value.get("type") in _MEDIA_TYPES or any(key in value for key in _MEDIA_KEYS):
        return True
    return any(has_media(item) for item in value.values() if isinstance(item, (list, dict)))


def task_signature(request: object) -> str | None:
    """What a request asks for beyond its prompt: the name of its structured-output schema.

    Two extraction tasks under one system prompt differ only in the schema
    the request carries, so it joins the workload key (N12). A FORCED tool is
    such a schema (OpenAI ``tool_choice.function.name``, Anthropic's and the
    Responses API's ``tool_choice.name``, the legacy ``function_call.name``;
    LangChain's ``with_structured_output`` forces one), so it joins too (RR-2).
    The tools a request OFFERS do not: an agent's tool set changes per call (a
    deploy adds one, permissions or retrieval pick them), and keying on it split
    one 2,400-call router into four workloads too small to audit (B2-2).
    ``None`` when the request names no schema.
    """
    if not isinstance(request, Mapping):
        return None
    schema = get(
        request,
        "response_format.json_schema.name",
        "response_format.name",
        "text.format.name",
        "tool_choice.function.name",
        "tool_choice.name",
        "function_call.name",
    )
    return None if schema is None else f"schema={text(schema)}"


__all__ = [
    "TOOL_CALL_NOTE",
    "content_text",
    "effective_response",
    "has_media",
    "is_reply",
    "reply_of",
    "reply_text",
    "split_prompt",
    "task_signature",
    "tool_calls",
]
