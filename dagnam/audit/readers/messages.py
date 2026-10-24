"""Chat content in every provider's shape -> the text, turns and tool calls a record keeps.

The readers meet one conversation in many spellings: OpenAI Chat messages,
Responses API items (``instructions`` + ``input``, ``function_call`` and
``function_call_output`` items, an ``output`` list), Anthropic content blocks
(``text``, ``tool_use``, ``tool_result``, ``thinking``), Gemini parts
(``functionCall``, ``functionResponse``, ``thought``), the Chat Completions
legacy ``function_call`` and ``custom`` tool calls, LangChain's messages
once the reader has flattened them, and OTel GenAI text parts. Every reader
goes through this module, so a shape learned here is read from every source.

Three rules hold for all of them:
- Reasoning never reaches a prompt or a reply, in any spelling
  (:mod:`~dagnam.audit.readers.reasoning` holds the one table). A reply made of
  nothing else, or a response object no reader recognises that holds some, is a
  ``reasoning_only`` call (:class:`Reply`): it is counted and priced like any
  call, with no answer and so no training row.
- A tool call is judged and trained as ``{"arguments": ..., "name": ...}``,
  and a reply with several calls as the ordered list of them.
- A Responses ``input`` list keeps only the items that carry conversation
  (messages, tool calls and their outputs); any other typed item (a hosted
  tool's record, a compaction, one a vendor adds next year) is skipped, never
  a reason to reject the row.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import json
from typing import Any, NamedTuple, TypeGuard, cast

from dagnam.audit.readers.base import MalformedRowError, get, text
from dagnam.audit.readers.calls import (
    CALL_OUTPUT_TYPES,
    CALL_TYPES,
    block_calls,
    effective_response,
    function_call,
    merge_calls,
    tool_calls,
)
from dagnam.audit.readers.reasoning import carries_reasoning, is_reasoning, strip_inline_reasoning
from dagnam.audit.record import Message

_SYSTEM_ROLES = frozenset({"system", "developer"})
_ASSISTANT_ROLES = frozenset({"assistant", "model"})
_TOOL_RESULT_TYPES = frozenset({"tool_result", "tool-result", "tool_call_response"})
_CONVERSATION_ITEM_TYPES = frozenset({"message"}) | CALL_TYPES | CALL_OUTPUT_TYPES
"""The only typed items of a Responses ``input`` list that are conversation; the rest are skipped."""
_TEXT_PART_TYPES = ("text", "output_text", "input_text")
_DECODES = 3
"""How many layers of JSON text a reply may be wrapped in."""
_JSON_OPENERS = ("[", "{", '"')
_MESSAGE_TYPES = frozenset({"message", "ai", "AIMessage", "AIMessageChunk"})
_REPLY_KEYS = ("content", "parts", "tool_calls", "function_call")
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
TOOL_CALL_NOTE = (
    "This workload answers with tool calls: the replacement returns the call as"
    ' {"name", "arguments"} JSON (an ordered list of them when the teacher made several)'
    " in message.content, not in tool_calls."
)
"""What a tool-calling workload's replacement answers with: every surface renders this sentence
verbatim (the scan's warning, the report's switch, the CLI, and the website's Markdown)."""


def _part_text(part: object) -> str:
    """The text one content part contributes: never reasoning, and a tool result's output."""
    if not isinstance(part, Mapping):
        return text(part)
    kind = part.get("type")
    if is_reasoning(part):
        return ""
    if kind in _TOOL_RESULT_TYPES or kind in CALL_OUTPUT_TYPES:
        return content_text(get(part, "content", "output", "result", "response"))
    if kind == "message":  # a Responses output item wrapping its own parts
        return content_text(part.get("content"))
    response = get(part, "functionResponse.response", "function_response.response")
    if response is not None:
        return text(response)
    return text(part.get("text") if "text" in part else part.get("content") if kind else None)


class Reply(NamedTuple):
    """What a record's response field held: its text, its tool calls, and whether that is all of it."""

    text: str
    calls: tuple[dict[str, Any], ...]
    reasoning_only: bool = False
    """Nothing but the model's reasoning (or an object no reader recognises that holds some):
    the call stays counted and priced, and gives no training row."""


def content_text(content: object) -> str:
    """Flatten a chat ``content`` value (a string or a list of typed parts) to plain text.

    A string that is JSON text of a reply (typed blocks with reasoning among them)
    is read as that reply, so a block never goes out as its own JSON.
    """
    if isinstance(content, str) and (decoded := _reply_json(content)) is not content:
        return reply_of(decoded)[0] if isinstance(decoded, Mapping) else content_text(decoded)
    if isinstance(content, list):
        return "".join(_part_text(part) for part in content)
    return text(content)


def reply_text(content: object) -> str:
    """:func:`content_text` of a reply, without the reasoning a model may write before it.

    Each text part is cleaned on its own (:func:`strip_inline_reasoning`), so a
    block in the second of two parts goes too.
    """
    if isinstance(content, str) and (decoded := _reply_json(content)) is not content:
        return reply_of(decoded)[0] if isinstance(decoded, Mapping) else reply_text(decoded)
    if isinstance(content, list):
        return "".join(strip_inline_reasoning(_part_text(part)) for part in content)
    return strip_inline_reasoning(text(content))


def _is_message(value: object) -> TypeGuard[Mapping[str, Any]]:
    """A mapping that is a chat message or Gemini content: its parts or content, and a role."""
    return (
        isinstance(value, Mapping)
        and any(key in value for key in _REPLY_KEYS)
        and ("role" in value or "parts" in value)
    )


def _inner_reply(value: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The message a response object wraps, when it is an envelope whose answer field is known.

    Chat Completions (``choices[].message``), Ollama and Cohere (``message``),
    Gemini (``candidates[].content``), Bedrock Converse (``output.message``),
    Ollama generate (``response`` beside ``done``). Everything beside the
    message (usage, ``thinking``, ids) is never read.
    """
    choices = value.get("choices")
    candidates = value.get("candidates")
    output = value.get("output")
    if (
        isinstance(choices, list)
        and choices
        and all(isinstance(c, Mapping) and isinstance(c.get("message"), Mapping) for c in choices)
    ):
        return choices[0]["message"]
    if _is_message(value.get("message")):
        return value["message"]
    if isinstance(candidates, list) and candidates:
        content = candidates[0].get("content") if isinstance(candidates[0], Mapping) else None
        if _is_message(content):
            return content
    if isinstance(output, Mapping) and _is_message(output.get("message")):
        return output["message"]
    if isinstance(value.get("response"), str) and isinstance(value.get("done"), bool):
        return {"role": "assistant", "content": value["response"]}
    return None


def is_reply(value: Mapping[str, Any]) -> bool:
    """Whether a mapping is a reply :func:`reply_of` unwraps, not an answer that is JSON itself.

    A reply is a tool-call item; a reasoning item or block; an envelope whose
    answer field is known (:func:`_inner_reply`); a Responses object, whose
    ``output`` holds a message, a tool call or reasoning (:func:`has_reply_parts`:
    an ``output`` list of ``{type, content}`` nodes is an answer); or a message -- a ``role`` or ``type`` (a
    LangChain ``AIMessage`` dump has only ``type: "ai"``) beside the
    ``content``, ``parts`` or ``tool_calls`` that carry it. An answer such as
    ``{"type": "refund", "amount": 12}`` or ``{"name": "Bob", "role": "admin"}``
    carries none of them: those keys are the model's own fields.
    """
    output = value.get("output")
    if (
        value.get("type") in CALL_TYPES
        or is_reasoning(value)
        or _inner_reply(value) is not None
        or (isinstance(output, list) and has_reply_parts(output))
    ):
        return True
    return ("role" in value or "type" in value) and any(key in value for key in _REPLY_KEYS)


def _is_reply_part(item: Mapping[str, Any]) -> bool:
    """Whether a list item is one of a reply's own parts: reasoning, a tool call, or a message."""
    message = any(key in item for key in _REPLY_KEYS) and (
        "role" in item or item.get("type") in _MESSAGE_TYPES
    )
    return is_reasoning(item) or item.get("type") in CALL_TYPES or message


def has_reply_parts(items: Sequence[object]) -> bool:
    """Whether a bare list holds a reply's own parts, not an answer that is a JSON array.

    It does when any item is reasoning, a tool call or a message, whatever
    else the list holds (a hosted tool's ``web_search_call``, an item type no
    reader knows): :func:`reply_of` then reads the answer out of it, so
    reasoning is never serialized into the answer. An item that merely has a
    ``type`` and a ``content`` is not one: ``[{"type": "paragraph", "content":
    "Hello"}]`` is an answer, and keeps every byte. An item typed as reasoning
    counts whatever it carries, so an answer array with such an object of its
    own is read as parts too, the safe side.
    """
    return any(isinstance(item, Mapping) and _is_reply_part(item) for item in items)


def is_text_parts(items: Sequence[object]) -> bool:
    """Whether a list is a reply's text parts (``{"text": ...}`` blocks, plain strings) and nothing else.

    A column mapped to ``message.content`` holds such a list; any other list
    is a JSON answer, whose objects keep their keys.
    """
    return bool(items) and all(
        isinstance(i, str)
        or (isinstance(i, Mapping) and "text" in i and i.get("type") in (None, *_TEXT_PART_TYPES))
        for i in items
    )


def _json_container(value: object, depth: int = _DECODES) -> object | None:
    """The object or list that JSON text holds (through up to ``depth`` layers of JSON strings).

    ``None`` when ``value`` is not text, is not JSON, holds a scalar, or nests
    too deeply to decode: a decode that raises ``RecursionError`` (a hundred
    thousand open brackets) is a text that is not JSON, never an error.
    """
    if not isinstance(value, str) or depth == 0 or value.lstrip()[:1] not in _JSON_OPENERS:
        return None
    try:
        decoded = json.loads(value)
    except (ValueError, RecursionError):
        return None
    if isinstance(decoded, str):  # a reply double-encoded as JSON text
        return _json_container(decoded, depth - 1)
    return decoded if isinstance(decoded, (Mapping, list)) else None


def _is_reply_container(value: object) -> bool:
    return (
        is_reply(value) if isinstance(value, Mapping) else has_reply_parts(cast("list[Any]", value))
    )


def _reply_json(value: object) -> object:
    """JSON text that holds a reply is that reply; any other text, JSON or not, is as it came.

    A CSV cell or a flattened export carries an object as text. Whatever the
    reply was, an object, a list or such a string, it goes through the same rules.
    """
    decoded = _json_container(value)
    return value if decoded is None or not _is_reply_container(decoded) else decoded


def reply_of(value: object) -> tuple[str, tuple[dict[str, Any], ...]]:
    """``(text, tool calls)`` of an assistant reply in any provider's shape.

    ``value`` may be a string (or JSON text of one of the rest), a message
    object (``content`` beside ``tool_calls``, or content blocks holding
    ``tool_use`` / ``functionCall``), an envelope that wraps one (Chat
    Completions, Ollama, Cohere, Gemini, Bedrock: see :func:`_inner_reply`), a
    Responses object (its ``output`` item list), one Responses item, or a list
    of typed blocks or items. Reasoning is never in the text.
    """
    value = _reply_json(value)
    if isinstance(value, Mapping):
        message = _inner_reply(value)
        if message is not None:
            return reply_of(message)
        if isinstance(value.get("output"), list):  # a Responses object
            value = value["output"]
        elif value.get("type") in CALL_TYPES:  # one Responses output item
            value = [value]
        else:
            content = value.get("content", value.get("parts"))  # OTel GenAI: ``parts``
            declared = tool_calls(value.get("tool_calls"))
            legacy = value.get("function_call")  # Chat Completions before ``tool_calls``
            if isinstance(legacy, Mapping):
                declared = merge_calls(declared, (dict(legacy),))
            return reply_text(content), merge_calls(declared, block_calls(content))
    return reply_text(value), block_calls(value)


def final_reply(value: object) -> Reply:
    """:func:`reply_of` of a response that is a record's answer.

    A reply with no text and no call, that holds the model's reasoning, is
    ``reasoning_only``: it has no answer to learn, and the call is still a call.
    """
    value = _reply_json(value)
    reply, calls = reply_of(value)
    return Reply(reply, calls, not reply and not calls and carries_reasoning(value))


def response_of(value: object) -> Reply:
    """:func:`final_reply` of a field that holds a reply or, as JSON, the answer itself.

    A mapping or list that is a reply (:func:`is_reply`, :func:`has_reply_parts`)
    is unwrapped, as is JSON text of one. Any other object is the model's own
    answer and stays its JSON text, byte for byte, **unless** reasoning sits
    anywhere in it (strict: a plain ``reasoning`` field may be the answer's own):
    then it is an envelope no reader recognises, whose answer cannot be told from
    its reasoning, and writing it out whole would train on the reasoning. It is
    ``reasoning_only`` instead (counted, priced, no row), never dumped and never
    partly dumped.
    """
    container = value if isinstance(value, (Mapping, list)) else _json_container(value)
    if container is not None and not _is_reply_container(container):
        if carries_reasoning(container, strict=True):
            return Reply("", (), reasoning_only=True)
        if not isinstance(value, str):
            value = text(value)
    return final_reply(value)


def _turns(items: list[Any]) -> Iterator[tuple[str, str]]:
    """``(role, content)`` for each chat message or Responses input item, in order.

    An assistant turn keeps its text and its tool calls, rendered the way the
    student will answer them (:func:`effective_response`), so a later step's
    tool result answers a call the student can see. A Responses
    ``function_call`` item is such a turn; its ``function_call_output`` is a
    ``tool`` turn. Any other typed item without a role (reasoning, a hosted
    tool's record, anything a vendor adds) is dropped.
    """
    for item in items:
        if not isinstance(item, Mapping):
            raise MalformedRowError(f"not a chat message: {item!r}")
        kind = item.get("type")
        if "role" not in item and (
            is_reasoning(item) or (isinstance(kind, str) and kind not in _CONVERSATION_ITEM_TYPES)
        ):
            continue
        if kind in CALL_OUTPUT_TYPES:
            yield "tool", _part_text(item)
        elif function_call(item) is not None and "role" not in item:
            yield "assistant", effective_response("", (item,))
        elif "role" not in item:
            raise MalformedRowError(f"not a chat message: {item!r}")
        else:
            role = text(item["role"])
            content = item.get("content", item.get("parts"))
            if role in _ASSISTANT_ROLES:
                reply, calls = reply_of(
                    {
                        "content": content,
                        "tool_calls": item.get("tool_calls"),
                        "function_call": item.get("function_call"),
                    }
                )
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
    the tool calls they made. Reasoning, and every typed item of a Responses
    ``input`` list that is not a message, a tool call or its output (a hosted
    tool's ``web_search_call`` or ``mcp_call``, a ``compaction``), are skipped.
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
    workload whose answers depend on it cannot be learned from the text.
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
    the request carries, so it joins the workload key. A FORCED tool is
    such a schema (OpenAI ``tool_choice.function.name``, Anthropic's and the
    Responses API's ``tool_choice.name``, the legacy ``function_call.name``;
    LangChain's ``with_structured_output`` forces one), so it joins too.
    The tools a request OFFERS do not: an agent's tool set changes per call (a
    deploy adds one, permissions or retrieval pick them), and keying on it split
    one 2,400-call router into four workloads too small to audit.
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
    "Reply",
    "content_text",
    "effective_response",
    "final_reply",
    "has_media",
    "has_reply_parts",
    "is_reply",
    "is_text_parts",
    "merge_calls",
    "reply_of",
    "reply_text",
    "response_of",
    "split_prompt",
    "task_signature",
    "tool_calls",
]
