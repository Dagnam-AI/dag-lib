"""Tool calls in every provider's spelling: read, matched and rendered as the one JSON the student answers.

A call is judged and trained as ``{"arguments": ..., "name": ...}``, and a reply
with several calls as the ordered list of them. A message can carry one call in
two places (LangChain ``tool_calls`` beside the ``tool_use`` block in its
``content``), and a reply may repeat a call the teacher really made twice;
:func:`merge_calls` tells the two apart.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

from dagnam.audit.readers.base import MalformedRowError, get

CALL_TYPES = frozenset({"tool_use", "function_call", "custom_tool_call", "tool-call", "tool_call"})
CALL_OUTPUT_TYPES = frozenset({"function_call_output", "custom_tool_call_output"})
"""Responses ``input`` items that carry a tool's answer to an earlier call."""


def function_call(part: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The tool call a content part or Responses item makes, if it makes one."""
    if part.get("type") in CALL_TYPES:
        return part
    call = get(part, "functionCall", "function_call")
    return call if isinstance(call, Mapping) else None


def tool_calls(value: object) -> tuple[dict[str, Any], ...]:
    """A response's tool calls as a tuple of JSON objects (empty when absent)."""
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(call, dict) for call in value):
        raise MalformedRowError(f"tool_calls is not a list of objects: {value!r}")
    return tuple(value)


def block_calls(content: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(content, list):
        return ()
    calls = (function_call(part) for part in content if isinstance(part, Mapping))
    return tuple(dict(call) for call in calls if call is not None)


def _call_id(call: Mapping[str, Any]) -> object:
    """What names one call: a Responses item's ``call_id``, else the ``id`` every other shape has."""
    return get(call, "call_id", "id")


def _same_call(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """By id when both carry a real one (an empty id names nothing), else by name and arguments."""
    ids = (_call_id(a), _call_id(b))
    return ids[0] == ids[1] if all(ids) else _signature(a) == _signature(b)


def merge_calls(
    calls: Sequence[dict[str, Any]], blocks: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], ...]:
    """``calls``, then every one of ``blocks`` that is not one of them over again.

    A message can carry one call in two places: a LangChain ``AIMessage`` from
    an Anthropic model lists it in ``tool_calls`` and keeps its ``tool_use``
    block in ``content``. Read from both it would be judged and trained as two.
    Each call absorbs at most one block, so a call the model really made twice
    stays two.
    """
    unmatched = list(calls)
    merged = list(calls)
    for block in blocks:
        twin = next((call for call in unmatched if _same_call(call, block)), None)
        if twin is None:
            merged.append(block)
        else:
            unmatched.remove(twin)
    return tuple(merged)


def _raw_arguments(call: Mapping[str, Any]) -> Any:
    return get(call, "function.arguments", "custom.input", "args", "arguments", "input")


def _name(call: Mapping[str, Any]) -> Any:
    return get(call, "function.name", "custom.name", "name")


def _parsed(arguments: Any) -> Any:
    """A JSON string's value (an object, a list, a number); anything else as it is."""
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except ValueError:
            return arguments
    return arguments


def _signature(call: Mapping[str, Any]) -> tuple[Any, Any]:
    """What makes two calls the same call when neither has an id: name and parsed arguments."""
    arguments = _raw_arguments(call)
    return _name(call), {} if arguments is None else _parsed(arguments)


def _call_object(call: Mapping[str, Any]) -> dict[str, Any]:
    """One tool call as ``{"arguments": ..., "name": ...}``, whatever the export's spelling.

    OpenAI ``function.name`` / ``function.arguments`` (a JSON string; a
    ``custom`` tool call's ``custom.name`` / ``custom.input``), LangChain
    ``name`` / ``args``, a bare ``arguments``, an Anthropic ``tool_use``
    block's ``input``, a Gemini ``functionCall``'s ``args``. Arguments that are
    not a JSON object stay what they are.
    """
    arguments = _raw_arguments(call)
    parsed = _parsed(arguments)
    if isinstance(parsed, dict):
        arguments = parsed
    return {"arguments": {} if arguments is None else arguments, "name": _name(call)}


def effective_response(response: str, calls: Sequence[Mapping[str, Any]]) -> str:
    """The text a reply is judged and trained on: its tool calls, else its text.

    One call is the sorted-keys object ``{"arguments": ..., "name": ...}``;
    several are the ordered list of them, so a reply that tags a ticket
    with a product *and* a team is judged on both. The name is part of it
    because a router or handoff step decides by *which* tool it calls, usually
    with ``{}`` arguments: judged on the arguments alone it has one output, and
    a student that always answers ``{}`` would score as agreeing.
    """
    if not calls:
        return response
    objects = [_call_object(call) for call in calls]
    return json.dumps(objects[0] if len(objects) == 1 else objects, sort_keys=True)


__all__ = [
    "CALL_OUTPUT_TYPES",
    "CALL_TYPES",
    "block_calls",
    "effective_response",
    "function_call",
    "merge_calls",
    "tool_calls",
]
