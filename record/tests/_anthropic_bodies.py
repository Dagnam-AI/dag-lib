"""Canned Anthropic Messages replies, in the shapes the API documents, for the mock server."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from _mock_server import Event

PATH = "/v1/messages"
MODEL = "claude-sonnet-4-5-20250929"
USAGE: Dict[str, Any] = {
    "input_tokens": 12,
    "cache_creation_input_tokens": 3,
    "cache_read_input_tokens": 5,
    "output_tokens": 7,
}


def message_body(
    content: Optional[List[Dict[str, Any]]] = None,
    stop_reason: str = "end_turn",
    usage: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """A non-stream ``message``; text "Hello there" unless ``content`` is given."""
    return {
        "id": "msg_01ABC",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": content if content is not None else [{"type": "text", "text": "Hello there"}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": dict(USAGE if usage is None else usage),
    }


def tool_use(index: int, name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """A ``tool_use`` content block."""
    return {"type": "tool_use", "id": f"toolu_0{index}", "name": name, "input": tool_input}


def error_body(kind: str = "overloaded_error") -> Dict[str, Any]:
    """The error object the API sends with an error status or in an ``error`` event."""
    return {"type": "error", "error": {"type": kind, "message": "try again"}}


def block_start(index: int, block: Dict[str, Any]) -> Event:
    """A ``content_block_start`` event."""
    event = {"type": "content_block_start", "index": index, "content_block": block}
    return "content_block_start", event


def block_delta(index: int, delta: Dict[str, Any]) -> Event:
    """A ``content_block_delta`` event."""
    return "content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta}


def block_stop(index: int) -> Event:
    """A ``content_block_stop`` event."""
    return "content_block_stop", {"type": "content_block_stop", "index": index}


def text_delta(text: str) -> Dict[str, Any]:
    """A ``text_delta``."""
    return {"type": "text_delta", "text": text}


def json_delta(partial: str) -> Dict[str, Any]:
    """An ``input_json_delta``."""
    return {"type": "input_json_delta", "partial_json": partial}


def head(usage: Optional[Dict[str, Any]] = None) -> List[Event]:
    """``message_start`` (with the input usage) and a ``ping``."""
    start = message_body(content=[], usage=usage or dict(USAGE, output_tokens=1))
    start["stop_reason"] = None
    return [
        ("message_start", {"type": "message_start", "message": start}),
        ("ping", {"type": "ping"}),
    ]


def tail(stop_reason: str = "end_turn", usage: Optional[Dict[str, Any]] = None) -> List[Event]:
    """``message_delta`` (stop reason, output usage) and the terminal ``message_stop``."""
    delta: Dict[str, Any] = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
    }
    delta["usage"] = usage if usage is not None else {"output_tokens": 7}
    return [("message_delta", delta), ("message_stop", {"type": "message_stop"})]


def text_events(*pieces: str, usage: Optional[Dict[str, Any]] = None) -> List[Event]:
    """A text answer split into ``pieces``."""
    deltas = [block_delta(0, text_delta(piece)) for piece in pieces or ("Hel", "lo")]
    return [
        *head(usage),
        block_start(0, {"type": "text", "text": ""}),
        *deltas,
        block_stop(0),
        *tail(),
    ]


def tool_events() -> List[Event]:
    """Text, then two parallel tool uses whose JSON pieces arrive interleaved, then a third with none."""

    def start(index: int, name: str) -> Event:
        return block_start(index, tool_use(index, name, {}))

    return [
        *head(),
        block_start(0, {"type": "text", "text": ""}),
        block_delta(0, text_delta("Let me check.")),
        block_stop(0),
        start(1, "weather"),
        start(2, "clock"),
        block_delta(1, json_delta("")),
        block_delta(1, json_delta('{"city": "Par')),
        block_delta(2, json_delta('{"tz":')),
        block_delta(1, json_delta('is"}')),
        block_delta(2, json_delta('"CET"}')),
        block_stop(1),
        block_stop(2),
        start(3, "ping"),
        block_delta(3, json_delta("")),
        block_stop(3),
        *tail("tool_use", {"output_tokens": 40}),
    ]
