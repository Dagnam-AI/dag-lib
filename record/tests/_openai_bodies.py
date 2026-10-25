"""Canned OpenAI replies, in the shapes the API documents, for the mock server."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from _mock_server import Event

CHAT_PATH = "/v1/chat/completions"
RESPONSES_PATH = "/v1/responses"
MODEL = "gpt-4o-mini-2024-07-18"


def chat_body(
    content: Optional[str] = "Hello there",
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    **message: Any,
) -> Dict[str, Any]:
    """A non-stream ``chat.completion`` with one choice."""
    reply: Dict[str, Any] = {"role": "assistant", "content": content, "refusal": None}
    if tool_calls is not None:
        reply["tool_calls"] = tool_calls
    reply.update(message)
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1_791_797_400,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": reply,
                "logprobs": None,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 3,
            "total_tokens": 14,
            "prompt_tokens_details": {"cached_tokens": 4, "audio_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
        "service_tier": "default",
        "system_fingerprint": "fp_1",
    }


def two_tool_calls() -> List[Dict[str, Any]]:
    """Two parallel tool calls, as the API returns them."""
    return [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "get_time", "arguments": '{"tz":"CET"}'},
        },
    ]


def _chunk(choices: List[Dict[str, Any]], **extra: Any) -> Event:
    body: Dict[str, Any] = {
        "id": "chatcmpl-stream1",
        "object": "chat.completion.chunk",
        "created": 1_791_797_400,
        "model": MODEL,
        "system_fingerprint": "fp_1",
        "choices": choices,
    }
    body.update(extra)
    return None, body


def _choice(delta: Dict[str, Any], finish: Optional[str] = None) -> Dict[str, Any]:
    return {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}


def chat_text_events(usage: bool = True) -> List[Event]:
    """A text stream: role, two content deltas, the stop chunk, then usage when asked for."""
    events = [
        _chunk([_choice({"role": "assistant", "content": "", "refusal": None})]),
        _chunk([_choice({"content": "Hel"})]),
        _chunk([_choice({"content": "lo"})]),
        _chunk([_choice({}, "stop")]),
    ]
    if usage:
        events.append(
            _chunk(
                [],
                usage={
                    "prompt_tokens": 11,
                    "completion_tokens": 2,
                    "total_tokens": 13,
                    "prompt_tokens_details": {"cached_tokens": 4},
                },
            )
        )
    events.append((None, "[DONE]"))
    return events


def chat_tool_events() -> List[Event]:
    """A stream with two tool calls whose argument pieces arrive interleaved by index."""

    def delta(index: int, **fields: Any) -> Event:
        call: Dict[str, Any] = {"index": index}
        call.update(fields)
        return _chunk([_choice({"tool_calls": [call]})])

    return [
        _chunk([_choice({"role": "assistant", "content": None})]),
        delta(0, id="call_1", type="function", function={"name": "get_weather", "arguments": ""}),
        delta(0, function={"arguments": '{"city"'}),
        delta(1, id="call_2", type="function", function={"name": "get_time", "arguments": ""}),
        delta(0, function={"arguments": ':"Paris"}'}),
        delta(1, function={"arguments": '{"tz":"CET"}'}),
        _chunk([_choice({}, "tool_calls")]),
        (None, "[DONE]"),
    ]


def response_body(text: str = "Hello there") -> Dict[str, Any]:
    """A non-stream Responses API ``response`` with one message and one function call."""
    return {
        "id": "resp_123",
        "object": "response",
        "created_at": 1_791_797_400,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": "Be brief.",
        "model": MODEL,
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            },
            {
                "id": "fc_1",
                "type": "function_call",
                "status": "completed",
                "call_id": "call_9",
                "name": "lookup",
                "arguments": '{"q":1}',
            },
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 12,
            "output_tokens": 5,
            "total_tokens": 17,
            "input_tokens_details": {"cached_tokens": 6},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def response_events(final: str = "response.completed") -> List[Event]:
    """A Responses stream that ends with the terminal event ``final``."""
    done = response_body()
    started = dict(done, status="in_progress", output=[], usage=None)
    return [
        (
            "response.created",
            {"type": "response.created", "sequence_number": 0, "response": started},
        ),
        (
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hello",
            },
        ),
        (final, {"type": final, "sequence_number": 2, "response": done}),
    ]
