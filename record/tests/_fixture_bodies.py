"""Canned replies for ``gen_fixtures.py``: documented provider shapes whose ids and times follow the call number.

Every body is built from the file's base time and the call's number, so the same script gives the
same bytes each run. (The streams reuse the other test modules' canned events, which are fixed.)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

CHAT_MODEL = "gpt-4o-mini-2024-07-18"
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
PDF = "JVBERi0xLjQKJSVFT0YK"
WAV = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA="


def tool_call(n: int, name: str, arguments: str) -> Dict[str, Any]:
    """An OpenAI function tool call as the API returns it."""
    return {
        "id": f"call_{n}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def chat_body(
    created: int,
    n: int,
    content: Optional[str],
    *,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    usage: Any = (40, 2, 0),
    finish: Optional[str] = None,
    **message: Any,
) -> Dict[str, Any]:
    """A non-stream ``chat.completion``; ``usage`` is (prompt, completion, cached) tokens."""
    reply: Dict[str, Any] = {"role": "assistant", "content": content, "refusal": None}
    if tool_calls is not None:
        reply["tool_calls"] = tool_calls
    reply.update(message)
    prompt, completion, cached = usage
    counts: Dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    if cached:
        counts["prompt_tokens_details"] = {"cached_tokens": cached}
    return {
        "id": f"chatcmpl-{n}",
        "object": "chat.completion",
        "created": created,
        "model": CHAT_MODEL,
        "choices": [
            {
                "index": 0,
                "message": reply,
                "finish_reason": finish or ("tool_calls" if tool_calls else "stop"),
            }
        ],
        "usage": counts,
    }


def response_body(
    created: int,
    n: int,
    output: List[Dict[str, Any]],
    *,
    usage: Any = (30, 2, 0),
) -> Dict[str, Any]:
    """A non-stream Responses API ``response``; ``usage`` is (input, output, cached) tokens."""
    prompt, completion, cached = usage
    return {
        "id": f"resp_{n}",
        "object": "response",
        "created_at": created,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "model": CHAT_MODEL,
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": prompt,
            "output_tokens": completion,
            "total_tokens": prompt + completion,
            "input_tokens_details": {"cached_tokens": cached},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def message_item(n: int, text: str) -> Dict[str, Any]:
    """A Responses ``message`` output item."""
    return {
        "id": f"msg_{n}",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def call_item(n: int, name: str, arguments: str) -> Dict[str, Any]:
    """A Responses ``function_call`` output item."""
    return {
        "id": f"fc_{n}",
        "type": "function_call",
        "status": "completed",
        "call_id": f"call_{n}",
        "name": name,
        "arguments": arguments,
    }


def claude_body(
    n: int,
    content: List[Dict[str, Any]],
    *,
    stop: str = "end_turn",
    usage: Any = (25, 4, 0, 0),
) -> Dict[str, Any]:
    """A non-stream Messages ``message``; ``usage`` is (input, output, cache read, cache write)."""
    plain, completion, read, write = usage
    return {
        "id": f"msg_{n}",
        "type": "message",
        "role": "assistant",
        "model": CLAUDE_MODEL,
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {
            "input_tokens": plain,
            "cache_creation_input_tokens": write,
            "cache_read_input_tokens": read,
            "output_tokens": completion,
        },
    }


def text_block(text: str) -> Dict[str, Any]:
    """A ``text`` content block."""
    return {"type": "text", "text": text}


def tool_block(n: int, name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """A ``tool_use`` content block."""
    return {"type": "tool_use", "id": f"toolu_{n}", "name": name, "input": tool_input}
