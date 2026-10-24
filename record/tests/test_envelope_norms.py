"""The ``norm`` block of an envelope line, for each provider shape."""

from __future__ import annotations

import json
from typing import Any

import pytest

from dagnam_record._envelope import (
    build_line,
    norm_anthropic,
    norm_openai_chat,
    norm_openai_responses,
)


def test_openai_chat_norm(chat_bodies: Any) -> None:
    norm = norm_openai_chat(*chat_bodies)
    assert norm == {
        "model": "gpt-4o-mini-2024",
        "system": "Be brief.\nNo jokes.",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": ""},
        ],
        "response": "Hello",
        "tool_calls": [
            {"type": "function", "function": {"name": "lookup", "arguments": '{"q": 1}'}},
            {"type": "function", "function": {"name": "other", "arguments": '{"a": 1}'}},
        ],
        "prompt_tokens": 9,
        "completion_tokens": 2,
        "cached_prompt_tokens": 4,
        "has_media": False,
    }


def test_openai_chat_norm_of_empty_or_odd_bodies() -> None:
    empty = norm_openai_chat({}, {})
    assert empty["model"] == "unknown"
    assert empty["system"] is None
    assert empty["messages"] == []
    assert empty["response"] == ""
    assert empty["tool_calls"] == []
    assert (empty["prompt_tokens"], empty["completion_tokens"], empty["cached_prompt_tokens"]) == (
        0,
        0,
        0,
    )
    odd = norm_openai_chat(
        {"model": "req-model", "messages": "no"},
        {"model": "", "choices": ["x"], "usage": {"prompt_tokens": True, "completion_tokens": -1}},
    )
    assert odd["model"] == "req-model"
    assert odd["response"] == ""
    assert odd["prompt_tokens"] == 0
    assert odd["completion_tokens"] == 0
    parts = norm_openai_chat({}, {"choices": [{"message": {"content": [{"text": "a"}, 1]}}]})
    assert parts["response"] == "a"
    assert norm_openai_chat({}, {"choices": []})["response"] == ""
    assert norm_openai_chat({}, {"choices": [{"message": {"tool_calls": "x"}}]})["tool_calls"] == []


def test_responses_norm() -> None:
    request = {
        "model": "gpt-5",
        "instructions": "Be brief.",
        "input": [
            {"role": "developer", "content": "Also kind."},
            {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
            {"type": "function_call", "name": "lookup", "arguments": "{}", "call_id": "c1"},
            {"type": "function_call_output", "call_id": "c1", "output": "42"},
            {"type": "function_call_output", "call_id": "c2", "output": {"x": 1}},
            {"type": "reasoning", "summary": []},
            "stray",
        ],
    }
    response = {
        "model": "gpt-5-2026",
        "output": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "hidden"}]},
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Hel"},
                    {"type": "output_text", "text": "lo"},
                ],
            },
            {"type": "function_call", "name": "lookup", "arguments": '{"q": 1}', "call_id": "c3"},
            {"type": "function_call", "name": 5, "arguments": "{}"},
            {"type": "function_call", "name": "noargs"},
            "stray",
        ],
        "usage": {
            "input_tokens": 20,
            "output_tokens": 5,
            "input_tokens_details": {"cached_tokens": 8},
        },
    }
    assert norm_openai_responses(request, response) == {
        "model": "gpt-5-2026",
        "system": "Be brief.\nAlso kind.",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "tool", "content": "42"},
            {"role": "tool", "content": ""},
        ],
        "response": "Hello",
        "tool_calls": [
            {"type": "function", "function": {"name": "lookup", "arguments": '{"q": 1}'}},
            {"type": "function", "function": {"name": "noargs", "arguments": ""}},
        ],
        "prompt_tokens": 20,
        "completion_tokens": 5,
        "cached_prompt_tokens": 8,
        "has_media": False,
    }


def test_responses_norm_of_a_string_input_and_odd_bodies() -> None:
    norm = norm_openai_responses({"input": "Hi", "model": "m"}, {"output": "no"})
    assert norm["system"] is None
    assert norm["messages"] == [{"role": "user", "content": "Hi"}]
    assert norm["response"] == ""
    assert norm["tool_calls"] == []
    assert norm["model"] == "m"
    assert norm_openai_responses({"input": 3}, {})["messages"] == []
    assert norm_openai_responses({"instructions": 3}, {})["system"] is None


def test_anthropic_norm() -> None:
    request = {
        "model": "claude-x",
        "system": [{"type": "text", "text": "Be"}, {"type": "text", "text": " brief."}],
        "messages": [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Calling"},
                    {"type": "tool_use", "id": "t1", "name": "lookup", "input": {"q": 1}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "42"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "text", "text": "ok"}],
                    },
                    {"type": "image", "source": {"data": "AAAA"}},
                ],
            },
        ],
    }
    response = {
        "model": "claude-x-2026",
        "content": [
            {"type": "thinking", "thinking": "secret", "signature": "s"},
            {"type": "text", "text": "Hel"},
            {"type": "text", "text": "lo"},
            {"type": "tool_use", "id": "t2", "name": "lookup", "input": {"q": "é"}},
            {"type": "tool_use", "id": "t3", "name": 5, "input": {}},
            {"type": "tool_use", "id": "t4", "name": "noinput"},
            "stray",
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 3,
            "cache_read_input_tokens": 4,
            "cache_creation_input_tokens": 2,
        },
    }
    assert norm_anthropic(request, response) == {
        "model": "claude-x-2026",
        "system": "Be brief.",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Calling"},
            {"role": "user", "content": "42ok"},
        ],
        "response": "Hello",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "lookup", "arguments": json.dumps({"q": "é"})},
            },
            {"type": "function", "function": {"name": "noinput", "arguments": "{}"}},
        ],
        "prompt_tokens": 16,
        "completion_tokens": 3,
        "cached_prompt_tokens": 4,
        "has_media": True,
    }


def test_anthropic_norm_of_a_string_system_and_empty_bodies() -> None:
    norm = norm_anthropic({"system": "Be brief.", "messages": []}, {"content": "no"})
    assert norm["system"] == "Be brief."
    assert norm["response"] == ""
    assert norm["prompt_tokens"] == 0
    assert norm_anthropic({}, {})["system"] is None


@pytest.mark.parametrize(
    ("provider", "endpoint", "req", "resp"),
    [
        ("openai", "chat.completions", {"messages": []}, {}),
        ("openai", "responses", {"input": "Hi"}, {"output": []}),
        ("anthropic", "messages", {"messages": []}, {"content": []}),
    ],
)
def test_the_norm_is_chosen_by_provider_and_endpoint(
    provider: str, endpoint: str, req: dict[str, Any], resp: dict[str, Any]
) -> None:
    line = json.loads(
        build_line(provider=provider, endpoint=endpoint, request=req, response=resp, latency_ms=1)
    )
    assert line["norm_v"] == 1
    assert set(line["norm"]) == {
        "model",
        "system",
        "messages",
        "response",
        "tool_calls",
        "prompt_tokens",
        "completion_tokens",
        "cached_prompt_tokens",
        "has_media",
    }
