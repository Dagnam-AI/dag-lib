"""Envelope v1 lines: the fields, media omission and the per-provider ``norm`` block."""

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

CHAT_REQUEST: dict[str, Any] = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "developer", "content": [{"type": "text", "text": "No jokes."}]},
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Hel"}, {"type": "text", "text": "lo"}],
        },
        {"role": "user", "content": None},
        "not a message",
        {"content": "no role"},
    ],
}
CHAT_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-1",
    "model": "gpt-4o-mini-2024",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "Hello",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q": 1}'},
                    },
                    {"type": "function", "function": {"name": "other", "arguments": {"a": 1}}},
                    {"type": "function", "function": {"arguments": "{}"}},
                    {"type": "function", "function": "bad"},
                    "bad",
                ],
            }
        }
    ],
    "usage": {
        "prompt_tokens": 9,
        "completion_tokens": 2,
        "prompt_tokens_details": {"cached_tokens": 4},
    },
}


def _line(**overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "provider": "openai",
        "endpoint": "chat.completions",
        "request": CHAT_REQUEST,
        "response": CHAT_RESPONSE,
        "latency_ms": 812.5,
        "trace_id": "t-1",
        "now": 1_791_797_400.125,
    }
    args.update(overrides)
    return json.loads(build_line(**args))


def test_a_line_carries_the_envelope_fields_raw_and_norm() -> None:
    line = _line()
    assert line["v"] == 1
    assert line["trace_id"] == "t-1"
    assert line["ts"] == "2026-10-12T09:30:00.125Z"
    assert line["provider"] == "openai"
    assert line["endpoint"] == "chat.completions"
    assert line["latency_ms"] == 812.5
    assert line["status"] == 200
    assert line["request"] == CHAT_REQUEST
    assert line["response"] == CHAT_RESPONSE
    assert line["norm_v"] == 1
    assert line["norm"] == norm_openai_chat(CHAT_REQUEST, CHAT_RESPONSE)


def test_a_line_is_one_compact_unicode_json_line() -> None:
    text = build_line(
        provider="openai",
        endpoint="chat.completions",
        request={"messages": [{"role": "user", "content": "café"}]},
        response={},
        latency_ms=1,
    )
    assert "\n" not in text
    assert ": " not in text
    assert "café" in text


def test_the_trace_id_and_the_time_default_when_not_given() -> None:
    first = json.loads(
        build_line(
            provider="other", endpoint="custom", request={}, response={}, latency_ms=1, status=500
        )
    )
    second = json.loads(
        build_line(provider="other", endpoint="custom", request={}, response={}, latency_ms=1)
    )
    assert first["trace_id"] != second["trace_id"]
    assert len(first["trace_id"]) > 8
    assert first["ts"].endswith("Z")
    assert first["status"] == 500
    assert "norm" not in first
    assert "norm_v" not in first


def test_no_raw_keeps_only_the_norm() -> None:
    line = _line(no_raw=True)
    assert "request" not in line
    assert "response" not in line
    assert line["norm"]["response"] == "Hello"


def test_no_raw_without_a_known_provider_shape_is_refused() -> None:
    with pytest.raises(ValueError, match="no_raw"):
        build_line(
            provider="other",
            endpoint="custom",
            request={},
            response={},
            latency_ms=1,
            no_raw=True,
        )


@pytest.mark.parametrize(
    ("provider", "endpoint"),
    [("gemini", "custom"), ("openai", "completions"), ("", "")],
)
def test_an_unknown_provider_or_endpoint_is_refused(provider: str, endpoint: str) -> None:
    with pytest.raises(ValueError, match="provider"):
        build_line(provider=provider, endpoint=endpoint, request={}, response={}, latency_ms=1)


@pytest.mark.parametrize("latency", [float("nan"), float("inf")])
def test_a_number_json_cannot_carry_is_refused(latency: float) -> None:
    with pytest.raises(ValueError, match="JSON"):
        build_line(provider="other", endpoint="custom", request={}, response={}, latency_ms=latency)


def test_a_lone_surrogate_is_refused() -> None:
    with pytest.raises(ValueError, match="utf-8"):
        build_line(
            provider="other",
            endpoint="custom",
            request={"text": "\ud800"},
            response={},
            latency_ms=1,
        )


def test_media_parts_are_replaced_by_their_size_everywhere_in_the_request() -> None:
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    audio = {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}
    nested = {"type": "tool_result", "content": [{"type": "document", "source": {"data": "B"}}]}
    request: dict[str, Any] = {
        "model": "m",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "look"}, image, audio, nested]}
        ],
        "tool_choice": {"type": "file"},
    }
    line = _line(request=request, response={})
    content = line["request"]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "look"}
    assert content[1] == {
        "type": "media_omitted",
        "bytes": len(json.dumps(image, ensure_ascii=False).encode("utf-8")),
    }
    assert content[2]["type"] == "media_omitted"
    assert content[3]["content"][0]["type"] == "media_omitted"
    assert line["request"]["tool_choice"] == {"type": "file"}  # not a list item: not a part
    assert line["norm"]["has_media"] is True
    assert "AAAA" not in json.dumps(line["request"])
    assert request["messages"][0]["content"][1] is image  # the caller's body is untouched


@pytest.mark.parametrize(
    "part_type",
    ["image_url", "input_image", "image", "input_audio", "audio", "file", "input_file", "document"],
)
def test_every_media_part_type_is_omitted_and_marks_the_call(part_type: str) -> None:
    request = {"messages": [{"role": "user", "content": [{"type": part_type, "data": "x"}]}]}
    line = _line(request=request, response={})
    assert line["request"]["messages"][0]["content"][0]["type"] == "media_omitted"
    assert line["norm"]["has_media"] is True


def test_an_already_omitted_part_still_marks_the_call_multimodal() -> None:
    omitted = {"type": "media_omitted", "bytes": 12}
    request = {"messages": [{"role": "user", "content": [omitted]}]}
    assert norm_openai_chat(request, {})["has_media"] is True
    plain = {"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}], "n": 1}]}
    assert norm_openai_chat(plain, {})["has_media"] is False


def test_openai_chat_norm() -> None:
    norm = norm_openai_chat(CHAT_REQUEST, CHAT_RESPONSE)
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
        ("openai", "chat.completions", CHAT_REQUEST, CHAT_RESPONSE),
        ("openai", "responses", {"input": "Hi"}, {"output": []}),
        ("anthropic", "messages", {"messages": []}, {"content": []}),
    ],
)
def test_the_norm_is_chosen_by_provider_and_endpoint(
    provider: str, endpoint: str, req: dict[str, Any], resp: dict[str, Any]
) -> None:
    line = _line(provider=provider, endpoint=endpoint, request=req, response=resp)
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
