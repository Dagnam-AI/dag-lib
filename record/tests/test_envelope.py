"""Envelope v1 lines: the fields, media omission and the errors for bad input."""

from __future__ import annotations

import json
from typing import Any

import pytest

from dagnam_record._envelope import build_line, norm_openai_chat


def _line(**overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "provider": "openai",
        "endpoint": "chat.completions",
        "request": {"messages": []},
        "response": {},
        "latency_ms": 812.5,
        "trace_id": "t-1",
        "now": 1_791_797_400.125,
    }
    args.update(overrides)
    return json.loads(build_line(**args))


def test_a_line_carries_the_envelope_fields_raw_and_norm(chat_bodies: Any) -> None:
    chat_request, chat_response = chat_bodies
    line = _line(request=chat_request, response=chat_response)
    assert line["v"] == 1
    assert line["trace_id"] == "t-1"
    assert line["ts"] == "2026-10-12T09:30:00.125Z"
    assert line["provider"] == "openai"
    assert line["endpoint"] == "chat.completions"
    assert line["latency_ms"] == 812.5
    assert line["status"] == 200
    assert line["request"] == chat_request
    assert line["response"] == chat_response
    assert line["norm_v"] == 1
    assert line["norm"] == norm_openai_chat(chat_request, chat_response)


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


def test_no_raw_keeps_only_the_norm(chat_bodies: Any) -> None:
    chat_request, chat_response = chat_bodies
    line = _line(request=chat_request, response=chat_response, no_raw=True)
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
    with pytest.raises(ValueError, match="latency"):
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
    [
        "image_url",
        "input_image",
        "image",
        "input_audio",
        "audio",
        "file",
        "input_file",
        "document",
        "video",
    ],
)
def test_every_media_part_type_is_omitted_and_marks_the_call(part_type: str) -> None:
    request = {"messages": [{"role": "user", "content": [{"type": part_type, "data": "x"}]}]}
    line = _line(request=request, response={})
    assert line["request"]["messages"][0]["content"][0]["type"] == "media_omitted"
    assert line["norm"]["has_media"] is True


def test_a_media_part_in_a_tuple_is_omitted_like_one_in_a_list() -> None:
    request = {"messages": [{"role": "user", "content": ({"type": "image", "data": "AAAA"},)}]}
    line = _line(request=request, response={})
    assert line["request"]["messages"][0]["content"][0]["type"] == "media_omitted"
    assert line["norm"]["has_media"] is True
    assert norm_openai_chat(request, {})["has_media"] is True


def test_an_already_omitted_part_still_marks_the_call_multimodal() -> None:
    omitted = {"type": "media_omitted", "bytes": 12}
    request = {"messages": [{"role": "user", "content": [omitted]}]}
    assert norm_openai_chat(request, {})["has_media"] is True
    plain = {"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}], "n": 1}]}
    assert norm_openai_chat(plain, {})["has_media"] is False
