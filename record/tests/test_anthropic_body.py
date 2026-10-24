"""Request and response bodies of Anthropic calls, apart from any client."""

from __future__ import annotations

from typing import Any

import pytest

from dagnam_record._anthropic_body import REDACTED, request_body, response_body


def test_transport_arguments_are_not_part_of_the_body() -> None:
    body = request_body(
        {
            "model": "m",
            "extra_headers": {"x-api-key": "k"},
            "extra_query": {"a": "b"},
            "extra_body": {"c": "d"},
            "timeout": 3,
            "api_key": "k",
            "auth_token": "t",
        }
    )
    assert body == {"model": "m"}


def test_only_the_tokens_of_server_definitions_and_the_credit_code_are_replaced() -> None:
    field = "authorization_" + "token"
    servers: Any = [
        {"type": "url", "url": "https://a", "name": "a", field: "secret-a"},
        {"type": "url", "url": "https://b", "name": "b"},  # no token to hide
        "not a definition",
    ]
    body = request_body(
        {"mcp_servers": servers, "fallback_credit_token": "secret-c", "metadata": {"user_id": "u"}}
    )
    assert body["mcp_servers"][0][field] == REDACTED
    assert body["mcp_servers"][1] == {"type": "url", "url": "https://b", "name": "b"}
    assert body["mcp_servers"][2] == "not a definition"
    assert body["fallback_credit_token"] == REDACTED
    assert body["metadata"] == {"user_id": "u"}
    assert servers[0][field] == "secret-a"  # the caller's own data is untouched


def test_a_request_without_servers_is_just_converted() -> None:
    assert request_body({"mcp_servers": "none", "max_tokens": 4}) == {
        "mcp_servers": "none",
        "max_tokens": 4,
    }


def test_base64_sources_anywhere_in_a_response_are_replaced_by_their_size() -> None:
    body = response_body(
        {
            "content": [
                {"type": "text", "text": "x"},
                {"type": "result", "parts": [{"source": {"type": "base64", "data": "QUJD"}}]},
                {"type": "other", "source": {"type": "url", "data": "https://x"}},
                {"type": "other", "source": {"type": "base64", "data": None}},
            ]
        }
    )
    parts = body["content"]
    assert parts[1]["parts"][0]["source"] == {
        "type": "base64",
        "data": {"type": "media_omitted", "bytes": 4},
    }
    assert parts[2]["source"] == {"type": "url", "data": "https://x"}
    assert parts[3]["source"] == {"type": "base64", "data": None}


def test_a_response_that_is_not_a_model_is_refused() -> None:
    with pytest.raises(ValueError, match="not a model"):
        response_body("<APIResponse>")
