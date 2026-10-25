"""Credentials in tool definitions never reach the file: for OpenAI and Anthropic, SDK or not."""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List

from _mock_server import Reply
from _openai_bodies import MODEL, RESPONSES_PATH, response_body
import pytest

from dagnam_record._credentials import REDACTED, hide_credentials

SECRETS = ("MCPSECRET1", "HDRSECRET2", "QUERYSECRET3", "userpw4")


def _mcp_tool() -> Dict[str, Any]:
    return {
        "type": "mcp",
        "server_label": "docs",
        "server_url": "https://user:userpw4@mcp.example.com/sse?key=QUERYSECRET3&x=1",
        "authorization": "Bearer MCPSECRET1",
        "headers": {"X-Api-Key": "HDRSECRET2", "X-Other": "kept?"},
        "require_approval": "never",
    }


def test_an_openai_mcp_tool_loses_its_authorization_headers_and_url_secrets() -> None:
    body = {"model": "m", "tools": [_mcp_tool()]}
    (tool,) = hide_credentials(body)["tools"]
    assert tool["authorization"] == REDACTED
    assert tool["headers"] == {"X-Api-Key": REDACTED, "X-Other": REDACTED}
    assert tool["server_url"] == f"https://{REDACTED}@mcp.example.com/sse?{REDACTED}"
    assert (tool["type"], tool["server_label"], tool["require_approval"]) == (
        "mcp",
        "docs",
        "never",
    )
    assert not any(secret in json.dumps(tool) for secret in SECRETS)


def test_the_body_it_was_given_is_not_changed() -> None:
    body = {"model": "m", "tools": [_mcp_tool()], "mcp_servers": [_mcp_tool()]}
    before = copy.deepcopy(body)
    hide_credentials(body)
    assert body == before


@pytest.mark.parametrize(
    "key", ["authorization", "authorization_token", "api_key", "token", "access_token", "secret"]
)
def test_every_credential_key_of_a_tool_or_server_entry_is_replaced_in_any_case(
    key: str,
) -> None:
    for spelling in (key, key.upper(), key.title()):
        found = hide_credentials(
            {"tools": [{spelling: "S3CRET"}], "mcp_servers": [{spelling: "S3CRET"}]}
        )
        assert found == {"tools": [{spelling: REDACTED}], "mcp_servers": [{spelling: REDACTED}]}


def test_a_function_tool_and_a_clean_server_url_are_left_exactly_as_they_were() -> None:
    tools: List[Any] = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"properties": {"token": {"type": "string"}}},
            },
        },
        {"type": "url", "url": "https://b.example.com/mcp", "name": "b", "headers": None},
        "not a definition",
    ]
    body = {"tools": tools, "messages": [{"role": "user", "content": "my token is hunter2"}]}
    assert hide_credentials(body) == body


def test_a_non_mapping_headers_value_is_replaced_whole() -> None:
    assert hide_credentials({"tools": [{"headers": ["Authorization: Bearer X"]}]}) == {
        "tools": [{"headers": REDACTED}]
    }


def test_a_body_without_tool_lists_comes_back_unchanged() -> None:
    body = {"model": "m", "tools": "none", "mcp_servers": None}
    assert hide_credentials(body) == body


def test_a_real_responses_call_with_an_mcp_tool_writes_no_credential(rig: Any) -> None:
    rig.server.on(RESPONSES_PATH, Reply(body=response_body()))
    with rig.client() as client:
        client.responses.create(model=MODEL, input="Hi", tools=[_mcp_tool()])
    raw = "\n".join(rig.raw())
    assert not any(secret in raw for secret in SECRETS)
    (line,) = rig.lines()
    assert line["request"]["tools"][0]["authorization"] == REDACTED
    assert line["request"]["tools"][0]["server_label"] == "docs"


def test_record_call_hides_them_too_and_so_does_a_norm_only_line(rig: Any) -> None:
    rig.recorder.no_raw = True
    rig.recorder.record(
        "openai",
        "responses",
        {"model": "m", "input": "Hi", "tools": [_mcp_tool()]},
        {"output": []},
        status=200,
        latency_ms=1.0,
    )
    assert not any(secret in "\n".join(rig.raw()) for secret in SECRETS)


def test_a_body_that_is_not_an_object_and_a_url_that_is_not_text_pass_through() -> None:
    assert hide_credentials(["not", "an", "object"]) == ["not", "an", "object"]
    assert hide_credentials({"tools": [{"url": None, "server_url": 7}]}) == {
        "tools": [{"url": None, "server_url": 7}]
    }
