"""The Anthropic patch on ``messages.create`` without streaming: bodies, norms, errors, secrets."""

from __future__ import annotations

import json
from typing import Any, Dict, cast

from _anthropic_bodies import MODEL, PATH, error_body, message_body, tool_use
from _anthropic_calls import ARGS, MESSAGES, run
from _mock_server import Reply
import anthropic
import pytest

from dagnam_record._anthropic_body import REDACTED

CREDENTIAL = "sk-ant-api03-must-never-be-recorded"


def test_a_call_is_recorded_with_its_bodies_and_the_sdk_result_is_returned(arig: Any) -> None:
    arig.server.on(PATH, Reply(body=message_body()))
    with arig.anthropic() as client:
        result = client.messages.create(**ARGS)
    assert isinstance(result, anthropic.types.Message)
    assert result.model_dump()["content"][0]["text"] == "Hello there"
    (line,) = arig.lines()
    assert (line["provider"], line["endpoint"], line["status"]) == ("anthropic", "messages", 200)
    assert line["request"] == ARGS
    assert line["response"] == message_body()
    norm = line["norm"]
    assert norm["response"] == "Hello there"
    assert norm["model"] == MODEL
    # input 12 + cache creation 3 + cache read 5; only the read part is "cached"
    assert (norm["prompt_tokens"], norm["completion_tokens"], norm["cached_prompt_tokens"]) == (
        20,
        7,
        5,
    )
    assert norm["messages"] == [{"role": "user", "content": "Hi"}]
    assert norm["system"] is None
    assert arig.server.seen[0].body == ARGS


def test_an_async_call_is_recorded_too(arig: Any) -> None:
    arig.server.on(PATH, Reply(body=message_body()))

    async def call(client: anthropic.AsyncAnthropic) -> Any:
        return await client.messages.create(**ARGS)

    assert run(call, arig).model_dump()["content"][0]["text"] == "Hello there"
    (line,) = arig.lines()
    assert line["norm"]["response"] == "Hello there"
    assert line["endpoint"] == "messages"


@pytest.mark.parametrize(
    "system",
    [
        "Be brief.",
        [
            {"type": "text", "text": "Be brief."},
            {"type": "text", "text": "No jokes.", "cache_control": {"type": "ephemeral"}},
        ],
    ],
)
def test_the_system_prompt_may_be_a_string_or_text_blocks(arig: Any, system: Any) -> None:
    arig.server.on(PATH, Reply(body=message_body()))
    with arig.anthropic() as client:
        client.messages.create(system=system, **ARGS)
    (line,) = arig.lines()
    assert line["request"]["system"] == system
    expected = system if isinstance(system, str) else "Be brief.No jokes."
    assert line["norm"]["system"] == expected


def test_tool_use_thinking_server_tool_and_citation_blocks_are_kept_as_sent(arig: Any) -> None:
    blocks = [
        {"type": "thinking", "thinking": "Hmm.", "signature": "EqQBCkYIA" * 40},
        {"type": "redacted_thinking", "data": "EmwKAhgB" * 30},
        {
            "type": "text",
            "text": "Paris.",
            "citations": [
                {
                    "type": "web_search_result_location",
                    "url": "https://example.com",
                    "title": "T",
                    "encrypted_index": "Eo8B",
                    "cited_text": "Paris",
                }
            ],
        },
        {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"q": "x"}},
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": [
                {
                    "type": "web_search_result",
                    "url": "https://example.com",
                    "title": "T",
                    "encrypted_content": "Ep" * 100,
                    "page_age": None,
                }
            ],
        },
        tool_use(1, "weather", {"city": "Paris", "days": 2}),
        tool_use(2, "clock", {}),
    ]
    arig.server.on(PATH, Reply(body=message_body(blocks, "tool_use")))
    with arig.anthropic() as client:
        client.messages.create(**ARGS)
    (line,) = arig.lines()
    assert line["response"]["content"] == blocks
    assert line["response"]["stop_reason"] == "tool_use"
    norm = line["norm"]
    assert norm["response"] == "Paris."  # only text blocks, not thinking
    assert norm["tool_calls"] == [  # only client tool uses, OpenAI-shaped
        {
            "type": "function",
            "function": {"name": "weather", "arguments": '{"city": "Paris", "days": 2}'},
        },
        {"type": "function", "function": {"name": "clock", "arguments": "{}"}},
    ]


@pytest.mark.parametrize(
    "stop_reason", ["end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"]
)
def test_every_stop_reason_is_recorded_in_the_raw_response(arig: Any, stop_reason: str) -> None:
    arig.server.on(PATH, Reply(body=message_body(stop_reason=stop_reason)))
    with arig.anthropic() as client:
        client.messages.create(**ARGS)
    (line,) = arig.lines()
    assert line["response"]["stop_reason"] == stop_reason


def test_images_and_documents_are_omitted_with_their_size_and_still_sent(arig: Any) -> None:
    image = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "QUJD" * 5},
    }
    document = {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": "UERG" * 9},
        "cache_control": {"type": "ephemeral"},
    }
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Look"}, image, document]},
        {
            "role": "assistant",
            "content": [tool_use(1, "shot", {})],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_01", "content": [image]},
            ],
        },
    ]
    arig.server.on(PATH, Reply(body=message_body()))
    with arig.anthropic() as client:
        client.messages.create(model=MODEL, max_tokens=8, messages=messages)
    (line,) = arig.lines()
    sent = arig.server.seen[0].body["messages"]
    assert sent[0]["content"][1] == image  # the server got the real thing
    parts = line["request"]["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "Look"}
    assert [part["type"] for part in parts[1:]] == ["media_omitted", "media_omitted"]
    assert parts[1]["bytes"] == len(json.dumps(image, ensure_ascii=False, separators=(", ", ": ")))
    result = line["request"]["messages"][2]["content"][0]["content"]
    assert [part["type"] for part in result] == ["media_omitted"]
    assert line["norm"]["has_media"] is True
    assert "QUJD" not in arig.raw()[0]


def test_base64_in_a_response_is_omitted_and_the_caller_still_gets_it(arig: Any) -> None:
    big = "QUJD" * 300_000  # more than a whole line may hold
    blocks = [
        {"type": "text", "text": "Done."},
        {
            "type": "container_file",
            "source": {"type": "base64", "media_type": "image/png", "data": big},
        },
    ]
    arig.server.on(PATH, Reply(body=message_body(blocks)))
    with arig.anthropic() as client:
        result = client.messages.create(**ARGS)
    assert big in json.dumps(result.model_dump(warnings=False))
    (line,) = arig.lines()
    source = line["response"]["content"][1]["source"]
    assert source == {
        "type": "base64",
        "media_type": "image/png",
        "data": {"type": "media_omitted", "bytes": len(big)},
    }


def test_no_key_header_or_credential_is_ever_recorded(arig: Any) -> None:
    arig.server.on(PATH, Reply(body=message_body()))
    servers = [
        {
            "type": "url",
            "url": "https://mcp.example.com",
            "name": "m",
            "authorization_token": CREDENTIAL,
        }
    ]
    with arig.anthropic(auth_token="bearer-" + CREDENTIAL) as client:
        client.beta.messages.create(
            betas=["mcp-client-2025-04-04"],
            mcp_servers=servers,
            fallback_credit_token=cast("Any", CREDENTIAL),
            extra_headers={"x-api-key": CREDENTIAL, "authorization": "Bearer " + CREDENTIAL},
            extra_query={"key": CREDENTIAL},
            extra_body={"secret": CREDENTIAL},
            **ARGS,
        )
    seen = arig.server.seen[0]
    assert seen.headers["x-api-key"] == CREDENTIAL  # the extra header won over the client's key
    assert seen.body["mcp_servers"][0]["authorization_token"] == CREDENTIAL  # the server got it
    (line,) = arig.lines()
    text = arig.raw()[0]
    assert CREDENTIAL not in text
    assert "anthropic-beta" not in text
    assert "sk-ant" not in text
    assert line["request"]["mcp_servers"][0]["authorization_token"] == REDACTED
    assert line["request"]["fallback_credit_token"] == REDACTED
    assert line["request"]["betas"] == ["mcp-client-2025-04-04"]
    assert "secret" not in line["request"]  # extra_body is not a parameter we know


@pytest.mark.parametrize(
    ("status", "kind", "error"),
    [
        (400, "invalid_request_error", anthropic.BadRequestError),
        (401, "authentication_error", anthropic.AuthenticationError),
        (429, "rate_limit_error", anthropic.RateLimitError),
        (500, "api_error", anthropic.InternalServerError),
        (529, "overloaded_error", anthropic.APIStatusError),
    ],
)
def test_an_error_status_propagates_unchanged_and_is_recorded_with_that_status(
    arig: Any, status: int, kind: str, error: Any
) -> None:
    arig.server.on(PATH, Reply(status, error_body(kind)))
    with arig.anthropic() as client, pytest.raises(error) as caught:
        client.messages.create(**ARGS)
    assert caught.value.status_code == status
    (line,) = arig.lines()
    assert line["status"] == status
    assert line["response"] == {"error": {"type": type(caught.value).__name__}}
    assert line["request"] == ARGS
    assert line["norm"]["response"] == ""


def test_a_connection_error_and_a_timeout_are_recorded_as_no_response(arig: Any) -> None:
    arig.server.on(PATH, Reply(body=message_body(), delay=1.0))
    with arig.anthropic(timeout=0.2) as client, pytest.raises(anthropic.APITimeoutError):
        client.messages.create(**ARGS)
    unreachable = arig.anthropic(base_url="http://127.0.0.1:1")
    with unreachable as client, pytest.raises(anthropic.APIConnectionError):
        client.messages.create(**ARGS)
    lines = arig.lines()
    assert [line["status"] for line in lines] == [599, 599]
    assert lines[0]["response"] == {"error": {"type": "APITimeoutError"}}
    assert lines[1]["response"] == {"error": {"type": "APIConnectionError"}}


def test_not_given_values_and_sdk_objects_in_a_request_are_plain_json(arig: Any) -> None:
    arig.server.on(PATH, Reply(body=message_body()), Reply(body=message_body()))
    with arig.anthropic() as client:
        first = client.messages.create(temperature=anthropic.NOT_GIVEN, **ARGS)
        again = [*MESSAGES, {"role": "assistant", "content": first.content}]
        client.messages.create(model=MODEL, max_tokens=8, messages=again)
    first_line, second = arig.lines()
    assert "temperature" not in first_line["request"]
    assert second["request"]["messages"][1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello there"}],
    }


def test_calls_that_are_not_message_calls_are_left_alone(arig: Any) -> None:
    arig.server.on("/v1/messages/count_tokens", Reply(body={"input_tokens": 4}))
    with arig.anthropic() as client:
        assert client.messages.count_tokens(model=MODEL, messages=MESSAGES).input_tokens == 4
    assert arig.lines() == []
    assert arig.writer.counts["dropped_error"] == 0


def test_raw_and_streaming_response_helpers_are_unaffected_and_counted(arig: Any) -> None:
    arig.server.on(PATH, Reply(body=message_body()), Reply(body=message_body()))
    with arig.anthropic() as client:
        raw = client.messages.with_raw_response.create(**ARGS)
        assert raw.parse().model_dump()["content"][0]["text"] == "Hello there"
        with client.messages.with_streaming_response.create(**ARGS) as response:
            assert response.parse().model_dump()["content"][0]["text"] == "Hello there"
    assert arig.lines() == []
    assert arig.writer.counts["dropped_error"] == 2


def test_the_request_is_taken_before_the_call_not_after(arig: Any) -> None:
    arig.server.on(PATH, Reply(body=message_body(), delay=0.3))
    messages: Any = [{"role": "user", "content": "Original"}]
    with arig.anthropic() as client:
        import threading

        timer = threading.Timer(0.1, lambda: messages[0].update(content="Changed"))
        timer.start()
        client.messages.create(model=MODEL, max_tokens=8, messages=messages)
        timer.join()
    (line,) = arig.lines()
    assert line["request"]["messages"][0]["content"] == "Original"


def test_a_request_that_loops_back_on_itself_does_not_hang_or_raise(arig: Any) -> None:
    loop: Dict[str, Any] = {"role": "user", "content": "Hi"}
    loop["self"] = loop
    arig.server.on(PATH, Reply(body=message_body()))
    with arig.anthropic() as client, pytest.raises(ValueError, match="Circular"):
        client.messages.create(model=MODEL, max_tokens=8, messages=[loop])
    assert arig.lines() == []
    assert arig.writer.counts["dropped_error"] == 1
