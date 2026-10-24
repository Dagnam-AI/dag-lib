"""The OpenAI patch (chat completions and responses, sync and async) against the real SDK.

The real ``openai`` client talks to a local mock server, so what is recorded is checked
against what the SDK really builds and parses. Streams are in ``test_openai_stream.py``.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

from _mock_server import Reply
from _openai_bodies import (
    CHAT_PATH,
    MODEL,
    RESPONSES_PATH,
    chat_body,
    response_body,
    two_tool_calls,
)
from _openai_calls import API_KEY, ORGANIZATION, run, unpatched
import openai
from openai.resources.chat.completions import AsyncCompletions, Completions
import openai.resources.responses as responses_module
import pytest

from dagnam_record import _openai

MESSAGES: List[Any] = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Hi"},
]


def test_a_chat_call_returns_what_an_unpatched_call_returns_and_records_one_line(
    rig: Any,
) -> None:
    rig.server.on(CHAT_PATH, Reply(body=chat_body()))
    with rig.client() as client:
        result = client.chat.completions.create(
            model=MODEL,
            messages=MESSAGES,
            temperature=0.5,
            extra_headers={"X-Trace": "1"},
            extra_query={"q": "1"},
            extra_body={"vendor": 1},
            timeout=5,
        )
        plain = unpatched(Completions.create)(
            client.chat.completions, model=MODEL, messages=MESSAGES, temperature=0.5
        )
    assert result == plain
    (line,) = rig.lines()
    assert (line["provider"], line["endpoint"], line["status"]) == (
        "openai",
        "chat.completions",
        200,
    )
    assert line["request"] == {"model": MODEL, "messages": MESSAGES, "temperature": 0.5}
    assert line["response"] == chat_body()
    norm = line["norm"]
    assert norm["response"] == "Hello there"
    assert norm["system"] == "Be brief."
    assert norm["messages"] == [{"role": "user", "content": "Hi"}]
    assert (norm["prompt_tokens"], norm["completion_tokens"], norm["cached_prompt_tokens"]) == (
        11,
        3,
        4,
    )
    assert line["latency_ms"] >= 0


def test_no_credential_reaches_a_recorded_line(rig: Any) -> None:
    rig.server.on(CHAT_PATH, Reply(body=chat_body()), Reply(429, {"error": {"message": "slow"}}))
    with rig.client() as client:
        client.chat.completions.create(model=MODEL, messages=MESSAGES)
        with pytest.raises(openai.RateLimitError):
            client.chat.completions.create(model=MODEL, messages=MESSAGES)
    seen = rig.server.seen[0].headers
    assert seen["authorization"] == f"Bearer {API_KEY}"  # the key was really sent...
    assert seen["openai-organization"] == ORGANIZATION
    text = "\n".join(rig.raw())
    assert len(rig.raw()) == 2
    assert API_KEY not in text  # ...and never recorded
    assert ORGANIZATION not in text
    assert "authorization" not in text.lower()
    assert "extra_headers" not in text


def test_the_async_chat_call_is_recorded_like_the_sync_one(rig: Any) -> None:
    rig.server.on(CHAT_PATH, Reply(body=chat_body()))

    async def call(client: openai.AsyncOpenAI) -> Any:
        result = await client.chat.completions.create(model=MODEL, messages=MESSAGES)
        plain = await unpatched(AsyncCompletions.create)(
            client.chat.completions, model=MODEL, messages=MESSAGES
        )
        return result, plain

    result, plain = run(call, rig)
    assert result == plain
    (line,) = rig.lines()
    assert line["norm"]["response"] == "Hello there"
    assert line["response"] == chat_body()


def test_parallel_tool_calls_are_two_entries_in_order(rig: Any) -> None:
    rig.server.on(CHAT_PATH, Reply(body=chat_body(None, two_tool_calls())))
    with rig.client() as client:
        client.chat.completions.create(model=MODEL, messages=MESSAGES)
    (line,) = rig.lines()
    assert [call["function"]["name"] for call in line["norm"]["tool_calls"]] == [
        "get_weather",
        "get_time",
    ]
    assert line["norm"]["tool_calls"][1]["function"]["arguments"] == '{"tz":"CET"}'


@pytest.mark.parametrize(
    ("status", "error"), [(429, "RateLimitError"), (500, "InternalServerError")]
)
def test_an_api_error_is_raised_unchanged_and_recorded_with_its_status(
    rig: Any, status: int, error: str
) -> None:
    rig.server.on(CHAT_PATH, Reply(status, {"error": {"message": "no", "type": "x"}}))
    with rig.client() as client, pytest.raises(getattr(openai, error)):
        client.chat.completions.create(model=MODEL, messages=MESSAGES)
    (line,) = rig.lines()
    assert line["status"] == status
    assert line["response"] == {"error": {"type": error}}
    assert line["request"]["model"] == MODEL


def test_an_async_api_error_is_raised_unchanged_and_recorded(rig: Any) -> None:
    rig.server.on(CHAT_PATH, Reply(429, {"error": {"message": "slow"}}))

    async def call(client: openai.AsyncOpenAI) -> None:
        with pytest.raises(openai.RateLimitError):
            await client.chat.completions.create(model=MODEL, messages=MESSAGES)

    run(call, rig)
    (line,) = rig.lines()
    assert (line["status"], line["response"]) == (429, {"error": {"type": "RateLimitError"}})


def test_a_call_that_gets_no_http_response_is_recorded_as_no_response(rig: Any) -> None:
    client = rig.client(base_url="http://127.0.0.1:1/v1")
    with client, pytest.raises(openai.APIConnectionError):
        client.chat.completions.create(model=MODEL, messages=MESSAGES)
    (line,) = rig.lines()
    assert line["status"] == 599
    assert line["response"] == {"error": {"type": "APIConnectionError"}}


def test_a_raw_response_call_works_and_is_counted_not_recorded(rig: Any) -> None:
    rig.server.on(CHAT_PATH, Reply(body=chat_body()))
    with rig.client() as client:
        raw = client.chat.completions.with_raw_response.create(model=MODEL, messages=MESSAGES)
    assert raw.parse().choices[0].message.content == "Hello there"
    assert rig.raw() == []
    assert rig.writer.counts["dropped_error"] == 1


def test_a_responses_call_is_recorded_sync(rig: Any) -> None:
    rig.server.on(RESPONSES_PATH, Reply(body=response_body()))
    with rig.client() as client:
        result = client.responses.create(model=MODEL, instructions="Be brief.", input="Hi")
        plain = unpatched(responses_module.Responses.create)(
            client.responses, model=MODEL, instructions="Be brief.", input="Hi"
        )
    assert result == plain
    (line,) = rig.lines()
    assert (line["provider"], line["endpoint"], line["status"]) == ("openai", "responses", 200)
    assert line["request"] == {"model": MODEL, "instructions": "Be brief.", "input": "Hi"}
    assert line["response"] == response_body()
    norm = line["norm"]
    assert (norm["system"], norm["response"]) == ("Be brief.", "Hello there")
    assert norm["messages"] == [{"role": "user", "content": "Hi"}]
    assert norm["tool_calls"][0]["function"]["name"] == "lookup"
    assert (norm["prompt_tokens"], norm["completion_tokens"], norm["cached_prompt_tokens"]) == (
        12,
        5,
        6,
    )


def test_a_responses_call_is_recorded_async(rig: Any) -> None:
    rig.server.on(RESPONSES_PATH, Reply(body=response_body("Async")))

    async def call(client: openai.AsyncOpenAI) -> Any:
        return await client.responses.create(model=MODEL, input="Hi")

    result = run(call, rig)
    assert result.output_text == "Async"
    (line,) = rig.lines()
    assert (line["endpoint"], line["norm"]["response"]) == ("responses", "Async")


def test_a_responses_error_is_recorded(rig: Any) -> None:
    rig.server.on(RESPONSES_PATH, Reply(500, {"error": {"message": "boom"}}))
    with rig.client() as client, pytest.raises(openai.InternalServerError):
        client.responses.create(model=MODEL, input="Hi")
    (line,) = rig.lines()
    assert (line["endpoint"], line["status"]) == ("responses", 500)


def test_not_given_values_and_sdk_objects_in_a_request_are_recorded_as_plain_json(
    rig: Any,
) -> None:
    rig.server.on(CHAT_PATH, Reply(body=chat_body()), Reply(body=chat_body("Second")))
    with rig.client() as client:
        first = client.chat.completions.create(
            model=MODEL, messages=MESSAGES, seed=openai.NOT_GIVEN
        )
        client.chat.completions.create(
            model=MODEL,
            messages=[*MESSAGES, first.choices[0].message, {"role": "user", "content": "More"}],
        )
    first_line, second = rig.lines()
    assert "seed" not in first_line["request"]
    assistant = second["request"]["messages"][2]
    assert (assistant["role"], assistant["content"]) == ("assistant", "Hello there")
    assert json.loads(json.dumps(second["request"])) == second["request"]
    assert rig.server.seen[1].body["messages"][3] == {"role": "user", "content": "More"}


def test_install_twice_patches_once_and_uninstall_restores_the_original(rig: Any) -> None:
    patched = Completions.create
    original = unpatched(patched)
    assert getattr(patched, "__dagnam_record__", False) is True
    assert not hasattr(original, "__dagnam_record__")
    assert _openai.install(rig.recorder) is True
    assert Completions.create is patched  # not wrapped a second time
    rig.server.on(CHAT_PATH, Reply(body=chat_body()))
    with rig.client() as client:
        client.chat.completions.create(model=MODEL, messages=MESSAGES)
    assert len(rig.lines()) == 1  # one call, one line
    _openai.uninstall()
    assert Completions.create is original
    assert AsyncCompletions.create is not patched
    assert not hasattr(responses_module.Responses.create, "__wrapped__")
    with rig.client() as client:
        client.chat.completions.create(model=MODEL, messages=MESSAGES)
    assert len(rig.lines()) == 1  # nothing recorded after the uninstall
    _openai.uninstall()  # a second uninstall is a no-op


def test_a_second_install_swaps_the_recorder_without_wrapping_again(
    rig: Any, make: Any, tmp_path: Any
) -> None:
    other = make(tmp_path / "other")
    assert _openai.install(type(rig.recorder)(other))
    rig.server.on(CHAT_PATH, Reply(body=chat_body()))
    with rig.client() as client:
        client.chat.completions.create(model=MODEL, messages=MESSAGES)
    assert rig.lines() == []
    other.drain()
    assert len(list((tmp_path / "other").glob("*.jsonl"))) == 1


def test_the_kill_switch_patches_nothing(rig: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _openai.uninstall()
    before = (Completions.create, AsyncCompletions.create, responses_module.Responses.create)
    monkeypatch.setenv("DAGNAM_RECORD", "0")
    assert _openai.install(rig.recorder) is False
    assert (
        Completions.create,
        AsyncCompletions.create,
        responses_module.Responses.create,
    ) == before
    assert not hasattr(Completions.create, "__dagnam_record__")


def test_an_absent_sdk_is_not_an_error(rig: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _openai.uninstall()
    for name in (
        "openai",
        "openai.resources",
        "openai.resources.beta",
        "openai.resources.chat",
        "openai.resources.chat.completions",
        "openai.resources.responses",
    ):
        monkeypatch.setitem(sys.modules, name, None)
    assert _openai.install(rig.recorder) is False


def test_an_sdk_without_one_resource_still_patches_the_others(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _openai.uninstall()
    monkeypatch.delattr(responses_module, "AsyncResponses")
    monkeypatch.delattr(responses_module.Responses, "create")
    assert _openai.install(rig.recorder) is True
    assert hasattr(Completions.create, "__dagnam_record__")
    assert not hasattr(responses_module.Responses, "create")
    _openai.uninstall()
    assert not hasattr(Completions.create, "__dagnam_record__")


def test_an_sdk_with_no_known_resource_is_not_patched(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _openai.uninstall()
    monkeypatch.delattr(responses_module, "AsyncResponses")
    for owner in (Completions, AsyncCompletions, responses_module.Responses):
        monkeypatch.delattr(owner, "create")
        monkeypatch.delattr(owner, "parse")
    assert _openai.install(rig.recorder) is False


def test_the_recorder_may_be_missing_after_an_uninstall_race(rig: Any) -> None:
    """A wrapper that outlives ``uninstall`` (a captured bound method) calls straight through."""
    kept: Dict[str, Any] = {"create": Completions.create}
    _openai.uninstall()
    rig.server.on(CHAT_PATH, Reply(body=chat_body()))
    with rig.client() as client:
        result = kept["create"](client.chat.completions, model=MODEL, messages=MESSAGES)
    assert result.choices[0].message.content == "Hello there"
    assert rig.lines() == []


def test_an_async_wrapper_that_outlives_the_uninstall_calls_straight_through(rig: Any) -> None:
    kept = AsyncCompletions.create
    _openai.uninstall()
    rig.server.on(CHAT_PATH, Reply(body=chat_body()))

    async def call(client: openai.AsyncOpenAI) -> Any:
        return await kept(client.chat.completions, model=MODEL, messages=MESSAGES)

    assert run(call, rig).choices[0].message.content == "Hello there"
    assert rig.lines() == []
