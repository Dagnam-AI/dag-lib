"""Streams: teed, assembled when exhausted, never altered, never recorded when abandoned."""

from __future__ import annotations

from typing import Any, List

from _mock_server import Event, Reply
from _openai_bodies import (
    CHAT_PATH,
    MODEL,
    RESPONSES_PATH,
    chat_text_events,
    chat_tool_events,
    response_body,
    response_events,
)
from _openai_calls import run, unpatched
import httpx
import openai
from openai.resources.chat.completions import AsyncCompletions, Completions
import openai.resources.responses as responses_module
import pytest

from dagnam_record import _patch

MESSAGES: List[Any] = [{"role": "user", "content": "Hi"}]
USAGE = {"include_usage": True}


def open_chat(client: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("stream_options", USAGE)
    return client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True, **kwargs)


def test_a_chat_stream_yields_the_same_chunks_and_is_recorded_when_exhausted(rig: Any) -> None:
    rig.server.on(CHAT_PATH, Reply(events=chat_text_events()))
    with rig.client() as client:
        plain = list(
            unpatched(Completions.create)(
                client.chat.completions,
                model=MODEL,
                messages=MESSAGES,
                stream=True,
                stream_options=USAGE,
            )
        )
        iterator = iter(open_chat(client))
        first = [next(iterator) for _ in range(4)]
        assert rig.raw() == []  # nothing is written while the stream is still open
        rest = list(iterator)
    assert first + rest == plain
    (line,) = rig.lines()
    assert line["request"] == {
        "model": MODEL,
        "messages": MESSAGES,
        "stream": True,
        "stream_options": USAGE,
    }
    message = line["response"]["choices"][0]["message"]
    assert (message["role"], message["content"]) == ("assistant", "Hello")
    assert line["response"]["choices"][0]["finish_reason"] == "stop"
    assert (line["response"]["id"], line["response"]["model"]) == ("chatcmpl-stream1", MODEL)
    norm = line["norm"]
    assert norm["response"] == "Hello"
    assert (norm["prompt_tokens"], norm["completion_tokens"], norm["cached_prompt_tokens"]) == (
        11,
        2,
        4,
    )
    assert line["status"] == 200


def test_a_stream_without_usage_records_no_token_counts(rig: Any) -> None:
    rig.server.on(CHAT_PATH, Reply(events=chat_text_events(usage=False)))
    with rig.client() as client:
        list(open_chat(client, stream_options=None))
    (line,) = rig.lines()
    assert "usage" not in line["response"]
    assert (line["norm"]["prompt_tokens"], line["norm"]["completion_tokens"]) == (0, 0)


def test_tool_call_deltas_are_assembled_by_index(rig: Any) -> None:
    rig.server.on(CHAT_PATH, Reply(events=chat_tool_events()))
    with rig.client() as client:
        list(open_chat(client))
    (line,) = rig.lines()
    calls = line["norm"]["tool_calls"]
    assert [call["function"]["name"] for call in calls] == ["get_weather", "get_time"]
    assert [call["function"]["arguments"] for call in calls] == [
        '{"city":"Paris"}',
        '{"tz":"CET"}',
    ]
    message = line["response"]["choices"][0]["message"]
    assert message["content"] is None
    assert [c["id"] for c in message["tool_calls"]] == ["call_1", "call_2"]
    assert message["tool_calls"][0]["function"]["arguments"] == '{"city":"Paris"}'
    assert line["response"]["choices"][0]["finish_reason"] == "tool_calls"


def test_choices_refusals_and_legacy_function_calls_are_assembled(rig: Any) -> None:
    def chunk(index: int, delta: Any, finish: Any = None) -> Event:
        choice = {"index": index, "delta": delta, "logprobs": None, "finish_reason": finish}
        body = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": MODEL}
        return None, dict(body, choices=[choice])

    events: List[Event] = [
        chunk(1, {"role": "assistant", "refusal": "No"}),
        chunk(0, {"role": "assistant", "function_call": {"name": "f", "arguments": "{"}}),
        chunk(1, {"refusal": " way"}, "stop"),
        chunk(0, {"function_call": {"arguments": "}"}}, "function_call"),
        (None, "[DONE]"),
    ]
    rig.server.on(CHAT_PATH, Reply(events=events))
    with rig.client() as client:
        list(open_chat(client, n=2))
    (line,) = rig.lines()
    first, second = line["response"]["choices"]
    assert first["message"]["function_call"] == {"name": "f", "arguments": "{}"}
    assert second["message"]["refusal"] == "No way"
    assert (first["index"], second["index"]) == (0, 1)


def test_the_stream_object_keeps_its_type_attributes_and_context_manager(rig: Any) -> None:
    rig.server.on(CHAT_PATH, Reply(events=chat_text_events()))
    with rig.client() as client, open_chat(client) as stream:
        assert isinstance(stream, openai.Stream)
        assert stream.response.status_code == 200
        assert next(stream).id == "chatcmpl-stream1"
        assert len(list(stream)) == 4
        assert list(stream) == []  # an exhausted stream stays exhausted
        with pytest.raises(StopIteration):
            next(stream)
    assert len(rig.lines()) == 1  # recorded once, not once per exhausted read


def test_a_stream_error_propagates_unchanged_and_is_recorded_as_failed(rig: Any) -> None:
    events: List[Event] = [
        *chat_text_events()[:2],
        (None, {"error": {"message": "overloaded", "type": "server_error"}}),
    ]
    rig.server.on(CHAT_PATH, Reply(events=events))
    with rig.client() as client:
        iterator = iter(open_chat(client))
        assert [next(iterator).id, next(iterator).id] == ["chatcmpl-stream1"] * 2
        with pytest.raises(openai.APIError, match="overloaded"):
            next(iterator)
    (line,) = rig.lines()
    assert line["status"] == 599  # a mid-stream failure is never a successful 200 row
    assert line["response"] == {"error": {"type": "APIError"}}


def test_a_dropped_connection_propagates_unchanged_and_is_recorded_as_failed(rig: Any) -> None:
    rig.server.on(CHAT_PATH, Reply(events=chat_text_events()[:2], cut=True))
    with rig.client() as client, pytest.raises(httpx.RemoteProtocolError):
        list(open_chat(client))
    (line,) = rig.lines()
    assert (line["status"], line["response"]) == (599, {"error": {"type": "RemoteProtocolError"}})


def test_a_stream_call_refused_by_the_api_is_recorded_with_its_status(rig: Any) -> None:
    rig.server.on(CHAT_PATH, Reply(429, {"error": {"message": "slow"}}))
    with rig.client() as client, pytest.raises(openai.RateLimitError):
        open_chat(client)
    (line,) = rig.lines()
    assert (line["status"], line["request"]["stream"]) == (429, True)


def test_an_async_stream_is_teed_and_recorded_when_exhausted(rig: Any) -> None:
    rig.server.on(CHAT_PATH, Reply(events=chat_text_events()))

    async def call(client: openai.AsyncOpenAI) -> Any:
        plain = [
            c
            async for c in await unpatched(AsyncCompletions.create)(
                client.chat.completions,
                model=MODEL,
                messages=MESSAGES,
                stream=True,
                stream_options=USAGE,
            )
        ]
        stream = await open_chat(client)
        assert isinstance(stream, openai.AsyncStream)
        chunks: List[Any] = []
        async with stream:
            async for chunk in stream:
                chunks.append(chunk)
                if len(chunks) == 4:
                    assert rig.raw() == []
            assert stream.response.status_code == 200
        return plain, chunks

    plain, chunks = run(call, rig)
    assert chunks == plain
    (line,) = rig.lines()
    assert line["norm"]["response"] == "Hello"
    assert line["norm"]["prompt_tokens"] == 11


def test_a_responses_stream_is_recorded_from_its_completed_event(rig: Any) -> None:
    rig.server.on(RESPONSES_PATH, Reply(events=response_events()))
    with rig.client() as client:
        plain = list(
            unpatched(responses_module.Responses.create)(
                client.responses, model=MODEL, input="Hi", stream=True
            )
        )
        stream = client.responses.create(model=MODEL, input="Hi", stream=True)
        events = [next(iter(stream))]
        assert rig.raw() == []
        events += list(stream)
    assert events == plain
    (line,) = rig.lines()
    assert (line["endpoint"], line["status"]) == ("responses", 200)
    assert line["request"] == {"model": MODEL, "input": "Hi", "stream": True}
    assert line["response"] == response_body()
    assert line["norm"]["response"] == "Hello there"
    assert line["norm"]["tool_calls"][0]["function"]["name"] == "lookup"


def test_an_async_responses_stream_is_recorded(rig: Any) -> None:
    rig.server.on(RESPONSES_PATH, Reply(events=response_events()))

    async def call(client: openai.AsyncOpenAI) -> int:
        stream = await client.responses.create(model=MODEL, input="Hi", stream=True)
        return len([event async for event in stream])

    assert run(call, rig) == 3
    (line,) = rig.lines()
    assert line["response"] == response_body()


def test_an_incomplete_responses_stream_is_still_the_final_answer(rig: Any) -> None:
    rig.server.on(RESPONSES_PATH, Reply(events=response_events("response.incomplete")))
    with rig.client() as client:
        list(client.responses.create(model=MODEL, input="Hi", stream=True))
    (line,) = rig.lines()
    assert (line["status"], line["norm"]["response"]) == (200, "Hello there")


def test_a_failed_responses_stream_is_recorded_as_a_failed_call(rig: Any) -> None:
    rig.server.on(RESPONSES_PATH, Reply(events=response_events("response.failed")))
    with rig.client() as client:
        list(client.responses.create(model=MODEL, input="Hi", stream=True))
    (line,) = rig.lines()
    assert (line["status"], line["response"]) == (599, {"error": {"type": "response.failed"}})


def test_a_stream_that_cannot_be_watched_is_handed_back_untouched_and_counted(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: Any) -> None:
        raise RuntimeError("cannot tee")

    monkeypatch.setattr(_patch, "TeeStream", refuse)
    monkeypatch.setattr(_patch, "AsyncTeeStream", refuse)
    rig.server.on(CHAT_PATH, Reply(events=chat_text_events()))
    with rig.client() as client:
        stream = open_chat(client)
        assert type(stream) is openai.Stream  # the SDK's own stream, unwrapped
        assert len(list(stream)) == 5

    async def call(client: openai.AsyncOpenAI) -> int:
        stream = await open_chat(client)
        assert type(stream) is openai.AsyncStream
        return len([chunk async for chunk in stream])

    assert run(call, rig) == 5
    assert rig.raw() == []
    assert rig.writer.counts["dropped_error"] == 2
