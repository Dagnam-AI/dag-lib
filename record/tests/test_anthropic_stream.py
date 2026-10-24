"""``messages.create(stream=True)``: the events are folded into the message a plain call returns."""

from __future__ import annotations

from typing import Any, Dict, List

from _anthropic_bodies import (
    MODEL,
    PATH,
    block_delta,
    block_start,
    block_stop,
    error_body,
    head,
    json_delta,
    message_body,
    tail,
    text_delta,
    text_events,
    tool_events,
)
from _anthropic_calls import ARGS, run, unpatched
from _mock_server import Event, Reply
import anthropic
from anthropic.resources import Messages
import pytest

from dagnam_record import _anthropic


def read(arig: Any, events: List[Event], **kwargs: Any) -> List[Any]:
    """Every event of a stream the mock server plays."""
    arig.server.on(PATH, Reply(events=events))
    with arig.anthropic() as client:
        return list(client.messages.create(stream=True, **ARGS, **kwargs))


def test_a_text_stream_is_recorded_as_the_message_a_plain_call_returns(arig: Any) -> None:
    events = read(arig, text_events("Hel", "lo"))
    assert [event.type for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]  # a ping is the SDK's to swallow, as it would be unrecorded
    (line,) = arig.lines()
    assert line["request"] == dict(ARGS, stream=True)
    assert line["response"] == message_body([{"type": "text", "text": "Hello"}])
    assert (line["provider"], line["endpoint"], line["status"]) == ("anthropic", "messages", 200)
    norm = line["norm"]
    assert norm["response"] == "Hello"
    assert (norm["prompt_tokens"], norm["completion_tokens"], norm["cached_prompt_tokens"]) == (
        20,
        7,
        5,
    )


def test_the_caller_sees_the_same_events_as_without_the_recorder(arig: Any) -> None:
    arig.server.on(PATH, Reply(events=tool_events()))
    with arig.anthropic() as client:
        recorded = [e.model_dump() for e in client.messages.create(stream=True, **ARGS)]
        plain = unpatched(Messages.create)
        original = [e.model_dump() for e in plain(client.messages, stream=True, **ARGS)]
    assert recorded == original
    assert len(arig.lines()) == 1  # only the recorded call left a line


def test_the_stream_object_reports_the_sdk_class_and_forwards_what_it_does_not_know(
    arig: Any,
) -> None:
    arig.server.on(PATH, Reply(events=text_events()))
    with arig.anthropic() as client:
        stream = client.messages.create(stream=True, **ARGS)
        assert stream.__class__ is anthropic.Stream  # the SDK's own isinstance() is False for all
        assert stream.response.status_code == 200
        with stream as same:
            assert same is stream
            assert len(list(same)) == 7


def test_an_async_stream_is_recorded(arig: Any) -> None:
    arig.server.on(PATH, Reply(events=tool_events()))

    async def call(client: anthropic.AsyncAnthropic) -> List[Any]:
        stream = await client.messages.create(stream=True, **ARGS)
        assert stream.__class__ is anthropic.AsyncStream
        return [event async for event in stream]

    assert len(run(call, arig)) == 18
    (line,) = arig.lines()
    assert [call["function"]["name"] for call in line["norm"]["tool_calls"]] == [
        "weather",
        "clock",
        "ping",
    ]


def test_parallel_tool_inputs_are_assembled_per_block_index(arig: Any) -> None:
    read(arig, tool_events())
    (line,) = arig.lines()
    content = line["response"]["content"]
    assert content[0] == {"type": "text", "text": "Let me check."}
    assert [(b["id"], b["name"], b["input"]) for b in content[1:]] == [
        ("toolu_01", "weather", {"city": "Paris"}),  # pieces came interleaved with clock's
        ("toolu_02", "clock", {"tz": "CET"}),
        ("toolu_03", "ping", {}),  # one empty piece
    ]
    assert line["response"]["stop_reason"] == "tool_use"
    assert line["response"]["usage"]["output_tokens"] == 40
    assert line["norm"]["response"] == "Let me check."
    assert line["norm"]["tool_calls"] == [
        {"type": "function", "function": {"name": "weather", "arguments": '{"city": "Paris"}'}},
        {"type": "function", "function": {"name": "clock", "arguments": '{"tz": "CET"}'}},
        {"type": "function", "function": {"name": "ping", "arguments": "{}"}},
    ]


def test_a_tool_without_any_json_piece_keeps_the_input_it_started_with(arig: Any) -> None:
    started = {
        "type": "server_tool_use",
        "id": "srvtoolu_1",
        "name": "web_search",
        "input": {"q": "x"},
    }
    stream = [*head(), block_start(0, started), block_stop(0), *tail()]
    read(arig, stream)
    (line,) = arig.lines()
    assert line["response"]["content"] == [started]


def test_a_tool_call_cut_short_by_max_tokens_has_an_empty_input(arig: Any) -> None:
    started = {"type": "tool_use", "id": "toolu_01", "name": "weather", "input": {}}
    stream = [*head(), block_start(0, started), block_delta(0, json_delta('{"city": "Pa'))]
    read(arig, [*stream, block_stop(0), *tail("max_tokens")])
    (line,) = arig.lines()
    assert line["response"]["content"][0]["input"] == {}
    assert line["response"]["stop_reason"] == "max_tokens"
    assert line["norm"]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_thinking_signature_citations_and_server_tool_blocks(arig: Any) -> None:
    citation = {
        "type": "web_search_result_location",
        "url": "https://example.com",
        "title": "T",
        "encrypted_index": "Eo8B",
        "cited_text": "Paris",
    }
    result = {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_1",
        "content": [{"type": "web_search_result", "url": "https://example.com", "title": "T"}],
    }
    signature = "EqQBCkYIA" * 40
    events = [
        *head(),
        block_start(0, {"type": "thinking", "thinking": "", "signature": ""}),
        block_delta(0, {"type": "thinking_delta", "thinking": "Let me "}),
        block_delta(0, {"type": "thinking_delta", "thinking": "think."}),
        block_delta(0, {"type": "signature_delta", "signature": signature}),
        block_stop(0),
        block_start(1, {"type": "redacted_thinking", "data": "EmwKAhgB"}),
        block_stop(1),
        block_start(
            2, {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}}
        ),
        block_delta(2, json_delta('{"query": "capital of France"}')),
        block_stop(2),
        block_start(3, result),
        block_stop(3),
        block_start(4, {"type": "text", "text": "", "citations": []}),
        block_delta(4, text_delta("Paris.")),
        block_delta(4, {"type": "citations_delta", "citation": citation}),
        block_stop(4),
        *tail(),
    ]
    read(arig, events)
    (line,) = arig.lines()
    assert line["response"]["content"] == [
        {"type": "thinking", "thinking": "Let me think.", "signature": signature},
        {"type": "redacted_thinking", "data": "EmwKAhgB"},
        {
            "type": "server_tool_use",
            "id": "srvtoolu_1",
            "name": "web_search",
            "input": {"query": "capital of France"},
        },
        result,
        {"type": "text", "text": "Paris.", "citations": [citation]},
    ]
    assert line["norm"]["response"] == "Paris."
    assert line["norm"]["tool_calls"] == []  # a server tool is not a call the client must answer


def test_usage_in_message_delta_updates_the_counts_from_message_start(arig: Any) -> None:
    final = {
        "input_tokens": 15,
        "cache_read_input_tokens": 6,
        "output_tokens": 9,
        "cache_creation_input_tokens": None,
    }
    events = text_events("Hi")
    events[-2] = ("message_delta", dict(events[-2][1], usage=final))
    read(arig, events)
    (line,) = arig.lines()
    assert line["response"]["usage"] == {
        "input_tokens": 15,
        "cache_creation_input_tokens": 3,  # null in the update: the earlier value stays
        "cache_read_input_tokens": 6,
        "output_tokens": 9,
    }
    norm = line["norm"]
    assert (norm["prompt_tokens"], norm["completion_tokens"], norm["cached_prompt_tokens"]) == (
        24,
        9,
        6,
    )


@pytest.mark.parametrize(
    "stop_reason", ["end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"]
)
def test_every_stop_reason_is_kept(arig: Any, stop_reason: str) -> None:
    events = [*head(), *tail(stop_reason)]
    read(arig, events)
    (line,) = arig.lines()
    assert line["response"]["stop_reason"] == stop_reason
    assert line["response"]["content"] == []


def test_unknown_future_delta_and_block_types_are_ignored_or_kept_whole(arig: Any) -> None:
    future: Dict[str, Any] = {"type": "future_block", "payload": {"a": 1}}
    events = [
        *head(),
        block_start(0, future),
        block_delta(0, {"type": "future_delta", "value": 1}),
        block_delta(0, {"type": "text_delta", "text": 5}),  # not text: skipped
        block_delta(9, text_delta("for a block that never started")),
        block_stop(0),
        ("future_event", {"type": "future_event"}),
        *tail(),
    ]
    read(arig, events)
    (line,) = arig.lines()
    assert line["response"]["content"] == [future]
    assert arig.writer.counts["dropped_error"] == 0


def test_an_error_event_mid_stream_is_a_failed_call_not_a_200_row(arig: Any) -> None:
    events = [*text_events("Hel")[:4], ("error", error_body("overloaded_error"))]
    arig.server.on(PATH, Reply(events=events))
    with arig.anthropic() as client:
        stream = client.messages.create(stream=True, **ARGS)
        with pytest.raises(anthropic.APIStatusError) as caught:
            list(stream)
    assert caught.value.status_code == 200  # the SDK reports the connection's status
    (line,) = arig.lines()
    assert line["status"] == 529  # what the error type stands for
    assert line["response"] == {"error": {"type": "APIStatusError"}}
    assert arig.writer.counts["dropped_partial"] == 0


def test_an_error_event_of_an_unknown_type_is_recorded_as_no_response(arig: Any) -> None:
    arig.server.on(PATH, Reply(events=[("error", error_body("brand_new_error"))]))
    with arig.anthropic() as client, pytest.raises(anthropic.APIStatusError):
        list(client.messages.create(stream=True, **ARGS))
    (line,) = arig.lines()
    assert line["status"] == 599


def test_a_stream_the_server_refuses_to_open_is_a_failed_call(arig: Any) -> None:
    arig.server.on(PATH, Reply(429, error_body("rate_limit_error")))
    with arig.anthropic() as client, pytest.raises(anthropic.RateLimitError):
        client.messages.create(stream=True, **ARGS)
    (line,) = arig.lines()
    assert line["status"] == 429
    assert line["request"]["stream"] is True


def test_the_request_is_taken_at_call_time_not_when_the_stream_ends(arig: Any) -> None:
    arig.server.on(PATH, Reply(events=text_events()))
    messages: List[Any] = [{"role": "user", "content": "before"}]
    with arig.anthropic() as client:
        stream = client.messages.create(model=MODEL, max_tokens=8, messages=messages, stream=True)
        messages[0]["content"] = "after"
        list(stream)
    (line,) = arig.lines()
    assert line["request"]["messages"] == [{"role": "user", "content": "before"}]


def test_a_stream_whose_events_cannot_be_folded_is_counted_and_still_passes_through(
    arig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(self: Any, event: Any) -> None:
        raise RuntimeError("assembler bug")

    monkeypatch.setattr(_anthropic.MessageAssembler, "add", refuse)
    assert len(read(arig, text_events())) == 7
    assert arig.raw() == []
    assert arig.writer.counts["dropped_error"] == 1


def test_an_error_event_that_reaches_the_recorder_as_an_item_is_a_failed_row(
    arig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transport that yields the error event instead of raising (the SDK raises it today)."""
    events: List[Any] = [
        {"type": "message_start", "message": {"id": "m", "usage": {"input_tokens": 1}}},
        {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}},
    ]
    _anthropic.uninstall()
    monkeypatch.setattr(Messages, "create", lambda self, **kwargs: iter(events))
    assert _anthropic.install(arig.recorder)
    with arig.anthropic() as client:
        assert list(client.messages.create(stream=True, **ARGS)) == events
    _anthropic.uninstall()  # before monkeypatch puts the SDK's function back
    (line,) = arig.lines()
    assert (line["status"], line["response"]) == (529, {"error": {"type": "overloaded_error"}})
    assert arig.writer.counts["dropped_partial"] == 0
