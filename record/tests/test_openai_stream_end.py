"""How a stream ends: finished, truncated, closed, let go of, cancelled, or interrupted."""

from __future__ import annotations

import asyncio
import contextlib
import gc
from typing import Any, List

from _mock_server import Event, Reply
from _openai_bodies import (
    CHAT_PATH,
    MODEL,
    RESPONSES_PATH,
    chat_text_events,
    response_events,
)
from _openai_calls import run
import openai
import pytest

MESSAGES: List[Any] = [{"role": "user", "content": "Hi"}]
USAGE = {"include_usage": True}


def open_chat(client: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("stream_options", USAGE)
    return client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True, **kwargs)


def through_the_stop_chunk(stream: Any) -> Any:
    """Read up to and including the chunk that carries ``finish_reason``; the usage is still unread."""
    iterator = iter(stream)
    return [next(iterator) for _ in range(4)]


def test_closing_before_the_stop_chunk_writes_nothing_and_is_counted(rig: Any) -> None:
    rig.server.on(CHAT_PATH, Reply(events=chat_text_events()))
    with rig.client() as client:
        stream = open_chat(client)
        next(iter(stream))
        stream.close()
        with open_chat(client) as other:
            for _chunk in other:
                break  # leaving the with-block early closes the stream
    assert rig.raw() == []
    assert rig.writer.counts["written"] == 0
    assert rig.writer.counts["dropped_partial"] == 2
    assert rig.writer.counts["dropped_error"] == 0


def test_closing_after_the_stop_chunk_records_the_finished_answer_once(rig: Any) -> None:
    rig.server.on(CHAT_PATH, Reply(events=chat_text_events()))
    with rig.client() as client:
        stream = open_chat(client)
        through_the_stop_chunk(stream)
        assert rig.raw() == []
        stream.close()
        stream.close()
        with contextlib.suppress(Exception):
            list(stream)  # the closed response may refuse more reading; nothing more is recorded
    (line,) = rig.lines()
    assert line["norm"]["response"] == "Hello"
    assert line["response"]["choices"][0]["finish_reason"] == "stop"
    assert "usage" not in line["response"]  # the usage chunk was never read
    assert rig.writer.counts["dropped_partial"] == 0


def test_leaving_a_with_block_after_the_stop_chunk_records_it_and_a_later_read_adds_nothing(
    rig: Any,
) -> None:
    rig.server.on(CHAT_PATH, Reply(events=chat_text_events()))
    with rig.client() as client:
        with open_chat(client) as stream:
            through_the_stop_chunk(stream)
        with contextlib.suppress(Exception):
            list(stream)
    assert len(rig.lines()) == 1


def test_letting_go_of_a_stream_records_it_only_when_it_had_finished(rig: Any) -> None:
    rig.server.on(CHAT_PATH, Reply(events=chat_text_events()))
    with rig.client() as client:
        finished = open_chat(client)
        through_the_stop_chunk(finished)
        del finished
        gc.collect()
        assert len(rig.lines()) == 1
        unfinished = open_chat(client)
        next(iter(unfinished))
        del unfinished
        gc.collect()
    assert len(rig.lines()) == 1
    assert rig.writer.counts["dropped_partial"] == 1


def test_a_truncated_chat_stream_is_not_a_200_row(rig: Any) -> None:
    rig.server.on(CHAT_PATH, Reply(events=[*chat_text_events()[:3], (None, "[DONE]")]))
    with rig.client() as client:
        assert len(list(open_chat(client))) == 3
    assert rig.raw() == []
    assert rig.writer.counts["dropped_partial"] == 1


def test_a_chat_stream_with_one_unfinished_choice_is_not_complete(rig: Any) -> None:
    def chunk(index: int, delta: Any, finish: Any = None) -> Event:
        choice = {"index": index, "delta": delta, "logprobs": None, "finish_reason": finish}
        body = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": MODEL}
        return None, dict(body, choices=[choice])

    events: List[Event] = [
        chunk(0, {"role": "assistant", "content": "a"}, "stop"),
        chunk(1, {"role": "assistant", "content": "b"}),
        (None, "[DONE]"),
    ]
    rig.server.on(CHAT_PATH, Reply(events=events))
    with rig.client() as client:
        list(open_chat(client, n=2))
    assert rig.raw() == []
    assert rig.writer.counts["dropped_partial"] == 1


def test_a_responses_stream_closed_after_its_terminal_event_is_recorded_once(rig: Any) -> None:
    rig.server.on(RESPONSES_PATH, Reply(events=response_events()))
    with rig.client() as client:
        stream = client.responses.create(model=MODEL, input="Hi", stream=True)
        iterator = iter(stream)
        for _ in range(3):
            next(iterator)
        stream.close()
        with contextlib.suppress(Exception):
            list(stream)
    (line,) = rig.lines()
    assert line["norm"]["response"] == "Hello there"


def test_a_responses_stream_abandoned_before_its_terminal_event_is_counted(rig: Any) -> None:
    rig.server.on(RESPONSES_PATH, Reply(events=response_events()))
    with rig.client() as client:
        with client.responses.create(model=MODEL, input="Hi", stream=True) as stream:
            next(iter(stream))
        stream2 = client.responses.create(model=MODEL, input="Hi", stream=True)
        assert len(list(stream2)) == 3
    assert len(rig.lines()) == 1
    assert rig.writer.counts["dropped_partial"] == 1


def test_a_responses_stream_that_ends_without_a_terminal_event_is_counted(rig: Any) -> None:
    rig.server.on(RESPONSES_PATH, Reply(events=response_events()[:2]))
    with rig.client() as client:
        assert len(list(client.responses.create(model=MODEL, input="Hi", stream=True))) == 2
    assert rig.raw() == []
    assert rig.writer.counts["dropped_partial"] == 1
    assert rig.writer.counts["dropped_error"] == 0


def test_an_async_stream_is_recorded_when_closed_after_the_stop_chunk_and_counted_before(
    rig: Any,
) -> None:
    rig.server.on(CHAT_PATH, Reply(events=chat_text_events()))

    async def early(client: openai.AsyncOpenAI) -> None:
        async with await open_chat(client) as stream:
            async for _chunk in stream:
                break

    run(early, rig)
    assert rig.raw() == []
    assert rig.writer.counts["dropped_partial"] == 1

    async def late(client: openai.AsyncOpenAI) -> None:
        stream = await open_chat(client)
        seen = 0
        async for _chunk in stream:
            seen += 1
            if seen == 4:
                break
        await stream.close()
        await stream.close()

    run(late, rig)
    (line,) = rig.lines()
    assert line["norm"]["response"] == "Hello"
    rig.server.on(CHAT_PATH, Reply(events=[(None, {"error": {"message": "bad"}})]))

    async def failing(client: openai.AsyncOpenAI) -> None:
        stream = await open_chat(client)
        with pytest.raises(openai.APIError):
            async for _chunk in stream:
                pass
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()

    run(failing, rig)
    assert [row["response"] for row in rig.lines()][-1] == {"error": {"type": "APIError"}}


def test_a_cancelled_async_call_propagates_cancellation_and_is_counted_not_failed(
    rig: Any,
) -> None:
    rig.server.on(CHAT_PATH, Reply(body={"id": "slow"}, delay=0.4))

    async def call(client: openai.AsyncOpenAI) -> None:
        task = asyncio.ensure_future(client.chat.completions.create(model=MODEL, messages=MESSAGES))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)

    run(call, rig)
    assert rig.raw() == []
    assert rig.writer.counts["dropped_partial"] == 1
    assert rig.writer.counts["dropped_error"] == 0


def test_an_interrupted_sync_call_propagates_and_is_counted_not_failed(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupt(*args: Any, **kwargs: Any) -> None:
        raise KeyboardInterrupt

    with rig.client() as client:
        monkeypatch.setattr(client.chat.completions, "_post", interrupt)
        with pytest.raises(KeyboardInterrupt):
            client.chat.completions.create(model=MODEL, messages=MESSAGES)
    assert rig.raw() == []
    assert rig.writer.counts["dropped_partial"] == 1


def test_an_interrupt_in_the_transport_mid_stream_propagates_and_leaves_state_consistent(
    rig: Any,
) -> None:
    rig.server.on(CHAT_PATH, Reply(events=chat_text_events()))

    def interrupted() -> Any:
        yield None
        raise KeyboardInterrupt

    with rig.client() as client:
        stream = open_chat(client)
        stream._stream._iterator = interrupted()
        iterator = iter(stream)
        next(iterator)
        with pytest.raises(KeyboardInterrupt):
            next(iterator)
        with pytest.raises(StopIteration):  # the generator is finished; nothing is recorded again
            next(iterator)
    assert rig.raw() == []
    assert rig.writer.counts["dropped_partial"] == 1
    assert rig.writer.counts["dropped_error"] == 0


def test_tool_call_deltas_without_an_index_are_split_by_id_and_name(rig: Any) -> None:
    def delta(**call: Any) -> Event:
        choice = {"index": 0, "delta": {"tool_calls": [call]}, "finish_reason": None}
        body = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": MODEL}
        return None, dict(body, choices=[choice])

    stop = {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
    last = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": MODEL}
    events: List[Event] = [
        delta(id="a", type="function", function={"name": "first", "arguments": ""}),
        delta(function={"arguments": '{"x"'}),
        delta(function={"arguments": ":1}"}),
        delta(id="b", type="function", function={"arguments": ""}),  # a new id: a second call
        delta(function={"name": "second", "arguments": '{"y":2}'}),  # its name arrives later
        delta(function={"name": "third", "arguments": "{}"}),  # a new name: a third call
        (None, dict(last, choices=[stop])),
        (None, "[DONE]"),
    ]
    rig.server.on(CHAT_PATH, Reply(events=events))
    with rig.client() as client:
        list(open_chat(client))
    (line,) = rig.lines()
    calls = line["response"]["choices"][0]["message"]["tool_calls"]
    assert [(c["id"], c["function"]["name"], c["function"]["arguments"]) for c in calls] == [
        ("a", "first", '{"x":1}'),
        ("b", "second", '{"y":2}'),
        (None, "third", "{}"),
    ]
    assert [c["function"]["name"] for c in line["norm"]["tool_calls"]] == [
        "first",
        "second",
        "third",
    ]


def test_the_request_is_taken_at_call_time_not_when_the_stream_ends(rig: Any) -> None:
    rig.server.on(CHAT_PATH, Reply(events=chat_text_events()))
    messages: List[Any] = [{"role": "user", "content": "before"}]
    with rig.client() as client:
        stream = client.chat.completions.create(model=MODEL, messages=messages, stream=True)
        messages[0]["content"] = "after"
        messages.append({"role": "user", "content": "extra"})
        list(stream)
    (line,) = rig.lines()
    assert line["request"]["messages"] == [{"role": "user", "content": "before"}]
