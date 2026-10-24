"""A stream cut, cancelled or interrupted after its end marker is still a complete answer."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, List

from _mock_server import Event, Reply
from _openai_bodies import (
    CHAT_PATH,
    MODEL,
    RESPONSES_PATH,
    chat_text_events,
    response_events,
)
from _openai_calls import run
import httpx
import pytest

if TYPE_CHECKING:
    import openai

MESSAGES: List[Any] = [{"role": "user", "content": "Hi"}]
THROUGH_STOP = chat_text_events(usage=False)[:4]  # role, two deltas, the chunk with finish_reason
THROUGH_COMPLETED = response_events()  # created, one delta, completed
BEFORE_STOP = chat_text_events()[:2]
BEFORE_COMPLETED = response_events()[:2]


def open_chat(client: Any) -> Any:
    return client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True)


def open_responses(client: Any) -> Any:
    return client.responses.create(model=MODEL, input="Hi", stream=True)


def only_row(rig: Any) -> Any:
    (line,) = rig.lines()
    assert rig.writer.counts["dropped_partial"] == 0
    return line


@pytest.mark.parametrize(
    ("path", "events", "opener", "reads"),
    [
        (CHAT_PATH, THROUGH_STOP, open_chat, 4),
        (RESPONSES_PATH, THROUGH_COMPLETED, open_responses, 3),
    ],
)
def test_a_transport_failure_after_the_end_marker_records_the_complete_answer(
    rig: Any, path: str, events: List[Event], opener: Any, reads: int
) -> None:
    rig.server.on(path, Reply(events=events, cut=True))
    with rig.client() as client:
        iterator = iter(opener(client))
        for _ in range(reads):
            next(iterator)
        with pytest.raises(httpx.RemoteProtocolError):
            next(iterator)  # the error still reaches the caller unchanged
    line = only_row(rig)
    assert line["status"] == 200
    assert line["norm"]["response"] in ("Hello", "Hello there")


@pytest.mark.parametrize(
    ("path", "events", "opener", "reads"),
    [
        (CHAT_PATH, THROUGH_STOP, open_chat, 4),
        (RESPONSES_PATH, THROUGH_COMPLETED, open_responses, 3),
    ],
)
def test_an_async_transport_failure_after_the_end_marker_records_the_complete_answer(
    rig: Any, path: str, events: List[Event], opener: Any, reads: int
) -> None:
    rig.server.on(path, Reply(events=events, cut=True))

    async def call(client: openai.AsyncOpenAI) -> None:
        iterator = (await opener(client)).__aiter__()
        for _ in range(reads):
            await iterator.__anext__()
        with pytest.raises(httpx.RemoteProtocolError):
            await iterator.__anext__()

    run(call, rig)
    assert only_row(rig)["status"] == 200


@pytest.mark.parametrize(
    ("path", "events", "opener"),
    [(CHAT_PATH, BEFORE_STOP, open_chat), (RESPONSES_PATH, BEFORE_COMPLETED, open_responses)],
)
def test_a_transport_failure_before_the_end_marker_is_still_a_failed_call(
    rig: Any, path: str, events: List[Event], opener: Any
) -> None:
    rig.server.on(path, Reply(events=events, cut=True))
    with rig.client() as client, pytest.raises(httpx.RemoteProtocolError):
        list(opener(client))
    (line,) = rig.lines()
    assert (line["status"], line["response"]) == (599, {"error": {"type": "RemoteProtocolError"}})


def test_an_interrupt_after_the_end_marker_records_the_complete_answer(rig: Any) -> None:
    rig.server.on(CHAT_PATH, Reply(events=chat_text_events()))
    with rig.client() as client:
        stream = open_chat(client)
        original = stream._stream._iterator

        def interrupted_after_the_stop_chunk() -> Any:
            for _ in range(4):
                yield next(original)
            raise KeyboardInterrupt

        stream._stream._iterator = interrupted_after_the_stop_chunk()
        iterator = iter(stream)
        for _ in range(4):
            next(iterator)
        with pytest.raises(KeyboardInterrupt):
            next(iterator)
        stream.close()
    assert only_row(rig)["norm"]["response"] == "Hello"


def test_an_interrupt_before_the_end_marker_records_nothing_and_counts(rig: Any) -> None:
    rig.server.on(CHAT_PATH, Reply(events=chat_text_events()))
    with rig.client() as client:
        stream = open_chat(client)
        original = stream._stream._iterator

        def interrupted_early() -> Any:
            yield next(original)
            raise KeyboardInterrupt

        stream._stream._iterator = interrupted_early()
        iterator = iter(stream)
        next(iterator)
        with pytest.raises(KeyboardInterrupt):
            next(iterator)
        stream.close()
    assert rig.raw() == []
    assert rig.writer.counts["dropped_partial"] == 1


@pytest.mark.parametrize(
    ("path", "events", "opener", "reads", "expected"),
    [
        (CHAT_PATH, THROUGH_STOP, open_chat, 4, "Hello"),
        (RESPONSES_PATH, THROUGH_COMPLETED, open_responses, 3, "Hello there"),
    ],
)
def test_a_cancelled_async_read_after_the_end_marker_still_records_the_answer_once(
    rig: Any, path: str, events: List[Event], opener: Any, reads: int, expected: str
) -> None:
    rig.server.on(path, Reply(events=events, cut=True, hold=1.0))

    async def call(client: openai.AsyncOpenAI) -> None:
        async with await opener(client) as stream:
            iterator = stream.__aiter__()
            for _ in range(reads):
                await iterator.__anext__()
            with pytest.raises(asyncio.TimeoutError):  # wait_for cancels the stalled read
                await asyncio.wait_for(iterator.__anext__(), timeout=0.1)

    run(call, rig)
    assert only_row(rig)["norm"]["response"] == expected


@pytest.mark.parametrize(
    ("path", "events", "opener", "reads"),
    [(CHAT_PATH, BEFORE_STOP, open_chat, 2), (RESPONSES_PATH, BEFORE_COMPLETED, open_responses, 2)],
)
def test_a_cancelled_async_read_before_the_end_marker_records_nothing_and_counts(
    rig: Any, path: str, events: List[Event], opener: Any, reads: int
) -> None:
    rig.server.on(path, Reply(events=events, cut=True, hold=1.0))

    async def call(client: openai.AsyncOpenAI) -> None:
        async with await opener(client) as stream:
            iterator = stream.__aiter__()
            for _ in range(reads):
                await iterator.__anext__()
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(iterator.__anext__(), timeout=0.1)

    run(call, rig)
    assert rig.raw() == []
    assert rig.writer.counts["dropped_partial"] == 1
    assert rig.writer.counts["dropped_error"] == 0


def test_the_complete_answer_of_a_failed_responses_stream_keeps_its_failed_status(
    rig: Any,
) -> None:
    failed = response_events("response.failed")
    rig.server.on(RESPONSES_PATH, Reply(events=failed, cut=True))
    with rig.client() as client:
        iterator = iter(open_responses(client))
        for _ in range(3):
            next(iterator)
        with pytest.raises(httpx.RemoteProtocolError):
            next(iterator)
    (line,) = rig.lines()
    assert (line["status"], line["response"]) == (599, {"error": {"type": "response.failed"}})
