"""How an Anthropic stream ends: finished, cut, closed, let go of, cancelled, or interrupted."""

from __future__ import annotations

import asyncio
import contextlib
import gc
from typing import TYPE_CHECKING, Any

from _anthropic_bodies import PATH, text_events
from _anthropic_calls import ARGS, run
from _mock_server import Reply
import httpx
import pytest

THROUGH_STOP = text_events("Hel", "lo")  # the SDK swallows the ping: 7 events, the last is the stop
if TYPE_CHECKING:
    import anthropic

READS_TO_STOP = 7
BEFORE_STOP = THROUGH_STOP[:4]


def open_stream(client: Any) -> Any:
    return client.messages.create(stream=True, **ARGS)


def read_to_the_stop(stream: Any) -> Any:
    iterator = iter(stream)
    return [next(iterator) for _ in range(READS_TO_STOP)]


def only_row(arig: Any) -> Any:
    (line,) = arig.lines()
    assert arig.writer.counts["dropped_partial"] == 0
    return line


def test_leaving_a_with_block_early_writes_nothing_and_counts(arig: Any) -> None:
    arig.server.on(PATH, Reply(events=THROUGH_STOP))
    with arig.anthropic() as client:
        with open_stream(client) as stream:
            for _event in stream:
                break
        stream = open_stream(client)
        next(iter(stream))
        stream.close()
    assert arig.raw() == []
    assert arig.writer.counts["dropped_partial"] == 2
    assert arig.writer.counts["dropped_error"] == 0


def test_closing_or_leaving_a_with_after_message_stop_records_the_answer_once(arig: Any) -> None:
    arig.server.on(PATH, Reply(events=THROUGH_STOP))
    with arig.anthropic() as client:
        stream = open_stream(client)
        read_to_the_stop(stream)  # never asked for the end of the iteration
        assert arig.raw() == []
        stream.close()
        stream.close()
        with contextlib.suppress(Exception):
            list(stream)  # nothing more is recorded
        with open_stream(client) as other:
            read_to_the_stop(other)
    assert [line["norm"]["response"] for line in arig.lines()] == ["Hello", "Hello"]
    assert arig.writer.counts["dropped_partial"] == 0


def test_letting_go_of_a_stream_records_it_only_when_it_had_finished(arig: Any) -> None:
    arig.server.on(PATH, Reply(events=THROUGH_STOP))
    with arig.anthropic() as client:
        finished = open_stream(client)
        read_to_the_stop(finished)
        del finished
        gc.collect()
        assert len(arig.lines()) == 1
        unfinished = open_stream(client)
        next(iter(unfinished))
        del unfinished
        gc.collect()
    assert len(arig.lines()) == 1
    assert arig.writer.counts["dropped_partial"] == 1


def test_a_stream_cut_before_message_stop_is_not_a_200_row(arig: Any) -> None:
    arig.server.on(PATH, Reply(events=BEFORE_STOP))
    with arig.anthropic() as client:
        assert len(list(open_stream(client))) == 3
    assert arig.raw() == []
    assert arig.writer.counts["dropped_partial"] == 1
    assert arig.writer.counts["dropped_error"] == 0


def test_a_stream_without_message_start_is_not_a_message(arig: Any) -> None:
    arig.server.on(PATH, Reply(events=[THROUGH_STOP[-1]]))
    with arig.anthropic() as client:
        assert len(list(open_stream(client))) == 1
    assert arig.raw() == []
    assert arig.writer.counts["dropped_partial"] == 1


def test_an_async_stream_is_recorded_when_closed_after_message_stop_and_counted_before(
    arig: Any,
) -> None:
    arig.server.on(PATH, Reply(events=THROUGH_STOP))

    async def early(client: anthropic.AsyncAnthropic) -> None:
        async with await open_stream(client) as stream:
            async for _event in stream:
                break

    run(early, arig)
    assert arig.raw() == []
    assert arig.writer.counts["dropped_partial"] == 1

    async def late(client: anthropic.AsyncAnthropic) -> None:
        stream = await open_stream(client)
        seen = 0
        async for _event in stream:
            seen += 1
            if seen == READS_TO_STOP:
                break
        await stream.close()
        await stream.close()

    run(late, arig)
    (line,) = arig.lines()
    assert line["norm"]["response"] == "Hello"
    assert arig.writer.counts["dropped_partial"] == 1  # still only the early one


def test_a_transport_failure_after_message_stop_records_the_complete_answer(arig: Any) -> None:
    arig.server.on(PATH, Reply(events=THROUGH_STOP, cut=True))
    with arig.anthropic() as client:
        iterator = iter(open_stream(client))
        for _ in range(READS_TO_STOP):
            next(iterator)
        with pytest.raises(httpx.RemoteProtocolError):
            next(iterator)  # the error still reaches the caller unchanged
    line = only_row(arig)
    assert (line["status"], line["norm"]["response"]) == (200, "Hello")


def test_an_async_transport_failure_after_message_stop_records_the_complete_answer(
    arig: Any,
) -> None:
    arig.server.on(PATH, Reply(events=THROUGH_STOP, cut=True))

    async def call(client: anthropic.AsyncAnthropic) -> None:
        iterator = (await open_stream(client)).__aiter__()
        for _ in range(READS_TO_STOP):
            await iterator.__anext__()
        with pytest.raises(httpx.RemoteProtocolError):
            await iterator.__anext__()

    run(call, arig)
    assert only_row(arig)["status"] == 200


def test_a_transport_failure_before_message_stop_is_a_failed_call(arig: Any) -> None:
    arig.server.on(PATH, Reply(events=BEFORE_STOP, cut=True))
    with arig.anthropic() as client, pytest.raises(httpx.RemoteProtocolError):
        list(open_stream(client))
    (line,) = arig.lines()
    assert (line["status"], line["response"]) == (599, {"error": {"type": "RemoteProtocolError"}})


def test_an_interrupt_after_message_stop_records_the_complete_answer(arig: Any) -> None:
    arig.server.on(PATH, Reply(events=THROUGH_STOP))
    with arig.anthropic() as client:
        stream = open_stream(client)
        original = stream._iterator._iterator  # the SDK's generator, inside the watcher

        def interrupted_after_the_stop() -> Any:
            for _ in range(READS_TO_STOP):
                yield next(original)
            raise KeyboardInterrupt

        stream._iterator._iterator = interrupted_after_the_stop()
        iterator = iter(stream)
        for _ in range(READS_TO_STOP):
            next(iterator)
        with pytest.raises(KeyboardInterrupt):
            next(iterator)
        stream.close()
    assert only_row(arig)["norm"]["response"] == "Hello"


def test_an_interrupt_before_message_stop_records_nothing_and_counts(arig: Any) -> None:
    arig.server.on(PATH, Reply(events=THROUGH_STOP))
    with arig.anthropic() as client:
        stream = open_stream(client)
        original = stream._iterator._iterator

        def interrupted_early() -> Any:
            yield next(original)
            raise KeyboardInterrupt

        stream._iterator._iterator = interrupted_early()
        iterator = iter(stream)
        next(iterator)
        with pytest.raises(KeyboardInterrupt):
            next(iterator)
        with pytest.raises(StopIteration):  # finished: nothing is reported again
            next(iterator)
        stream.close()
    assert arig.raw() == []
    assert arig.writer.counts["dropped_partial"] == 1
    assert arig.writer.counts["dropped_error"] == 0


def test_a_cancelled_async_read_after_message_stop_still_records_the_answer_once(
    arig: Any,
) -> None:
    arig.server.on(PATH, Reply(events=THROUGH_STOP, cut=True, hold=1.0))

    async def call(client: anthropic.AsyncAnthropic) -> None:
        async with await open_stream(client) as stream:
            iterator = stream.__aiter__()
            for _ in range(READS_TO_STOP):
                await iterator.__anext__()
            with pytest.raises(asyncio.TimeoutError):  # wait_for cancels the stalled read
                await asyncio.wait_for(iterator.__anext__(), timeout=0.1)

    run(call, arig)
    assert only_row(arig)["norm"]["response"] == "Hello"


def test_a_cancelled_async_read_before_message_stop_records_nothing_and_counts(
    arig: Any,
) -> None:
    arig.server.on(PATH, Reply(events=BEFORE_STOP, cut=True, hold=1.0))

    async def call(client: anthropic.AsyncAnthropic) -> None:
        async with await open_stream(client) as stream:
            iterator = stream.__aiter__()
            for _ in range(3):
                await iterator.__anext__()
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(iterator.__anext__(), timeout=0.1)

    run(call, arig)
    assert arig.raw() == []
    assert arig.writer.counts["dropped_partial"] == 1
    assert arig.writer.counts["dropped_error"] == 0


def test_a_cancelled_async_call_propagates_cancellation_and_is_counted_not_failed(
    arig: Any,
) -> None:
    arig.server.on(PATH, Reply(body={"id": "slow"}, delay=0.4))

    async def call(client: anthropic.AsyncAnthropic) -> None:
        task = asyncio.ensure_future(client.messages.create(**ARGS))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)

    run(call, arig)
    assert arig.raw() == []
    assert arig.writer.counts["dropped_partial"] == 1
    assert arig.writer.counts["dropped_error"] == 0


def test_an_interrupted_sync_call_propagates_and_is_counted_not_failed(
    arig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupt(*args: Any, **kwargs: Any) -> None:
        raise KeyboardInterrupt

    with arig.anthropic() as client:
        monkeypatch.setattr(client.messages, "_post", interrupt)
        with pytest.raises(KeyboardInterrupt):
            client.messages.create(**ARGS)
    assert arig.raw() == []
    assert arig.writer.counts["dropped_partial"] == 1
    assert arig.writer.counts["dropped_error"] == 0
