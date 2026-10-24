"""The patch never changes the call: broken recording, huge lines, raw responses, many threads."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, List

from _mock_server import Reply
from _openai_bodies import CHAT_PATH, MODEL, chat_body, chat_text_events
from _openai_calls import run
import openai
import pytest

from dagnam_record import _openai, _openai_body
from dagnam_record._recorder import Recorder

MESSAGES: List[Any] = [{"role": "user", "content": "Hi"}]


def boom(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("the recorder has a bug")


def test_a_recorder_bug_never_changes_a_result(rig: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    rig.server.on(CHAT_PATH, Reply(body=chat_body()))
    monkeypatch.setattr(_openai_body, "to_plain", boom)
    with rig.client() as client:
        result = client.chat.completions.create(model=MODEL, messages=MESSAGES)
    assert result.choices[0].message.content == "Hello there"
    assert rig.raw() == []
    assert rig.writer.counts["dropped_error"] == 1


def test_a_recorder_bug_never_masks_the_sdks_own_exception(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig.server.on(CHAT_PATH, Reply(429, {"error": {"message": "slow"}}))
    monkeypatch.setattr(_openai_body, "to_plain", boom)
    with rig.client() as client, pytest.raises(openai.RateLimitError):
        client.chat.completions.create(model=MODEL, messages=MESSAGES)
    assert rig.writer.counts["dropped_error"] == 1


def test_a_recorder_bug_never_changes_a_stream(rig: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    rig.server.on(CHAT_PATH, Reply(events=chat_text_events()))
    with rig.client() as client:
        plain = list(client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True))
        monkeypatch.setattr(_openai_body, "to_plain", boom)
        again = list(client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True))
    assert again == plain
    assert len(rig.lines()) == 1  # only the first stream, recorded before the bug
    assert rig.writer.counts["dropped_error"] == 1


def test_cyclic_arguments_are_refused_in_a_blink_and_the_sdks_own_error_is_raised(
    rig: Any,
) -> None:
    loop: Any = {"role": "user"}
    loop["content"] = loop
    loop["again"] = loop  # two ways back: a depth cap alone would take 2**64 steps
    started = time.perf_counter()
    with rig.client() as client, pytest.raises((ValueError, RecursionError, TypeError)):
        client.chat.completions.create(model=MODEL, messages=[loop])
    assert time.perf_counter() - started < 10
    assert rig.raw() == []
    assert rig.writer.counts["dropped_error"] == 1  # counted once, at the snapshot


def test_an_oversized_response_is_dropped_and_counted_not_truncated(rig: Any) -> None:
    rig.server.on(CHAT_PATH, Reply(body=chat_body("x" * 1_100_000)))
    with rig.client() as client:
        result = client.chat.completions.create(model=MODEL, messages=MESSAGES)
    assert len(result.choices[0].message.content or "") == 1_100_000
    assert rig.raw() == []
    assert rig.writer.counts["dropped_large"] == 1


@pytest.mark.parametrize("access", ["with_raw_response", "with_streaming_response"])
def test_raw_and_streaming_response_helpers_work_and_are_counted_not_recorded(
    rig: Any, access: str
) -> None:
    rig.server.on(CHAT_PATH, Reply(body=chat_body()))
    with rig.client() as client:
        helper = getattr(client.chat.completions, access)
        if access == "with_raw_response":
            parsed = helper.create(model=MODEL, messages=MESSAGES).parse()
        else:
            with helper.create(model=MODEL, messages=MESSAGES) as response:
                parsed = response.parse()
    assert parsed.choices[0].message.content == "Hello there"
    assert rig.raw() == []
    assert rig.writer.counts["dropped_error"] == 1


def test_concurrent_threads_each_record_their_own_call_once(rig: Any) -> None:
    rig.server.on(CHAT_PATH, Reply(body=chat_body()))
    threads = 8
    barrier = threading.Barrier(threads, timeout=10)
    failures: List[BaseException] = []

    def worker(number: int) -> None:
        try:
            with rig.client() as client:
                barrier.wait()  # every thread is inside the patch at the same moment
                client.chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": f"call {number}"}]
                )
        except BaseException as exc:
            failures.append(exc)

    pool = [threading.Thread(target=worker, args=(n,)) for n in range(threads)]
    for thread in pool:
        thread.start()
    for thread in pool:
        thread.join(30)
        assert not thread.is_alive()
    assert failures == []
    lines = rig.lines()
    assert sorted(line["request"]["messages"][0]["content"] for line in lines) == sorted(
        f"call {n}" for n in range(threads)
    )
    assert len({line["trace_id"] for line in lines}) == threads


def test_concurrent_async_calls_and_streams_are_each_recorded_once(rig: Any) -> None:
    rig.server.on(CHAT_PATH, Reply(body=chat_body()))

    async def plain(client: openai.AsyncOpenAI, number: int) -> None:
        await client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": f"plain {number}"}]
        )

    async def streamed(client: openai.AsyncOpenAI, number: int) -> None:
        stream = await client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": f"stream {number}"}], stream=True
        )
        async for _chunk in stream:
            await asyncio.sleep(0)

    async def main(client: openai.AsyncOpenAI) -> None:
        await asyncio.wait_for(asyncio.gather(*[plain(client, n) for n in range(6)]), timeout=30)

    run(main, rig)
    assert len(rig.lines()) == 6
    rig.server.on(CHAT_PATH, Reply(events=chat_text_events()))

    async def streams(client: openai.AsyncOpenAI) -> None:
        await asyncio.wait_for(asyncio.gather(*[streamed(client, n) for n in range(6)]), timeout=30)

    run(streams, rig)
    lines = rig.lines()
    assert len(lines) == 12
    assert {line["norm"]["response"] for line in lines[6:]} == {"Hello"}


def test_the_real_writer_thread_records_through_a_whole_run(
    rig: Any, make: Any, tmp_path: Any
) -> None:
    live = make(tmp_path / "live", start=True)
    assert _openai.install(Recorder(live))
    rig.server.on(CHAT_PATH, Reply(body=chat_body()))
    with rig.client() as client:
        client.chat.completions.create(model=MODEL, messages=MESSAGES)
    live.close(timeout=5)
    (path,) = (tmp_path / "live").glob("*.jsonl")
    assert path.read_text(encoding="utf-8").count("\n") == 1
    assert live.counts["written"] == 1
