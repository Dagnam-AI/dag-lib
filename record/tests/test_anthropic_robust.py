"""The Anthropic patch never changes the call: broken recording, huge lines, many threads."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, List

from _anthropic_bodies import MODEL, PATH, message_body, text_events
from _anthropic_calls import ARGS, run
from _mock_server import Reply
import anthropic
import pytest

from dagnam_record import _anthropic, _anthropic_assemble, _anthropic_body, _openai, _patch
from dagnam_record._envelope import LineTooLargeError
from dagnam_record._recorder import Recorder


def boom(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("the recorder has a bug")


def test_a_recorder_bug_never_changes_a_result(arig: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    arig.server.on(PATH, Reply(body=message_body()))
    monkeypatch.setattr(_anthropic_body, "to_plain", boom)
    with arig.anthropic() as client:
        result = client.messages.create(**ARGS)
    assert result.model_dump()["content"][0]["text"] == "Hello there"
    assert arig.raw() == []
    assert arig.writer.counts["dropped_error"] == 1


def test_a_response_the_recorder_cannot_convert_never_changes_the_result(
    arig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    arig.server.on(PATH, Reply(body=message_body()))
    monkeypatch.setattr(_anthropic_body, "omit_media", boom)
    with arig.anthropic() as client:
        assert client.messages.create(**ARGS).stop_reason == "end_turn"
    assert arig.raw() == []
    assert arig.writer.counts["dropped_error"] == 1


def test_a_recorder_bug_never_masks_the_sdks_own_exception(
    arig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    arig.server.on(PATH, Reply(429, {"type": "error", "error": {"type": "rate_limit_error"}}))
    monkeypatch.setattr(_anthropic_body, "to_plain", boom)
    with arig.anthropic() as client, pytest.raises(anthropic.RateLimitError):
        client.messages.create(**ARGS)
    assert arig.writer.counts["dropped_error"] == 1


def test_a_recorder_bug_never_changes_a_stream(arig: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    arig.server.on(PATH, Reply(events=text_events()))
    with arig.anthropic() as client:
        plain = [e.model_dump() for e in client.messages.create(stream=True, **ARGS)]
        monkeypatch.setattr(_anthropic_body, "to_plain", boom)
        again = [e.model_dump() for e in client.messages.create(stream=True, **ARGS)]
    assert again == plain
    assert len(arig.lines()) == 1  # only the first stream, recorded before the bug
    assert arig.writer.counts["dropped_error"] == 1


def test_a_stream_that_cannot_be_assembled_at_the_end_is_counted_and_passes_through(
    arig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    arig.server.on(PATH, Reply(events=text_events()))
    monkeypatch.setattr(_anthropic_assemble, "response_body", boom)
    with arig.anthropic() as client:
        assert len(list(client.messages.create(stream=True, **ARGS))) == 7
    assert arig.raw() == []
    assert arig.writer.counts["dropped_error"] == 1


def test_a_stream_that_cannot_be_watched_is_handed_back_untouched_and_counted(
    arig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise TypeError("cannot tee")

    arig.server.on(PATH, Reply(events=text_events()))
    monkeypatch.setattr(_patch, "TeeStream", refuse)
    monkeypatch.setattr(_patch, "AsyncTeeStream", refuse)
    with arig.anthropic() as client:
        stream = client.messages.create(stream=True, **ARGS)
        assert type(stream) is anthropic.Stream
        assert len(list(stream)) == 7

    async def call(client: anthropic.AsyncAnthropic) -> None:
        stream = await client.messages.create(stream=True, **ARGS)
        assert type(stream) is anthropic.AsyncStream

    run(call, arig)
    assert arig.raw() == []
    assert arig.writer.counts["dropped_error"] == 2


def test_cyclic_arguments_never_reach_the_recorders_walk_unbounded(arig: Any) -> None:
    loop: Any = {"role": "user"}
    loop["content"] = loop
    loop["again"] = loop  # two ways back
    with arig.anthropic() as client:
        with pytest.raises((ValueError, RecursionError, TypeError)):
            client.messages.create(model=MODEL, max_tokens=8, messages=[loop])
        manager = client.messages.stream(model=MODEL, max_tokens=8, messages=[loop])
        with pytest.raises((ValueError, RecursionError, TypeError)), manager:
            pytest.fail("the SDK refuses the request when it is opened")
    assert arig.raw() == []
    assert arig.writer.counts["dropped_error"] == 2  # counted once per call, at the snapshot


def test_a_request_that_expands_without_end_is_refused_by_the_node_budget() -> None:
    shared: Any = ["x"]
    for _ in range(60):
        shared = [shared, shared]  # 2**60 paths, a handful of objects
    with pytest.raises(LineTooLargeError):
        _anthropic_body.request_body({"messages": shared})


def test_an_oversized_response_is_dropped_and_counted_not_truncated(arig: Any) -> None:
    big = message_body([{"type": "text", "text": "x" * 1_100_000}])
    arig.server.on(PATH, Reply(body=big))
    with arig.anthropic() as client:
        result = client.messages.create(**ARGS)
    assert len(result.model_dump()["content"][0]["text"]) == 1_100_000
    assert arig.raw() == []
    assert arig.writer.counts["dropped_large"] == 1


def test_concurrent_threads_each_record_their_own_call_once(arig: Any) -> None:
    arig.server.on(PATH, Reply(body=message_body()))
    threads = 8
    barrier = threading.Barrier(threads, timeout=10)
    failures: List[BaseException] = []

    def worker(number: int) -> None:
        try:
            with arig.anthropic() as client:
                barrier.wait()  # every thread is inside the patch at the same moment
                client.messages.create(
                    model=MODEL,
                    max_tokens=8,
                    messages=[{"role": "user", "content": f"call {number}"}],
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
    lines = arig.lines()
    assert sorted(line["request"]["messages"][0]["content"] for line in lines) == sorted(
        f"call {n}" for n in range(threads)
    )
    assert len({line["trace_id"] for line in lines}) == threads


def test_concurrent_async_calls_streams_and_helpers_are_each_recorded_once(arig: Any) -> None:
    arig.server.on(PATH, Reply(events=text_events()))

    async def streamed(client: anthropic.AsyncAnthropic, number: int) -> None:
        stream = await client.messages.create(
            model=MODEL,
            max_tokens=8,
            messages=[{"role": "user", "content": f"s {number}"}],
            stream=True,
        )
        async for _event in stream:
            await asyncio.sleep(0)

    async def helped(client: anthropic.AsyncAnthropic, number: int) -> None:
        messages: Any = [{"role": "user", "content": f"h {number}"}]
        async with client.messages.stream(model=MODEL, max_tokens=8, messages=messages) as stream:
            async for _text in stream.text_stream:
                await asyncio.sleep(0)

    async def main(client: anthropic.AsyncAnthropic) -> None:
        jobs = [streamed(client, n) for n in range(5)] + [helped(client, n) for n in range(5)]
        await asyncio.wait_for(asyncio.gather(*jobs), timeout=30)

    run(main, arig)
    lines = arig.lines()
    assert len(lines) == 10
    assert {line["norm"]["response"] for line in lines} == {"Hello"}
    assert len({line["request"]["messages"][0]["content"] for line in lines}) == 10


def test_both_providers_can_be_installed_together(arig: Any) -> None:
    assert _openai.install(Recorder(arig.writer))
    try:
        arig.server.on(PATH, Reply(body=message_body()))
        with arig.anthropic() as client:
            client.messages.create(**ARGS)
        assert len(arig.lines()) == 1
    finally:
        _openai.uninstall()
    _anthropic.uninstall()
    assert _anthropic.install(arig.recorder)


def test_the_real_writer_thread_records_through_a_whole_run(
    arig: Any, make: Any, tmp_path: Any
) -> None:
    live = make(tmp_path / "live", start=True)
    assert _anthropic.install(Recorder(live))
    arig.server.on(PATH, Reply(body=message_body()))
    with arig.anthropic() as client:
        client.messages.create(**ARGS)
    live.close(timeout=5)
    (path,) = (tmp_path / "live").glob("*.jsonl")
    assert path.read_text(encoding="utf-8").count("\n") == 1
    assert live.counts["written"] == 1
