"""What happens when a watched stream is let go: nothing under a lock, everything at a safe point.

A stream can die at any allocation (the garbage collector runs finalizers on whatever thread
triggered it, possibly inside a writer call that holds the writer's non-reentrant locks). The
hook that learns of the death only queues the stream's watch state; the outcome is delivered
later from a place that holds no writer lock: the next wrapped call, the writer's loop, its
``close()``, ``uninstall()`` and the exit hook. Each outcome is delivered exactly once.
"""

from __future__ import annotations

import gc
import os
import queue
import signal
import threading
from typing import Any, Callable, Iterator, List

from _anthropic_bodies import PATH, text_events
from _anthropic_calls import ARGS, run
from _mock_server import Reply
from _openai_bodies import CHAT_PATH, MODEL, chat_text_events
import pytest
from typing_extensions import override

from dagnam_record import _anthropic, _stream
from dagnam_record._envelope import build_line
from dagnam_record._stream import (
    TeeStream,
    deliver_pending,
    finish_live,
    forget_live,
    instrument,
    live_count,
)

OPENAI_ARGS = {"model": MODEL, "messages": [{"role": "user", "content": "Hi"}], "stream": True}


class Told:
    """A watcher that notes every call."""

    def __init__(self) -> None:
        self.events: List[str] = []

    def item(self, value: Any) -> None:
        """Item."""
        self.events.append("item")

    def done(self) -> None:
        """Done."""
        self.events.append("done")

    def error(self, exc: BaseException) -> None:
        """Error."""
        self.events.append("error")

    def cancelled(self) -> None:
        """Cancelled."""
        self.events.append("cancelled")


def guard(step: Any, *arguments: Any) -> bool:
    step(*arguments)
    return True


class Fake:
    """A synchronous SDK-style stream."""

    def __init__(self, count: int = 2, cyclic: bool = False) -> None:
        self._iterator = iter(range(count))
        self.me = self if cyclic else None  # a cycle: only the collector can free it

    def __next__(self) -> Any:
        return self._iterator.__next__()

    def __iter__(self) -> Iterator[Any]:
        yield from self._iterator

    def close(self) -> None:
        """Close."""


@pytest.fixture(autouse=True)
def quiet() -> Iterator[None]:
    """Start and end each test with nothing pending, and the collector on."""
    deliver_pending()
    yield
    gc.enable()
    deliver_pending()


def test_a_stream_that_dies_tells_the_watcher_nothing_until_a_safe_point() -> None:
    told = Told()
    stream = instrument(Fake(), guard, told, sync=True)
    next(stream)
    del stream
    gc.collect()  # the hook runs here, as it would inside any allocation
    assert told.events == ["item"]
    deliver_pending()
    assert told.events == ["item", "done"]
    deliver_pending()
    gc.collect()
    assert told.events == ["item", "done"]  # exactly once


def test_a_proxy_that_dies_tells_the_watcher_nothing_until_a_safe_point() -> None:
    told = Told()
    proxy = TeeStream(Fake(), guard, told)
    next(proxy)
    del proxy
    assert told.events == ["item"]
    deliver_pending()
    assert told.events == ["item", "done"]


def test_a_stream_that_ended_or_was_closed_is_not_delivered_again() -> None:
    told = Told()
    stream = instrument(Fake(), guard, told, sync=True)
    list(stream)
    stream.close()
    del stream
    deliver_pending()
    finish_live()
    assert told.events == ["item", "item", "done"]


def test_finish_live_ends_every_stream_that_is_still_alive_exactly_once() -> None:
    told = Told()
    stream = instrument(Fake(), guard, told, sync=True)
    proxy = TeeStream(Fake(), guard, told)
    next(stream)
    next(proxy)
    finish_live()
    finish_live()
    stream.close()
    list(proxy)
    del stream, proxy
    deliver_pending()
    assert sorted(told.events) == ["done", "done", "item", "item"]


def test_a_stream_dying_in_a_cycle_is_delivered_by_finish_live_even_before_the_collector_runs() -> (
    None
):
    told = Told()
    gc.disable()

    def make() -> None:
        stream = instrument(Fake(cyclic=True), guard, told, sync=True)
        next(stream)

    make()
    assert told.events == ["item"]
    finish_live()
    assert told.events == ["item", "done"]
    gc.enable()
    gc.collect()
    deliver_pending()
    assert told.events == ["item", "done"]


def test_the_exit_hook_is_registered_last_each_time(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Any] = []

    class FakeAtexit:
        @staticmethod
        def register(function: Callable[[], None]) -> None:
            calls.append(("register", function))

        @staticmethod
        def unregister(function: Callable[[], None]) -> None:
            calls.append(("unregister", function))

    monkeypatch.setattr(_stream, "atexit", FakeAtexit)
    _stream.register_exit()
    _stream.register_exit()
    assert calls == [("unregister", finish_live), ("register", finish_live)] * 2


class Stubborn:
    """Takes the new iterator, refuses ``close`` (no slot), and then refuses to give the old one back."""

    __slots__ = ("__weakref__", "_state")

    def __init__(self) -> None:
        self._state = [iter([1, 2]), 0]

    @property
    def _iterator(self) -> Any:
        return self._state[0]

    @_iterator.setter
    def _iterator(self, value: Any) -> None:
        if self._state[1]:
            raise AttributeError("read-only now")
        self._state[0] = value
        self._state[1] = 1

    def close(self) -> None:
        """Close."""


def test_a_stream_that_cannot_be_given_back_is_disarmed_and_never_reports() -> None:
    told = Told()
    stream = Stubborn()
    with pytest.raises(AttributeError):
        instrument(stream, guard, told, sync=True)
    inside: Any = stream
    assert list(inside._iterator) == [1, 2]  # our reader is stuck in there, passing items on
    stream.close()
    del stream
    gc.collect()
    deliver_pending()
    finish_live()
    assert told.events == []


@pytest.fixture
def clean_arig(arig: Any) -> Any:
    gc.collect()
    arig.server.on(PATH, *[Reply(events=text_events("Hel", "lo")) for _ in range(4)])
    return arig


def finished_dropped_stream(client: Any) -> None:
    """A stream read to message_stop and dropped in a cycle, never closed."""
    stream = client.messages.create(stream=True, **ARGS)
    for event in stream:
        if event.type == "message_stop":
            break


def collect() -> None:
    gc.collect()


def collect_while_holding(writer: Any, action: Callable[[], None]) -> bool:
    """Run ``action`` on another thread while this one holds the writer's gate; True if it finished."""
    worker = threading.Thread(target=action, daemon=True)
    with writer.gate():
        worker.start()
        worker.join(5)
        finished = not worker.is_alive()
    worker.join(5)
    return finished


def test_the_collector_running_while_a_writer_lock_is_held_cannot_block_on_it(
    clean_arig: Any,
) -> None:
    arig = clean_arig
    gc.disable()
    with arig.anthropic() as client:
        finished_dropped_stream(client)
        assert collect_while_holding(arig.writer, collect)
        assert arig.writer.pending().qsize() == 0  # nothing reached the writer yet
        gc.enable()
        assert len(arig.lines()) == 1  # delivered at the writer's drain, once
        assert len(arig.lines()) == 1
    assert arig.writer.counts["dropped_partial"] == 0


def test_a_proxy_dying_while_a_writer_lock_is_held_cannot_block_on_it(rig: Any) -> None:
    rig.server.on(CHAT_PATH, Reply(events=chat_text_events()))
    with rig.client() as client:
        stream = client.chat.completions.create(**OPENAI_ARGS)
        iterator = iter(stream)
        for _ in range(4):
            next(iterator)
        del iterator
        holder = [stream]
        del stream
        assert collect_while_holding(rig.writer, holder.clear)
        assert rig.writer.pending().qsize() == 0
        assert len(rig.lines()) == 1
    assert rig.writer.counts["dropped_partial"] == 0


def test_the_next_wrapped_call_delivers_what_was_deferred(clean_arig: Any) -> None:
    arig = clean_arig
    gc.disable()
    with arig.anthropic() as client:
        finished_dropped_stream(client)
        gc.collect()
        assert arig.writer.pending().qsize() == 0
        client.messages.create(stream=True, **ARGS).close()  # this call is a safe point
        assert arig.writer.pending().qsize() == 1
    assert len(arig.lines()) == 1  # the deferred stream's row (the second call was closed early)


def test_the_writer_loop_delivers_on_an_idle_tick(clean_arig: Any, scripted_queue: Any) -> None:
    arig = clean_arig
    gc.disable()
    with arig.anthropic() as client:
        finished_dropped_stream(client)
        gc.collect()
        assert arig.writer.pending().qsize() == 0
        arig.writer.loop(scripted_queue([queue.Empty, None]))
        assert arig.writer.pending().qsize() == 1
    assert len(arig.lines()) == 1


def test_close_delivers_before_it_stops_taking_lines(clean_arig: Any) -> None:
    arig = clean_arig
    gc.disable()
    with arig.anthropic() as client:
        finished_dropped_stream(client)
        gc.collect()
        arig.writer.close(timeout=5)
    assert len(arig.lines()) == 1
    assert arig.writer.counts["dropped_error"] == 0


def test_uninstall_delivers_what_was_deferred(clean_arig: Any) -> None:
    arig = clean_arig
    gc.disable()
    with arig.anthropic() as client:
        finished_dropped_stream(client)
        gc.collect()
        _anthropic.uninstall()
        assert arig.writer.pending().qsize() == 1
    assert len(arig.lines()) == 1


def test_a_wrapped_async_call_is_a_safe_point_too(clean_arig: Any) -> None:
    arig = clean_arig
    gc.disable()

    async def call(client: Any) -> None:
        await client.messages.create(**ARGS)

    with arig.anthropic() as client:
        finished_dropped_stream(client)
    gc.collect()
    assert arig.writer.pending().qsize() == 0
    arig.server.on(PATH, Reply(body={"id": "m", "content": []}))
    run(call, arig)
    assert arig.writer.pending().qsize() == 2  # the call's own row and the deferred stream's


def test_the_writer_loop_delivers_after_each_line_it_writes(
    clean_arig: Any, scripted_queue: Any
) -> None:
    arig = clean_arig
    gc.disable()
    with arig.anthropic() as client:
        finished_dropped_stream(client)
        gc.collect()
        line = build_line(
            provider="other", endpoint="custom", request={}, response={}, latency_ms=1.0
        )
        arig.writer.loop(scripted_queue([line, None]))
        assert arig.writer.pending().qsize() == 1
    assert len(arig.lines()) == 2  # the line the loop wrote and the delivered stream's row


def test_the_registry_holds_a_stream_only_until_it_has_reported() -> None:
    before = live_count()
    told = Told()
    stream = instrument(Fake(), guard, told, sync=True)
    proxy = TeeStream(Fake(), guard, told)
    assert live_count() == before + 2
    list(stream)
    proxy.close()
    assert live_count() == before


def test_a_forked_child_forgets_the_streams_of_its_parent() -> None:
    told = Told()
    stream = instrument(Fake(), guard, told, sync=True)
    next(stream)
    forget_live()
    finish_live()
    assert told.events == ["item"]  # the parent's stream is not the child's to report
    del stream


def test_creating_a_stream_helper_is_a_safe_point_too(clean_arig: Any) -> None:
    arig = clean_arig
    gc.disable()
    with arig.anthropic() as client:
        finished_dropped_stream(client)
        gc.collect()
        assert arig.writer.pending().qsize() == 0
        client.messages.stream(**ARGS)  # nothing is asked of the server yet
        assert arig.writer.pending().qsize() == 1


def test_a_guard_that_raises_cannot_make_delivery_raise() -> None:
    class Failing(Told):
        @override
        def done(self) -> None:
            """Fail."""
            raise RuntimeError("bug")

    def raising(step: Any, *arguments: Any) -> bool:
        step(*arguments)
        return True

    stream = instrument(Fake(), raising, Failing(), sync=True)
    next(stream)
    del stream
    gc.collect()
    deliver_pending()  # does not raise
    finish_live()


def test_a_proxy_whose_watch_cannot_be_queued_at_shutdown_is_silent() -> None:
    class Gone:
        def expire(self) -> None:
            raise AttributeError("module globals are None at shutdown")

    proxy: Any = TeeStream(Fake(), guard, Told())
    proxy._watch = Gone()
    del proxy  # would print "Exception ignored in __del__" if it raised
    gc.collect()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_a_forked_child_neither_reports_nor_collects_into_a_report_of_an_inherited_stream() -> None:
    told = Told()
    own = Told()
    held = [instrument(Fake(), guard, told, sync=True)]
    next(held[0])
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            signal.alarm(10)  # a child that hangs is ended by the alarm
            held.clear()  # the child lets go of the parent's stream and collects it
            gc.collect()
            deliver_pending()
            finish_live()
            mine = instrument(Fake(), guard, own, sync=True)
            list(mine)
            finish_live()
            code = 0 if (told.events == ["item"] and own.events == ["item", "item", "done"]) else 2
        finally:
            os._exit(code)
    _, wait_status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(wait_status) == 0
    assert told.events == ["item"]  # the parent has not let go yet
    held.clear()
    gc.collect()
    deliver_pending()
    finish_live()
    assert told.events == ["item", "done"]  # the parent reports it, once
