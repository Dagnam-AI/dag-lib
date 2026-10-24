"""The recorder never makes the program wait, and its cost stays small and bounded.

The calls go through the real SDKs over an in-process transport (no socket), so thousands of them
are quick and the time measured is the SDK's own plus the recorder's.
"""

from __future__ import annotations

from pathlib import Path
import statistics
import threading
import time
import tracemalloc
from typing import Iterator

from _guarantee_support import flood, quick
from _openai_bodies import chat_body
import httpx
import openai
import pytest

import dagnam_record
from dagnam_record import _state
from dagnam_record._writer import Writer

pytestmark = pytest.mark.usefixtures("fresh_state")
CALLS = 10_000
FLOOD = 50_000
BOUND = 120.0
"""Seconds that thousands of calls may take with a stuck writer: far above an honest run."""
QUEUE = 1000


def client() -> openai.OpenAI:
    """A real client whose transport answers in-process."""

    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=chat_body())

    return openai.OpenAI(
        api_key="k",
        base_url="http://127.0.0.1:9/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(answer), trust_env=False),
    )


def call(sdk: openai.OpenAI) -> None:
    sdk.chat.completions.create(model="m", messages=[{"role": "user", "content": "Hi"}])


@pytest.fixture
def stuck_writer(monkeypatch: pytest.MonkeyPatch) -> Iterator[threading.Event]:
    """Every writer thread blocks on its first line until the event is set."""
    release = threading.Event()

    def blocked(self: Writer, line: str) -> None:
        release.wait(15)

    monkeypatch.setattr(Writer, "_write", blocked)
    yield release
    release.set()
    dagnam_record.uninstall()


def test_ten_thousand_patched_calls_with_a_stuck_writer_finish_and_drop_what_does_not_fit(
    tmp_path: Path, stuck_writer: threading.Event
) -> None:
    dagnam_record.install(dir=str(tmp_path / "traces"))
    installed = _state.STATE.installed
    assert installed is not None
    sdk = client()
    began = time.perf_counter()
    for number in range(CALLS):
        quick(lambda: call(sdk), f"patched call number {number}")
    assert time.perf_counter() - began < BOUND
    counts = installed.writer.counts
    assert counts["dropped_full"] >= CALLS - QUEUE - 1  # one line is in the stuck thread's hand
    assert counts["written"] == 0


def test_a_patched_call_adds_under_five_milliseconds_at_the_median_even_on_a_slow_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def slow(self: Writer, line: str) -> None:
        time.sleep(0.01)  # a write that takes 10 ms: a synchronous recorder would pay it per call

    sdk = client()
    plain = []
    for _ in range(200):
        began = time.perf_counter()
        call(sdk)
        plain.append(time.perf_counter() - began)
    monkeypatch.setattr(Writer, "_write", slow)
    dagnam_record.install(dir=str(tmp_path / "traces"))
    recorded = []
    for _ in range(200):
        began = time.perf_counter()
        call(sdk)
        recorded.append(time.perf_counter() - began)
    added = statistics.median(recorded) - statistics.median(plain)
    assert added < 0.005, f"a recorded call took {added * 1000:.2f} ms longer at the median"


def test_fifty_thousand_record_calls_with_a_stuck_writer_stay_bounded_in_time_and_memory(
    tmp_path: Path, stuck_writer: threading.Event
) -> None:
    dagnam_record.install(dir=str(tmp_path / "traces"))
    installed = _state.STATE.installed
    assert installed is not None
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        began = time.perf_counter()
        flood(FLOOD)
        elapsed = time.perf_counter() - began
        grown = tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()
    assert elapsed < BOUND
    assert grown < 20_000_000  # about 1000 queued lines: bytes each, not 50,000 of them
    counts = installed.writer.counts
    assert counts["dropped_full"] >= FLOOD - QUEUE - 1
