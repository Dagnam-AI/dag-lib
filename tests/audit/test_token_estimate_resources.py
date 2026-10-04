"""Memory and time of the estimate on text built to be costly: one symbol, one emoji, underscores,
alternating quotes, spaces, and a long word.

A symbol run once cost about a gigabyte for each megabyte of symbols, because the pattern kept a
stack frame for every character. The pieces are streamed now, and a run of any one kind is
priced without being copied or listed.
"""

from __future__ import annotations

import time
import tracemalloc
import unicodedata

import pytest

from dagnam.audit.token_estimate import clear_caches, estimate
from dagnam.audit.token_tables import student_chars, student_fragile, student_words

TWO_MB = 2 * (1 << 20)
PEAK_BOUND = 64 * 1024
"""Bytes: what a call may allocate beyond the text itself. Measured about 5 KiB."""


def peak_of(text: str) -> tuple[int, int]:
    """``(estimate, peak traced bytes)`` of one call; the text is built before tracing starts."""
    estimate("warm up the tables and the caches")
    tracemalloc.start()
    try:
        tokens = estimate(text)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return tokens, peak


@pytest.mark.parametrize(
    ("name", "text", "tokens"),
    [
        ("one symbol", "=" * TWO_MB, 699_051),
        ("one emoji", "\U0001f600" * TWO_MB, 2_097_152),
        ("underscores", "_" * TWO_MB, 699_051),
        ("alternating quotes", "\"'" * (TWO_MB // 2), 2_097_151),
        ("spaces", " " * TWO_MB, 16_385),
    ],
    ids=["symbol", "emoji", "underscores", "quotes", "spaces"],
)
def test_two_megabytes_of_one_kind_of_character_allocate_a_few_kilobytes(
    name: str, text: str, tokens: int
) -> None:
    got, peak = peak_of(text)
    assert got == tokens, name
    assert peak < PEAK_BOUND, (name, peak)


def test_a_long_word_amid_other_text_is_held_a_few_times_and_no_more() -> None:
    # The one thing that grows with the input: a run of letters is matched whole, and read as a
    # word, so it is copied (twice). That is bounded by the word, not the text.
    word = "a" * TWO_MB
    got, peak = peak_of("x " + word + " y")
    assert got > 1_000_000
    assert peak < 4 * len(word)


def test_alternating_case_in_one_word_does_not_list_its_seams() -> None:
    # Counting the lower-to-upper seams of a long word used to build a list of them (54 MB for
    # 2 MB); a word over a thousand letters is streamed.
    _, peak = peak_of("x " + "aB" * (TWO_MB // 4))
    assert peak < 4 * TWO_MB


@pytest.mark.parametrize(
    ("name", "unit"),
    [
        ("one symbol", "="),
        ("spaces", " "),
        (
            "mixed text",
            'The parcel arrived late and the box was damaged, so 我们想 a refund {"id": 1}\n',
        ),
    ],
    ids=["symbol", "spaces", "mixed"],
)
def test_time_is_linear_in_the_length_of_the_text(name: str, unit: str) -> None:
    # CPU seconds for 1, 2 and 4 MB: a quadratic estimate takes sixteen times as long for four
    # times the text, a linear one four; the bound is ten.
    estimate("warm up")
    seconds = []
    for megabytes in (1, 2, 4):
        text = unit * (megabytes * (1 << 20) // len(unit.encode("utf-8")))
        start = time.process_time()
        estimate(text)
        seconds.append(time.process_time() - start)
    assert seconds[2] < 10 * max(seconds[0], 0.02), (name, seconds)
    assert seconds[1] < 5 * max(seconds[0], 0.02), (name, seconds)


def test_the_first_use_of_the_tables_takes_milliseconds_and_a_few_megabytes() -> None:
    # Both tables decode into sets on first use (about 9 ms and 6 ms, 13 MB resident in a fresh
    # process). Cleared and decoded again here, with the allocations traced.
    clear_caches()
    try:
        tracemalloc.start()
        started = time.perf_counter()
        words = student_words()
        chars = student_chars()
        fragile = student_fragile()
        seconds = time.perf_counter() - started
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        clear_caches()
    assert (len(words), len(chars), len(fragile)) == (49_926, 39_637, 678)
    assert seconds < 1.0
    assert current < 30 * (1 << 20)
    assert peak < 40 * (1 << 20)


@pytest.mark.parametrize("megabytes", [2, 4])
def test_text_in_nfd_form_costs_one_normalised_copy_and_no_more(megabytes: int) -> None:
    # The copy NFC needs is the only allocation: about 4.6 bytes for each byte of input at
    # peak (the conversion buffer), linear in the input.
    unit = unicodedata.normalize("NFD", "Tất cả mọi người sinh ra đều được tự do. ")
    text = unit * (megabytes * (1 << 20) // len(unit.encode("utf-8")))
    tokens, peak = peak_of(text)
    assert tokens == estimate(unicodedata.normalize("NFC", text))
    assert peak < 6 * len(text.encode("utf-8"))
