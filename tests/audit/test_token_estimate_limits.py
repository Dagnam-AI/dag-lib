"""The structural limits of the estimate, and a seeded property check of its ceiling.

The estimate starts every character at its UTF-8 byte count and discounts it only where the
vocabulary shows a merge exists, each discount with a limit. These tests pin the limits one by
one (what a word, a whitespace run, an unknown character, a repeated unit costs, however long)
and then check, on seeded random text of every Unicode category and many scripts, the bound they
add up to: the estimate is never above the UTF-8 bytes of the text in NFC form, and a character
the vocabulary does not hold is priced at its bytes. No test here needs the tokenizer: the real
counts are pinned in the other test modules.
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Any
import unicodedata

import pytest

from dagnam.audit import token_estimate
from dagnam.audit.derive import _row_tokens
from dagnam.audit.token_estimate import estimate
from dagnam.audit.token_tables import student_chars


class SpyOnTheWordTable:
    """Stands in for the table of whole words: says ``answer`` and records what it was asked."""

    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.asked: list[str] = []

    def __contains__(self, word: object) -> bool:
        assert isinstance(word, str)
        self.asked.append(word)
        return self.answer


def spy_on_the_word_table(monkeypatch: pytest.MonkeyPatch, answer: bool) -> SpyOnTheWordTable:
    spy = SpyOnTheWordTable(answer)
    monkeypatch.setattr(token_estimate, "student_words", lambda: spy)
    return spy


# --- a word longer than any whole-word token never consults the table --------------------------


@pytest.mark.parametrize(
    ("letter", "longest"),
    [
        ("a", 36),  # Latin: the longest whole-word token has 36 letters
        ("\u0434", 14),  # Cyrillic
        ("\u03b1", 14),  # Greek
        ("\u05d0", 14),  # Hebrew
        ("\u0627", 14),  # Arabic
    ],
    ids=["latin", "cyrillic", "greek", "hebrew", "arabic"],
)
def test_a_word_longer_than_the_longest_whole_word_token_never_consults_the_table(
    letter: str, longest: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = spy_on_the_word_table(monkeypatch, answer=False)
    estimate(" " + letter * longest)
    assert spy.asked == [letter * longest]
    spy.asked.clear()
    estimate(" " + letter * (longest + 1))
    assert spy.asked == []


def test_a_table_that_accepted_every_word_would_still_price_a_long_one_by_its_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The tables are exact, so this cannot happen; it is the limit that would make it harmless.
    # Up to the longest whole-word token a word costs one token; longer, its tail up to the
    # natural length and the floor of random letters (0.65 for Latin) after it: 60 letters cost
    # 35 and 400 cost 256.
    spy_on_the_word_table(monkeypatch, answer=True)
    assert estimate(" " + "a" * 36) == 1
    assert estimate(" " + "a" * 37) == 20
    assert estimate(" " + "a" * 60) == 35
    assert estimate(" " + "a" * 400) == 256
    # Cyrillic: 14 letters at the tail (1.336 + 0.325 * 14), the 15th at 0.9.
    assert estimate(" " + "\u0434" * 15) == 7


# --- whitespace is priced per run of one character --------------------------------------------

LONGEST_TOKEN = {" ": 128, "\n": 32, "\t": 20, "\xa0": 8, "\u3000": 2, "\r\n": 4}
"""The longest token of only that whitespace in the vocabulary, in characters: a run of n of them
cannot take fewer than n divided by this many tokens."""


@pytest.mark.parametrize(
    ("char", "count", "tokens"),
    [
        (" ", 81, 1),  # one token up to 81 spaces
        (" ", 82, 2),  # the second at 82
        (" ", 209, 2),
        (" ", 210, 3),
        (" ", 2048, 17),
        ("\n", 12, 1),
        ("\n", 13, 2),
        ("\n", 28, 2),
        ("\n", 29, 3),
        ("\n", 60, 3),
        ("\n", 61, 4),
        ("\n", 64, 4),
        ("\n", 65, 4),
        ("\n", 4096, 138),
        ("\t", 16, 1),
        ("\t", 17, 2),
        ("\t", 4096, 256),
        ("\xa0", 4, 1),
        ("\xa0", 9, 2),
        ("\u3000", 2, 1),
        ("\u3000", 3, 2),
        ("\r\n", 4, 1),
        ("\r\n", 5, 2),
    ],
)
def test_a_whitespace_run_is_priced_at_the_period_measured_for_its_character(
    char: str, count: int, tokens: int
) -> None:
    assert estimate(char * count) == tokens


@pytest.mark.parametrize("char", list(LONGEST_TOKEN))
def test_a_whitespace_run_is_never_priced_below_what_the_longest_token_allows(char: str) -> None:
    for count in [*range(1, 300), 512, 1000, 2048, 4096]:
        assert estimate(char * count) >= math.ceil(count / LONGEST_TOKEN[char])


def test_whitespace_of_two_kinds_is_priced_by_segment() -> None:
    # 100 spaces take two tokens and 13 newlines take two: 4. Tabs and spaces in one run are two
    # segments, and a newline followed by indentation is two.
    assert estimate(" " * 100 + "\n" * 13) == 4
    assert estimate("\t" * 17 + " " * 82) == 4
    assert estimate("\n" + " " * 50) == 2
    assert estimate("\n" + " " * 95) == 3
    assert estimate("\r\n" * 2 + "\r") == 1
    # The other space characters (thin, hair, em, line separator) are not tokens: their bytes.
    assert estimate("\u2009" * 100) == 300


# --- random letters, beyond the length of a natural word --------------------------------------

FLOORS = {"latin": 0.65, "cyrillic": 0.9, "hebrew": 0.85, "arabic": 1.0, "greek": 1.1}
NATURAL = {"latin": 16, "cyrillic": 14, "hebrew": 8, "arabic": 8, "greek": 36}
ALPHABETS = {
    "latin": (0x61, 0x7A),
    "cyrillic": (0x430, 0x44F),
    "hebrew": (0x5D0, 0x5EA),
    "arabic": (0x621, 0x64A),
    "greek": (0x3B1, 0x3C9),
}


def random_letters(script: str, count: int, seed: int) -> str:
    low, high = ALPHABETS[script]
    rnd = random.Random(seed)
    return "".join(chr(rnd.randint(low, high)) for _ in range(count))


@pytest.mark.parametrize(
    ("script", "count", "expected"),
    [
        ("latin", 416, 266),
        ("cyrillic", 114, 96),
        ("hebrew", 108, 89),
        ("arabic", 108, 108),
        ("greek", 136, 147),
    ],
)
def test_letters_beyond_the_natural_length_cost_the_floor_of_random_letters(
    script: str, count: int, expected: int
) -> None:
    text = random_letters(script, count, seed=20261003)
    got = estimate(text)
    assert got == expected
    assert got >= FLOORS[script] * (count - NATURAL[script])
    assert got <= count * 2  # and far below the bytes of the letters


def test_random_upper_case_letters_are_natural_for_eight_letters_then_the_floor() -> None:
    rnd = random.Random(20261003)
    text = "".join(chr(rnd.randint(0x41, 0x5A)) for _ in range(408))
    # 1.308 + 0.258 * 8 for the acronym-length start, then 0.65 for each of the other 400.
    assert estimate(text) == math.ceil(1.308 + 0.258 * 8 + 0.65 * 400) == 264


# --- characters with no better rule -----------------------------------------------------------


def test_a_character_the_vocabulary_does_not_hold_costs_its_bytes() -> None:
    # Private-use characters are in no vocabulary, and the tables are exact: every one costs its
    # bytes, 400 of 400 drawn.
    rnd = random.Random(7)
    chars = student_chars()
    for low, high in ((0xE000, 0xF8FF), (0xF0000, 0xFFFFD), (0x100000, 0x10FFFD), (0xE000, 0xF8FF)):
        for _ in range(100):
            char = chr(rnd.randint(low, high))
            assert char not in chars
            assert estimate(char) == len(char.encode("utf-8"))
    # In a run, too: 100 private-use characters of 3 bytes, 100 of plane 15 of 4.
    assert estimate("\ue000" * 100) == 300
    assert estimate("\U000f0000" * 100) == 400


def test_a_character_that_is_a_token_costs_one() -> None:
    assert estimate("\u2500" * 100) == 100  # box drawing
    assert estimate("\u3002" * 3) == 3


def test_a_format_character_costs_one_alone_and_one_and_a_half_in_a_run() -> None:
    # Zero-width space is a token; BPE splits it into bytes when it repeats (1.5 a character).
    assert estimate("\u200b") == 1
    assert estimate("\u200b" * 2) == 3
    assert estimate("\u200b" * 1000) == 1500
    # Zero-width joiner is no token: its three bytes.
    assert estimate("\u200d" * 2) == 6


def test_a_lone_surrogate_costs_three_and_never_raises() -> None:
    # The real tokenizer refuses such a string; the estimate must not, and prices each at 3.
    assert estimate("\ud800") == 3
    assert estimate("\ud800" * 10) == 30
    assert estimate("a\udfffb") == 5


def test_a_han_character_that_is_no_token_costs_its_bytes() -> None:
    assert estimate("\U00020000") == 4
    assert estimate("\U00020000" * 10) == 40
    assert estimate("\u9fa6") == 3


# --- the bounds, on seeded random text --------------------------------------------------------


def _code_points_by_category() -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for code in range(0x20, 0x110000):
        if 0xD800 <= code <= 0xDFFF:
            continue
        groups.setdefault(unicodedata.category(chr(code)), []).append(code)
    return groups


CATEGORIES = _code_points_by_category()
SCRIPT_BLOCKS = {
    "latin": (0x41, 0x24F),
    "greek": (0x370, 0x3FF),
    "cyrillic": (0x400, 0x52F),
    "hebrew": (0x591, 0x5F4),
    "arabic": (0x600, 0x6FF),
    "devanagari": (0x900, 0x97F),
    "bengali": (0x980, 0x9FF),
    "thai": (0xE01, 0xE5B),
    "tibetan": (0xF00, 0xFFF),
    "myanmar": (0x1000, 0x109F),
    "hangul": (0xAC00, 0xD7A3),
    "hiragana": (0x3041, 0x3096),
    "katakana": (0x30A1, 0x30FA),
    "han": (0x4E00, 0x9FFF),
    "han-ext-b": (0x20000, 0x2A6DF),
    "emoji": (0x1F300, 0x1F64F),
    "math-alnum": (0x1D400, 0x1D7FF),
    "braille": (0x2800, 0x28FF),
    "box-drawing": (0x2500, 0x257F),
}


def _generated(rnd: random.Random, codes: list[int], variant: str, size: int) -> str:
    if variant == "repeat":
        return chr(rnd.choice(codes)) * size
    if variant == "spaced":
        words = (
            "".join(chr(rnd.choice(codes)) for _ in range(rnd.randint(1, 8)))
            for _ in range(max(1, size // 4))
        )
        return " ".join(words)
    return "".join(chr(rnd.choice(codes)) for _ in range(size))


def _every_pool() -> list[tuple[str, list[int]]]:
    pools = [(f"category {name}", codes) for name, codes in sorted(CATEGORIES.items())]
    for name, (low, high) in SCRIPT_BLOCKS.items():
        codes = [c for c in range(low, high + 1) if unicodedata.category(chr(c)) != "Cn"]
        pools.append((f"script {name}", codes))
    return pools


def nfc_bytes(text: str) -> int:
    return len(unicodedata.normalize("NFC", text).encode("utf-8", "surrogatepass"))


@pytest.mark.parametrize("seed", [7, 20261003])
def test_the_estimate_is_never_above_the_utf8_bytes_of_the_nfc_form_and_never_zero(
    seed: int,
) -> None:
    # A byte-level BPE starts from the bytes of its (NFC) input and only merges, so the real
    # count is at most that; the estimate is capped at it for every piece. Strings of every
    # Unicode category and 19 scripts: a character repeated, a random run, random short words.
    rnd = random.Random(seed)
    checked = 0
    for name, codes in _every_pool():
        for variant in ("run", "spaced", "repeat"):
            for size in (1, 2, 5, 40, 400):
                text = _generated(rnd, codes, variant, size)
                got = estimate(text)
                assert 1 <= got <= nfc_bytes(text), (name, variant, size, got)
                checked += 1
    assert checked > 700


def test_the_estimate_of_a_row_is_below_the_bytes_of_its_text_plus_the_template() -> None:
    rnd = random.Random(11)
    messages: list[dict[str, str]] = []
    for _, codes in _every_pool()[::9]:
        messages.append({"role": "user", "content": _generated(rnd, codes, "run", 60)})
    row: dict[str, Any] = {"messages": messages}
    count = len(messages)
    total_bytes = sum(nfc_bytes(message["content"]) for message in messages)
    assert _row_tokens(row) <= total_bytes + 5 * count + 21


def test_a_run_of_characters_is_never_below_a_token_each_and_its_bytes_where_none_is_held() -> None:
    rnd = random.Random(5)
    for low, high in ((0xE000, 0xF8FF), (0xF0000, 0xFFFFD)):
        for size in (2, 9, 100, 1000):
            text = "".join(chr(rnd.randint(low, high)) for _ in range(size))
            chars = student_chars()
            floor = sum(1 if char in chars else len(char.encode("utf-8")) for char in text)
            assert estimate(text) >= floor


def test_a_text_of_whitespace_is_never_below_what_its_runs_need() -> None:
    rnd = random.Random(3)
    kinds = [" ", "\n", "\t", "\xa0", "\u3000"]
    for _ in range(50):
        text = "".join(rnd.choice(kinds) * rnd.randint(1, 500) for _ in range(rnd.randint(1, 6)))
        floor = sum(
            math.ceil(len(list(run)) / LONGEST_TOKEN[char]) for char, run in itertools.groupby(text)
        )
        assert estimate(text) >= floor
