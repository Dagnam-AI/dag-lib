"""The estimate on 213 adversarial texts: kinds of text no corpus holds, built from recipes.

Each text is rebuilt from code points, repetition and a seeded generator (``_attacks*``), never
kept as text. ``tests/audit/fixtures/token_estimate_attacks.json`` holds, in the same order, the
real count of Qwen2.5 for each text and the estimate that is pinned. The claim the estimate is
built on is that it is low only by a bounded amount: no text here is below 0.59 of its real
count, and every one outside the documented list is at 0.95 or above.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tests.audit._attacks import attacks, label

from dagnam.audit.token_estimate import estimate

FIXTURES = Path(__file__).parent / "fixtures"
ATTACKS: list[dict[str, Any]] = json.loads(
    (FIXTURES / "token_estimate_attacks.json").read_text(encoding="utf-8")
)
NATURAL = {
    pin["kind"]: pin["text"]
    for pin in json.loads((FIXTURES / "token_estimate_pins.json").read_text(encoding="utf-8"))
}
LANGUAGES = {
    "en": "english system prompt",
    "ja": "japanese",
    "ar": "arabic",
    "ru": "russian",
    "hi": "hindi",
    "zh-hans": "simplified chinese",
    "zh-hant": "traditional chinese",
    "ko": "korean",
    "th": "thai",
    "de": "german",
    "fa": "persian",
    "he": "hebrew",
    "el": "greek",
    "vi": "vietnamese",
}
RECIPES = attacks({code: NATURAL[kind] for code, kind in LANGUAGES.items()})

DOCUMENTED_LOW = {
    "base32",
    "base32 lower",
    "random lowercase words 8-12 letters",
    "random UPPERCASE words 8-12 letters",
    "license keys XXXXX-XXXXX",
    "consonant-only lowercase words",
    "emoji spaced x600",
    "leet/small caps",
    "ruled script, random words: Hiragana",
    "ruled script, random words: Katakana",
    "ruled script, random words: Halfwidth katakana",
    "ruled script, random words: Thai",
    "ruled script, random words: Cyrillic basic",
    "ruled script, random words: Arabic basic",
    "ruled script, random words: Hebrew",
    "ruled script, random words: Latin-1 letters",
    "ruled script, random words: Latin Ext-A/B",
    "Arabic random words + ASCII digits",
}
"""The 18 texts under 0.95: random letters of natural word length and random words of a script
that has a rate (16), and two crafted ones (small capitals, spaced emoji). The design lists them
as the known floor; the lowest is 0.59."""
FLOOR_OF_THE_LOW = 0.59
WORST_OVER_COUNT = 21.5  # a symbol repeated 8,000 times: 21.3 (the student merges it in 64s)


def ratio(index: int) -> float:
    return ATTACKS[index]["estimate"] / ATTACKS[index]["real"]


def kinds_containing(*parts: str) -> list[int]:
    return [i for i, row in enumerate(ATTACKS) if any(part in row["kind"] for part in parts)]


def test_the_recipes_are_the_213_pinned_rows_in_order() -> None:
    assert len(RECIPES) == len(ATTACKS) == 213
    assert [name for name, _ in RECIPES] == [row["kind"] for row in ATTACKS]
    assert len({row["kind"] for row in ATTACKS}) == 213


def test_the_fixture_is_ascii_and_the_recipes_name_no_character_outside_ascii() -> None:
    raw = (FIXTURES / "token_estimate_attacks.json").read_bytes()
    assert raw.isascii()
    assert all(name.isascii() for name, _ in RECIPES)
    assert label("a哈") == "aU+54C8"


@pytest.mark.parametrize("index", range(213), ids=[row["kind"][:50] for row in ATTACKS])
def test_each_attack_is_priced_exactly_and_within_the_bound_the_design_claims(index: int) -> None:
    name, text = RECIPES[index]
    row = ATTACKS[index]
    got = estimate(text)
    assert got == row["estimate"]
    if not row["real"]:
        # The tokenizer refuses a lone surrogate, so there is no real count; each is priced at
        # its three bytes, and the SDK decides whether such a message is counted at all.
        assert got == 3 * len(text)
        return
    assert got / row["real"] <= WORST_OVER_COUNT
    if name in DOCUMENTED_LOW:
        assert got / row["real"] >= FLOOR_OF_THE_LOW
    else:
        assert got / row["real"] >= 0.95


def test_the_documented_low_texts_are_exactly_the_18_under_95_percent() -> None:
    low = {row["kind"] for row in ATTACKS if row["real"] and row["estimate"] < 0.95 * row["real"]}
    assert low == DOCUMENTED_LOW
    assert len(low) == 18
    assert min(ratio(i) for i in range(213) if ATTACKS[i]["real"]) > FLOOR_OF_THE_LOW


def test_whitespace_of_every_kind_is_never_below_the_real_count() -> None:
    # Whitespace used to cost one token for a run of any length: 0.0002 of the real count.
    indexes = kinds_containing(
        "spaces x",
        "tabs x",
        "newlines x",
        "CRLF",
        "NBSP",
        "ideographic space x",
        "em/thin",
        "line/para",
        "vertical tab",
        "words padded",
        "trailing 80",
        "indented 7",
        "short English sentence",
        "scraped page",
        "fixed-width",
        "alternating space/tab",
    )
    assert len(indexes) == 19
    assert all(ATTACKS[i]["estimate"] >= ATTACKS[i]["real"] for i in indexes)


def test_characters_outside_the_vocabulary_are_priced_at_their_real_count_or_above() -> None:
    indexes = kinds_containing(
        "private use",
        "unassigned",
        "noncharacters",
        "C0 controls",
        "C1 controls",
        "NUL",
        "DEL",
        "ZWSP",
        "ZWJ",
        "ZWNJ",
        "BOM/WJ",
        "bidi controls",
        "variation selectors",
        "tag characters",
        "braille",
        "math bold",
        "math script",
        "keycaps",
        "han, random",
        "han ext B random",
        "common han random",
    )
    assert len(indexes) >= 21
    # At or above the real count, or 0.2% below it (a private-use or unassigned character is
    # priced at its four bytes, a character the tokenizer sometimes takes a byte cheaper).
    assert all(ATTACKS[i]["estimate"] >= 0.99 * ATTACKS[i]["real"] for i in indexes)


def test_unknown_scripts_stay_within_five_percent_of_the_real_count() -> None:
    indexes = kinds_containing("unknown script")
    assert len(indexes) == 18
    assert min(ratio(i) for i in indexes) > 0.95  # Syriac 0.98


def test_a_long_random_word_is_priced_at_the_floor_of_random_letters() -> None:
    # The tables are exact, so no random word is a token (the earlier search for one that a
    # filter wrongly accepted, 200,000 tries of 60 letters, finds nothing): 60 letters cost
    # 34 each time, 0.65 a letter beyond 16.
    index = kinds_containing("one random 60-letter word")[0]
    assert 1.0 < ratio(index) < 1.12


def test_the_worst_over_counts_are_the_documented_ones() -> None:
    assert max(ratio(i) for i in range(213) if ATTACKS[i]["real"]) == pytest.approx(21.34, abs=0.01)
    by_ratio = sorted((i for i in range(213) if ATTACKS[i]["real"]), key=ratio, reverse=True)
    assert {ATTACKS[i]["kind"] for i in by_ratio[:3]} == {
        "repeated '=' x8000",
        "repeated '-' x8000",
        "repeated '.' x8000",
    }
    code = next(i for i, row in enumerate(ATTACKS) if row["kind"] == "code, long identifiers")
    assert ratio(code) == pytest.approx(3.02, abs=0.01)
