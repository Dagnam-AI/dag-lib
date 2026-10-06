"""Where two classes meet: every kind of character next to a letter of every script, and the
rules for a run on its own, a mark on its own, and a space and one letter.

The estimate once priced the lead of a word as one token whatever it was, and a private-use or
unassigned character as a lead has no token: its three bytes are three tokens (0.40 of the real
count). Each rule below closes one such boundary. The generator puts one character of each of the
29 general categories before and after a letter of each of the 53 blocks that have letters, 3,074
units a seed, each repeated 300 times, and checks the bounds that hold of the estimate itself: it
is never above the UTF-8 bytes of the NFC form, never below one token a unit, and a private-use
or unassigned character costs at least its bytes. (Against the tokenizer the design measured 12
of 3,074 under 0.95, the lowest 0.50: one sign next to one script, a neighbour that makes both
split.)
"""

from __future__ import annotations

import random
import unicodedata

import pytest

from dagnam.audit.token_estimate import estimate
from dagnam.audit.token_tables import student_chars

BLOCKS = {
    "latin": (0x41, 0x24F),
    "latin-ext-additional": (0x1E00, 0x1EFF),
    "ipa": (0x250, 0x2AF),
    "greek": (0x370, 0x3FF),
    "cyrillic": (0x400, 0x52F),
    "armenian": (0x530, 0x58F),
    "hebrew": (0x591, 0x5F4),
    "arabic": (0x600, 0x6FF),
    "syriac": (0x700, 0x74F),
    "thaana": (0x780, 0x7B1),
    "nko": (0x7C0, 0x7FA),
    "devanagari": (0x900, 0x97F),
    "bengali": (0x980, 0x9FF),
    "gurmukhi": (0xA00, 0xA7F),
    "gujarati": (0xA80, 0xAFF),
    "oriya": (0xB00, 0xB7F),
    "tamil": (0xB80, 0xBFF),
    "telugu": (0xC00, 0xC7F),
    "kannada": (0xC80, 0xCFF),
    "malayalam": (0xD00, 0xD7F),
    "sinhala": (0xD80, 0xDFF),
    "thai": (0xE01, 0xE5B),
    "lao": (0xE81, 0xEDF),
    "tibetan": (0xF00, 0xFFF),
    "myanmar": (0x1000, 0x109F),
    "georgian": (0x10A0, 0x10FF),
    "hangul-jamo": (0x1100, 0x11FF),
    "ethiopic": (0x1200, 0x137F),
    "cherokee": (0x13A0, 0x13F5),
    "canadian-syllabics": (0x1400, 0x167F),
    "khmer": (0x1780, 0x17F9),
    "mongolian": (0x1800, 0x18AA),
    "general-punct": (0x2000, 0x206F),
    "super-sub": (0x2070, 0x209F),
    "currency": (0x20A0, 0x20C0),
    "combining-symbols": (0x20D0, 0x20F0),
    "letterlike": (0x2100, 0x214F),
    "number-forms": (0x2150, 0x218B),
    "arrows": (0x2190, 0x21FF),
    "math-ops": (0x2200, 0x22FF),
    "misc-technical": (0x2300, 0x23FF),
    "enclosed-alnum": (0x2460, 0x24FF),
    "box-drawing": (0x2500, 0x257F),
    "blocks-shapes": (0x2580, 0x25FF),
    "misc-symbols": (0x2600, 0x26FF),
    "dingbats": (0x2700, 0x27BF),
    "braille": (0x2800, 0x28FF),
    "cjk-symbols": (0x3000, 0x303F),
    "hiragana": (0x3041, 0x3096),
    "katakana": (0x30A1, 0x30FA),
    "hangul-compat": (0x3131, 0x318E),
    "cjk-ext-a": (0x3400, 0x4DBF),
    "cjk": (0x4E00, 0x9FFF),
    "yi": (0xA000, 0xA48C),
    "vai": (0xA500, 0xA62B),
    "hangul": (0xAC00, 0xD7A3),
    "cjk-compat": (0xF900, 0xFAD9),
    "arabic-pres-a": (0xFB50, 0xFDFF),
    "halfwidth-fullwidth": (0xFF01, 0xFFEE),
    "linear-b": (0x10000, 0x1005D),
    "gothic": (0x10330, 0x1034A),
    "cuneiform": (0x12000, 0x12399),
    "hieroglyphs": (0x13000, 0x1342E),
    "math-alnum": (0x1D400, 0x1D7FF),
    "emoji-misc": (0x1F300, 0x1F5FF),
    "emoticons": (0x1F600, 0x1F64F),
    "transport": (0x1F680, 0x1F6FF),
    "supplemental-symbols": (0x1F900, 0x1F9FF),
    "symbols-ext-a": (0x1FA70, 0x1FAFF),
    "cjk-ext-b": (0x20000, 0x2A6DF),
    "tags": (0xE0001, 0xE007F),
    "private-use-a": (0xF0000, 0xFFFFD),
}


def nfc_bytes(text: str) -> int:
    return len(unicodedata.normalize("NFC", text).encode("utf-8", "surrogatepass"))


def _by_category() -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for code in range(0x20, 0x110000):
        if 0xD800 <= code <= 0xDFFF:
            continue
        groups.setdefault(unicodedata.category(chr(code)), []).append(code)
    return groups


CATEGORIES = _by_category()


def _letters() -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    for name, (low, high) in BLOCKS.items():
        letters = [
            c
            for c in range(low, high + 1)
            if unicodedata.category(chr(c)) not in ("Cn", "Cs")
            and unicodedata.category(chr(c)).startswith("L")
        ]
        if letters:
            found[name] = letters
    return found


LETTERS = _letters()


@pytest.mark.parametrize("seed", [20261004, 7])
def test_every_category_next_to_every_script_stays_between_a_token_a_unit_and_the_bytes(
    seed: int,
) -> None:
    rnd = random.Random(seed)
    checked = 0
    for category, codes in sorted(CATEGORIES.items()):
        if category == "Cs":
            continue
        for block, letters in LETTERS.items():
            for side in ("before", "after"):
                char = chr(rnd.choice(codes))
                letter = chr(rnd.choice(letters))
                unit = char + letter if side == "before" else letter + char
                text = unit * 300
                got = estimate(text)
                where = (category, side, block, got)
                assert 300 <= got <= nfc_bytes(text), where
                if category in ("Co", "Cn"):
                    assert estimate(unit) >= len(char.encode("utf-8")), where
                checked += 1
    assert checked == 3074


def test_the_generator_covers_29_categories_and_53_blocks_with_letters() -> None:
    assert len(CATEGORIES) == 29  # the surrogates, category Cs, are left out
    assert len(LETTERS) == 53


def test_a_private_use_or_unassigned_lead_costs_its_bytes_not_one_token() -> None:
    # A letter after a private-use character (3 bytes) or an unassigned one (3-4): the lead is
    # a token of its own and its bytes are tokens too. It was priced 1: 0.40 of the real count.
    assert estimate("\ue000\u0283") == 4
    assert estimate("\u0378\u0283") == 3
    assert estimate("\u0283\ue000") == 4
    assert estimate(("\ue000\u0283") * 300) == 1200
    assert estimate(("\u0378\u0283") * 300) == 900


def test_a_run_of_a_script_on_its_own_costs_at_least_one_token() -> None:
    # A lone Hangul syllable (0.804) or kana (0.446) between symbols was under one token.
    assert estimate("!\ud55c!") == 3
    assert estimate(("!\ud55c!") * 100) == 201
    assert estimate(("!\u3042!") * 100) == 201


def test_kana_and_kanji_touching_each_other_share_tokens_and_are_exempt_from_the_floor() -> None:
    assert estimate("\u3042\u4e2d") == 2
    assert estimate(("\u3042\u4e2d") * 100) == 145


def test_letters_of_a_script_that_are_no_token_cost_their_bytes_even_at_a_rate_above_one() -> None:
    # U+0904 is a Devanagari letter with no token of its own: 3 bytes, not 1.02 a character.
    assert "\u0904" not in student_chars()
    assert estimate("\u0904" * 5) == 15
    assert estimate("\u0904" * 50) == 150


def test_marks_and_signs_with_no_letter_cost_a_token_or_more_each() -> None:
    # A vowel sign alone has no consonant to merge into.
    assert estimate("\u093e" * 10) == 10
    assert estimate("\u0cbe") == 3
    assert estimate("\u0cbe" * 50) == 150


def test_a_space_and_one_letter_that_are_not_one_token_cost_at_least_two() -> None:
    # " \u00f1" is not a token of the vocabulary (" a" is): the space stays a token of its own.
    assert estimate(" a") == 1
    assert estimate(" \u00f1") == 2
    assert estimate(" \u00f1" * 100) == 200
    assert estimate(" \ud55c" * 50) == 66  # a token, priced by the Hangul rate and its lead


def test_the_replacement_character_is_a_token_of_its_own() -> None:
    # U+FFFD x 2,000 is 500 real tokens (it merges in fours); it was priced at its bytes, 6,000.
    assert "\ufffd" in student_chars()
    assert estimate("\ufffd") == 1
    assert estimate("\ufffd" * 2000) == 2000
