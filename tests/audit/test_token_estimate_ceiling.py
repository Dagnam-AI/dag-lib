"""NFC, the ceiling at the bytes, and the fragile-unit rule.

The student's tokenizer normalises its input to NFC before it cuts it into pieces, so the
estimate does too, and a byte-level BPE spends at most as many tokens on a piece as the piece has
UTF-8 bytes in that form: each piece is capped at them. A few hundred units of the vocabulary
(the fragile ones) are not re-merged when they recur, alone or in short cycles, the way a longest
match assumes, so one seen again within eight units costs its characters.
"""

from __future__ import annotations

import math
import random
from typing import Literal
import unicodedata

import pytest

from dagnam.audit.token_estimate import estimate
from dagnam.audit.token_tables import student_fragile

# Text in several scripts, each as written below in NFC; NFD is built from it in the tests.
SAMPLES = {
    "french": "Les \u00e9l\u00e8ves ont \u00e9t\u00e9 tr\u00e8s \u00e9tonn\u00e9s \u00e0 l'id\u00e9e d'\u00eatre re\u00e7us pr\u00e8s de l'h\u00f4tel o\u00f9 na\u00eet le ma\u00efs. ",
    "german": "\u00dcber die gr\u00f6\u00dften Sch\u00e4den m\u00fcssen \u00f6ffentliche \u00c4mter fr\u00fch h\u00f6ren. ",
    "vietnamese": "T\u1ea5t c\u1ea3 m\u1ecdi ng\u01b0\u1eddi sinh ra \u0111\u1ec1u \u0111\u01b0\u1ee3c t\u1ef1 do v\u00e0 b\u00ecnh \u0111\u1eb3ng v\u1ec1 nh\u00e2n ph\u1ea9m. ",
    "korean": "\ubaa8\ub4e0 \uc778\uac04\uc740 \ud0dc\uc5b4\ub0a0 \ub54c\ubd80\ud130 \uc790\uc720\ub85c\uc6b0\uba70 \uadf8 \uc874\uc5c4\uacfc \uad8c\ub9ac\uc5d0 \uc788\uc5b4 \ub3d9\ub4f1\ud558\ub2e4. ",
    "japanese": "\u3059\u3079\u3066\u306e\u4eba\u9593\u306f\u3001\u751f\u307e\u308c\u306a\u304c\u3089\u306b\u3057\u3066\u81ea\u7531\u3067\u3042\u308a\u3001\u30ac\u30ae\u30b0\u30b2\u30b4\u3002",
    "greek": "\u039a\u03b1\u03bb\u03b7\u03bc\u03ad\u03c1\u03b1, \u03c0\u03ce\u03c2 \u03b5\u03af\u03c3\u03c4\u03b5 \u03c3\u03ae\u03bc\u03b5\u03c1\u03b1; ",
    "gurmukhi": "\u0a38\u0a3c\u0a3e\u0a02\u0a24\u0a40 \u0a16\u0a3c\u0a41\u0a36\u0a40 \u0a32\u0a3c\u0a3e\u0a30\u0a3e ",
}


@pytest.mark.parametrize("name", list(SAMPLES))
def test_text_in_nfd_and_nfc_form_is_estimated_identically(name: str) -> None:
    nfc = unicodedata.normalize("NFC", SAMPLES[name]) * 12
    nfd = unicodedata.normalize("NFD", nfc)
    if name != "gurmukhi":  # its letters are ones NFC writes decomposed, so both forms are one
        assert nfd != nfc
    assert estimate(nfd) == estimate(nfc)


def test_random_decomposed_text_is_estimated_like_its_composed_form() -> None:
    rnd = random.Random(20261003)
    composed = [chr(code) for code in range(0xC0, 0x250) if unicodedata.decomposition(chr(code))]
    base = [chr(code) for code in range(0x61, 0x7B)]
    for _ in range(30):
        text = "".join(rnd.choice(composed if rnd.random() < 0.4 else base) for _ in range(300))
        assert estimate(unicodedata.normalize("NFD", text)) == estimate(
            unicodedata.normalize("NFC", text)
        )


def test_text_that_is_already_normalised_is_not_copied(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    real = unicodedata.normalize

    def spy(form: Literal["NFC", "NFD", "NFKC", "NFKD"], text: str) -> str:
        calls.append(form)
        return real(form, text)

    monkeypatch.setattr(unicodedata, "normalize", spy)
    estimate("Already composed: \u00e9\u00e8\u00fc, \u4e2d\u6587, \ud55c\uae00.")
    assert calls == []
    estimate(unicodedata.normalize("NFD", "Already composed: \u00e9\u00e8\u00fc."))
    assert calls == ["NFD", "NFC"]  # the test's own NFD, then the estimate's one copy


def test_a_precomposed_character_nfc_splits_is_priced_on_its_parts() -> None:
    # U+0F76 is a Tibetan vowel sign that NFC writes as two characters: the real count is 6 a
    # pair, more than the 3 bytes it has as written. 50 of them were estimated 150.
    assert estimate("\u0f76" * 50) == 300
    assert estimate("\u0f77" * 8) == 24


def test_a_lone_point_or_digit_costs_its_bytes_and_no_more() -> None:
    # A Hebrew point was priced 8 for its 2 bytes, an Arabic-Indic digit 6 for 2: a fitted line
    # that was never capped. Each piece is now capped at the UTF-8 bytes of its NFC form.
    assert estimate("\u05bf") == 2
    assert estimate("\u05bf" * 40) == 80
    assert estimate("\u0663") == 2
    assert estimate("\u0663" * 8) == 16


def test_a_piece_is_capped_at_its_bytes_but_a_text_is_the_sum_of_its_pieces() -> None:
    # "a" is a token (1.1 bare, so capped at its byte); a sentence of such pieces is below its bytes.
    assert estimate("a") == 1
    text = "1a" * 50
    assert estimate(text) <= len(text)


# --- the fragile-unit rule -------------------------------------------------------------------

# Units of the fragile table: BPE does not re-merge them when they recur (eight copies take more
# than 8.8 real tokens). One seen again within the last eight units costs its characters.
HAN4 = "\u4e00\u5207\u90fd\u662f"
HAN4_B = "\u4e00\u65b9\u9762\u662f"
HAN4_C = "\u4e00\u76f4\u90fd\u662f"
HAN3 = "\u4e00\u4e2a\u662f"
HAN2 = "\u4e00\u4e07"
HAN2_B = "\u4e00\u540c"
CLEAN4 = "\u4e2d\u534e\u4eba\u6c11"  # a 4-character token that repeats cleanly: not fragile
SYMBOL3 = "!=("
SYMBOL2 = "!?"
COMMA = "\uff0c"


def test_the_units_used_below_are_fragile_or_not_as_the_cases_need() -> None:
    fragile = student_fragile()
    assert {HAN4, HAN4_B, HAN4_C, HAN3, HAN2, HAN2_B, SYMBOL3, SYMBOL2} <= fragile
    assert CLEAN4 not in fragile
    assert {"---", "===", "###", "..."}.isdisjoint(fragile)


def test_a_fragile_han_unit_repeated_costs_its_characters() -> None:
    # One 4-character token repeated 100 times is 100 real tokens in the worst units and up to
    # 2.98 a copy; longest match said 1.08 a copy. The first copy is 1.08, the rest 4.
    assert estimate(HAN4) == 2
    assert estimate(HAN4 * 2) == 6  # 1.08 + 4, rounded up
    assert estimate(HAN4 * 100) == 398  # 1.08 + 4 x 99


def test_a_fragile_unit_of_three_or_two_characters_costs_that_many() -> None:
    assert estimate(HAN3 * 100) == 299  # 1.08 + 3 x 99
    assert estimate(HAN2 * 50) == 100  # 1.08 + 2 x 49, rounded up


def test_a_cycle_of_different_fragile_units_within_the_window_costs_their_characters() -> None:
    # Cycles of 3 and 2 different units walked past a rule that only looked at the unit before:
    # 0.44-0.50 of the real count. Each unit seen again within 8 units costs its characters.
    assert estimate((HAN4 + HAN4_B + HAN4_C) * 40) == 472  # 3 x 1.08 + 4 x 117
    assert estimate((HAN4 + HAN4_B) * 50) == 395  # 2 x 1.08 + 4 x 98
    assert estimate((HAN2 + HAN2_B) * 50) == 199  # 2 x 1.08 + 2 x 98


def test_a_cycle_longer_than_the_window_is_not_charged() -> None:
    # Nine different fragile units: each was last seen nine units ago, outside the window of
    # eight, so each costs 1.08 (a cycle of 8 is still inside it).
    fragile = sorted(
        u for u in student_fragile() if len(u) == 4 and all(ord(c) >= 0x2E80 for c in u)
    )
    assert len(fragile) >= 9
    cycle8 = "".join(fragile[:8]) * 20
    cycle9 = "".join(fragile[:9]) * 20
    assert estimate(cycle8) == math.ceil(8 * 1.08 + 4 * (160 - 8))
    assert estimate(cycle9) == math.ceil(180 * 1.08)


def test_a_unit_that_repeats_cleanly_is_not_penalised() -> None:
    assert estimate(CLEAN4 * 100) == 108
    assert estimate("\u4e2d\u56fd" * 50) == 54


def test_units_that_are_not_adjacent_in_one_run_are_not_repeats() -> None:
    # A comma between the copies ends the run of han: each is a fresh unit.
    assert estimate((HAN4 + COMMA) * 50) == 104
    assert estimate((HAN4 + " x ") * 3) < estimate(HAN4 * 3) + 10


def test_a_fragile_symbol_unit_repeated_or_cycled_costs_its_characters() -> None:
    assert estimate(SYMBOL3) == 1
    assert estimate(SYMBOL3 * 2) == 4  # 1 + 3
    assert estimate(SYMBOL3 * 100) == 298  # 1 + 3 x 99
    assert estimate(SYMBOL2 * 100) == 199  # 1 + 2 x 99
    assert estimate((SYMBOL3 + SYMBOL2) * 50) == 247  # 2 + 3 x 49 + 2 x 49


def test_a_run_of_one_symbol_that_is_not_fragile_costs_a_third_of_a_token_a_character() -> None:
    # BPE merges '---' into long tokens (300 dashes are 5 real tokens): the units are not fragile.
    assert estimate("-" * 300) == 100
    assert estimate("---" * 100) == 100
    assert estimate("..." * 100) == 100


def test_a_fragile_symbol_unit_repeated_with_something_between_is_priced_once_each() -> None:
    assert estimate(SYMBOL3 + "a" + SYMBOL3) == 3
