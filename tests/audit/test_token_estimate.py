"""The student token estimate: pinned against texts counted by the real Qwen2.5 tokenizer.

Every ``real`` below is the count of ``Qwen/Qwen2.5-0.5B-Instruct`` for exactly that text, taken
once and kept as a literal, so no test needs the tokenizer. The estimate is a fit, so what is
pinned is the exact value it returns (a changed constant is a changed estimate and should be a
decision) and, beside it, how close that is to the real count.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any
import unicodedata

import pytest

from dagnam.audit.derive import _DEFAULT_SYSTEM_TOKENS, _MESSAGE_TOKENS, _row_tokens
from dagnam.audit.token_classes import PER_CHAR
from dagnam.audit.token_estimate import estimate, pieces
from dagnam.audit.token_rules import RATES

FIXTURES = Path(__file__).parent / "fixtures"
PINS: list[dict[str, Any]] = json.loads(
    (FIXTURES / "token_estimate_pins.json").read_text(encoding="utf-8")
)
ROWS: dict[str, Any] = json.loads(
    (FIXTURES / "token_estimate_rows.json").read_text(encoding="utf-8")
)
SCRIPTS: dict[str, dict[str, Any]] = json.loads(
    (FIXTURES / "token_estimate_scripts.json").read_text(encoding="utf-8")
)

SCRIPTS_NATURAL: dict[str, dict[str, Any]] = json.loads(
    (FIXTURES / "token_estimate_scripts_natural.json").read_text(encoding="utf-8")
)

# Texts the design documents as over-counts or as empty; every other pin is within the band.
OVER_COUNTS = {"camelCase identifiers", "single label"}


def split(text: str) -> list[str]:
    return [piece.group() for piece in pieces(text)]


def row_of(messages: list[dict[str, str]]) -> dict[str, Any]:
    return {"messages": messages}


def row_ratio(case: str) -> float:
    """The estimate of a three-message row (system, user, reply) over its real, template included."""
    entry = ROWS["cases"][case]
    return _row_tokens(row_of(entry["messages"])) / entry["real"]


def letters_of(script: str) -> str:
    """The first 45 letters of ``script``, in code point order (37 for Tamil)."""
    chars: list[str] = []
    for low, high in re.findall(r"(.)-(.)", PER_CHAR[script], flags=re.S):
        for code in range(ord(low), ord(high) + 1):
            char = chr(code)
            if unicodedata.category(char).startswith("L") and unicodedata.name(char, ""):
                chars.append(char)
            if len(chars) == 45:
                return "".join(chars)
    return "".join(chars)


# --- pinned texts ---------------------------------------------------------------------------


def test_the_pins_are_the_48_texts_of_the_design() -> None:
    assert len(PINS) == 48
    assert len({pin["kind"] for pin in PINS}) == 48
    assert [pin["kind"] for pin in PINS][:3] == [
        "english prose",
        "english system prompt",
        "terse english",
    ]


@pytest.mark.parametrize("pin", PINS, ids=[pin["kind"] for pin in PINS])
def test_each_pinned_text_is_counted_exactly_and_within_its_band(pin: dict[str, Any]) -> None:
    got = estimate(pin["text"])
    assert got == pin["estimate"]
    if not pin["real"]:
        assert got == 0
    elif pin["kind"] in OVER_COUNTS:
        # A list of camelCase names (real 13) and a one-word text (real 1) are over-counted by
        # design: the seam cost is fitted to base64, which must not be under-counted, and a
        # one-token text rounds up to two.
        assert got > 1.35 * pin["real"]
    else:
        assert 0.95 * pin["real"] <= got <= 1.35 * pin["real"]


def test_two_pinned_texts_are_a_little_below_their_real_count_and_none_below_95_percent() -> None:
    # Kazakh 76 of 78 and Thai 66 of 69: the lowest ratios of the natural pins, 0.957 at the
    # lowest.
    below = {
        pin["kind"]: (pin["real"], pin["estimate"]) for pin in PINS if pin["estimate"] < pin["real"]
    }
    assert below == {"kazakh": (78, 76), "thai": (69, 66)}
    assert min(pin["estimate"] / pin["real"] for pin in PINS if pin["real"]) > 0.95


# --- behaviour --------------------------------------------------------------------------------


def test_empty_and_whitespace_texts() -> None:
    assert estimate("") == 0
    assert estimate(" ") == 1
    assert estimate("\n\n\n") == 1
    assert estimate(" " * 8) == 1


def test_every_ascii_digit_is_one_token() -> None:
    assert estimate("1234567890") == 10
    assert estimate("3.14159") == 7
    # A Devanagari digit is three bytes and not a token (real 2): priced at its bytes.
    assert estimate("\u0969") == 3


def test_a_whole_word_token_costs_one_with_a_space_and_a_little_more_without() -> None:
    # Qwen2.5 counts 1, 1, 2 and 8. A word with no space before it is a token the vocabulary
    # knows less often (+0.1), after an ASCII symbol the symbol is usually a token of its own
    # (+0.5); both round up from a single token here.
    assert estimate(" paid") == 1
    assert estimate("paid") == 2
    assert estimate(",paid") == 2
    assert estimate(",paid ,paid ,paid ,paid") == 8


def test_case_is_not_folded_when_the_word_is_looked_up() -> None:
    # " men" and " MEN" are tokens of the vocabulary (1 each); " WOMEN" is two (estimate 3) and
    # " CHAIRS" three (estimate 3): lowercasing them all made the shouted ones cost 1.
    assert estimate(" men") == 1
    assert estimate(" MEN") == 1
    assert estimate(" WOMEN") == 3
    assert estimate(" CHAIRS") == 3


def test_han_is_priced_by_the_longest_vocabulary_token_not_by_a_rate_or_a_variant() -> None:
    # Each longest match of 2-4 han characters costs 1.08; a character matching none costs 1 if
    # it is a token and its bytes if not. Simplified and Traditional need no table: the
    # difference is which pairs exist. Real counts: 2, 2, 3 and 50.
    assert estimate("\u6771\u4eac") == 2
    assert estimate("\u4e2d\u56fd") == 2
    assert estimate("\u4e2d\u534e\u4eba\u6c11\u5171\u548c\u56fd") == 3
    assert estimate("\u4e2d" * 50) == 50
    # Kanji next to kana are priced the same way, and the kana by their rate.
    assert estimate("\u3067\u3059") == 1
    assert estimate("\u6771\u4eac\u3067\u3059") == 3
    assert estimate("\u3067\u3059\u6771\u4eac") == 3


def test_a_rate_below_one_token_a_character_holds_for_a_natural_length_only() -> None:
    # Natural Japanese words stop after about 8 kana (Hangul syllables: 6); random text does
    # not, so each character beyond costs a whole token.
    assert estimate("\u3042" * 8) == 4  # 8 x 0.446
    assert estimate("\u3042" * 9) == 5  # + 1
    assert estimate("\u3042" * 20) == 16
    assert estimate("\ud55c" * 6) == 5  # 6 x 0.804
    assert estimate("\ud55c" * 7) == 6
    assert estimate("\ud55c" * 20) == 19


def test_pieces_follow_qwen_s_pre_tokenizer() -> None:
    assert split("it's don't I'll") == ["it", "'s", " don", "'t", " I", "'ll"]
    assert split('{"name": "get_weather"}') == ['{"', "name", '":', ' "', "get", "_weather", '"}']
    # The lead of a word may be a symbol or an underscore: dropping the underscore from the
    # lead made "_status" one piece cheaper.
    assert split("_status") == ["_status"]
    assert split("a\n\n  b") == ["a", "\n\n", " ", " b"]
    assert split("12") == ["1", "2"]


def test_symbol_runs_are_priced_by_the_vocabulary_units_they_are_made_of() -> None:
    # One token for each longest match against the vocabulary's symbol tokens of up to three
    # characters, so a repeated symbol costs a third of a token a character, as it does for the
    # student ("))))" is one token there). Mixed symbols are what Qwen2.5 counts (real 1, 1, 3).
    assert estimate("{") == 1
    assert estimate("),") == 1
    assert estimate("!?#%") == 3
    assert estimate("))))") == 2
    assert estimate("=" * 9) == 3
    assert estimate("=" * 30) == 10
    # Wide characters are tokens of their own (real 1, 2, 3, and 2 with an ASCII symbol before).
    assert estimate("\u00ab") == 1
    assert estimate("\u201c\u201d") == 2
    assert estimate("\u201c\u201d\u201e") == 3
    assert estimate('."\u201d') == 2
    # Emoji that are tokens cost one each (real 1, 2, 3).
    assert estimate("\U0001f600") == 1
    assert estimate("\U0001f600" * 2) == 2
    assert estimate("\U0001f600" * 3) == 3


def test_a_script_with_no_rule_costs_its_bytes_a_letter_and_one_for_a_new_word() -> None:
    # Tifinagh letters are not tokens: 3 bytes each (real 4 for the three).
    assert estimate("\u2d30\u2d31\u2d32") == 5


def test_camel_case_seams_are_priced_but_all_lower_and_all_upper_words_are_not() -> None:
    # A seam costs 1.39: getUserName is 2 tokens for the student and 7 here, while the same
    # letters in one case are 5 or 6. The price is fitted to base64 and random-case strings,
    # which are the under-count risk, so camelCase names are the known over-count.
    assert estimate("getUserName") == 7
    assert estimate("getusername") == 5
    assert estimate("GETUSERNAME") == 6
    # Real 38 (the first estimate of the project, which priced it as English, said 19).
    assert estimate("q83vEjRWeJC6/tI0VniQur7SNFZ4kLq+0jRWeJC6vtI0VniQ") == 44


@pytest.mark.parametrize(
    ("text", "real", "expected"),
    [
        ("\u0432\u043e\u0437\u0432\u0440\u0430\u0442", 3, 4),  # a Russian word
        ("\u039a\u03b1\u03bb\u03b7\u03bc\u03ad\u03c1\u03b1", 8, 9),  # Greek: a token a letter
        ("\u0562\u0561\u0580\u0565\u0582", 5, 6),  # Armenian
        ("\u10d2\u10d0\u10db\u10d0\u10e0\u10ef\u10dd\u10d1\u10d0", 9, 10),  # Georgian
        ("\u05e9\u05dc\u05d5\u05dd", 1, 3),  # Hebrew: a word the vocabulary knows, bare
        ("\u0645\u0631\u062d\u0628\u0627\u064b", 4, 4),  # Arabic with its vowel mark
    ],
    ids=["cyrillic", "greek", "armenian", "georgian", "hebrew", "arabic"],
)
def test_a_word_of_each_alphabet_is_priced_per_letter_and_never_below_the_real_count(
    text: str, real: int, expected: int
) -> None:
    assert estimate(text) == expected
    assert estimate(text) >= real


def test_cyrillic_is_looked_up_lowercased_and_its_extra_letters_cost_more() -> None:
    # " \u0434\u043b\u044f" (for) is one token and so is its capitalised form: the lookup folds
    # Cyrillic case, which the vocabulary does for every Russian word it holds.
    assert estimate(" \u0434\u043b\u044f") == 1
    assert estimate(" \u0414\u043b\u044f") == 1
    # Ukrainian "\u0457" is outside Russian's alphabet: real 5 for five of them, and 3 for the
    # five Russian "\u044f" the basic set covers.
    assert estimate("\u0457\u0457\u0457\u0457\u0457") == 10
    assert estimate("\u044f\u044f\u044f\u044f\u044f") == 3


@pytest.mark.parametrize("script", list(SCRIPTS))
def test_every_script_written_without_spaces_has_its_own_rate_and_lead(script: str) -> None:
    # 45 letters of the script (its first ones, so 37 for Tamil): changing one script's rate or
    # lead cost changes this number, and no other test would notice. They are not natural
    # text, so there is no real count to compare with; the natural texts are in the pins.
    text = letters_of(script)
    assert len(text) == (37 if script == "tamil" else 45)
    assert estimate(text) == SCRIPTS[script]["estimate"]
    assert estimate(" " + text) == SCRIPTS[script]["estimate_after_a_space"]


@pytest.mark.parametrize("script", ["kannada", "oriya", "tibetan"])
def test_oriya_kannada_and_tibetan_are_priced_at_their_measured_rates_not_their_bytes(
    script: str,
) -> None:
    # Short sentences of natural text (the first article of the Universal Declaration of Human
    # Rights and two sentences about a capital), counted by Qwen2.5: Kannada 383, Oriya 452,
    # Tibetan 300 tokens. At their bytes (3 a character, the price before they were measured)
    # they were 754, 607 and 667: 1.97, 1.34 and 2.22 times. On the Wikipedia text the rates were
    # fitted to, held-out rows of 500 or more tokens are 0.976-1.017 (Kannada, 248 rows),
    # 0.992-1.052 (Oriya, 37) and 0.962-1.039 (Tibetan, 15) of the real count.
    case = SCRIPTS_NATURAL[script]
    got = estimate(case["text"])
    assert got == case["estimate"]
    assert 0.95 * case["real"] <= got <= 1.15 * case["real"]


def test_the_rates_of_the_three_scripts_are_the_fitted_ones() -> None:
    assert (RATES["oriya"], RATES["kannada"], RATES["tibetan"]) == (2.0, 1.42, 1.4)


def test_the_scripts_table_covers_all_twenty_scripts() -> None:
    assert len(SCRIPTS) == 20
    assert {"han", "hiragana", "lao", "myanmar", "khmer", "ethiopic", "tibetan"} <= set(SCRIPTS)


# --- the cases the first estimates got wrong, with the template ------------------------------


def test_english_instructions_over_a_finnish_policy_are_not_counted_as_english() -> None:
    # A 470-word English preamble over a Finnish returns policy, a short question and a JSON
    # reply. Qwen2.5 reads 1,728 tokens (policy twice) and 2,736 (four times). An estimate that
    # read the whole system prompt as English said 1,470 (0.85) and 2,187 (0.80), so a workload
    # whose every row is over the student's 2,048 tokens passed as trainable.
    twice = ROWS["cases"]["english_instructions_over_finnish_policy_x2"]
    four = ROWS["cases"]["english_instructions_over_finnish_policy_x4"]
    assert (twice["real"], four["real"]) == (1728, 2736)
    assert _row_tokens(row_of(twice["messages"])) == 1755
    assert _row_tokens(row_of(four["messages"])) == 2783
    assert 0.95 <= row_ratio("english_instructions_over_finnish_policy_x2") <= 1.10
    assert 0.95 <= row_ratio("english_instructions_over_finnish_policy_x4") <= 1.10
    # The row that does not fit is over the 2,048-token budget the scan checks.
    assert _row_tokens(row_of(four["messages"])) > 2048


def test_terse_english_and_lists_of_labels_are_not_counted_at_twice_their_size() -> None:
    # Terse instructions with 262 category lines (real 1,103), the same categories comma
    # separated (997) and 60 product titles (1,367). With no function words to read a language
    # from, a language-guessing estimate priced them as an unknown language: 899 (0.82, an
    # under-count once the lines were separate), 1,502 (1.51) and 1,926 (1.41), refusing
    # workloads that fit.
    expected = {
        "terse_instruction_with_category_lines": (1103, 1211),
        "comma_separated_category_list": (997, 1171),
        "product_titles": (1367, 1461),
    }
    for case, (real, rowed) in expected.items():
        assert ROWS["cases"][case]["real"] == real
        assert _row_tokens(row_of(ROWS["cases"][case]["messages"])) == rowed
    assert 0.95 <= row_ratio("terse_instruction_with_category_lines") <= 1.25
    assert 0.95 <= row_ratio("comma_separated_category_list") <= 1.25
    assert 0.95 <= row_ratio("product_titles") <= 1.25


def test_traditional_chinese_is_not_under_counted() -> None:
    # Eight conversational messages in Traditional Chinese: Qwen2.5 reads 691 tokens in the row,
    # an estimate at one rate for all Chinese said 662 (0.96). A 118-character message (real
    # 96) was 89 (0.93).
    case = ROWS["cases"]["traditional_chinese_conversation"]
    assert case["real"] == 691
    assert _row_tokens(row_of(case["messages"])) == 718
    assert row_ratio("traditional_chinese_conversation") >= 1.0
    pin = next(pin for pin in PINS if pin["kind"] == "traditional chinese")
    assert (pin["real"], estimate(pin["text"])) == (96, 98)


def test_the_documented_under_counts_of_base64_emoji_and_persian_are_gone() -> None:
    # Real counts: base64 88 (an estimate by language said 39), a line of emoji chat 24 (15),
    # Persian 65 (50). Each was 0.4-0.77 of the real count.
    pins = {pin["kind"]: pin for pin in PINS}
    for kind, real in (("base64", 88), ("emoji", 24), ("persian", 65)):
        assert pins[kind]["real"] == real
        assert 0.95 * real <= estimate(pins[kind]["text"]) <= 1.35 * real


def test_conversational_simplified_chinese_is_counted_close_to_its_real_count() -> None:
    # Qwen2.5 reads 601 tokens in the row; the estimate is 634 (1.05).
    case = ROWS["cases"]["simplified_chinese_conversation"]
    assert case["real"] == 601
    assert _row_tokens(row_of(case["messages"])) == 634


# --- the chat template ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "template",
    ROWS["templates"],
    ids=[
        f"{len(t['messages'])}-messages-{'system' if t['messages'][0]['role'] == 'system' else 'no-system'}"
        for t in ROWS["templates"]
    ],
)
def test_the_template_costs_five_tokens_a_message_and_21_when_there_is_no_system(
    template: dict[str, Any],
) -> None:
    # Rendered with Qwen2.5's template and counted by its tokenizer: the rendered
    # conversation minus the bare contents. The estimate of a row is its contents' estimates
    # plus exactly this.
    messages = template["messages"]
    overhead = 5 * len(messages) + (0 if messages[0]["role"] == "system" else 21)
    assert template["real_rendered"] - sum(template["real_contents"]) == overhead
    assert _row_tokens(row_of(messages)) - sum(estimate(m["content"]) for m in messages) == overhead


def test_the_template_constants_are_the_measured_ones() -> None:
    assert _MESSAGE_TOKENS == 5
    assert _DEFAULT_SYSTEM_TOKENS == 21
