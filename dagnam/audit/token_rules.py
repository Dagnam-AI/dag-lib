"""What each kind of piece costs: the fitted constants and the pricing rules of the token estimate.

See :mod:`dagnam.audit.token_estimate` for the principle (every character starts at its UTF-8
bytes and is discounted only on evidence from the vocabulary, each discount with a limit).
"""

from __future__ import annotations

import functools
import math
import re
from typing import Final
import unicodedata

from dagnam.audit.token_classes import (
    CAMEL_SEAM,
    HAS_LETTER,
    NONBASIC,
    PER_CHAR,
    SYMBOL_CHARS,
    THAI_MARK,
)

_LISTED: Final = 1024
"""Longest text whose matches are listed to count them; a longer one is streamed, so memory does
not grow with a word of any length."""

_WORD_TABLE_MAX: Final[dict[str, int]] = {"latin": 36}
_WORD_TABLE_MAX_OTHER: Final = 14
"""The longest whole-word token of each alphabet, in letters: a longer word never consults the
table."""
_WORD_TAIL: Final[dict[str, tuple[float, float, float]]] = {
    "latin": (1.308, 0.258, 0.368),
    "cyrillic": (1.336, 0.325, 1.208),
    "greek": (0.18, 1.0, 0),
    "armenian": (0.982, 0.997, 0),
    "georgian": (0.937, 1.004, 0),
    "hebrew": (0.991, 0.335, 2.127),
    "arabic": (1.041, 0.336, 0.825),
}
"""Per alphabet ``(a, per_letter, per_nonbasic)``: a word that is not a whole-word token costs
``max(1, a + per_letter * letters + per_nonbasic * letters outside the basic set)`` for its
natural letters."""
_NATURAL_LETTERS: Final[dict[str, int]] = {
    "latin": 16,
    "cyrillic": 14,
    "greek": 36,
    "armenian": 36,
    "georgian": 36,
    "hebrew": 8,
    "arabic": 8,
}
"""Letters of a word priced by the natural tail; each letter beyond costs the alphabet's floor."""
_FLOOR: Final[dict[str, float]] = {
    "latin": 0.65,
    "cyrillic": 0.9,
    "greek": 1.1,
    "armenian": 1.0,
    "georgian": 1.0,
    "hebrew": 0.85,
    "arabic": 1.0,
}
"""The lowest price a letter of a random string of the alphabet measured at."""
_CAPS_NATURAL: Final = 8
"""Letters of an ALL-CAPS Latin word priced by the tail (acronyms stop there), then the floor."""
_CAMEL_SEAM_COST: Final = 1.39
"""Added per lowercase-to-uppercase seam in a Latin word of mixed case (camelCase, base64)."""
_LEAD_SYMBOL: Final = 0.5
"""Added to a whole-word token led by an ASCII symbol, not a space (``,paid``): the symbol is
usually a token of its own."""
_NO_LEAD: Final = 0.1
"""Added to a whole-word token with nothing before it: the vocabulary knows fewer words bare."""
RATES: Final[dict[str, float]] = {
    "hiragana": 0.446,
    "katakana": 0.734,
    "hangul": 0.804,
    "thai": 0.566,
    "lao": 1.292,
    "myanmar": 1.609,
    "khmer": 1.227,
    "devanagari": 1.019,
    "bengali": 1.091,
    "gurmukhi": 1.761,
    "gujarati": 1.743,
    "oriya": 2.0,
    "tamil": 1.099,
    "telugu": 1.686,
    "kannada": 1.42,
    "malayalam": 1.305,
    "sinhala": 1.75,
    "ethiopic": 1.6,
    "tibetan": 1.4,
}
"""Tokens per character of a run of a script written without spaces, as measured on natural text
(han is priced by its tokens instead). Oriya (2.0), Kannada (1.42) and Tibetan (1.40) were fitted
on Wikipedia text of their own, with a margin. Sinhala and Ethiopic, thinly held, sit at their
neighbours' level."""
_RUN_CAP: Final[dict[str, int]] = {"hiragana": 8, "katakana": 8, "hangul": 6}
"""Characters of a run priced at a rate below one token; natural words stop there, random text
does not, so each character beyond costs 1."""
SCRIPT_LEAD: Final[dict[str, float]] = {
    "bengali": 1.0,
    "devanagari": 0.6,
    "gujarati": 0.3,
    "gurmukhi": 0.5,
    "han": 1.6,
    "hangul": 0.5,
    "hiragana": 1.2,
    "katakana": 1.5,
    "khmer": 1.3,
    "lao": 1.7,
    "malayalam": 1.9,
    "myanmar": 0.8,
    "tamil": 1.2,
    "telugu": 0.2,
    "thai": 0.9,
}
SCRIPT_LEAD_DEFAULT: Final = 1.0
"""What the space before a run of such a script costs: its own token unless the script has
space-prefixed tokens."""
_HAN_MAX_UNIT: Final = 4
_HAN_UNIT: Final = 1.08
"""A longest match of 2-4 han characters costs a little over one token: the scan sometimes picks
a unit that the BPE does not take."""
_SYMBOL_MAX_UNIT: Final = 3
FRAGILE_WINDOW: Final = 8
"""A fragile unit seen again within this many units costs its characters."""
_FORMAT_CHAR: Final = 1.5
"""A format character (zero-width joiner, byte-order mark, bidi control) in a run of two or more
costs this much even when it is a token: BPE splits the token's bytes."""

# --- pricing ------------------------------------------------------------------------------------


def _bytes(char: str) -> int:
    code = ord(char)
    return 1 if code < 0x80 else 2 if code < 0x800 else 3 if code < 0x10000 else 4


def char_cost(char: str, chars: frozenset[str]) -> float:
    """A character with no better rule: 1 if it is a token, else its UTF-8 bytes.

    A lone surrogate counts 3. The bytes are the ceiling of a byte-level BPE.
    """
    if char in chars:
        return _FORMAT_CHAR if unicodedata.category(char) == "Cf" else 1.0
    return float(_bytes(char))


def _count(pattern: re.Pattern[str], text: str) -> int:
    """How many matches of ``pattern`` ``text`` holds; a long text is streamed, not listed."""
    if len(text) <= _LISTED:
        return len(pattern.findall(text))
    return sum(1 for _ in pattern.finditer(text))


def word_cost(
    piece: str, word: str, alphabet: str, words: frozenset[str], chars: frozenset[str]
) -> float:
    """A word of an alphabet with the space or symbol before it (the whole piece)."""
    letters = len(word)
    lead = piece[0]
    if letters <= _WORD_TABLE_MAX.get(alphabet, _WORD_TABLE_MAX_OTHER) and (
        (word.lower() if alphabet == "cyrillic" else word) in words
    ):
        if lead == " ":
            return 1.0
        if lead.isalpha():
            return 1.0 + _NO_LEAD
        if lead < "\x80":  # an ASCII symbol merges with the word about a third of the time
            return 1.0 + (_LEAD_SYMBOL if "\x1f" < lead != "\x7f" else 1.0)  # a control, never
        return 1.0 + char_cost(lead, chars)
    base, per_letter, per_nonbasic = _WORD_TAIL[alphabet]
    pattern = NONBASIC.get(alphabet)
    nonbasic = _count(pattern, word) if pattern is not None and not word.isascii() else 0
    natural = _NATURAL_LETTERS[alphabet]
    if alphabet == "latin" and letters > 1 and word.isupper():
        natural = min(natural, _CAPS_NATURAL)
    extra = 0.0  # a letter that is no token (an archaic Greek letter, a rare Cyrillic one): bytes
    if not word.isascii():
        for char in word:
            if char > "\x7f" and char not in chars:
                extra += _bytes(char)
                letters -= 1
    priced = min(letters, natural)
    cost = base + per_letter * priced + per_nonbasic * nonbasic
    cost = max(1.0, cost) + _FLOOR[alphabet] * (letters - priced) + extra
    if alphabet == "latin" and not word.islower() and not word.isupper():
        cost += _CAMEL_SEAM_COST * _count(CAMEL_SEAM, word)
    if lead > "\x7f" and not lead.isalpha():  # a zero-width space, a CJK bracket: its own token
        cost += char_cost(lead, chars)
    return cost


def symbols_cost(piece: str, chars: frozenset[str], fragile: frozenset[str]) -> float:
    """A run of ASCII symbols, with its leading space and trailing newlines.

    One for each longest match against the vocabulary's symbol tokens of up to three characters,
    one for each unmatched symbol or control; any other character is :func:`char_cost`. A fragile
    unit seen again within the last 8 units costs its characters (see :func:`han_cost`).
    """
    total = 0.0
    i = 0
    size = len(piece)
    seen: dict[str, int] = {}  # fragile unit -> the index of the unit where it was last seen
    index = 0
    while i < size:
        char = piece[i]
        if char in SYMBOL_CHARS or (char == " " and i == 0) or char in "\r\n":
            for length in range(min(_SYMBOL_MAX_UNIT, size - i), 1, -1):
                unit = piece[i : i + length]
                if unit in chars:
                    total += _unit_cost(unit, 1.0, index, seen, fragile)
                    index += 1
                    i += length
                    break
            else:
                total += 1.0
                i += 1
            continue
        if char < "\x80" or (size == 1 and char in chars):
            total += 1.0  # a control; or a lone token, where no run merges it
        else:
            total += char_cost(char, chars)
        i += 1
    return total


def _unit_cost(
    unit: str, plain: float, index: int, seen: dict[str, int], fragile: frozenset[str]
) -> float:
    """What one matched unit costs.

    ``plain``, or its characters when it is a fragile unit seen within the last
    ``FRAGILE_WINDOW`` units. Records where a fragile unit was seen.
    """
    if unit not in fragile:
        return plain
    cost = (
        float(len(unit)) if index - seen.get(unit, -1 - FRAGILE_WINDOW) <= FRAGILE_WINDOW else plain
    )
    seen[unit] = index
    return cost


def _segment_cost(char: str, size: int, chars: frozenset[str]) -> float:
    """Tokens of ``size`` copies of one whitespace character.

    From the exhaustive measurement of runs of 1 to 4,096; never below the real count.
    """
    if char == " ":  # one token up to 81 spaces, then one more for each 128
        return float(math.ceil((size + 47) / 128))
    if char == "\n":  # 1 up to 12 newlines, 2 up to 28, 3 up to 60, 4 up to 64, then 1 per 30
        for limit, tokens in ((12, 1), (28, 2), (60, 3), (64, 4)):
            if size <= limit:
                return float(tokens)
        return float(math.ceil((size + 28) / 30))
    if char == "\t":
        return float(math.ceil(size / 16))
    if char == "\xa0":
        return float(math.ceil((size + 4) / 8))
    if char == "　":
        return float(math.ceil(size / 2))
    return size * char_cost(char, chars)  # CR, VT, FF and the rest: 1 each if a token, else bytes


def whitespace_cost(run: str, chars: frozenset[str]) -> float:
    """A run of whitespace, cut into maximal segments of one character.

    A CRLF pair is one kind of character. Each segment is priced by :func:`_segment_cost`.
    """
    total = 0.0
    i = 0
    size = len(run)
    while i < size:
        char = run[i]
        j = i + 1
        if char == "\r" and j < size and run[j] == "\n":
            while j + 1 < size and run[j] == "\n" and run[j - 1] == "\r" and run[j + 1] == "\r":
                j += 2
            if j < size and run[j] == "\n" and run[j - 1] == "\r":
                j += 1
            total += math.ceil((j - i) / 2 / 4)
        else:
            while j < size and run[j] == char:
                j += 1
            total += _segment_cost(char, j - i, chars)
        i = j
    return total


def han_cost(run: str, chars: frozenset[str], fragile: frozenset[str]) -> float:
    """A run of han characters.

    The longest match against the vocabulary's han tokens of 2-4 characters costs 1.08; any other
    character is :func:`char_cost`. A fragile unit, one that BPE does not re-merge when it recurs,
    seen again within the last 8 units of the run costs its characters: one 4-character token
    repeated costs up to 2.98 real tokens a copy, and short cycles of different units cost as much.
    """
    total = 0.0
    i = 0
    size = len(run)
    seen: dict[str, int] = {}  # fragile unit -> the index of the unit where it was last seen
    index = 0
    while i < size:
        for length in range(min(_HAN_MAX_UNIT, size - i), 1, -1):
            unit = run[i : i + length]
            if unit in chars:
                total += _unit_cost(unit, _HAN_UNIT, index, seen, fragile)
                index += 1
                i += length
                break
        else:
            total += char_cost(run[i], chars)
            i += 1
    return total


@functools.cache
def _non_token_letters(chars: frozenset[str]) -> re.Pattern[str]:
    """One regex for the letters of the scripts at a rate of one or more that are no token.

    Built once per table (a few hundred characters).
    """
    letters: list[str] = []
    for kind, rate in RATES.items():
        if rate >= 1.0:
            for low, high in re.findall(r"(.)-(.)", PER_CHAR[kind], flags=re.DOTALL):
                letters.extend(
                    char
                    for char in map(chr, range(ord(low), ord(high) + 1))
                    if char.isalpha() and char not in chars
                )
    return re.compile("[" + "".join(map(re.escape, letters)) + "]" if letters else "(?!)")


def script_cost(kind: str, run: str, chars: frozenset[str]) -> float:
    """A run of a script written without spaces.

    A rate of one token a character or more is the measured one, for natural text, whose letters
    are tokens; marks or signs with no letter to merge into cost a token or more each, and a
    letter that is no token costs its bytes. Below one the rate holds for the first characters of
    a run only; after that, and for a letter that is no token, a character costs 1 or its bytes
    (Thai's vowel signs and tone marks keep the rate: they merge with their consonant).
    """
    rate = RATES[kind]
    if rate >= 1.0:
        if not HAS_LETTER.search(run):
            return float(sum(char_cost(char, chars) for char in run))
        odd = _non_token_letters(chars).findall(run)
        floor = len(run) + sum(_bytes(char) - 1 for char in odd)
        return max(rate * len(run), floor) if odd else rate * len(run)
    cap = _RUN_CAP.get(kind, len(run))
    total = 0.0
    for i, char in enumerate(run):
        if char not in chars and not THAI_MARK.match(char):
            total += _bytes(char)
        elif i >= cap:
            total += 1.0
        else:
            total += rate
    return total
