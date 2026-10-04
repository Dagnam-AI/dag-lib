r"""Character classes and the piece pattern of the token estimate.

The student's pre-tokenizer pattern, translated to Python's ``re`` (``\p{L}`` as ``[^\W\d_]``),
with a named group for every class the estimate prices differently. Ranges are written as
escapes: several are look-alikes of ASCII.
"""

from __future__ import annotations

from collections.abc import Iterator
import re
from typing import Final

PER_CHAR: Final[dict[str, str]] = {
    "han": "\u2e80-\u2fdf\u3005-\u3007\u3021-\u3029\u3038-\u303b\u3400-\u4dbf\u4e00-\u9fff"
    "\uf900-\ufaff\U00020000-\U0002ffff",
    "hiragana": "\u3041-\u309f",
    "katakana": "\u30a0-\u30ff\u31f0-\u31ff\uff66-\uff9f",
    "hangul": "\u1100-\u11ff\u3130-\u318f\uac00-\ud7af",
    "thai": "\u0e01-\u0e3a\u0e40-\u0e4e",
    "lao": "\u0e81-\u0edf",
    "myanmar": "\u1000-\u109f",
    "khmer": "\u1780-\u17dd",
    "devanagari": "\u0900-\u0963\u0972-\u097f",
    "bengali": "\u0980-\u09e3\u09f0-\u09fb",
    "gurmukhi": "\u0a00-\u0a63\u0a70-\u0a75",
    "gujarati": "\u0a80-\u0ae3\u0af9-\u0aff",
    "oriya": "\u0b00-\u0b63\u0b70-\u0b77",
    "tamil": "\u0b80-\u0be3",
    "telugu": "\u0c00-\u0c63",
    "kannada": "\u0c80-\u0ce3",
    "malayalam": "\u0d00-\u0d63\u0d7a-\u0d7f",
    "sinhala": "\u0d80-\u0df4",
    "ethiopic": "\u1200-\u139f",
    "tibetan": "\u0f00-\u0fff",
}
"""Scripts priced per character: letters and their marks together, one rate each."""

ALPHABETS: Final[dict[str, str]] = {
    "latin": "A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u024f\u1e00-\u1eff",
    "cyrillic": "\u0400-\u052f",
    "greek": "\u0370-\u03ff\u1f00-\u1fff",
    "armenian": "\u0530-\u058f",
    "georgian": "\u10a0-\u10ff",
    "hebrew": "\u0591-\u05f4",
    "arabic": "\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff",
}
"""Scripts priced per word: a run of letters, with the space or symbol before it."""

BASIC: Final[dict[str, str]] = {
    "latin": "\x00-\x7f",
    "cyrillic": "\u0410-\u044f\u0401\u0451",
    "hebrew": "\u05d0-\u05ea",
    "arabic": "\u0621-\u064a",
}
"""The letters the vocabulary knows best (ASCII, Russian's, Hebrew's and Arabic's own); a letter of
the alphabet outside its set costs extra."""

HAN_CLASS: Final = PER_CHAR["han"]
"""The character class of han characters, as the body of a ``[...]`` regex class."""
PER_CHAR_CLASS: Final = "".join(PER_CHAR.values())
PIECE: Final = re.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    "| ?(?:" + "|".join(f"(?P<{n}>[{c}]+)" for n, c in PER_CHAR.items()) + ")"
    r"|(?:[^\r\n\w]|_)?(?:"
    + "|".join(f"(?P<{n}>[{c}]+)" for n, c in ALPHABETS.items())
    + rf"|(?P<letters>[^\W\d_{PER_CHAR_CLASS}]+))"
    r"|(?P<digit>\d)"
    r"| ?(?P<symbols>(?:[^\s\w]|_)++)[\r\n]*"
    r"|(?P<space>\s*[\r\n]+|\s+(?!\S)|\s+)"
)
"""Qwen2's pre-tokenizer regex, with ``[^\\W\\d_]`` for ``\\p{L}``, a script-run alternative placed
before the word alternatives, and a named group per class. The order of the alternatives is
Qwen's, and the lead ``[^\\r\\n\\w]|_`` keeps ``_status`` two pieces. The repeat of the symbol run
is possessive (``++``): the backtracking form keeps a stack frame for each character, about a
gigabyte for a megabyte of one symbol. A match with no named group is a contraction."""

NONBASIC: Final[dict[str, re.Pattern[str]]] = {
    name: re.compile(f"[^{chars}]") for name, chars in BASIC.items()
}
CAMEL_SEAM: Final = re.compile(r"[a-z][A-Z]")
THAI_MARK: Final = re.compile("[\u0e31\u0e34-\u0e3a\u0e47-\u0e4e]")
"""Thai vowel signs and tone marks: they merge with their consonant, so they keep the Thai rate."""
SYMBOL_CHARS: Final = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
CJK: Final = frozenset({"han", "hiragana", "katakana"})
"""Scripts whose runs share tokens where they touch (kanji and kana)."""
CJK_CHAR: Final = re.compile(f"[{PER_CHAR['han']}{PER_CHAR['hiragana']}{PER_CHAR['katakana']}]")
HAS_LETTER: Final = re.compile(r"[^\W\d_]")


def pieces(text: str) -> Iterator[re.Match[str]]:
    """The pieces of ``text``, streamed; nothing is kept but the current match."""
    return PIECE.finditer(text)
