"""Estimate how many tokens the student model reads in a text, without its tokenizer.

The student is ``Qwen2.5-0.5B-Instruct``. The scan needs to know whether a training row fits its
2,048-token context, and installing or downloading the tokenizer to find out is not an option for
an SDK. The estimate is pure standard library, one pass over the text, and guesses no language.

How it is built. The student's tokenizer normalises its input to NFC and then, being a byte-level
BPE, spends on each piece at most as many tokens as the piece has UTF-8 bytes in that form. The
estimate does the same normalisation, starts every character at its byte count, discounts it only
where the vocabulary itself shows a merge exists, each discount with a structural limit, and caps
every piece at its bytes (:mod:`dagnam.audit.token_rules` has the rules and their constants):

* a character that is a token of its own costs 1 (``student_chars.z``), any other character its
  bytes (a lone surrogate counts 3); a format character in a run costs 1.5;
* a word that is a whole-word token (``student_words.z``, in the case it is written) costs 1, but
  only up to the longest such token (36 letters, 14 in other alphabets);
* any other word of an alphabet costs the natural-language tail (a line in its length) up to the
  length where natural words stop (16 Latin letters, 14 Cyrillic, 8 Arabic and Hebrew), and
  every letter after that the lowest price a string of random letters measured at;
* a run of han characters costs 1.08 for each longest match against the vocabulary's tokens of
  2-4 han characters; a run of ASCII symbols costs 1 for each longest match of up to 3; a unit
  of either kind that the tokenizer does not re-merge when it recurs (678 of them,
  ``student_fragile.z``) costs its characters when seen again within the last 8 units;
* whitespace is priced per run of one character at the period measured for runs of 1-4,096;
* a script written without spaces costs the rate measured on natural text of that script; the
  rates below one token a character (hiragana, katakana, Hangul, Thai) hold for the first 8
  characters of a run (6 for Hangul), and a letter that is no token costs its bytes.

What it guarantees, and what it does not, measured on 2,498 rows of 500 or more real tokens held
out of the fit and on 213 texts built to be hard:

* at least 0.95 of the real count on natural text of the measured kinds: 0.954 at the lowest,
  0.971 at the 1st percentile, 1.04 at the median and 1.22 at the 95th percentile. Measured: the
  Latin, Cyrillic, Greek, Armenian, Georgian, Hebrew and Arabic alphabets; Simplified and
  Traditional Chinese, Japanese and Korean; Thai, Lao, Khmer and Myanmar; Devanagari, Bengali,
  Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam and Sinhala; Ethiopic and Tibetan;
  code, JSON, identifiers, tables and emoji. Not measured, because no natural text was available,
  and priced at the safe byte price for every letter that is not a token: Syriac, Thaana, N'Ko,
  Mongolian, Cherokee, Canadian syllabics, Tifinagh, Vai, Bamum, Javanese, Balinese, Sundanese,
  Tai Le, Tai Viet, Lisu, Yi, Ol Chiki, Osmanya, Deseret, Gothic, Glagolitic and Coptic. It is high,
  by design, for code (median 1.12, 1.26 at the 95th percentile, 1.77 at most), German, Spanish,
  French, Portuguese, Italian and Indonesian (1.2) and lists of labels (1.17); short texts round up
  (a one-word text of one token counts 2);
* on random or crafted text it can count about half the real tokens, and no lower bound is
  promised. Measured on independent generators: random letters of natural word length, lowest 0.486
  (random Cyrillic words of 14 letters); random text of the scripts priced per character, lowest
  0.504 (random Thai); the scripts priced at one token a character or more, 0.547 to 0.62 (one
  number sign repeated, random Tibetan); licence keys 0.75; base32 on plausible text 0.88 to 0.90.
  Text crafted against the rules goes lower: 0.36 for two han units that are not in the fragile
  table, alternated. A row estimated to fit the student's 2,048 tokens can therefore exceed that
  cap by up to about 2.8 times on crafted text;
* a known limit of the fragile window: it sees a unit that recurs within 8 units, not pairs of
  units that break each other (two han units outside the fragile table, alternated: 0.36) nor a
  cycle of 9 or more different fragile han units (0.68 to 0.82);
* never above the UTF-8 byte count of the NFC form of the text, for any text;
* memory flat in the input size, apart from the three tables (about 13 MB once in use), the
  normalised copy of text that is not already NFC, and a few copies of one unbroken word.

Over-counts are bounded by the byte ceiling above and measured: a symbol repeated 8,000 times is 21
times its real count, a fragile 4-character han unit repeated 4 times, code with long identifiers
3. A single character that is a token on its own but splits beside one neighbour (one currency
sign after a Hangul syllable, 0.50) is not modelled.

The tables are exact lists of the vocabulary's keys, compressed, read through
``dagnam.audit.token_tables``; the pieces are cut by Qwen's own pre-tokenizer pattern
(:mod:`dagnam.audit.token_classes`), with the repeat over symbols made possessive so that a run
of any length is matched in constant memory.
"""

from __future__ import annotations

import math
import unicodedata

from dagnam.audit.token_classes import CJK, CJK_CHAR, pieces
from dagnam.audit.token_rules import (
    RATES,
    SCRIPT_LEAD,
    SCRIPT_LEAD_DEFAULT,
    char_cost,
    han_cost,
    script_cost,
    symbols_cost,
    whitespace_cost,
    word_cost,
)
from dagnam.audit.token_tables import student_chars, student_fragile, student_words

__all__ = ["estimate", "pieces"]


def estimate(text: str) -> int:
    """Estimated Qwen2.5 token count of ``text`` alone (no template, no special tokens).

    The text is first brought to NFC, as the tokenizer does (its normalizer is NFC);
    ``is_normalized`` avoids the copy when it already is. Every piece is then capped at the UTF-8
    bytes of its NFC form, which a byte-level BPE never exceeds in tokens. Rounded up once, so a
    text of one token can count two. The estimate is not below the real count by more than the
    limits in the module docstring. Each call stands alone: a caller counting a conversation
    message by message gets the sum of the messages.

    Raises:
        RuntimeError: a vocabulary table is missing or corrupt (see
            :func:`dagnam.audit.token_tables.student_words`).
    """
    words = student_words()
    chars = student_chars()
    fragile = student_fragile()
    if not unicodedata.is_normalized("NFC", text):
        text = unicodedata.normalize("NFC", text)
    total = 0.0
    previous = ""  # the previous piece's kind: kanji and kana touching each other share tokens
    for match in pieces(text):
        kind = match.lastgroup
        piece = match.group()
        if kind is None:  # a contraction
            cost = 1.0
        elif kind == "space":
            cost = whitespace_cost(piece, chars)
        elif kind == "han" or kind in RATES:
            run = match.group(kind)
            cost = han_cost(run, chars, fragile) if kind == "han" else script_cost(kind, run, chars)
            end = match.end()
            if piece[0] == " ":
                cost += SCRIPT_LEAD.get(kind, SCRIPT_LEAD_DEFAULT)
                if len(run) == 1 and piece not in chars:
                    cost = max(cost, 2.0)  # a space and one letter that are not one token
            elif not (
                kind in CJK and (previous in CJK or (end < len(text) and CJK_CHAR.match(text[end])))
            ):
                cost = max(cost, 1.0)  # a run costs a token, unless kana or kanji touch each other
        elif kind == "digit":
            cost = 1.0 if piece.isascii() else char_cost(piece, chars)
        elif kind == "symbols":
            cost = symbols_cost(piece, chars, fragile)
        elif kind == "letters":
            run = match.group("letters")
            if piece[0] == " ":  # a space before a 4-byte letter merges with its first byte
                lead = 2.0 if ord(run[0]) >= 0x10000 else 1.0
            elif piece[0].isalpha():
                lead = 0.0
            else:  # a symbol 1, a private-use or unassigned character its bytes
                lead = char_cost(piece[0], chars)
            cost = sum(char_cost(char, chars) for char in run) + lead
        else:
            word = match.group(kind)
            cost = word_cost(piece, word, kind, words, chars)
            if len(word) == 1 and piece[0] == " " and piece not in chars:
                cost = max(cost, 2.0)
        if cost > len(piece):  # the ceiling: never more tokens than the piece's UTF-8 bytes
            cost = min(cost, float(len(piece.encode("utf-8", "surrogatepass"))))
        total += cost
        previous = kind or ""
    return math.ceil(total)
