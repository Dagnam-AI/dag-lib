#!/usr/bin/env python3
"""Rebuild the vocabulary tables of the token estimate from the student's ``tokenizer.json``.

Writes ``dagnam/audit/student_words.z``, ``student_chars.z`` and ``student_fragile.z`` for
``Qwen/Qwen2.5-0.5B-Instruct``. Each is the exact list of its keys, sorted, front-coded and
compressed with zlib (see ``dagnam/audit/token_tables.py``); none holds a token id.

``student_words.z`` holds every vocabulary token that is one space followed only by letters:
Latin in the case it is written (``the``, ``The`` and ``THE`` are three entries), Cyrillic
lowercased, Hebrew and Arabic as they are.

``student_chars.z`` holds: every token that is one non-ASCII character (not led by a space),
U+FFFD included; every token of two to four han characters; every token of two to eight ASCII
symbols with its leading space and trailing newlines; and every token that is a space and one
letter.

``student_fragile.z`` holds the han and symbol units that the tokenizer does not re-merge when
they recur: those whose eight copies take more than 8.8 tokens. That needs the tokenizer's own
tokenization, so this script carries a small byte-level BPE (the pre-tokenizer pattern and the
merges of ``tokenizer.json``) and uses it for that and for the two ``encode`` checks above. It
matches the real tokenizer on the units the tables need, not on text in general: it counts numerals
such as fractions and superscripts as letters, where the tokenizer splits them, which only
over-counts.

It prints the entry counts, the sizes and the SHA-256 of each table. After a rebuild for a new
tokenizer set the ``STUDENT_*_BYTES`` and ``STUDENT_*_SHA256`` constants in
``dagnam/audit/token_tables.py`` to them and refit the constants of ``token_rules.py``: the
tables alone do not make the estimate fit another tokenizer.

Standard library only; nothing is downloaded. Fetch ``tokenizer.json`` yourself, from the model's
page.

Usage:
    python scripts/make_student_tables.py path/to/tokenizer.json [--out-dir DIR]
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
import unicodedata

from dagnam.audit.token_classes import HAN_CLASS, SYMBOL_CHARS
from dagnam.audit.token_tables import (
    STUDENT_CHARS_FILE,
    STUDENT_FRAGILE_FILE,
    STUDENT_WORDS_FILE,
    pack,
)

ROOT = Path(__file__).resolve().parents[1]
HAN = re.compile(f"^[{HAN_CLASS}]+$")
QWEN_PIECES = re.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|(?:[^\r\n\w]|_)?[^\W\d_]+|\d"
    r"| ?(?:[^\s\w]|_)+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
"""The tokenizer's pre-tokenizer pattern, in Python's ``re``."""
FRAGILE_COPIES = 8
FRAGILE_ABOVE = 8.8


def byte_decoder() -> dict[str, int]:
    """The byte-level BPE alphabet's inverse: the character a vocabulary file writes for each byte.

    Printable Latin-1 bytes stand for themselves; the other bytes are written as the characters
    from U+0100 up, in order.
    """
    kept = [*range(ord("!"), ord("~") + 1), *range(0xA1, 0xAD), *range(0xAE, 0x100)]
    extra = (b for b in range(256) if b not in kept)
    characters = {b: chr(b) for b in kept}
    characters.update({b: chr(256 + i) for i, b in enumerate(extra)})
    return {char: byte for byte, char in characters.items()}


def decoded(vocab: Mapping[str, int]) -> list[tuple[int, str]]:
    """``(id, text)`` of each vocabulary token that is whole characters of UTF-8."""
    decoder = byte_decoder()
    found: list[tuple[int, str]] = []
    for token, index in vocab.items():
        try:
            found.append((index, bytes(decoder[char] for char in token).decode("utf-8")))
        except (KeyError, UnicodeDecodeError):  # not bytes of the alphabet, or half a character
            continue
    return found


def make_encoder(vocab: Mapping[str, int], merges: list[str]) -> Callable[[str], list[int]]:
    """A byte-level BPE: NFC, the pre-tokenizer pattern, then the merges by rank."""
    encode_byte = {byte: char for char, byte in byte_decoder().items()}
    ranks = {tuple(merge.split(" ")): rank for rank, merge in enumerate(merges)}
    cache: dict[str, list[int]] = {}

    def merged(piece: str) -> list[int]:
        symbols = [encode_byte[byte] for byte in piece.encode("utf-8", "surrogatepass")]
        while len(symbols) > 1:
            ranked = [
                (ranks[pair], i)
                for i, pair in enumerate(itertools.pairwise(symbols))
                if pair in ranks
            ]
            if not ranked:
                break
            i = min(ranked)[1]
            symbols[i : i + 2] = [symbols[i] + symbols[i + 1]]
        return [vocab[symbol] for symbol in symbols]

    def encode(text: str) -> list[int]:
        ids: list[int] = []
        for match in QWEN_PIECES.finditer(unicodedata.normalize("NFC", text)):
            piece = match.group()
            if piece not in cache:
                cache[piece] = merged(piece)
            ids.extend(cache[piece])
        return ids

    return encode


def whole_words(texts: list[str]) -> list[str]:
    """The sorted, distinct whole-word entries among the texts of a byte-level BPE vocabulary."""
    words: set[str] = set()
    for text in texts:
        word = text[1:]
        if not text.startswith(" ") or not word.isalpha():
            continue
        first = ord(word[0])
        if word.isascii() or first < 0x250 or 0x1E00 <= first <= 0x1EFF:
            words.add(word)  # Latin, in its own case: " The", " USB" and " the" differ
        elif 0x400 <= first <= 0x52F:
            words.add(word.lower())  # Cyrillic is looked up lowercased
        elif 0x5D0 <= first <= 0x5EA or 0x600 <= first <= 0x6FF or 0x750 <= first <= 0x77F:
            words.add(word)  # Hebrew, Arabic
    return sorted(words)


def char_units(
    tokens: list[tuple[int, str]], encode: Callable[[str], list[int]]
) -> tuple[list[str], list[str], list[str]]:
    """The sorted ``(single characters, han tokens of 2-4, ASCII symbol tokens)`` of a vocabulary.

    U+FFFD counts as a character when it is the token its own encoding gives, and so does a space
    and one letter that encodes to that one token; any other text holding U+FFFD is a byte
    fallback, not a character.
    """
    singles: set[str] = set()
    han: set[str] = set()
    symbols: set[str] = set()
    for index, text in tokens:
        if text == "�" and encode(text) == [index]:
            singles.add(text)
        elif "�" in text:
            continue
        elif len(text) == 2 and text[0] == " " and text[1].isalpha() and encode(text) == [index]:
            singles.add(text)
        else:
            led = text.startswith(" ")
            word = text[1:] if led else text
            if len(word) == 1 and not word.isascii() and not led:
                singles.add(word)
            elif 2 <= len(word) <= 4 and HAN.match(word):
                han.add(word)
            elif (
                2 <= len(text) <= 8
                and all(c in SYMBOL_CHARS or c in "\r\n" for c in word)
                and any(c in SYMBOL_CHARS for c in word)
                and (len(word) >= 2 or led)
            ):
                symbols.add(text)  # with its space and newlines: ' {"', '{"' and ';\n' differ
    return sorted(singles), sorted(han), sorted(symbols)


def fragile_units(
    han: list[str], symbols: list[str], encode: Callable[[str], list[int]]
) -> list[str]:
    """The units whose eight copies take more than 8.8 tokens, sorted.

    Greedy matching assumes one token a copy; these are the units the tokenizer does not re-merge
    when they recur, alone or in short cycles. Symbol units led by a space or holding a line break
    are not measured.
    """
    candidates = [
        *han,
        *(s for s in symbols if not s.startswith(" ") and "\n" not in s and "\r" not in s),
    ]
    return sorted(unit for unit in candidates if len(encode(unit * FRAGILE_COPIES)) > FRAGILE_ABOVE)


def _script(word: str) -> str:
    first = ord(word[0])
    if 0x400 <= first <= 0x52F:
        return "cyrillic"
    if 0x5D0 <= first <= 0x5EA:
        return "hebrew"
    return "arabic" if 0x600 <= first <= 0x77F else "latin"


def main(argv: list[str] | None = None) -> int:
    """Write the three tables and report them; exit status 1 when the vocabulary has no entries."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tokenizer", type=Path, help="the student's tokenizer.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "dagnam" / "audit")
    args = parser.parse_args(argv)
    model = json.loads(args.tokenizer.read_text(encoding="utf-8"))["model"]
    vocab: dict[str, int] = model["vocab"]
    tokens = decoded(vocab)
    words = whole_words([text for _, text in tokens])
    singles, han, symbols = char_units(tokens, make_encoder(vocab, model["merges"]))
    if not words or not (singles and han and symbols):
        sys.stderr.write(
            f"{args.tokenizer}: no whole-word or character tokens: is it a byte-level BPE file?\n"
        )
        return 1
    fragile = fragile_units(han, symbols, make_encoder(vocab, model["merges"]))
    tables = {
        STUDENT_WORDS_FILE: pack(words),
        STUDENT_CHARS_FILE: pack(singles + han + symbols),
        STUDENT_FRAGILE_FILE: pack(fragile),
    }
    by_script = dict(Counter(_script(word) for word in words))
    sys.stdout.write(
        f"{len(words)} whole words {by_script}\n"
        f"{len(singles)} characters + {len(han)} han tokens + {len(symbols)} symbol tokens\n"
        f"{len(fragile)} fragile units\n"
    )
    for name, blob in tables.items():
        (args.out_dir / name).write_bytes(blob)
        sys.stdout.write(
            f"wrote {name}: {len(blob)} bytes, sha256 {hashlib.sha256(blob).hexdigest()}\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
