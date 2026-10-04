r"""The three vocabulary tables the token estimate reads, and how it reads them.

Both are exact lists of keys taken from the vocabulary of the student model,
``Qwen2.5-0.5B-Instruct``, sorted, front-coded and compressed with zlib, shipped as package data and
decoded into a ``frozenset`` the first time they are used (about 10 ms and 4-5 MB each). A key is
a member only if it is a vocabulary token: there are no false positives.

* ``student_words.z``: every whole-word token (a space followed only by letters), the word without
  the space; Latin in the case it is written, Cyrillic lowercased, Hebrew and Arabic as they are.
  49,926 keys.
* ``student_chars.z``: every token that is one non-ASCII character (a character in it costs one
  token, any other costs its UTF-8 bytes, U+FFFD included), every token of two to four han
  characters, every token of two to eight ASCII symbols (with its leading space and trailing
  newlines, when it has them), and every token that is a space and one letter (501 of them).
  39,637 keys.
* ``student_fragile.z``: the 678 han and symbol units that BPE does not re-merge when they recur
  (eight copies cost more than 8.8 real tokens): one seen again within the last eight units of a
  run costs its characters, not one token.

Each file is read once, through :mod:`importlib.resources`, and its length and SHA-256 are checked
against the constants below, so a missing or changed table stops the estimate with a clear error
instead of silently giving other numbers. ``scripts/make_student_tables.py`` rebuilds them from a
``tokenizer.json``; change the pinned digests with it.

File format. Each key is one line: a printable ASCII character holding 0x20 plus the number of
leading characters it shares with the previous key (at most 94), then the rest of the key in
UTF-8 with backslash, newline and carriage return written as ``\\``, ``\n`` and ``\r``, then a
newline.
"""

from __future__ import annotations

from collections.abc import Iterable
import functools
import hashlib
from importlib import resources
import re
from typing import Final
import zlib

STUDENT_WORDS_FILE: Final = "student_words.z"
STUDENT_WORDS_BYTES: Final = 94_338
STUDENT_WORDS_SHA256: Final = "e762f4642356db255f9c1a9cbe76eeda980e15de089c17332120ca7d83bdf981"
STUDENT_CHARS_FILE: Final = "student_chars.z"
STUDENT_CHARS_BYTES: Final = 94_857
STUDENT_CHARS_SHA256: Final = "71b57243ccedcf3a94b6c72e336fbe96f225a3579c484beafe8816b4b6b5ffad"
STUDENT_FRAGILE_FILE: Final = "student_fragile.z"
STUDENT_FRAGILE_BYTES: Final = 2_018
STUDENT_FRAGILE_SHA256: Final = "b21b4b02a53e449c589c2f920b0d834444c6aade5200b500ddbfaf0ec4b2dc60"
MAX_SHARED_PREFIX: Final = 94
_ESCAPES: Final = {"\\": "\\", "n": "\n", "r": "\r"}
_ESCAPE: Final = re.compile(r"\\([\\nr])")


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")


def _unescape(text: str) -> str:
    r"""Undo :func:`_escape` in one left-to-right pass: ``\\n`` is a backslash and an ``n``."""
    return _ESCAPE.sub(lambda match: _ESCAPES[match.group(1)], text) if "\\" in text else text


def pack(keys: Iterable[str]) -> bytes:
    """The table file of ``keys``: sorted, front-coded, zlib at level 9. Used by the generator."""
    out = bytearray()
    previous = ""
    for key in sorted(set(keys)):
        shared = 0
        limit = min(len(key), len(previous), MAX_SHARED_PREFIX)
        while shared < limit and key[shared] == previous[shared]:
            shared += 1
        out.append(0x20 + shared)
        out += _escape(key[shared:]).encode("utf-8")
        out.append(10)
        previous = key
    return zlib.compress(bytes(out), 9)


def unpack(blob: bytes) -> frozenset[str]:
    """The keys of a table file."""
    keys: list[str] = []
    previous = ""
    for line in zlib.decompress(blob).decode("utf-8").split("\n"):
        if line:
            previous = previous[: ord(line[0]) - 0x20] + _unescape(line[1:])
            keys.append(previous)
    return frozenset(keys)


def _read(name: str, length: int, digest: str) -> frozenset[str]:
    path = f"dagnam.audit/{name}"
    try:
        blob = resources.files("dagnam.audit").joinpath(name).read_bytes()
    except OSError as exc:
        raise RuntimeError(f"the token estimate's table {path} cannot be read: {exc}") from exc
    if len(blob) != length or hashlib.sha256(blob).hexdigest() != digest:
        raise RuntimeError(
            f"the token estimate's table {path} is not the one it was fitted with"
            f" ({len(blob)} bytes, expected {length} with sha256 {digest}); reinstall dagnam"
        )
    return unpack(blob)


@functools.cache
def student_words() -> frozenset[str]:
    """The whole-word tokens, read once from the package data.

    Raises:
        RuntimeError: the file is missing or is not the table the estimate was fitted with.
            There is no fallback, because another table would silently give other estimates.
    """
    return _read(STUDENT_WORDS_FILE, STUDENT_WORDS_BYTES, STUDENT_WORDS_SHA256)


@functools.cache
def student_chars() -> frozenset[str]:
    """The character, han, symbol and space-plus-letter tokens, read once from the package data.

    Raises:
        RuntimeError: the file is missing or is not the table the estimate was fitted with.
    """
    return _read(STUDENT_CHARS_FILE, STUDENT_CHARS_BYTES, STUDENT_CHARS_SHA256)


@functools.cache
def student_fragile() -> frozenset[str]:
    """The fragile units, read once from the package data.

    Raises:
        RuntimeError: the file is missing or is not the table the estimate was fitted with.
    """
    return _read(STUDENT_FRAGILE_FILE, STUDENT_FRAGILE_BYTES, STUDENT_FRAGILE_SHA256)
