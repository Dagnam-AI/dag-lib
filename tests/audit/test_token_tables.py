"""The vocabulary tables behind the token estimate: contents, format, loading and failures."""

from __future__ import annotations

from collections.abc import Iterator
import hashlib
from importlib import resources
from pathlib import Path
import random
import string
from typing import Any, override
import zipfile
import zlib

import pytest

from dagnam.audit import token_tables
from dagnam.audit.token_estimate import estimate
from dagnam.audit.token_tables import (
    STUDENT_CHARS_BYTES,
    STUDENT_CHARS_FILE,
    STUDENT_CHARS_SHA256,
    STUDENT_FRAGILE_BYTES,
    STUDENT_FRAGILE_FILE,
    STUDENT_FRAGILE_SHA256,
    STUDENT_WORDS_BYTES,
    STUDENT_WORDS_FILE,
    STUDENT_WORDS_SHA256,
    pack,
    student_chars,
    student_fragile,
    student_words,
    unpack,
)

PACKAGE = resources.files("dagnam.audit")
WORDS = PACKAGE.joinpath(STUDENT_WORDS_FILE).read_bytes()
CHARS = PACKAGE.joinpath(STUDENT_CHARS_FILE).read_bytes()
FRAGILE = PACKAGE.joinpath(STUDENT_FRAGILE_FILE).read_bytes()
BOTH = {STUDENT_WORDS_FILE: WORDS, STUDENT_CHARS_FILE: CHARS, STUDENT_FRAGILE_FILE: FRAGILE}


class FakePackage:
    """What ``importlib.resources.files`` returns: the files in ``data``, or ``error`` on a read."""

    def __init__(self, data: dict[str, bytes], error: OSError | None = None) -> None:
        self._data = data
        self._error = error
        self._name = ""

    def joinpath(self, name: str) -> FakePackage:
        served = type(self)(self._data, self._error)
        served._name = name
        return served

    def read_bytes(self) -> bytes:
        if self._error is not None:
            raise self._error
        if self._name not in self._data:
            raise FileNotFoundError(self._name)
        return self._data[self._name]


@pytest.fixture
def fresh_tables() -> Iterator[None]:
    """Forget the loaded tables, before and after."""
    for function in (student_words, student_chars, student_fragile):
        function.cache_clear()
    yield
    for function in (student_words, student_chars, student_fragile):
        function.cache_clear()


def serve(monkeypatch: pytest.MonkeyPatch, package: Any) -> None:
    monkeypatch.setattr(resources, "files", lambda name: package)


# --- the tables -------------------------------------------------------------------------------


def test_the_tables_are_the_ones_the_estimate_was_fitted_with() -> None:
    assert len(WORDS) == STUDENT_WORDS_BYTES == 94_338
    assert hashlib.sha256(WORDS).hexdigest() == STUDENT_WORDS_SHA256
    assert (
        STUDENT_WORDS_SHA256 == "e762f4642356db255f9c1a9cbe76eeda980e15de089c17332120ca7d83bdf981"
    )
    assert len(CHARS) == STUDENT_CHARS_BYTES == 94_857
    assert hashlib.sha256(CHARS).hexdigest() == STUDENT_CHARS_SHA256
    assert (
        STUDENT_CHARS_SHA256 == "71b57243ccedcf3a94b6c72e336fbe96f225a3579c484beafe8816b4b6b5ffad"
    )
    assert len(FRAGILE) == STUDENT_FRAGILE_BYTES == 2_018
    assert hashlib.sha256(FRAGILE).hexdigest() == STUDENT_FRAGILE_SHA256
    assert (
        STUDENT_FRAGILE_SHA256 == "b21b4b02a53e449c589c2f920b0d834444c6aade5200b500ddbfaf0ec4b2dc60"
    )
    assert len(student_words()) == 49_926
    # 17,675 characters, U+FFFD, 501 space-and-letter tokens, 16,382 han and 5,078 symbol tokens.
    assert len(student_chars()) == 17_675 + 1 + 501 + 16_382 + 5_078 == 39_637
    assert len(student_fragile()) == 678


def test_the_shipped_files_are_the_packed_form_of_their_keys() -> None:
    # Packing the decoded keys again gives the file back (the format has one spelling).
    assert pack(student_words()) == WORDS
    assert pack(student_chars()) == CHARS
    assert pack(student_fragile()) == FRAGILE


@pytest.mark.parametrize(
    "word",
    [
        "the",
        "and",
        "und",
        "der",
        "les",
        "para",
        "och",
        "dan",
        "ett\u00e4",
        "\u0434\u043b\u044f",
        "\u0645\u0646",
        "\u05e9\u05dc",
        "Paris",
        "MEN",
    ],
)
def test_the_words_of_many_languages_are_in_the_word_table(word: str) -> None:
    # Keys carry no leading space, and Latin is kept in the case it is written: "MEN", "Men"
    # and "men" are separate entries, so a word in a case the vocabulary lacks is not found.
    assert word in student_words()


@pytest.mark.parametrize("key", [" the", "the ", "THEE", "tHe", ""])
def test_a_key_with_a_space_or_in_another_case_is_not_in_the_word_table(key: str) -> None:
    assert key not in student_words()


@pytest.mark.parametrize(
    "unit",
    [
        "\u00e9",
        "\u4e2d",
        "\ud55c",
        "\u044f",
        "\u20ac",
        "\U0001f600",
        "\u200b",
        "\u4e2d\u56fd",
        "\u6211\u4eec",
        "\u4e2d\u534e\u4eba\u6c11",
        "==",
        "...",
        ' {"',
        '"}',
        ";\n",
        '",\n',
        "\ufffd",
        " a",
        " \u00e9",
        " \ud55c",
    ],
)
def test_characters_han_and_symbol_tokens_are_in_the_unit_table(unit: str) -> None:
    assert unit in student_chars()


@pytest.mark.parametrize(
    "unit",
    [
        "a",
        " ",
        '"',
        "ab",
        "zzzz",
        "\u2460",
        "\n\n",
        "\u4e2d\u56fd\u4eba\u6c11\u5171\u548c",
        "\x85",
        "\ud800",
    ],
)
def test_other_strings_are_not_in_the_unit_table(unit: str) -> None:
    # An ASCII letter or symbol alone is priced by its byte; a long han string by its units; a
    # character that is no token (here an enclosed digit) by its bytes.
    assert unit not in student_chars()


def test_a_space_and_one_letter_are_in_the_unit_table_only_when_they_are_one_token() -> None:
    # The vocabulary has 501 tokens of a space and one letter; a letter after a space that is not
    # one of them is priced as a space token and the letter.
    spaced = [
        unit for unit in student_chars() if len(unit) == 2 and unit[0] == " " and unit[1].isalpha()
    ]
    assert len(spaced) == 501
    assert " \u00f1" not in student_chars()
    assert "\ufffd" in student_chars()


def test_the_fragile_units_are_han_and_symbol_units_and_all_in_the_unit_table() -> None:
    fragile = student_fragile()
    assert fragile <= student_chars()
    assert all(len(unit) >= 2 for unit in fragile)
    assert not any(unit.startswith(" ") or "\n" in unit for unit in fragile)
    assert "\u4e00\u5207\u90fd\u662f" in fragile  # a 4-character token that BPE does not re-merge
    assert "\u4e2d\u56fd" not in fragile  # a 2-character unit that repeats cleanly


def test_a_unit_ending_in_a_newline_is_kept() -> None:
    # A symbol token that ends a JSON line (a comma and its newline) is one key: a line-based
    # listing once dropped it and cost every line break a second token.
    assert '",\n' in student_chars()


@pytest.mark.parametrize(
    "alphabet", [string.ascii_lowercase, string.ascii_uppercase, string.ascii_letters]
)
def test_no_random_string_of_12_to_36_letters_is_a_member(alphabet: str) -> None:
    # The tables are exact: the 0.8% of strings a Bloom filter accepted are not members, so a
    # random word costs what its letters cost. 20,000 of each alphabet, none accepted.
    rng = random.Random(20261003)
    words = student_words()
    assert not any(
        "".join(rng.choices(alphabet, k=rng.randint(12, 36))) in words for _ in range(20_000)
    )


def _words_read_by_hand() -> set[str]:
    """The whole-word table decoded by a second, minimal reader: words hold no escapes."""
    words: set[str] = set()
    previous = ""
    for line in zlib.decompress(WORDS).decode("utf-8").split("\n"):
        if line:
            previous = previous[: ord(line[0]) - 0x20] + line[1:]
            words.add(previous)
    return words


def test_a_random_string_of_3_to_36_letters_is_a_member_exactly_when_it_is_a_table_word() -> None:
    # Exact membership, against an independent reading of the file: strings of 3 to 36 letters
    # in each case pattern. The short ones are sometimes real words; the 12-letter and longer
    # ones never are, so a random long word is never priced as a token.
    by_hand = _words_read_by_hand()
    rng = random.Random(20261003)
    words = student_words()
    members = 0
    for _ in range(100_000):
        letters = rng.choice((string.ascii_lowercase, string.ascii_uppercase, string.ascii_letters))
        candidate = "".join(rng.choices(letters, k=rng.randint(3, 36)))
        assert (candidate in words) == (candidate in by_hand)
        if candidate in words:
            members += 1
            assert len(candidate) < 12
    assert 0 < members < 2_000


@pytest.mark.parametrize(
    "bounds",
    [(0x430, 0x44F), (0x5D0, 0x5EA), (0x621, 0x64A), (0x3B1, 0x3C9)],
    ids=["cyrillic", "hebrew", "arabic", "greek"],
)
def test_no_random_string_of_12_to_36_letters_of_another_alphabet_is_a_member(
    bounds: tuple[int, int],
) -> None:
    rng = random.Random(7)
    words = student_words()
    for _ in range(20_000):
        word = "".join(chr(rng.randint(*bounds)) for _ in range(rng.randint(12, 36)))
        assert word not in words


def test_a_member_of_every_length_is_found_and_only_those() -> None:
    words = student_words()
    by_length: dict[int, str] = {}
    for word in sorted(words):
        by_length.setdefault(len(word), word)
    assert max(by_length) >= 30  # the longest Latin whole-word token has 33 letters
    assert all(word in words for word in by_length.values())
    assert all(word + "\u00a7" not in words for word in by_length.values())


# --- the format -------------------------------------------------------------------------------


def test_a_packed_list_decodes_to_exactly_its_keys() -> None:
    keys = {
        "",
        "a",
        "ab",
        "abc",
        "abd",
        "back\\slash",
        "new\nline",
        "carriage\rreturn",
        "\\n literally",
        "\u4e2d\u534e",
        "\u4e2d\u534e\u4eba",
        ' {"',
        "x" * 200,
        "x" * 200 + "y",
        "x" * 94 + "z",
    }
    assert unpack(pack(keys)) == keys
    assert unpack(pack(sorted(keys, reverse=True))) == keys  # the order given does not matter


def test_a_packed_list_of_random_unicode_keys_decodes_to_exactly_its_keys() -> None:
    rng = random.Random(3)
    pool = [chr(code) for code in (*range(0x20, 0x7F), 0x5C, 0x0A, 0x0D, 0xE9, 0x4E2D, 0x1F600)]
    keys = {"".join(rng.choices(pool, k=rng.randint(1, 12))) for _ in range(3_000)}
    assert unpack(pack(keys)) == keys


def test_the_file_is_front_coded_one_printable_character_a_key() -> None:
    lines = zlib.decompress(pack(["abc", "abd", "b\nc"])).split(b"\n")
    assert lines == [b" abc", b'"d', b" b\\nc", b""]  # 0x20 + shared prefix, then the rest


def test_a_shared_prefix_longer_than_94_characters_is_cut_at_94() -> None:
    lines = zlib.decompress(pack(["x" * 120, "x" * 121])).split(b"\n")
    assert lines[1][:1] == bytes([0x20 + 94])


# --- loading ----------------------------------------------------------------------------------


def test_each_table_is_read_once(monkeypatch: pytest.MonkeyPatch, fresh_tables: None) -> None:
    reads: list[str] = []

    class Counting(FakePackage):
        @override
        def read_bytes(self) -> bytes:
            reads.append(self._name)
            return super().read_bytes()

    serve(monkeypatch, Counting(BOTH))
    assert estimate("The parcel arrived late.") == estimate("The parcel arrived late.")
    assert "the" in student_words()
    assert "\u00e9" in student_chars()
    assert sorted(reads) == sorted(BOTH)


def test_tables_inside_a_zip_are_read_like_ones_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_tables: None
) -> None:
    # A wheel can be imported from a zip, where a path next to __file__ does not exist.
    archive = tmp_path / "dagnam.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name, data in BOTH.items():
            zf.writestr(f"dagnam/audit/{name}", data)
    with zipfile.ZipFile(archive) as zf:
        serve(monkeypatch, zipfile.Path(zf, "dagnam/audit/"))
        assert len(student_words()) == 49_926
        assert len(student_chars()) == 39_637
        assert len(student_fragile()) == 678
        assert estimate("The parcel arrived late and the box was damaged.") == 11


@pytest.mark.parametrize("name", list(BOTH))
def test_a_missing_table_fails_loudly_and_never_falls_back(
    name: str, monkeypatch: pytest.MonkeyPatch, fresh_tables: None
) -> None:
    served = {key: data for key, data in BOTH.items() if key != name}
    serve(monkeypatch, FakePackage(served))
    with pytest.raises(RuntimeError, match=rf"table dagnam.audit/{name} cannot be read"):
        estimate("")


@pytest.mark.parametrize("name", list(BOTH))
@pytest.mark.parametrize(
    "damage",
    [
        lambda data: data[:-1],
        lambda data: bytes([data[0] ^ 1]) + data[1:],
        lambda data: data + b" ",
    ],
    ids=["short", "one-changed-bit", "long"],
)
def test_a_changed_table_fails_loudly_whatever_the_text(
    name: str, damage: Any, monkeypatch: pytest.MonkeyPatch, fresh_tables: None
) -> None:
    serve(monkeypatch, FakePackage({**BOTH, name: damage(BOTH[name])}))
    with pytest.raises(
        RuntimeError, match=r"is not the one it was fitted with .* reinstall dagnam"
    ):
        estimate("12345")


def test_a_failed_load_is_retried_and_does_not_poison_later_estimates(
    monkeypatch: pytest.MonkeyPatch, fresh_tables: None
) -> None:
    serve(monkeypatch, FakePackage({}, error=OSError("disk gone")))
    with pytest.raises(RuntimeError, match="disk gone"):
        student_words()
    serve(monkeypatch, FakePackage(BOTH))
    assert len(student_words()) == 49_926
    assert token_tables.MAX_SHARED_PREFIX == 94
