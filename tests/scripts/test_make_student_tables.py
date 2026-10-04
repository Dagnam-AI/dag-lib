"""``scripts/make_student_tables.py``: the rebuild of the token estimate's vocabulary tables.

Run on a tiny synthetic vocabulary written the way a byte-level BPE ``tokenizer.json`` writes it,
with merges that build each token from its bytes, so nothing here needs the real tokenizer or
the network. That the script reproduces the three shipped tables from the real vocabulary byte
for byte, and that its BPE gives the real tokenizer's ids on the units the tables need, are manual checks, made when a table
or the script changes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from dagnam.audit import token_tables

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "make_student_tables.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("make_student_tables", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_tables = _load()
ENCODE = {byte: char for char, byte in make_tables.byte_decoder().items()}


def token(text: str) -> str:
    """``text`` as a byte-level BPE vocabulary writes it: one character per UTF-8 byte."""
    return "".join(ENCODE[byte] for byte in text.encode("utf-8"))


WORDS = [
    " the",
    " The",
    " THE",
    " caf\u00e9",
    " \u0434\u043b\u044f",
    " \u041f\u0440\u0438\u0432\u0435\u0442",
    " \u05e9\u05dc\u05d5\u05dd",
    " \u0645\u0631\u062d\u0628\u0627",
    " \u1ea1n",  # Latin Extended Additional: kept as written
]
NOT_WORDS = [
    "the",  # no leading space
    " ",
    " it's",  # not only letters
    " 123",
    " the1",
    "\n the",
]
CHARACTERS = ["\u00e9", "\u044f", "\u4e2d", "\U0001f600", "\u200b", "\ufffd"]
SPACE_AND_LETTER = [" a", " \u00f1"]  # a space and one letter that the encoder gives as one token
SPACE_AND_LETTER_SPLIT = " b"  # in the vocabulary, but the encoder gives a space and a "b"
NOT_CHARACTERS = [
    "a",  # ASCII is priced by its byte
    "\u00e9\u00e9",  # not a han token, a symbol token or a single character
]
HAN_UNITS = [
    "\u4e2d\u56fd",
    "\u4e2d\u534e\u4eba\u6c11",
    " \u4e2d\u56fd",  # led by a space: the same unit
    "\u65e5\u672c",
    " \u65e5\u672c",
]
NOT_HAN_UNITS = ["\u4e2d\u56fd\u4eba\u6c11\u5171\u548c"]  # six characters
SYMBOLS = ["==", ' {"', '{"', ";\n", " (", "=>", ">="]
NOT_SYMBOLS = ["=", "(", "\n\n", "=========", "a=", " "]
FRAGILE = {"=>"}  # its eight copies take nine tokens: the merge of ">=" comes first
HALF_CHARACTER = ENCODE[0x20] + ENCODE["\u00e9".encode()[0]]  # a space and half of an accent
BYTE_FALLBACK = ENCODE[0x80]  # one byte of a character: not text


def vocabulary() -> tuple[dict[str, int], list[str]]:
    """Byte symbols, then every token above, each built from its bytes by merges ranked after the
    two that make ``=>`` fragile (``>=`` is merged before ``=>``)."""
    vocab = {ENCODE[byte]: byte for byte in range(256)}
    merges = [f"{ENCODE[ord('>')]} {ENCODE[ord('=')]}", f"{ENCODE[ord('=')]} {ENCODE[ord('>')]}"]
    texts = [
        *WORDS,
        *NOT_WORDS,
        *CHARACTERS,
        *SPACE_AND_LETTER,
        SPACE_AND_LETTER_SPLIT,
        *NOT_CHARACTERS,
        *HAN_UNITS,
        *NOT_HAN_UNITS,
        *SYMBOLS,
        *NOT_SYMBOLS,
    ]
    for text in texts:
        symbols = [ENCODE[byte] for byte in text.encode("utf-8")]
        merged = symbols[0]
        for symbol in symbols[1:]:
            if text != SPACE_AND_LETTER_SPLIT and f"{merged} {symbol}" not in merges:
                merges.append(f"{merged} {symbol}")
            merged += symbol
        vocab.setdefault(merged, len(vocab))
    vocab[HALF_CHARACTER] = len(vocab)
    return vocab, merges


VOCAB, MERGES = vocabulary()
TOKENS = make_tables.decoded(VOCAB)
ENCODER = make_tables.make_encoder(VOCAB, MERGES)


def test_the_byte_alphabet_is_the_byte_level_bpe_one() -> None:
    decoder = make_tables.byte_decoder()
    assert len(decoder) == 256
    assert sorted(decoder.values()) == list(range(256))
    # The characters a vocabulary file shows: a space is "\u0120", a newline "\u010a".
    assert decoder["\u0120"] == 0x20
    assert decoder["\u010a"] == 0x0A
    assert decoder["a"] == ord("a")
    assert decoder["\u00e9"] == 0xE9


def test_a_token_that_is_not_whole_characters_of_text_is_not_decoded() -> None:
    texts = [text for _, text in TOKENS]
    assert HALF_CHARACTER not in VOCAB or all(text != HALF_CHARACTER for text in texts)
    assert BYTE_FALLBACK in VOCAB  # byte 0x80 alone is a token, and is half of a character
    assert all(text != "\u0080" for text in texts)
    assert len(texts) == len(VOCAB) - 128 - 1  # the 128 bytes above ASCII and the half character


def test_the_encoder_gives_a_token_its_own_id_and_merges_by_rank() -> None:
    assert ENCODER("the") == [VOCAB[token("the")]]
    assert ENCODER(" the") == [VOCAB[token(" the")]]
    assert ENCODER("\u4e2d\u56fd") == [VOCAB[token("\u4e2d\u56fd")]]
    assert ENCODER("=>") == [VOCAB[token("=>")]]
    assert len(ENCODER("=>" * 8)) == 9  # ">=" is merged before "=>", so the copies do not re-merge
    assert ENCODER(" b") == [VOCAB[token(" ")], VOCAB[token("b")]]
    assert ENCODER("") == []


def test_the_encoder_normalises_to_nfc_like_the_tokenizer() -> None:
    assert ENCODER("e\u0301") == ENCODER("\u00e9")


def test_only_words_of_the_priced_alphabets_with_a_leading_space_are_kept() -> None:
    words = make_tables.whole_words([text for _, text in TOKENS])
    expected = {
        "\u00f1",  # " \u00f1" is a word as well as a space and one letter
        "a",
        "b",
        "THE",
        "The",
        "the",
        "caf\u00e9",
        "\u1ea1n",
        "\u0434\u043b\u044f",
        "\u043f\u0440\u0438\u0432\u0435\u0442",  # Cyrillic is lowercased
        "\u05e9\u05dc\u05d5\u05dd",
        "\u0645\u0631\u062d\u0628\u0627",
    }
    assert words == sorted(expected)


def test_characters_han_tokens_symbol_tokens_and_the_space_and_letter_tokens_are_picked_apart() -> (
    None
):
    singles, han, symbols = make_tables.char_units(TOKENS, ENCODER)
    assert singles == sorted([*CHARACTERS, *SPACE_AND_LETTER])
    assert " b" not in singles  # a token, but not the one its own encoding gives
    assert han == sorted({unit.strip() for unit in HAN_UNITS})
    assert symbols == sorted(SYMBOLS)
    assert not set(NOT_CHARACTERS) & set(singles)
    assert not set(NOT_HAN_UNITS) & set(han)
    assert not set(NOT_SYMBOLS) & set(symbols)


def test_a_replacement_character_is_a_character_only_as_its_own_token() -> None:
    # U+FFFD is a token of its own; a token that merely holds it is a byte fallback and is dropped.
    holding = [(len(VOCAB), "x\ufffd"), (len(VOCAB) + 1, "\ufffd\ufffd")]
    singles, _, _ = make_tables.char_units([*TOKENS, *holding], ENCODER)
    assert "\ufffd" in singles
    assert not any("x\ufffd" in unit or "\ufffd\ufffd" in unit for unit in singles)
    wrong_id = [(len(VOCAB), "\ufffd")]
    assert "\ufffd" not in make_tables.char_units(wrong_id, ENCODER)[0]


def test_the_fragile_units_are_those_whose_eight_copies_take_more_than_8_8_tokens() -> None:
    _, han, symbols = make_tables.char_units(TOKENS, ENCODER)
    assert make_tables.fragile_units(han, symbols, ENCODER) == sorted(FRAGILE)
    # Units led by a space or holding a line break are not measured.
    assert make_tables.fragile_units([], [" (", ";\n"], ENCODER) == []


def test_tables_it_builds_are_read_back_exactly_by_the_estimate() -> None:
    words = make_tables.whole_words([text for _, text in TOKENS])
    singles, han, symbols = make_tables.char_units(TOKENS, ENCODER)
    assert token_tables.unpack(token_tables.pack(words)) == frozenset(words)
    units = [*singles, *han, *symbols]
    assert token_tables.unpack(token_tables.pack(units)) == frozenset(units)


@pytest.fixture
def tokenizer_json(tmp_path: Path) -> Path:
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({"model": {"vocab": VOCAB, "merges": MERGES}}), encoding="utf-8")
    return path


def test_main_writes_the_three_tables_and_reports_their_sizes_and_digests(
    tokenizer_json: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "tables"
    out.mkdir()
    assert make_tables.main([str(tokenizer_json), "--out-dir", str(out)]) == 0
    words = (out / "student_words.z").read_bytes()
    chars = (out / "student_chars.z").read_bytes()
    fragile = (out / "student_fragile.z").read_bytes()
    assert token_tables.unpack(words) == frozenset(make_tables.whole_words([t for _, t in TOKENS]))
    singles, han, symbols = make_tables.char_units(TOKENS, ENCODER)
    assert token_tables.unpack(chars) == frozenset([*singles, *han, *symbols])
    assert token_tables.unpack(fragile) == frozenset(FRAGILE)
    report = capsys.readouterr().out
    assert "{'latin': 8, 'cyrillic': 2, 'hebrew': 1, 'arabic': 1}" in report
    assert f"{len(singles)} characters + 3 han tokens + {len(symbols)} symbol tokens" in report
    assert "1 fragile units" in report
    for name, blob in (
        ("student_words.z", words),
        ("student_chars.z", chars),
        ("student_fragile.z", fragile),
    ):
        assert f"{name}: {len(blob)} bytes, sha256 {hashlib.sha256(blob).hexdigest()}" in report


def test_main_refuses_a_vocabulary_with_no_word_or_character_tokens(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "tokenizer.json"
    path.write_text(
        json.dumps({"model": {"vocab": {"a": 0, "b": 1}, "merges": []}}), encoding="utf-8"
    )
    assert make_tables.main([str(path), "--out-dir", str(tmp_path)]) == 1
    assert "no whole-word or character tokens" in capsys.readouterr().err
    assert not (tmp_path / "student_words.z").exists()


def test_the_default_output_is_the_shipped_tables() -> None:
    for name, digest in (
        ("student_words.z", token_tables.STUDENT_WORDS_SHA256),
        ("student_chars.z", token_tables.STUDENT_CHARS_SHA256),
        ("student_fragile.z", token_tables.STUDENT_FRAGILE_SHA256),
    ):
        shipped = ROOT / "dagnam" / "audit" / name
        assert hashlib.sha256(shipped.read_bytes()).hexdigest() == digest
        assert make_tables.ROOT / "dagnam" / "audit" / name == shipped
