from __future__ import annotations

from pathlib import Path

import pytest

from dagnam.data.loaders.system.decoders import get_decoder
from dagnam.data.loaders.system.decoders.base import DecodeError


def test_system_text_decoder_reads_declared_corpus_file_lines(tmp_path: Path) -> None:
    (tmp_path / "wiki.train.tokens").write_text("alpha\nbeta\n\ngamma\n", encoding="utf-8")

    store = get_decoder("text").decode(tmp_path, {"text": {"file": "wiki.train.tokens"}}, "train")

    assert len(store) == 3
    assert str(store.column("text")[0]) == "alpha"
    assert str(store.column("text")[2]) == "gamma"


def test_system_text_decoder_builds_next_token_pairs(tmp_path: Path) -> None:
    (tmp_path / "wiki.train.tokens").write_text("alpha beta gamma delta epsilon", encoding="utf-8")

    store = get_decoder("text").decode(
        tmp_path,
        {
            "text": {
                "file": "wiki.train.tokens",
                "self_supervised": "next_token",
                "sequence_length": 2,
                "vocab_size": 10,
            }
        },
        "train",
    )

    assert store.column("text")[0].tolist() == [1, 2]
    assert store.column("target")[0].tolist() == [2, 3]


def test_system_text_decoder_raises_decode_error_when_corpus_too_short(tmp_path: Path) -> None:
    # One token cannot form a single (seq_len=8)+1 next-token window:
    # build_lm_sequences raises a ValueError that must surface as DecodeError.
    (tmp_path / "wiki.train.tokens").write_text("alpha", encoding="utf-8")

    with pytest.raises(DecodeError, match="text format"):
        get_decoder("text").decode(
            tmp_path,
            {
                "text": {
                    "file": "wiki.train.tokens",
                    "self_supervised": "next_token",
                    "sequence_length": 8,
                    "vocab_size": 10,
                }
            },
            "train",
        )


def test_system_text_decoder_rejects_traversal_file(tmp_path: Path) -> None:
    # A malicious server descriptor must not read files outside the artifact dir.
    with pytest.raises(DecodeError):
        get_decoder("text").decode(tmp_path, {"text": {"file": "../../../etc/passwd"}}, "train")


def test_system_text_decoder_rejects_absolute_file(tmp_path: Path) -> None:
    with pytest.raises(DecodeError):
        get_decoder("text").decode(tmp_path, {"text": {"file": "/etc/passwd"}}, "train")
