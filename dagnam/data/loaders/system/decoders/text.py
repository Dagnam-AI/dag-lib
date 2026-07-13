"""Plain-text corpus system-dataset decoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from dagnam.data.loaders.system.column_store import Column, ColumnStore
from dagnam.data.loaders.system.decoders._helpers import safe_subpath
from dagnam.data.loaders.system.decoders.base import DecodeError
from dagnam.data.loaders.text_lm import build_lm_sequences


class TextDecoder:
    """Decode a declared text file into a non-empty text column."""

    def decode(self, artifact_dir: Path, layout: dict[str, object], split: str) -> ColumnStore:
        del split
        raw_spec = layout.get("text")
        if not isinstance(raw_spec, dict):
            raise DecodeError("text format requires text layout")
        spec = cast("dict[str, Any]", raw_spec)
        filename = spec.get("file")
        if not isinstance(filename, str):
            raise DecodeError("text format requires layout.text.file")
        path = safe_subpath(artifact_dir, filename)
        if not path.exists():
            raise DecodeError(f"text format: file does not exist: {path}")
        corpus = path.read_text(encoding="utf-8")
        if spec.get("self_supervised") == "next_token":
            seq_len = spec.get("sequence_length", 128)
            vocab_size = spec.get("vocab_size")
            try:
                x, y = build_lm_sequences(
                    corpus,
                    seq_len=int(seq_len),
                    vocab_size=vocab_size if isinstance(vocab_size, int) else None,
                )
            except ValueError as exc:
                raise DecodeError(f"text format: {exc}") from exc
            return ColumnStore({"text": Column.eager(x), "target": Column.eager(y)})

        lines = [line for line in corpus.splitlines() if line]
        return ColumnStore({"text": Column.eager(np.asarray(lines, dtype=object))})
