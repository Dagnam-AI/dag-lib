"""Framework-neutral helpers for next-token language-modeling datasets."""

from __future__ import annotations

from collections import Counter

import numpy as np
import numpy.typing as npt


class DatasetLoadError(ValueError):
    """Raised when a text corpus cannot produce LM training sequences."""


def build_lm_sequences(
    text: str,
    *,
    seq_len: int,
    vocab_size: int | None = None,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Tokenize text and build ``(tokens[:-1], tokens[1:])`` LM pairs."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    tokens = text.split()
    if not tokens:
        raise DatasetLoadError("WikiText-2 corpus is empty or unreadable")

    cap = vocab_size if vocab_size is not None and vocab_size > 1 else None
    most_common = Counter(tokens).most_common(None if cap is None else cap - 1)
    vocab = {token: idx + 1 for idx, (token, _count) in enumerate(most_common)}
    encoded = np.asarray([vocab.get(token, 0) for token in tokens], dtype=np.int64)

    chunk = seq_len + 1
    usable = (len(encoded) // chunk) * chunk
    if usable == 0:
        raise DatasetLoadError("WikiText-2 corpus is too short for the requested sequence length")

    windows = encoded[:usable].reshape(-1, chunk)
    return windows[:, :-1], windows[:, 1:]
