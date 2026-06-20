import numpy as np
import pytest

from dagnam.data.loaders.text_lm import DatasetLoadError, build_lm_sequences


def test_build_lm_sequences_shifts_next_token_targets():
    x, y = build_lm_sequences("a b c d e f g h i j k l", seq_len=4, vocab_size=50)

    assert x.shape == y.shape == (2, 4)
    np.testing.assert_array_equal(x[:, 1:], y[:, :-1])
    assert int(x.max()) < 50


def test_build_lm_sequences_raises_for_empty_corpus():
    with pytest.raises(DatasetLoadError, match="empty or unreadable"):
        build_lm_sequences("   \n\t", seq_len=4, vocab_size=50)


def test_build_lm_sequences_validates_sequence_length():
    with pytest.raises(ValueError, match="seq_len"):
        build_lm_sequences("a b c", seq_len=0, vocab_size=50)


def test_build_lm_sequences_raises_when_corpus_too_short():
    with pytest.raises(DatasetLoadError, match="too short"):
        build_lm_sequences("a b c", seq_len=4, vocab_size=50)


def test_build_lm_sequences_uncapped_vocab_keeps_all_tokens():
    x, y = build_lm_sequences("a b c d e", seq_len=4, vocab_size=None)

    assert x.shape == y.shape == (1, 4)
    # No cap: every distinct token gets its own id (1..N), so none collapse to 0.
    assert int(x.min()) >= 1
