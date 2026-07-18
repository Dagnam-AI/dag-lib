from pathlib import Path
from typing import cast

import numpy as np
import pytest

from dagnam.data.loaders.system.decoders import get_decoder
from dagnam.data.loaders.system.decoders.base import DecodeError


def test_array_decoder_maps_keys_to_columns(tmp_path: Path) -> None:
    np.savez(
        tmp_path / "d.npz",
        x=np.zeros((5, 4, 4, 3), np.uint8),
        y=np.arange(5),
        x_test=np.zeros((2, 4, 4, 3), np.uint8),
        y_test=np.arange(2),
    )
    layout = cast(
        "dict[str, object]",
        {
            "image": {"key": "x", "test_key": "x_test"},
            "label": {"key": "y", "test_key": "y_test"},
        },
    )

    train = get_decoder("array").decode(tmp_path, layout, "train")
    test = get_decoder("array").decode(tmp_path, layout, "test")

    assert len(train) == 5
    assert train.column("image")[0].shape == (4, 4, 3)
    assert len(test) == 2


def test_get_decoder_unknown_format_raises() -> None:
    with pytest.raises(DecodeError, match="unknown format"):
        get_decoder("nope")


def test_array_decoder_no_npz_artifact_raises(tmp_path: Path) -> None:
    """An artifact dir with no ``.npz`` file fails cleanly."""
    layout = cast("dict[str, object]", {"image": {"key": "x", "test_key": "x"}})
    with pytest.raises(DecodeError, match=r"no \.npz artifact"):
        get_decoder("array").decode(tmp_path, layout, "train")


def test_array_decoder_missing_key_spec_raises(tmp_path: Path) -> None:
    """A column spec that names no resolvable key (non-string) is rejected."""
    np.savez(tmp_path / "d.npz", y=np.arange(3))
    layout = cast("dict[str, object]", {"label": {"test_key": "y"}})
    with pytest.raises(DecodeError, match="key None missing"):
        get_decoder("array").decode(tmp_path, layout, "train")


def test_array_decoder_missing_column_key_raises(tmp_path: Path) -> None:
    """A non-ragged column whose key is absent from the npz is rejected."""
    np.savez(tmp_path / "d.npz", y=np.arange(3))
    layout = cast("dict[str, object]", {"image": {"key": "x", "test_key": "x"}})
    with pytest.raises(DecodeError, match="key 'x' missing"):
        get_decoder("array").decode(tmp_path, layout, "train")


def test_array_decoder_refuses_object_arrays_no_pickle(tmp_path: Path) -> None:
    """A server-supplied .npz with an object (pickled) array must be refused,
    never deserialized through pickle (arbitrary-code-execution vector)."""
    np.savez(
        tmp_path / "d.npz",
        x=np.array([{"payload": "arbitrary"}], dtype=object),
        y=np.arange(1),
    )
    layout = cast(
        "dict[str, object]",
        {
            "image": {"key": "x", "test_key": "x"},
            "label": {"key": "y", "test_key": "y"},
        },
    )

    with pytest.raises(DecodeError, match="object arrays are not supported"):
        get_decoder("array").decode(tmp_path, layout, "train")


def _ragged_offsets(rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Encode ragged int rows as flat ``values`` + ``offsets`` (pickle-free)."""
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    for i, row in enumerate(rows):
        offsets[i + 1] = offsets[i] + len(row)
    values = np.array([tok for row in rows for tok in row], dtype=np.int64)
    return values, offsets


def test_array_decoder_reconstructs_ragged_column_no_pickle(tmp_path: Path) -> None:
    """A ragged column stored as ``<key>_values``+``<key>_offsets`` regular int
    arrays (imdb-scale token ids) round-trips row-by-row under allow_pickle=False."""
    train_rows = [[1, 88586, 4], [7], [0, 2, 3, 88585, 10]]
    test_rows = [[88584], [5, 6]]
    xtr_values, xtr_offsets = _ragged_offsets(train_rows)
    xte_values, xte_offsets = _ragged_offsets(test_rows)
    np.savez(
        tmp_path / "d.npz",
        x_train_values=xtr_values,
        x_train_offsets=xtr_offsets,
        x_test_values=xte_values,
        x_test_offsets=xte_offsets,
        y_train=np.array([1, 0, 1]),
        y_test=np.array([0, 1]),
    )
    layout = cast(
        "dict[str, object]",
        {
            "review": {"key": "x_train", "test_key": "x_test", "ragged": True},
            "sentiment": {"key": "y_train", "test_key": "y_test"},
        },
    )

    train = get_decoder("array").decode(tmp_path, layout, "train")
    test = get_decoder("array").decode(tmp_path, layout, "test")

    assert len(train) == 3
    for i, row in enumerate(train_rows):
        assert train.column("review")[i].tolist() == row
    assert train.column("sentiment")[1] == 0

    assert len(test) == 2
    for i, row in enumerate(test_rows):
        assert test.column("review")[i].tolist() == row


def test_array_decoder_ragged_missing_values_offsets_raises(tmp_path: Path) -> None:
    """A ragged column whose values/offsets arrays are absent fails cleanly."""
    np.savez(tmp_path / "d.npz", y_train=np.arange(2))
    layout = cast(
        "dict[str, object]",
        {"review": {"key": "x_train", "test_key": "x_test", "ragged": True}},
    )
    with pytest.raises(DecodeError, match="requires 'x_train_values'"):
        get_decoder("array").decode(tmp_path, layout, "train")


def test_array_decoder_ragged_malformed_offsets_raises(tmp_path: Path) -> None:
    """A ragged column with an empty/multi-dim offsets array is rejected."""
    np.savez(
        tmp_path / "d.npz",
        x_train_values=np.array([1, 2, 3]),
        x_train_offsets=np.zeros((0,), dtype=np.int64),
    )
    layout = cast(
        "dict[str, object]",
        {"review": {"key": "x_train", "test_key": "x_test", "ragged": True}},
    )
    with pytest.raises(DecodeError, match="malformed offsets"):
        get_decoder("array").decode(tmp_path, layout, "train")
