from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dagnam.data.loaders.system.column_store import Column, ColumnStore


def test_system_column_store_eager_column_indexes_rows() -> None:
    column = Column.eager(np.arange(6).reshape(3, 2))

    assert len(column) == 3
    assert column[1].tolist() == [2, 3]


def test_system_column_store_lazy_column_decodes_path_on_index(tmp_path: Path) -> None:
    path = tmp_path / "row.npy"
    np.save(path, np.array([7, 8]))

    column = Column.lazy([path], lambda item: np.load(item))

    assert len(column) == 1
    assert column[0].tolist() == [7, 8]


def test_system_column_store_rejects_columns_with_unequal_lengths() -> None:
    with pytest.raises(ValueError, match="length"):
        ColumnStore({"a": Column.eager(np.zeros(3)), "b": Column.eager(np.zeros(2))})


def test_system_column_store_exposes_names_length_and_column_access() -> None:
    store = ColumnStore({"x": Column.eager(np.zeros(4)), "y": Column.eager(np.ones(4))})

    assert len(store) == 4
    assert store.names == ("x", "y")
    assert store.column("y")[0] == 1.0
