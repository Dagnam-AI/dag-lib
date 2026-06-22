"""Uniform native representation for descriptor-driven system datasets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

DecodeFn = Callable[[Path], npt.NDArray[np.generic]]


@dataclass(frozen=True)
class Column:
    """One dataset column, either eager ndarray values or lazy file paths."""

    _values: npt.NDArray[np.generic] | None
    _paths: tuple[Path, ...] | None
    _decode: DecodeFn | None

    @classmethod
    def eager(cls, values: npt.ArrayLike) -> Column:
        return cls(np.asarray(values), None, None)

    @classmethod
    def lazy(cls, paths: list[Path] | tuple[Path, ...], decode: DecodeFn) -> Column:
        return cls(None, tuple(paths), decode)

    def __len__(self) -> int:
        if self._values is not None:
            return len(self._values)
        return len(self._paths or ())

    def __getitem__(self, index: int) -> npt.NDArray[np.generic]:
        if self._values is not None:
            return self._values[index]
        if self._paths is None or self._decode is None:
            raise IndexError("lazy column is not configured")
        return self._decode(self._paths[index])


@dataclass(frozen=True)
class ColumnStore:
    """A split represented as named equal-length columns."""

    columns: dict[str, Column]

    def __post_init__(self) -> None:
        lengths = {name: len(column) for name, column in self.columns.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"ColumnStore columns have unequal length: {lengths}")

    def __len__(self) -> int:
        if not self.columns:
            return 0
        return len(next(iter(self.columns.values())))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.columns)

    def column(self, name: str) -> Column:
        try:
            return self.columns[name]
        except KeyError as exc:
            raise KeyError(f"column {name!r} not in store; have {self.names}") from exc
