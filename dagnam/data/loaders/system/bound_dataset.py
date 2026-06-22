"""Indexable dataset view binding a ColumnStore to architecture columns."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dagnam.data.loaders.system.column_store import ColumnStore
from dagnam.data.loaders.system.transform_executor import apply_transform

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

# Roles that mark a column as the architecture's input / target. Used to resolve a
# column by intent when the binding's literal name does not match the dataset's
# actual column names (e.g. binding "text" vs descriptor "review").
_INPUT_ROLES = ("image_input", "text_input", "audio_input")
_TARGET_ROLES = ("target", "text_target")
# Input modality inferred from the input transform's kind, so a framework converter
# can apply its layout (pytorch transposes images to channels-first).
_KIND_TO_MODALITY = {"image_resize": "image", "tokenize": "text", "audio_mel": "audio"}


class BoundNativeDataset:
    """A ``len``/``getitem`` native dataset consumed by existing converters."""

    def __init__(
        self,
        store: ColumnStore,
        binding: dict[str, Any],
        descriptor_columns: list[dict[str, Any]],
        column_roles: dict[str, str] | None = None,
    ) -> None:
        self._store = store
        self._input_transform = binding.get("input_transform", {"kind": "identity", "params": {}})
        self._target_transform = binding.get("target_transform", {"kind": "identity", "params": {}})
        roles = column_roles or {}
        self._input_column = self._resolve(binding.get("input_column"), _INPUT_ROLES, roles)
        self._target_column = self._resolve(binding.get("target_column"), _TARGET_ROLES, roles)
        self._normalize = next(
            (
                column.get("normalize")
                for column in descriptor_columns
                if column.get("name") == self._input_column
            ),
            None,
        )
        # Input modality, so the pytorch converter knows to transpose images to CHW.
        self.input_kind = _KIND_TO_MODALITY.get(self._input_transform.get("kind", ""), "other")

    def _resolve(
        self, binding_name: object, roles_wanted: tuple[str, ...], column_roles: dict[str, str]
    ) -> str | None:
        """Resolve a column by the binding's literal name, else by declared role.

        The binding may carry generic architecture-side names (e.g. "text"/"label")
        that don't exist in the dataset; fall back to the column whose role matches
        the wanted input/target intent so loading is name-agnostic.
        """
        if isinstance(binding_name, str) and binding_name in self._store.columns:
            return binding_name
        for name, role in column_roles.items():
            if role in roles_wanted and name in self._store.columns:
                return name
        return None

    def __len__(self) -> int:
        return len(self._store)

    def __getitem__(self, index: int) -> tuple[npt.NDArray[np.generic], npt.NDArray[np.generic]]:
        if not isinstance(self._input_column, str):
            raise ValueError("binding.input_column is required")
        raw_x = self._store.column(self._input_column)[index]
        x = apply_transform(raw_x, self._input_transform, self._normalize)

        if not isinstance(self._target_column, str):
            return x, x
        raw_y = self._store.column(self._target_column)[index]
        y = apply_transform(raw_y, self._target_transform, None)
        return x, y
