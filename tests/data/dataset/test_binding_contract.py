"""Binding contract: the optional ``binding=`` param on the public loaders.

``binding`` is additive/optional (dag-lib is PUBLIC). When ``column_roles`` is not
given but a ``binding`` names ``input_column``/``target_column``, the tabular
converters derive ``column_roles`` from it; otherwise behaviour is unchanged.
"""

from collections.abc import Callable
from inspect import signature
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from dagnam.data.dataset import DagnamDataset
from dagnam.data.dataset.to_flax import _column_roles_from_binding as _roles_flax
from dagnam.data.dataset.to_pytorch import _column_roles_from_binding as _roles_pt
from dagnam.data.dataset.to_tensorflow import _column_roles_from_binding as _roles_tf
from dagnam.data.load import load_dataset

_RolesFn = Callable[[dict[str, Any]], dict[str, str] | None]
_ALL_ROLE_FNS: list[_RolesFn] = [_roles_pt, _roles_tf, _roles_flax]


def _csv_dataset(tmp_path: Path) -> DagnamDataset:
    (tmp_path / "data.csv").write_text("x,label\n1,a\n2,b\n", encoding="utf-8")
    return DagnamDataset(
        {
            "id": "bind-1",
            "name": "Tabular",
            "format": "csv",
            "dataset_type": "tabular",
            "num_samples": 2,
            "num_classes": 2,
            "class_names": ["a", "b"],
            "feature_schema": None,
        },
        tmp_path,
    )


def test_public_loader_and_converter_signatures_accept_binding() -> None:
    assert "binding" in signature(load_dataset).parameters
    assert "binding" in signature(DagnamDataset.to_pytorch_loader).parameters
    assert "binding" in signature(DagnamDataset.to_tensorflow_dataset).parameters
    assert "binding" in signature(DagnamDataset.to_flax_dataset).parameters


@pytest.mark.parametrize("roles_fn", _ALL_ROLE_FNS)
def test_column_roles_from_binding_maps_input_and_target(roles_fn: _RolesFn) -> None:
    assert roles_fn({"input_column": "x", "target_column": "label"}) == {
        "x": "feature",
        "label": "target",
    }


@pytest.mark.parametrize("roles_fn", _ALL_ROLE_FNS)
def test_column_roles_from_binding_ignores_missing_and_nonstring(roles_fn: _RolesFn) -> None:
    # No usable column names -> None (the loader keeps its default behaviour).
    assert roles_fn({}) is None
    assert roles_fn({"input_column": 123, "target_column": None}) is None


def test_to_pytorch_loader_derives_column_roles_from_binding(tmp_path: Path) -> None:
    ds = _csv_dataset(tmp_path)
    with patch(
        "dagnam.data.loaders.csv.create_pytorch_loader",
        return_value="loader",
    ) as mock_create:
        ds.to_pytorch_loader(
            split="train",
            batch_size=1,
            num_workers=0,
            binding={"input_column": "x", "target_column": "label"},
        )
    assert mock_create.call_args.kwargs["column_roles"] == {"x": "feature", "label": "target"}


def test_to_tensorflow_dataset_derives_column_roles_from_binding(tmp_path: Path) -> None:
    ds = _csv_dataset(tmp_path)
    with patch(
        "dagnam.data.loaders.tf.create_tensorflow_dataset",
        return_value="loader",
    ) as mock_create:
        ds.to_tensorflow_dataset(
            split="train",
            batch_size=1,
            binding={"input_column": "x", "target_column": "label"},
        )
    assert mock_create.call_args.kwargs["column_roles"] == {"x": "feature", "label": "target"}


def test_to_flax_dataset_derives_column_roles_from_binding(tmp_path: Path) -> None:
    ds = _csv_dataset(tmp_path)
    with patch(
        "dagnam.data.loaders.flax.create_flax_dataset",
        return_value=["batch"],
    ) as mock_create:
        ds.to_flax_dataset(
            split="train",
            batch_size=1,
            binding={"input_column": "x", "target_column": "label"},
        )
    assert mock_create.call_args.kwargs["column_roles"] == {"x": "feature", "label": "target"}


def test_explicit_column_roles_take_precedence_over_binding(tmp_path: Path) -> None:
    """When column_roles is given, binding is NOT consulted (column_roles wins)."""
    ds = _csv_dataset(tmp_path)
    with patch(
        "dagnam.data.loaders.csv.create_pytorch_loader",
        return_value="loader",
    ) as mock_create:
        ds.to_pytorch_loader(
            split="train",
            batch_size=1,
            num_workers=0,
            column_roles={"x": "feature"},
            binding={"input_column": "x", "target_column": "label"},
        )
    assert mock_create.call_args.kwargs["column_roles"] == {"x": "feature"}
