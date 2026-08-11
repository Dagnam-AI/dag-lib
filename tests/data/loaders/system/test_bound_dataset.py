from __future__ import annotations

import numpy as np
import pytest

from dagnam.data.loaders.system.bound_dataset import BoundNativeDataset
from dagnam.data.loaders.system.column_store import Column, ColumnStore


def test_system_bound_dataset_selects_bound_columns_and_applies_transforms() -> None:
    store = ColumnStore(
        {
            "image": Column.eager(np.zeros((2, 10, 10, 3), np.uint8)),
            "segmentation_mask": Column.eager(np.ones((2, 10, 10), np.uint8) * 2),
        }
    )
    binding = {
        "input_column": "image",
        "target_column": "segmentation_mask",
        "input_transform": {"kind": "image_resize", "params": {"size": [8, 8]}},
        "target_transform": {
            "kind": "mask",
            "params": {"resize": [8, 8], "remap": "contiguous_long", "value_set": [1, 2, 3]},
        },
    }

    dataset = BoundNativeDataset(store, binding, descriptor_columns=[{"name": "image"}])

    assert len(dataset) == 2
    x, y = dataset[0]
    assert x.shape == (8, 8, 3)
    assert y.shape == (8, 8)
    assert y.dtype == np.int64


def test_system_bound_dataset_resolves_columns_by_role_when_name_mismatches() -> None:
    # binding uses generic IR names (text/label) absent from the dataset, which has
    # review/sentiment; resolution falls back to the declared roles.
    store = ColumnStore(
        {
            "review": Column.eager(np.array([[1, 2, 3], [4, 5, 6]], np.int64)),
            "sentiment": Column.eager(np.array([0, 1], np.int64)),
        }
    )
    binding = {
        "input_column": "text",
        "target_column": "label",
        "input_transform": {"kind": "tokenize", "params": {"sequence_length": 3}},
        "target_transform": {"kind": "class_index", "params": {}},
    }

    dataset = BoundNativeDataset(
        store,
        binding,
        descriptor_columns=[{"name": "review"}, {"name": "sentiment"}],
        column_roles={"review": "text_input", "sentiment": "target"},
    )

    assert dataset.input_kind == "text"
    x, y = dataset[1]
    assert x.tolist() == [4, 5, 6]
    assert int(y) == 1


def test_system_bound_dataset_input_kind_image_and_self_supervised_target() -> None:
    store = ColumnStore({"image": Column.eager(np.zeros((2, 4, 4, 3), np.uint8))})
    binding = {
        "input_column": "image",
        "target_column": None,  # self-supervised: target derived from the input
        "input_transform": {"kind": "image_resize", "params": {"size": [4, 4]}},
    }

    dataset = BoundNativeDataset(store, binding, descriptor_columns=[{"name": "image"}])

    assert dataset.input_kind == "image"
    x, y = dataset[0]
    assert x.shape == (4, 4, 3)
    assert np.array_equal(x, y)  # self-supervised returns (x, x)


def test_system_bound_dataset_uses_loader_next_token_target_column() -> None:
    store = ColumnStore(
        {
            "text": Column.eager(np.array([[1, 2, 3]], dtype=np.int64)),
            "target": Column.eager(np.array([[2, 3, 4]], dtype=np.int64)),
        }
    )
    binding = {
        "input_column": "text",
        "target_column": None,
        "input_transform": {"kind": "tokenize", "params": {"sequence_length": 3}},
        "self_supervised": {"kind": "next_token", "where": "loader"},
    }

    dataset = BoundNativeDataset(store, binding, descriptor_columns=[{"name": "text"}])

    x, y = dataset[0]
    assert x.tolist() == [1, 2, 3]
    assert y.tolist() == [2, 3, 4]


def test_system_bound_dataset_raises_when_no_input_column_resolves() -> None:
    store = ColumnStore({"feature": Column.eager(np.zeros((1, 3)))})
    binding = {"input_column": "missing", "input_transform": {"kind": "identity", "params": {}}}

    dataset = BoundNativeDataset(store, binding, descriptor_columns=[], column_roles={})

    with pytest.raises(ValueError, match="input_column"):
        _ = dataset[0]


def test_system_bound_dataset_applies_descriptor_normalize_to_input_column() -> None:
    store = ColumnStore(
        {
            "image": Column.eager(np.full((1, 2, 2, 3), 255, np.uint8)),
            "label": Column.eager(np.array([1])),
        }
    )
    binding = {
        "input_column": "image",
        "target_column": "label",
        "input_transform": {"kind": "image_resize", "params": {"size": [2, 2]}},
    }

    dataset = BoundNativeDataset(
        store,
        binding,
        descriptor_columns=[
            {"name": "image", "normalize": {"mean": [1.0, 1.0, 1.0], "std": [1.0, 1.0, 1.0]}}
        ],
    )

    x, y = dataset[0]
    assert np.allclose(x, 0.0)
    assert int(y) == 1


def test_system_bound_dataset_input_kind_video_and_role_fallback() -> None:
    # The binding carries the architecture-side name "video" while the dataset's
    # column is "clip": resolution falls back to the declared video_input role.
    store = ColumnStore(
        {
            "clip": Column.eager(np.zeros((2, 8, 4, 4, 3), np.uint8)),
            "label": Column.eager(np.arange(2, dtype=np.int64)),
        }
    )
    binding = {
        "input_column": "video",
        "target_column": "label",
        "input_transform": {"kind": "video", "params": {"size": [4, 4]}},
        "target_transform": {"kind": "class_index", "params": {}},
    }

    dataset = BoundNativeDataset(
        store,
        binding,
        descriptor_columns=[{"name": "clip"}, {"name": "label"}],
        column_roles={"clip": "video_input", "label": "target"},
    )

    assert dataset.input_kind == "video"
    x, y = dataset[0]
    assert x.shape == (8, 4, 4, 3)  # channels-last [T, H, W, C]
    assert int(y) == 0
