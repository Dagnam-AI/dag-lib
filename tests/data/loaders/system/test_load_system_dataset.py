from pathlib import Path
from typing import cast

import numpy as np
from tests.typing_helpers import PytestMonkeyPatch

from dagnam._types import JsonObject
from dagnam.data.loaders.system import load_system_dataset
from dagnam.data.loaders.system.dispatch import _text_layout_for_binding


def test_system_load_system_dataset_array_descriptor_end_to_end(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    np.savez(
        tmp_path / "d.npz",
        x=np.zeros((3, 4, 4, 3), np.uint8),
        y=np.arange(3),
        x_test=np.zeros((1, 4, 4, 3), np.uint8),
        y_test=np.arange(1),
    )
    meta: JsonObject = {
        "id": "system-array",
        "name": "System Array",
        "format": "array",
        "dataset_type": "image",
        "source_type": "system",
        "num_samples": 3,
        "num_classes": 3,
        "layout": {
            "image": {"key": "x", "test_key": "x_test"},
            "label": {"key": "y", "test_key": "y_test"},
        },
        "columns": [{"name": "image"}, {"name": "label"}],
    }
    binding = cast(
        "dict[str, object]",
        {
            "input_column": "image",
            "target_column": "label",
            "input_transform": {"kind": "image_resize", "params": {"size": [4, 4]}},
            "target_transform": {"kind": "identity", "params": {}},
        },
    )
    monkeypatch.setattr("dagnam.data.loaders.system.dispatch._artifact_dir", lambda m: tmp_path)

    dataset = load_system_dataset(meta, binding=binding)

    assert dataset.native_train is not None
    x, y = cast("tuple[np.ndarray, np.ndarray]", dataset.native_train[0])
    assert x.shape == (4, 4, 3)
    assert int(y) == 0
    assert len(dataset.native_train) == 3
    assert dataset.native_test is not None
    assert len(dataset.native_test) == 1


def test_system_load_system_dataset_text_lm_builds_loader_targets(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    (tmp_path / "wiki.train.tokens").write_text(
        "alpha beta gamma delta epsilon zeta eta theta",
        encoding="utf-8",
    )
    meta: JsonObject = {
        "id": "wikitext-2",
        "name": "WikiText-2",
        "format": "text",
        "dataset_type": "text",
        "source_type": "system",
        "num_samples": 8,
        "num_classes": 20,
        "task_type_hint": "language_modeling",
        "layout": {"text": {"file": "wiki.train.tokens"}},
        "columns": [{"name": "text"}],
        "column_roles": {"text": "text_input"},
    }
    binding = cast(
        "dict[str, object]",
        {
            "input_column": "text",
            "target_column": None,
            "input_transform": {
                "kind": "tokenize",
                "params": {"sequence_length": 3, "vocab_size": 20},
            },
            "target_transform": {"kind": "identity", "params": {}},
            "self_supervised": {"kind": "next_token", "where": "loader"},
        },
    )
    monkeypatch.setattr("dagnam.data.loaders.system.dispatch._artifact_dir", lambda m: tmp_path)

    dataset = load_system_dataset(meta, binding=binding)

    assert dataset.native_train is not None
    x, y = cast("tuple[np.ndarray, np.ndarray]", dataset.native_train[0])
    assert x.tolist() == [1, 2, 3]
    assert y.tolist() == [2, 3, 4]


def test_text_layout_for_binding_returns_layout_when_no_text_dict() -> None:
    layout: dict[str, object] = {"image": {"dir": "imgs/"}}

    result = _text_layout_for_binding(layout, {}, {})

    assert result is layout


def test_text_layout_for_binding_returns_layout_when_not_language_modeling() -> None:
    layout: dict[str, object] = {"text": {"file": "wiki.tokens"}}

    result = _text_layout_for_binding(layout, {"task_type_hint": "classification"}, {})

    assert result is layout


def test_text_layout_for_binding_sets_sequence_and_vocab_from_params() -> None:
    layout: dict[str, object] = {"text": {"file": "wiki.tokens"}}
    meta: JsonObject = {"task_type_hint": "language_modeling"}
    binding = {
        "self_supervised": {"kind": "next_token", "where": "loader"},
        "input_transform": {
            "kind": "tokenize",
            "params": {"sequence_length": 16, "vocab_size": 5000},
        },
    }

    result = _text_layout_for_binding(layout, meta, binding)
    text_spec = cast("dict[str, object]", result["text"])

    assert text_spec["self_supervised"] == "next_token"
    assert text_spec["sequence_length"] == 16
    assert text_spec["vocab_size"] == 5000


def test_text_layout_for_binding_omits_non_int_or_absent_params() -> None:
    layout: dict[str, object] = {"text": {"file": "wiki.tokens"}}
    meta: JsonObject = {"task_type_hint": "language_modeling"}
    # input_transform has no params dict -> the param block is skipped entirely.
    no_params = {
        "self_supervised": {"kind": "next_token", "where": "loader"},
        "input_transform": {"kind": "tokenize"},
    }
    spec_no_params = cast(
        "dict[str, object]", _text_layout_for_binding(layout, meta, no_params)["text"]
    )
    assert "sequence_length" not in spec_no_params
    assert "vocab_size" not in spec_no_params

    # params present but non-int values -> each optional field is individually skipped.
    bad_params = {
        "self_supervised": {"kind": "next_token", "where": "loader"},
        "input_transform": {
            "kind": "tokenize",
            "params": {"sequence_length": None, "vocab_size": "x"},
        },
    }
    spec_bad = cast("dict[str, object]", _text_layout_for_binding(layout, meta, bad_params)["text"])
    assert "sequence_length" not in spec_bad
    assert "vocab_size" not in spec_bad
