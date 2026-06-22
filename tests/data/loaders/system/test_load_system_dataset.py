from pathlib import Path
from typing import cast

import numpy as np
from tests.typing_helpers import PytestMonkeyPatch

from dagnam._types import JsonObject
from dagnam.data.loaders.system import load_system_dataset


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
