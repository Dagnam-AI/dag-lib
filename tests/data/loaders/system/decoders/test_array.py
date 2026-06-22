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
