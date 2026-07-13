from __future__ import annotations

from pathlib import Path

import pytest

PIL_Image = pytest.importorskip("PIL.Image")

from dagnam.data.loaders.system.decoders import get_decoder  # noqa: E402


def test_system_image_folder_assigns_sorted_class_subdir_labels(tmp_path: Path) -> None:
    root = tmp_path / "imgs"
    for cls in ("dog", "cat"):
        (root / cls).mkdir(parents=True)
        PIL_Image.new("RGB", (4, 4)).save(root / cls / "0.png")

    store = get_decoder("image_folder").decode(
        tmp_path,
        {"image": {"dir": "imgs/", "ext": [".png"]}, "label": {"from": "subdir"}},
        "train",
    )

    assert len(store) == 2
    assert sorted(int(store.column("label")[i]) for i in range(2)) == [0, 1]
    assert store.column("image")[0].shape == (4, 4, 3)


def test_system_image_folder_rejects_traversal_dir(tmp_path: Path) -> None:
    from dagnam.data.loaders.system.decoders.base import DecodeError

    with pytest.raises(DecodeError):
        get_decoder("image_folder").decode(
            tmp_path,
            {"image": {"dir": "../../../etc", "ext": [".png"]}, "label": {"from": "subdir"}},
            "train",
        )
