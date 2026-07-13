from __future__ import annotations

from pathlib import Path
import tarfile
from typing import cast

import numpy as np
import pytest

PIL_Image = pytest.importorskip("PIL.Image")

from dagnam.data.loaders.system.decoders import get_decoder  # noqa: E402


def _make_tarball(tmp_path: Path) -> Path:
    root = tmp_path / "oxford-pets"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)
    for basename in ("a", "b"):
        PIL_Image.new("RGB", (8, 8), (1, 2, 3)).save(root / "images" / f"{basename}.jpg")
        PIL_Image.new("L", (8, 8), 2).save(root / "masks" / f"{basename}.png")
    tarball = tmp_path / "oxford-pets.tar.gz"
    with tarfile.open(tarball, "w:gz") as archive:
        archive.add(root, arcname="oxford-pets")
    return tmp_path


def test_system_image_mask_folder_pairs_images_and_masks_from_tarball(tmp_path: Path) -> None:
    artifact_dir = _make_tarball(tmp_path)
    layout = cast(
        "dict[str, object]",
        {
            "image": {"dir": "images/", "ext": [".jpg"]},
            "segmentation_mask": {"dir": "masks/", "ext": [".png"], "value_set": [1, 2, 3]},
        },
    )

    store = get_decoder("image_mask_folder").decode(artifact_dir, layout, "train")

    assert len(store) == 2
    assert store.column("image")[0].shape == (8, 8, 3)
    assert store.column("segmentation_mask")[0].shape == (8, 8)
    assert np.asarray(store.column("segmentation_mask")[0]).dtype == np.uint8


def test_system_image_mask_folder_rejects_traversal_dir(tmp_path: Path) -> None:
    from dagnam.data.loaders.system.decoders.base import DecodeError

    artifact_dir = _make_tarball(tmp_path)
    layout = cast(
        "dict[str, object]",
        {
            "image": {"dir": "../../../etc", "ext": [".jpg"]},
            "segmentation_mask": {"dir": "masks/", "ext": [".png"]},
        },
    )
    with pytest.raises(DecodeError):
        get_decoder("image_mask_folder").decode(artifact_dir, layout, "train")
