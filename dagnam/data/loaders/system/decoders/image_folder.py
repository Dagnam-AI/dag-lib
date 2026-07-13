"""Class-subdirectory image folder decoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dagnam.data.loaders.system.column_store import Column, ColumnStore
from dagnam.data.loaders.system.decoders._helpers import (
    extensions,
    read_rgb,
    safe_subpath,
    spec_dict,
)
from dagnam.data.loaders.system.decoders.base import DecodeError


class ImageFolderDecoder:
    """Decode image class subdirectories into image and label columns."""

    def decode(self, artifact_dir: Path, layout: dict[str, object], split: str) -> ColumnStore:
        del split
        image_spec = spec_dict(layout, "image")
        root = safe_subpath(artifact_dir, str(image_spec["dir"]))
        image_exts = extensions(image_spec)
        if not root.exists():
            raise DecodeError(f"image_folder: image root does not exist: {root}")

        classes = sorted(item for item in root.iterdir() if item.is_dir())
        if not classes:
            raise DecodeError(f"image_folder: no class subdirectories under {root}")

        image_paths: list[Path] = []
        labels: list[int] = []
        for label, class_dir in enumerate(classes):
            for image in sorted(item for item in class_dir.iterdir() if item.suffix in image_exts):
                image_paths.append(image)
                labels.append(label)
        if not image_paths:
            raise DecodeError(f"image_folder: no images under {root}")
        return ColumnStore(
            {
                "image": Column.lazy(image_paths, read_rgb),
                "label": Column.eager(np.asarray(labels, dtype=np.int64)),
            }
        )
