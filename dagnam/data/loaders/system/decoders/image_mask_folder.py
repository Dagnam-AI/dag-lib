"""Image + segmentation mask folder decoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from dagnam.data.loaders.system.column_store import Column, ColumnStore
from dagnam.data.loaders.system.decoders._helpers import (
    extensions,
    read_mask,
    read_rgb,
    safe_extract_tar,
    safe_subpath,
    spec_dict,
)
from dagnam.data.loaders.system.decoders.base import DecodeError


def _artifact_root(artifact_dir: Path) -> Path:
    tarballs = sorted(artifact_dir.glob("*.tar.gz"))
    if not tarballs:
        return artifact_dir
    unpacked = artifact_dir / "_unpacked_image_mask_folder"
    if not unpacked.exists():
        return safe_extract_tar(tarballs[0], unpacked)
    roots = [item for item in unpacked.iterdir() if item.is_dir()]
    return roots[0] if len(roots) == 1 else unpacked


class ImageMaskFolderDecoder:
    """Decode paired image and mask files by matching basename."""

    def decode(self, artifact_dir: Path, layout: dict[str, object], split: str) -> ColumnStore:
        del split
        root = _artifact_root(artifact_dir)
        image_spec = spec_dict(layout, "image")
        mask_name = next((name for name in layout if name != "image"), None)
        if mask_name is None:
            raise DecodeError("image_mask_folder requires a mask column")
        mask_spec = cast("dict[str, Any]", layout[mask_name])
        image_dir = safe_subpath(root, str(image_spec["dir"]))
        mask_dir = safe_subpath(root, str(mask_spec["dir"]))
        image_exts = extensions(image_spec)
        mask_exts = extensions(mask_spec)
        if not image_dir.exists() or not mask_dir.exists():
            raise DecodeError(f"image_mask_folder: missing image/mask dirs under {root}")

        image_paths: list[Path] = []
        mask_paths: list[Path] = []
        for image in sorted(item for item in image_dir.iterdir() if item.suffix in image_exts):
            mask = next(
                (
                    mask_dir / f"{image.stem}{ext}"
                    for ext in mask_exts
                    if (mask_dir / f"{image.stem}{ext}").exists()
                ),
                None,
            )
            if mask is None:
                continue
            image_paths.append(image)
            mask_paths.append(mask)

        if not image_paths:
            raise DecodeError(f"image_mask_folder: no paired image/mask files under {root}")
        return ColumnStore(
            {
                "image": Column.lazy(image_paths, read_rgb),
                mask_name: Column.lazy(mask_paths, read_mask),
            }
        )
