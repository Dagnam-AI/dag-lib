"""Shared helpers for descriptor-driven system decoders."""

from __future__ import annotations

from pathlib import Path
import tarfile
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from PIL import Image

from dagnam.data.loaders.system.decoders.base import DecodeError


def spec_dict(layout: dict[str, object], name: str) -> dict[str, Any]:
    value = layout.get(name)
    if not isinstance(value, dict):
        raise DecodeError(f"missing layout for column {name!r}")
    return cast("dict[str, Any]", value)


def extensions(spec: dict[str, Any]) -> tuple[str, ...]:
    raw = spec.get("ext", [])
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise DecodeError("layout ext must be a list of strings")
    return tuple(raw)


def safe_extract_tar(tarball: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    with tarfile.open(tarball, "r:gz") as archive:
        for member in archive.getmembers():
            resolved_member = (destination / member.name).resolve()
            if resolved_destination not in (resolved_member, *resolved_member.parents):
                raise DecodeError(f"tar member escapes destination: {member.name}")
        archive.extractall(destination, filter="data")
    roots = [item for item in destination.iterdir() if item.is_dir()]
    return roots[0] if len(roots) == 1 else destination


def read_rgb(path: Path) -> npt.NDArray[np.uint8]:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def read_mask(path: Path) -> npt.NDArray[np.uint8]:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))
