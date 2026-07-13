"""Shared helpers for descriptor-driven system decoders."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import tarfile
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from PIL import Image

from dagnam.data.loaders.system.decoders.base import DecodeError

# Decompression-bomb bounds, matching dagnam.data.loaders.media: a malicious
# server can serve a checksum-matching tar that expands to exhaust disk/inodes.
_MAX_TAR_MEMBERS = 200_000
_MAX_TAR_UNCOMPRESSED_BYTES = 8 * 1024 * 1024 * 1024

# Per-image pixel ceiling for the decode helpers below. A hostile dataset image
# can otherwise declare enormous dimensions and force a huge allocation before
# any resize (a decompression-bomb DoS). ~64 MP is far above any real training
# image (a 64 MP RGB decode is already ~192 MB).
_MAX_IMAGE_PIXELS = 64_000_000


def spec_dict(layout: dict[str, object], name: str) -> dict[str, Any]:
    value = layout.get(name)
    if not isinstance(value, dict):
        raise DecodeError(f"missing layout for column {name!r}")
    return cast("dict[str, Any]", value)


def safe_subpath(base: Path, relative: str) -> Path:
    """Resolve a server-supplied sub-path strictly within ``base``.

    The ``layout[...]["file"]`` / ``["dir"]`` values in a system-dataset
    descriptor are server-controlled. An absolute value or a ``..`` component
    would escape the artifact cache and let a malicious server read arbitrary
    local files (``/etc/passwd``, ``~/.aws/credentials``) into training data,
    so reject anything that does not resolve inside ``base``. Backslashes are
    normalised to forward slashes so a Windows-style parent-dir payload is
    caught on POSIX too, and the post-resolve containment check also defeats
    symlink escapes.
    """
    normalized = relative.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if not normalized or pure.is_absolute() or ".." in pure.parts:
        raise DecodeError(f"unsafe layout path (absolute or traversal): {relative!r}")
    candidate = base.joinpath(*pure.parts).resolve()
    base_resolved = base.resolve()
    if base_resolved != candidate and base_resolved not in candidate.parents:
        raise DecodeError(f"layout path escapes dataset directory: {relative!r}")
    return candidate


def _check_image_pixels(image: Image.Image) -> None:
    """Reject an image whose declared dimensions exceed the decode ceiling.

    ``Image.open`` only parses the header, so ``image.size`` is available before
    the (expensive) pixel decode — bound the allocation here rather than after.
    """
    width, height = image.size
    if width * height > _MAX_IMAGE_PIXELS:
        raise DecodeError(
            f"image too large to decode safely: {width}x{height} exceeds {_MAX_IMAGE_PIXELS} pixels"
        )


def extensions(spec: dict[str, Any]) -> tuple[str, ...]:
    raw = spec.get("ext", [])
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise DecodeError("layout ext must be a list of strings")
    return tuple(raw)


def safe_extract_tar(tarball: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    with tarfile.open(tarball, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) > _MAX_TAR_MEMBERS:
            raise DecodeError(f"tar has too many members: {len(members)}")
        total_size = 0
        for member in members:
            resolved_member = (destination / member.name).resolve()
            if resolved_destination not in (resolved_member, *resolved_member.parents):
                raise DecodeError(f"tar member escapes destination: {member.name}")
            total_size += member.size
            if total_size > _MAX_TAR_UNCOMPRESSED_BYTES:
                raise DecodeError(
                    f"tar uncompressed size exceeds limit "
                    f"({total_size} > {_MAX_TAR_UNCOMPRESSED_BYTES} bytes)"
                )
        archive.extractall(destination, filter="data")
    roots = [item for item in destination.iterdir() if item.is_dir()]
    return roots[0] if len(roots) == 1 else destination


def read_rgb(path: Path) -> npt.NDArray[np.uint8]:
    with Image.open(path) as image:
        _check_image_pixels(image)
        return np.asarray(image.convert("RGB"))


def read_mask(path: Path) -> npt.NDArray[np.uint8]:
    with Image.open(path) as image:
        _check_image_pixels(image)
        return np.asarray(image.convert("L"))
