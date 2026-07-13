"""Coverage for the shared system-decoder helpers.

Focus on the security-relevant surface: ``safe_subpath`` containment against
server-controlled ``layout`` paths, and the ``read_rgb``/``read_mask`` pixel
ceiling. The tar/extension helpers are exercised via the decoder tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from dagnam.data.loaders.system.decoders import _helpers
from dagnam.data.loaders.system.decoders._helpers import (
    read_mask,
    read_rgb,
    safe_subpath,
)
from dagnam.data.loaders.system.decoders.base import DecodeError


def test_safe_subpath_allows_nested_relative_path(tmp_path: Path) -> None:
    target = tmp_path / "images" / "train"
    target.mkdir(parents=True)
    assert safe_subpath(tmp_path, "images/train") == target.resolve()


def test_safe_subpath_allows_dot(tmp_path: Path) -> None:
    # "." (or "") means "the artifact dir itself" — legitimate for flat datasets.
    assert safe_subpath(tmp_path, ".") == tmp_path.resolve()


@pytest.mark.parametrize(
    "malicious",
    [
        "../escape",
        "a/../../escape",
        "/etc/passwd",
        "..\\..\\windows",  # backslash payload must be caught on POSIX too
        "",
    ],
)
def test_safe_subpath_rejects_traversal_and_absolute(tmp_path: Path, malicious: str) -> None:
    with pytest.raises(DecodeError):
        safe_subpath(tmp_path, malicious)


def test_safe_subpath_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_target"
    outside.mkdir()
    base = tmp_path / "cache"
    base.mkdir()
    (base / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(DecodeError):
        safe_subpath(base, "link")


def _make_image(path: Path, size: tuple[int, int]) -> None:
    Image.new("RGB", size, (10, 20, 30)).save(path)


def test_read_rgb_reads_small_image(tmp_path: Path) -> None:
    p = tmp_path / "img.png"
    _make_image(p, (4, 3))
    arr = read_rgb(p)
    assert arr.shape == (3, 4, 3)
    assert arr.dtype == np.uint8


def test_read_mask_reads_small_image(tmp_path: Path) -> None:
    p = tmp_path / "mask.png"
    _make_image(p, (4, 3))
    arr = read_mask(p)
    assert arr.shape == (3, 4)


def test_read_rgb_rejects_oversized_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Lower the ceiling instead of writing a 64 MP file. The check reads only
    # the header (image.size), so a normal small file trips the lowered cap.
    monkeypatch.setattr(_helpers, "_MAX_IMAGE_PIXELS", 3)
    p = tmp_path / "img.png"
    _make_image(p, (4, 3))
    with pytest.raises(DecodeError, match="too large to decode"):
        read_rgb(p)


def test_read_mask_rejects_oversized_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_helpers, "_MAX_IMAGE_PIXELS", 3)
    p = tmp_path / "mask.png"
    _make_image(p, (4, 3))
    with pytest.raises(DecodeError, match="too large to decode"):
        read_mask(p)
