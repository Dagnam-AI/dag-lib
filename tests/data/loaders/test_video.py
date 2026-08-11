"""Tests for dagnam.data.loaders.video."""

from __future__ import annotations

import numpy as np
import pytest

from dagnam.data.loaders.video import resize_frames


def test_resize_frames_resizes_each_frame_independently() -> None:
    clip = np.zeros((4, 16, 16, 3), dtype=np.uint8)
    clip[:, :8, :8, :] = 255  # a quadrant marker per frame

    out = resize_frames(clip, (8, 8))

    assert out.shape == (4, 8, 8, 3)
    assert out.dtype == np.float32
    # The marked quadrant survives the resize on every frame (the resize is
    # per-frame, not a collapse of the time axis into the batch of pixels).
    assert bool(np.all(out[:, :4, :4, :] > 0))


def test_resize_frames_preserves_frame_count_for_grayscale_clips() -> None:
    clip = np.zeros((3, 10, 10), dtype=np.uint8)

    out = resize_frames(clip, (5, 5))

    assert out.shape == (3, 5, 5)


def test_resize_frames_keeps_single_channel_axis() -> None:
    # PIL cannot build an image from a [H, W, 1] frame; the channel axis is
    # squeezed for the resize and restored so downstream rank stays [T, H, W, C].
    clip = np.zeros((3, 10, 10, 1), dtype=np.uint8)

    out = resize_frames(clip, (5, 5))

    assert out.shape == (3, 5, 5, 1)


def test_resize_frames_rejects_non_video_rank() -> None:
    flat = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="video frames array must be rank 3 or 4"):
        resize_frames(flat, (5, 5))
