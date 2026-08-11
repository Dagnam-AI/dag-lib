"""Video frame-stack loader helpers for video datasets.

Video clips arrive as a decoded channels-last ``[T, H, W, C]`` (or grayscale
``[T, H, W]``) numpy array — one call to :func:`resize_frames` per clip resizes
every frame identically. Kept pickle-free and dependency-light (PIL only,
already a dag-lib dependency) rather than pulling in a video-codec library:
video fixtures are stored as plain numeric ``.npz`` frame-tensor arrays decoded
by the existing ``system/decoders/array.py``, never as encoded video files.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from PIL import Image

_VIDEO_RANKS = (3, 4)


def resize_frames(frames: npt.ArrayLike, size: tuple[int, int]) -> npt.NDArray[np.float32]:
    """Resize every frame of a ``[T, H, W, C]``/``[T, H, W]`` clip to ``size``.

    Mirrors ``system.transform_executor._resize``'s single-image PIL resize,
    applied once per frame along the leading time axis, so a video clip resizes
    exactly the way a single image does — including the ``[H, W, 1]`` squeeze
    (PIL cannot build an image from a single-channel 3-D array), which keeps the
    clip's channel axis intact for downstream framework converters.
    """
    array = np.asarray(frames)
    if array.ndim not in _VIDEO_RANKS:
        raise ValueError(f"video frames array must be rank 3 or 4, got rank {array.ndim}")

    height, width = size
    has_channel_axis = array.ndim == 4 and array.shape[-1] == 1
    source = array[..., 0] if has_channel_axis else array
    resized = np.stack(
        [
            np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.BILINEAR))
            for frame in source
        ]
    ).astype(np.float32)
    return resized[..., np.newaxis] if has_channel_axis else resized
