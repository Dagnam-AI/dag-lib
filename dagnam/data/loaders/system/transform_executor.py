"""Generic binding-driven transforms for system dataset columns."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt
from PIL import Image

from dagnam.data.loaders.video import resize_frames


def _hw(size: object) -> tuple[int, int]:
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("transform size must be [height, width]")
    return int(size[0]), int(size[1])


def _resize(
    value: npt.ArrayLike,
    size: object,
    *,
    nearest: bool,
) -> npt.NDArray[np.generic]:
    height, width = _hw(size)
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    array = np.asarray(value)
    # PIL cannot build an image from a single-channel ``[H, W, 1]`` array; squeeze
    # the trailing channel for the resize and restore it so the channel contract is
    # preserved for downstream framework converters.
    has_channel_axis = array.ndim == 3 and array.shape[-1] == 1
    source = array[..., 0] if has_channel_axis else array
    resized = np.asarray(Image.fromarray(source).resize((width, height), mode))
    return resized[..., np.newaxis] if has_channel_axis else resized


def _normalize(
    value: npt.NDArray[np.float32],
    normalize: dict[str, Any] | None,
) -> npt.NDArray[np.float32]:
    if normalize is None:
        return value
    mean = np.asarray(normalize["mean"], dtype=np.float32)
    std = np.asarray(normalize["std"], dtype=np.float32)
    return cast("npt.NDArray[np.float32]", (value - mean) / std)


def _remap_contiguous(
    value: npt.NDArray[np.generic],
    value_set: object,
) -> npt.NDArray[np.int64]:
    array = np.asarray(value).astype(np.int64)
    values = (
        sorted({int(item) for item in value_set})
        if isinstance(value_set, list)
        else sorted({int(item) for item in np.unique(array).tolist()})
    )
    if not values:
        return array
    max_value = max(values)
    # Vectorized LUT: index a lookup table by the raw pixel value (O(pixels) in C,
    # not a Python call per pixel). Values outside the declared set — gaps,
    # negatives, or > max — map to 0, matching the prior ``dict.get(item, 0)``.
    lookup = np.zeros(max_value + 1, dtype=np.int64)
    for index, raw in enumerate(values):
        lookup[raw] = index
    in_range = (array >= 0) & (array <= max_value)
    return cast(
        "npt.NDArray[np.int64]",
        np.where(in_range, lookup[np.clip(array, 0, max_value)], 0),
    )


def _fit_frame_count(clip: npt.NDArray[np.float32], frame_count: object) -> npt.NDArray[np.float32]:
    """Trim/zero-pad a clip's leading time axis to the declared ``frame_count``.

    The binding declares the architecture's clip length; a dataset whose clips
    are longer or shorter would otherwise reach ``Conv3d`` with the wrong depth.
    Mirrors the ``audio`` branch's ``target_length`` handling; a missing or
    non-positive declaration leaves the clip untouched.
    """
    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count <= 0:
        return clip
    if clip.shape[0] > frame_count:
        return clip[:frame_count]
    if clip.shape[0] < frame_count:
        padding = [(0, frame_count - clip.shape[0]), *[(0, 0)] * (clip.ndim - 1)]
        return np.pad(clip, padding).astype(np.float32)
    return clip


def apply_transform(
    value: npt.ArrayLike,
    transform: dict[str, Any],
    normalize: dict[str, Any] | None = None,
) -> npt.NDArray[np.generic]:
    """Apply one binding transform to one column value."""
    kind = transform.get("kind", "identity")
    params = cast("dict[str, Any]", transform.get("params", {}))
    if kind == "image_resize":
        resized = _resize(value, params["size"], nearest=False) if params.get("size") else value
        image = np.asarray(resized, dtype=np.float32) / 255.0
        # Canonical channels-last [H, W, C]: a grayscale [H, W] gains an explicit
        # channel axis so every framework converter sees the same rank (pytorch
        # then transposes to [C, H, W]; tf/flax keep channels-last as-is).
        if image.ndim == 2:
            image = image[..., np.newaxis]
        return _normalize(image, normalize)
    if kind == "video":
        frames = resize_frames(value, _hw(params["size"])) if params.get("size") else value
        clip = np.asarray(frames, dtype=np.float32) / 255.0
        # Canonical channels-last [T, H, W, C]: a grayscale [T, H, W] clip gains an
        # explicit channel axis so every framework converter sees the same rank
        # (pytorch then moves it to [C, T, H, W] for Conv3d; tf/flax keep it last).
        if clip.ndim == 3:
            clip = clip[..., np.newaxis]
        return _normalize(_fit_frame_count(clip, params.get("frame_count")), normalize)
    if kind == "mask":
        mask = _resize(value, params["resize"], nearest=True) if params.get("resize") else value
        mask_array = np.asarray(mask)
        if params.get("remap") == "contiguous_long":
            return _remap_contiguous(mask_array, params.get("value_set"))
        return mask_array.astype(np.int64)
    if kind == "class_index":
        # Classification target: a 0-based integer class index (CrossEntropy needs long).
        return np.asarray(value).astype(np.int64)
    if kind == "numeric":
        # Regression target: a float scalar/vector.
        return np.asarray(value).astype(np.float32)
    if kind == "tokenize":
        sequence_length = int(params.get("sequence_length", len(np.asarray(value))))
        sequence = np.asarray(value, dtype=np.int64)[:sequence_length]
        if len(sequence) < sequence_length:
            sequence = np.pad(sequence, (0, sequence_length - len(sequence)))
        vocab_size = params.get("vocab_size")
        if isinstance(vocab_size, int) and not isinstance(vocab_size, bool) and vocab_size > 0:
            # Clamp token ids into the embedding's vocabulary; out-of-vocab/high ids
            # would otherwise raise "index out of range" inside nn.Embedding.
            sequence = np.clip(sequence, 0, vocab_size - 1)
        return sequence
    if kind == "audio":
        waveform = np.asarray(value, dtype=np.float32)
        target_length = params.get("target_length")
        if (
            not isinstance(target_length, int)
            or isinstance(target_length, bool)
            or target_length <= 0
        ):
            return waveform
        if waveform.shape[0] > target_length:
            return waveform[:target_length]
        if waveform.shape[0] < target_length:
            return np.pad(waveform, (0, target_length - waveform.shape[0])).astype(np.float32)
        return waveform
    return np.asarray(value)
