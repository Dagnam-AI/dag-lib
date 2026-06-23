from __future__ import annotations

import numpy as np

from dagnam.data.loaders.system.transform_executor import apply_transform


def test_system_transform_executor_resizes_image_to_declared_hw() -> None:
    image = np.zeros((10, 4, 3), np.uint8)

    out = apply_transform(image, {"kind": "image_resize", "params": {"size": [8, 8]}}, None)

    assert out.shape == (8, 8, 3)


def test_system_transform_executor_resizes_mask_and_remaps_to_contiguous_long() -> None:
    mask = np.array([[1, 2], [3, 1]], np.uint8)

    out = apply_transform(
        mask,
        {
            "kind": "mask",
            "params": {"resize": [2, 2], "remap": "contiguous_long", "value_set": [1, 2, 3]},
        },
        None,
    )

    assert out.dtype == np.int64
    assert set(np.unique(out).tolist()) <= {0, 1, 2}


def test_system_transform_executor_resizes_single_channel_image_and_keeps_channel() -> None:
    image = np.zeros((10, 4, 1), np.uint8)

    out = apply_transform(image, {"kind": "image_resize", "params": {"size": [8, 8]}}, None)

    assert out.shape == (8, 8, 1)


def test_system_transform_executor_mask_remap_is_exact_and_zeroes_unknown_values() -> None:
    mask = np.array([[1, 2, 3], [0, 9, 1]], np.int64)  # 0 and 9 are outside {1, 2, 3}

    out = apply_transform(
        mask,
        {"kind": "mask", "params": {"remap": "contiguous_long", "value_set": [1, 2, 3]}},
        None,
    )

    # 1->0, 2->1, 3->2; out-of-set 0 and 9 -> 0
    assert out.tolist() == [[0, 1, 2], [0, 0, 0]]


def test_system_transform_executor_mask_remap_infers_value_set_when_absent() -> None:
    mask = np.array([[2, 4], [4, 2]], np.int64)

    out = apply_transform(mask, {"kind": "mask", "params": {"remap": "contiguous_long"}}, None)

    # inferred set {2, 4} -> {0, 1}
    assert out.tolist() == [[0, 1], [1, 0]]


def test_system_transform_executor_mask_remap_empty_value_set_returns_array() -> None:
    mask = np.array([[5, 6]], np.int64)

    out = apply_transform(
        mask,
        {"kind": "mask", "params": {"remap": "contiguous_long", "value_set": []}},
        None,
    )

    assert out.tolist() == [[5, 6]]


def test_system_transform_executor_image_resize_adds_channel_axis_to_grayscale() -> None:
    image = np.zeros((10, 4), np.uint8)  # grayscale, no channel axis

    out = apply_transform(image, {"kind": "image_resize", "params": {"size": [8, 8]}}, None)

    assert out.shape == (8, 8, 1)  # canonical channels-last [H, W, C]


def test_system_transform_executor_class_index_casts_to_int64() -> None:
    out = apply_transform(
        np.array(3, np.uint8), {"kind": "class_index", "params": {"dtype": "long"}}, None
    )

    assert out.dtype == np.int64
    assert int(out) == 3


def test_system_transform_executor_numeric_casts_to_float32() -> None:
    out = apply_transform(np.array([1, 2], np.int64), {"kind": "numeric", "params": {}}, None)

    assert out.dtype == np.float32
    assert out.tolist() == [1.0, 2.0]


def test_system_transform_executor_tokenize_pads_and_truncates_sequence() -> None:
    sequence = np.array([5, 6, 7], dtype=np.int64)

    out = apply_transform(sequence, {"kind": "tokenize", "params": {"sequence_length": 5}}, None)

    assert out.tolist() == [5, 6, 7, 0, 0]


def test_system_transform_executor_tokenize_clamps_ids_to_vocab_size() -> None:
    seq = np.array([3, 50000, 9999, 10000], np.int64)  # 50000 and 10000 exceed vocab 10000

    out = apply_transform(
        seq, {"kind": "tokenize", "params": {"sequence_length": 4, "vocab_size": 10000}}, None
    )

    assert out.tolist() == [3, 9999, 9999, 9999]  # clamped to [0, vocab_size - 1]


def test_system_transform_executor_audio_pads_and_truncates_to_target_length() -> None:
    short = np.zeros(10, dtype=np.float32)
    long = np.ones(100, dtype=np.float32)
    transform = {"kind": "audio", "params": {"target_length": 50}}

    padded = apply_transform(short, transform)
    truncated = apply_transform(long, transform)

    assert padded.shape == (50,)
    assert truncated.shape == (50,)
    assert truncated.tolist() == [1.0] * 50


def test_system_transform_executor_audio_without_target_length_is_identity() -> None:
    waveform = np.ones(10, dtype=np.float32)

    out = apply_transform(waveform, {"kind": "audio", "params": {"target_length": None}})

    assert out.shape == (10,)


def test_system_transform_executor_identity_returns_array_value() -> None:
    value = np.array([1, 2])

    out = apply_transform(value, {"kind": "identity", "params": {}}, None)

    assert out.tolist() == [1, 2]


def test_system_transform_executor_audio_equal_length_returns_waveform() -> None:
    waveform = np.arange(50, dtype=np.float32)

    out = apply_transform(waveform, {"kind": "audio", "params": {"target_length": 50}})

    assert out.shape == (50,)
    assert out.tolist() == waveform.tolist()
