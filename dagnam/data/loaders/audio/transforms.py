"""Framework adapters for audio folder datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dagnam.data.loaders.audio.dataset import (
    AudioFolderDataset,
    _collect_audio_files,
    _resolve_audio_split_dir,
)
from dagnam.data.loaders.audio.io import _collect_audio_samples, _load_waveform_py
from dagnam.data.loaders.media import discover_class_folders, ensure_extracted, split_indices
from dagnam.data.loaders.torch_utils import should_pin_memory

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from dagnam.data.dataset import DagnamDataset


def create_pytorch_loader(
    dagnam_ds: DagnamDataset,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool | None = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    sample_rate: int | None = None,
    n_mels: int = 64,
    max_duration_sec: float = 5.0,
    waveform_transform=None,
    spectrogram_transform=None,
    target_transform=None,
    collate_fn=None,
) -> DataLoader:
    """Create a PyTorch DataLoader from an audio-folder dataset.

    Requires ``torchaudio`` to be installed.

    Args:
        dagnam_ds: The DagnamDataset instance.
        split: One of 'train', 'val', 'test'.
        batch_size: Batch size for the DataLoader.
        num_workers: Number of data loading workers.
        shuffle: Whether to shuffle. Defaults to True for train.
        val_ratio: Fraction for validation when using deterministic splits.
        test_ratio: Fraction for test when using deterministic splits.
        seed: Random seed for deterministic splitting.
        sample_rate: Target sample rate for resampling. If None, uses
            metadata or defaults to 16000.
        n_mels: Number of mel filterbanks for spectrogram.
        max_duration_sec: Maximum audio duration in seconds (clips longer
            audio, pads shorter).

    Returns:
        A PyTorch DataLoader yielding (spectrogram_tensor, label) batches.

    Raises:
        ImportError: If torch or torchaudio is not installed.
    """
    try:
        import torch  # noqa: F401
        from torch.utils.data import DataLoader, Subset
    except ImportError:
        raise ImportError(
            "PyTorch is required for audio loading. Install with: pip install dagnam[audio]"
        )

    try:
        import torchaudio  # noqa: F401
    except ImportError:
        raise ImportError(
            "torchaudio is required for audio folder loading. "
            "Install with: pip install dagnam[audio]"
        )

    if shuffle is None:
        shuffle = split == "train"

    # Resolve audio parameters from metadata
    meta_audio = getattr(dagnam_ds, "_meta_audio", None)
    if meta_audio is None and hasattr(dagnam_ds, "__dict__"):
        # Check if audio config was passed in the original meta dict
        meta_audio = {}

    if sample_rate is None:
        sample_rate = 16000
        if hasattr(dagnam_ds, "_raw_meta"):
            audio_cfg = dagnam_ds._raw_meta.get("audio", {})
            if audio_cfg:
                sample_rate = audio_cfg.get("sample_rate", 16000)
                n_mels = audio_cfg.get("n_mels", n_mels)

    # Ensure archives are extracted
    data_root = ensure_extracted(dagnam_ds._data_dir)

    # Discover folder layout
    layout = discover_class_folders(data_root)

    # Build the dataset
    if layout.has_explicit_splits:
        split_dir = _resolve_audio_split_dir(data_root, split, layout.splits)
        files, labels, class_names = _collect_audio_files(split_dir)
    else:
        files, labels, class_names = _collect_audio_files(data_root)

    dataset = AudioFolderDataset(
        file_paths=files,
        labels=labels,
        target_sample_rate=sample_rate,
        n_mels=n_mels,
        max_duration_sec=max_duration_sec,
        waveform_transform=waveform_transform,
        spectrogram_transform=spectrogram_transform,
        target_transform=target_transform,
    )

    # Apply deterministic split if unsplit
    if not layout.has_explicit_splits:
        n = len(dataset)
        train_idx, val_idx, test_idx = split_indices(
            n, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
        split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
        dataset = Subset(dataset, split_map[split])

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=should_pin_memory(),
        drop_last=(split == "train"),
        collate_fn=collate_fn,
    )


def create_tensorflow_dataset(
    dagnam_ds: DagnamDataset,
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool | None = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    sample_rate: int = 16000,
    max_duration_sec: float = 5.0,
    map_fn=None,
    batch_map_fn=None,
):
    """Create a tf.data.Dataset from an audio-folder dataset.

    Uses ``tf.audio.decode_wav`` for WAV; MP3/FLAC files are decoded on the
    Python side via ``soundfile`` if installed (fallback to PyTorch
    torchaudio). Returns raw waveforms (not spectrograms) so that the model
    can apply the generated MelSpectrogram layer. Pads/truncates to a fixed
    sample count per batch.
    """
    import numpy as np
    import tensorflow as tf

    if shuffle is None:
        shuffle = split == "train"

    samples, _classes = _collect_audio_samples(dagnam_ds, split, val_ratio, test_ratio, seed)
    target_len = int(max_duration_sec * sample_rate)

    paths = np.array([str(s[0]) for s in samples])
    labels = np.array([s[1] for s in samples], dtype=np.int64)

    def _load_one(path_tensor, label_tensor):
        path_str = (
            path_tensor.numpy().decode("utf-8") if hasattr(path_tensor, "numpy") else path_tensor
        )
        waveform = _load_waveform_py(path_str, sample_rate, target_len)
        return waveform.astype(np.float32), np.int64(
            label_tensor.numpy() if hasattr(label_tensor, "numpy") else label_tensor
        )

    def _map(path, label):
        waveform, lbl = tf.py_function(_load_one, [path, label], [tf.float32, tf.int64])
        waveform.set_shape([target_len])
        lbl.set_shape([])
        return waveform, lbl

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=max(len(paths), 1024), seed=seed)
    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    if map_fn is not None:
        ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    if batch_map_fn is not None:
        ds = ds.map(batch_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def create_flax_dataset(
    dagnam_ds: DagnamDataset,
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool | None = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    sample_rate: int = 16000,
    max_duration_sec: float = 5.0,
    transform_fn=None,
    batch_transform_fn=None,
) -> list:
    """Create a list of FlaxBatch from an audio-folder dataset.

    Loads waveforms eagerly into a list of JAX arrays. Suitable for small
    audio benchmarks; for large corpora use ``to_tensorflow_dataset`` + a
    per-batch ``jnp.asarray`` in the training loop.
    """
    import jax.numpy as jnp
    import numpy as np

    from dagnam.data.loaders.flax import FlaxBatch

    if shuffle is None:
        shuffle = split == "train"

    samples, _classes = _collect_audio_samples(dagnam_ds, split, val_ratio, test_ratio, seed)
    target_len = int(max_duration_sec * sample_rate)

    if shuffle:
        import random as _random

        _random.Random(seed).shuffle(samples)

    batches: list[FlaxBatch] = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        waves = []
        labels = []
        for path, label in chunk:
            w = _load_waveform_py(str(path), sample_rate, target_len)
            if transform_fn is not None:
                w = transform_fn(w)
            waves.append(w)
            labels.append(label)
        x = jnp.asarray(np.stack(waves).astype(np.float32))
        y = jnp.asarray(np.array(labels, dtype=np.int64))
        batch = FlaxBatch(features=x, labels=y)
        if batch_transform_fn is not None:
            feat, lbl = batch_transform_fn(batch.features, batch.labels)
            batch = FlaxBatch(features=feat, labels=lbl)
        batches.append(batch)

    return batches
