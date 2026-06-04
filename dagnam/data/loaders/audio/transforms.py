"""Framework adapters for audio folder datasets."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import import_module
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import numpy.typing as npt

from dagnam._types import SupportsNumpy, TensorflowDataset
from dagnam.data.loaders.audio.dataset import (
    AudioFolderDataset,
    collect_audio_files,
    resolve_audio_split_dir,
)
from dagnam.data.loaders.audio.io import collect_audio_samples, load_waveform_py
from dagnam.data.loaders.media import discover_class_folders, ensure_extracted, split_indices
from dagnam.data.loaders.torch_utils import should_pin_memory

if TYPE_CHECKING:
    import jax
    from torch.utils.data import DataLoader, Dataset

    from dagnam.data.dataset._typing import DatasetMixinBase
    from dagnam.data.loaders.flax import FlaxBatch

WaveformArray = npt.NDArray[np.float32]
SampleTransform = Callable[[object], object]
TensorflowMapTransform = Callable[[object, object], object]
WaveformTransform = Callable[[npt.ArrayLike], npt.ArrayLike]
JaxArrayFactory = Callable[[npt.ArrayLike], "jax.Array"]
BatchTransform = Callable[["jax.Array", "jax.Array"], tuple["jax.Array", "jax.Array"]]


class TensorValue(Protocol):
    """TensorFlow tensor surface used by this adapter."""

    def numpy(self) -> object: ...

    def set_shape(self, shape: Sequence[int | None]) -> None: ...


class TensorflowAudioDatasetFactory(Protocol):
    """TensorFlow dataset factory used by this adapter."""

    def from_tensor_slices(self, tensors: object) -> TensorflowDataset: ...


class TensorflowAudioDataNamespace(Protocol):
    """TensorFlow data namespace used by this adapter."""

    AUTOTUNE: object
    Dataset: TensorflowAudioDatasetFactory


class TensorflowAudioModule(Protocol):
    """TensorFlow module surface used by the audio adapter."""

    data: TensorflowAudioDataNamespace
    float32: object
    int64: object

    def py_function(
        self,
        func: Callable[..., object],
        inp: Sequence[object],
        Tout: Sequence[object],
    ) -> tuple[TensorValue, TensorValue]: ...


def _load_tensorflow() -> TensorflowAudioModule:
    return cast("TensorflowAudioModule", import_module("tensorflow"))


def _decode_path_tensor(value: object) -> str:
    if isinstance(value, SupportsNumpy):
        value = value.numpy()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    return str(value)


def _decode_label_tensor(value: object) -> np.int64:
    if isinstance(value, SupportsNumpy):
        value = value.numpy()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str | bytes | int | float | bool):
        return np.int64(value)
    raise TypeError(f"Expected TensorFlow scalar label, got {type(value).__name__}")


def _int_setting(value: object, default: int) -> int:
    if isinstance(value, bool | dict | list) or value is None:
        return default
    if isinstance(value, str | int | float):
        return int(value)
    return default


def create_pytorch_loader(
    dagnam_ds: DatasetMixinBase,
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
    waveform_transform: SampleTransform | None = None,
    spectrogram_transform: SampleTransform | None = None,
    target_transform: SampleTransform | None = None,
    collate_fn: Callable[[object], object] | None = None,
) -> DataLoader[object]:
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
        from torch.utils.data import DataLoader, Subset
    except ImportError:
        raise ImportError(
            "PyTorch is required for audio loading. Install with: pip install dagnam[audio]"
        )

    try:
        import_module("torchaudio")
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
            audio_cfg = dagnam_ds.raw_meta.get("audio", {})
            if isinstance(audio_cfg, dict):
                sample_rate = _int_setting(audio_cfg.get("sample_rate"), 16000)
                n_mels = _int_setting(audio_cfg.get("n_mels"), n_mels)

    # Ensure archives are extracted
    data_root = ensure_extracted(dagnam_ds.data_dir)

    # Discover folder layout
    layout = discover_class_folders(data_root)

    # Build the dataset
    if layout.has_explicit_splits:
        split_dir = resolve_audio_split_dir(data_root, split, layout.splits)
        files, labels, _class_names = collect_audio_files(split_dir)
    else:
        files, labels, _class_names = collect_audio_files(data_root)

    dataset: object = AudioFolderDataset(
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
        dataset = cast(
            "Dataset[object]", Subset(cast("Dataset[object]", dataset), split_map[split])
        )

    loader = DataLoader(
        cast("Dataset[object]", dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=should_pin_memory(),
        drop_last=(split == "train"),
        collate_fn=collate_fn,
    )
    return loader


def create_tensorflow_dataset(
    dagnam_ds: DatasetMixinBase,
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool | None = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    sample_rate: int = 16000,
    max_duration_sec: float = 5.0,
    map_fn: TensorflowMapTransform | None = None,
    batch_map_fn: TensorflowMapTransform | None = None,
) -> TensorflowDataset:
    """Create a tf.data.Dataset from an audio-folder dataset.

    Uses ``tf.audio.decode_wav`` for WAV; MP3/FLAC files are decoded on the
    Python side via ``soundfile`` if installed (fallback to PyTorch
    torchaudio). Returns raw waveforms (not spectrograms) so that the model
    can apply the generated MelSpectrogram layer. Pads/truncates to a fixed
    sample count per batch.
    """
    tf = _load_tensorflow()
    if shuffle is None:
        shuffle = split == "train"

    samples, _classes = collect_audio_samples(dagnam_ds, split, val_ratio, test_ratio, seed)
    target_len = int(max_duration_sec * sample_rate)

    paths = np.array([str(s[0]) for s in samples])
    labels = np.array([s[1] for s in samples], dtype=np.int64)

    def _load_one(path_tensor: object, label_tensor: object) -> tuple[WaveformArray, np.int64]:
        path_str = _decode_path_tensor(path_tensor)
        waveform: WaveformArray = load_waveform_py(path_str, sample_rate, target_len)
        return waveform.astype(np.float32), _decode_label_tensor(label_tensor)

    def _map(path: object, label: object) -> tuple[TensorValue, TensorValue]:
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
    dagnam_ds: DatasetMixinBase,
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool | None = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    sample_rate: int = 16000,
    max_duration_sec: float = 5.0,
    transform_fn: WaveformTransform | None = None,
    batch_transform_fn: BatchTransform | None = None,
) -> list[FlaxBatch]:
    """Create a list of FlaxBatch from an audio-folder dataset.

    Loads waveforms eagerly into a list of JAX arrays. Suitable for small
    audio benchmarks; for large corpora use ``to_tensorflow_dataset`` + a
    per-batch ``jnp.asarray`` in the training loop.
    """
    import jax.numpy as jnp

    from dagnam.data.loaders.flax import FlaxBatch

    as_jax_array = cast("JaxArrayFactory", jnp.asarray)

    if shuffle is None:
        shuffle = split == "train"

    samples, _classes = collect_audio_samples(dagnam_ds, split, val_ratio, test_ratio, seed)
    target_len = int(max_duration_sec * sample_rate)

    if shuffle:
        import random as _random

        _random.Random(seed).shuffle(samples)

    batches: list[FlaxBatch] = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        waves: list[WaveformArray] = []
        labels: list[int] = []
        for path, label in chunk:
            w: WaveformArray = load_waveform_py(str(path), sample_rate, target_len)
            if transform_fn is not None:
                w = cast("WaveformArray", transform_fn(w))
            waves.append(w)
            labels.append(label)
        x = as_jax_array(np.stack(waves).astype(np.float32))
        y = as_jax_array(np.array(labels, dtype=np.int64))
        batch = FlaxBatch(features=x, labels=y)
        if batch_transform_fn is not None:
            feat, lbl = batch_transform_fn(batch.features, batch.labels)
            batch = FlaxBatch(features=feat, labels=lbl)
        batches.append(batch)

    return batches
