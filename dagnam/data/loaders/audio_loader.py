"""Audio folder dataset loader for PyTorch.

Loads audio classification datasets organized in class-folder structure:
- Split layout: root/{split}/{class}/*.wav
- Unsplit layout: root/{class}/*.wav

Supports WAV, MP3, and FLAC formats. Applies optional resampling and
mel spectrogram conversion.

Requires: torch, torchaudio (install with: pip install dagnam[audio])
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

from dagnam.data.loaders.media_utils import (
    AUDIO_EXTENSIONS,
    discover_class_folders,
    ensure_extracted,
    split_indices,
)

if TYPE_CHECKING:
    from dagnam.data.dataset import DagnamDataset
    from torch.utils.data import DataLoader


def create_pytorch_loader(
    dagnam_ds: "DagnamDataset",
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
) -> "DataLoader":
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
            "PyTorch is required for audio loading. "
            "Install with: pip install dagnam[audio]"
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
        pin_memory=True,
        drop_last=(split == "train"),
        collate_fn=collate_fn,
    )


class AudioFolderDataset:
    """PyTorch Dataset for audio classification from folder structure.

    Loads audio files, converts to mono, resamples to target rate,
    and applies mel spectrogram transform.
    """

    def __init__(
        self,
        file_paths: List[Path],
        labels: List[int],
        target_sample_rate: int = 16000,
        n_mels: int = 64,
        max_duration_sec: float = 5.0,
        waveform_transform=None,
        spectrogram_transform=None,
        target_transform=None,
    ) -> None:
        import torchaudio

        self.file_paths = file_paths
        self.labels = labels
        self.target_sample_rate = target_sample_rate
        self.n_mels = n_mels
        self.max_samples = int(target_sample_rate * max_duration_sec)
        self.waveform_transform = waveform_transform
        self.spectrogram_transform = spectrogram_transform
        self.target_transform = target_transform

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_mels=n_mels,
            n_fft=1024,
            hop_length=512,
        )

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple:
        import torch
        import torchaudio

        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load audio
        waveform, sr = torchaudio.load(str(file_path))

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            waveform = resampler(waveform)

        # Pad or truncate to fixed length
        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, : self.max_samples]
        elif waveform.shape[1] < self.max_samples:
            padding = self.max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        if self.waveform_transform is not None:
            waveform = self.waveform_transform(waveform)

        # Apply mel spectrogram
        mel_spec = self.mel_transform(waveform)

        if self.spectrogram_transform is not None:
            mel_spec = self.spectrogram_transform(mel_spec)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if not torch.is_tensor(label):
            label = torch.tensor(label, dtype=torch.long)

        return mel_spec.squeeze(0), label


def _collect_audio_files(
    root: Path,
) -> Tuple[List[Path], List[int], List[str]]:
    """Collect audio files from class subdirectories.

    Returns:
        Tuple of (file_paths, labels, class_names).
    """
    class_dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    file_paths: List[Path] = []
    labels: List[int] = []
    class_names: List[str] = []

    for idx, class_dir in enumerate(class_dirs):
        class_names.append(class_dir.name)
        for audio_file in sorted(class_dir.iterdir()):
            if audio_file.suffix.lower() in AUDIO_EXTENSIONS and audio_file.is_file():
                file_paths.append(audio_file)
                labels.append(idx)

    return file_paths, labels, class_names


def _resolve_audio_split_dir(
    root: Path, split: str, available_splits: list[str]
) -> Path:
    """Resolve the actual directory for a requested split name."""
    if split in available_splits:
        return root / split

    aliases = {
        "val": ["validation", "dev"],
        "validation": ["val"],
        "test": ["dev"],
    }

    for alias in aliases.get(split, []):
        if alias in available_splits:
            return root / alias

    if "train" in available_splits:
        return root / "train"

    raise FileNotFoundError(
        f"No directory found for split '{split}' in {root}. "
        f"Available splits: {available_splits}"
    )


def create_tensorflow_dataset(
    dagnam_ds: "DagnamDataset",
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

    samples, _classes = _collect_audio_samples(
        dagnam_ds, split, val_ratio, test_ratio, seed
    )
    target_len = int(max_duration_sec * sample_rate)

    paths = np.array([str(s[0]) for s in samples])
    labels = np.array([s[1] for s in samples], dtype=np.int64)

    def _load_one(path_tensor, label_tensor):
        path_str = path_tensor.numpy().decode("utf-8") if hasattr(path_tensor, "numpy") else path_tensor
        waveform = _load_waveform_py(path_str, sample_rate, target_len)
        return waveform.astype(np.float32), np.int64(label_tensor.numpy() if hasattr(label_tensor, "numpy") else label_tensor)

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
    dagnam_ds: "DagnamDataset",
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
    import numpy as np
    import jax.numpy as jnp

    from dagnam.data.loaders.flax_loader import FlaxBatch

    if shuffle is None:
        shuffle = split == "train"

    samples, _classes = _collect_audio_samples(
        dagnam_ds, split, val_ratio, test_ratio, seed
    )
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


def _collect_audio_samples(
    dagnam_ds: "DagnamDataset",
    split: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[str]]:
    """Enumerate (audio_path, class_idx) pairs for the requested split."""
    data_root = ensure_extracted(dagnam_ds._data_dir)
    layout = discover_class_folders(data_root)

    if layout.has_explicit_splits:
        split_dir = _resolve_audio_split_dir(data_root, split, layout.splits)
        samples, classes = _enumerate_audio_samples(split_dir)
    else:
        samples, classes = _enumerate_audio_samples(data_root)
        train_idx, val_idx, test_idx = split_indices(
            len(samples), val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
        split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
        samples = [samples[i] for i in split_map[split]]
    return samples, classes


def _enumerate_audio_samples(root: Path) -> Tuple[List[Tuple[Path, int]], List[str]]:
    classes = sorted(
        e.name for e in root.iterdir()
        if e.is_dir() and not e.name.startswith(".")
    )
    class_to_idx = {c: i for i, c in enumerate(classes)}
    samples: List[Tuple[Path, int]] = []
    for cls in classes:
        for p in sorted((root / cls).iterdir()):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
                samples.append((p, class_to_idx[cls]))
    return samples, classes


def _resolve_audio_split_dir(root: Path, split: str, available: list[str]) -> Path:
    if split in available:
        return root / split
    aliases = {"val": ["validation", "dev"], "validation": ["val"], "test": ["dev"]}
    for a in aliases.get(split, []):
        if a in available:
            return root / a
    if "train" in available:
        return root / "train"
    raise FileNotFoundError(f"No directory for split '{split}' in {root}")


def _load_waveform_py(path: str, target_sr: int, target_len: int):
    """Load an audio file as a 1-D float32 numpy array at *target_sr*.

    Tries soundfile first (pure-Python), then falls back to torchaudio.
    Pads/truncates to *target_len* samples.
    """
    import numpy as np
    try:
        import soundfile as sf
        waveform, sr = sf.read(path, dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
    except (ImportError, Exception):
        try:
            import torchaudio
            t, sr = torchaudio.load(path)
            if t.ndim > 1:
                t = t.mean(dim=0)
            waveform = t.numpy().astype(np.float32)
        except ImportError:
            raise ImportError(
                "Either 'soundfile' or 'torchaudio' is required to load audio. "
                "Install with: pip install soundfile  OR  pip install dagnam[audio]"
            )

    if sr != target_sr:
        # Cheap linear-interpolation resample (adequate for model smoke tests).
        ratio = target_sr / float(sr)
        new_len = int(round(len(waveform) * ratio))
        if new_len > 0:
            xp = np.linspace(0, 1, len(waveform), endpoint=False)
            xq = np.linspace(0, 1, new_len, endpoint=False)
            waveform = np.interp(xq, xp, waveform).astype(np.float32)

    if len(waveform) < target_len:
        pad = target_len - len(waveform)
        waveform = np.pad(waveform, (0, pad))
    else:
        waveform = waveform[:target_len]

    return waveform
