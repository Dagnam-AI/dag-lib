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
    ) -> None:
        import torchaudio

        self.file_paths = file_paths
        self.labels = labels
        self.target_sample_rate = target_sample_rate
        self.n_mels = n_mels
        self.max_samples = int(target_sample_rate * max_duration_sec)

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

        # Apply mel spectrogram
        mel_spec = self.mel_transform(waveform)

        return mel_spec.squeeze(0), torch.tensor(label, dtype=torch.long)


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
