"""Audio folder dataset primitives."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from dagnam.data.loaders.media import AUDIO_EXTENSIONS


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
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))

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


def _resolve_audio_split_dir(root: Path, split: str, available_splits: list[str]) -> Path:
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
        f"No directory found for split '{split}' in {root}. Available splits: {available_splits}"
    )
