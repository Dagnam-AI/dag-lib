"""Audio class-subdirectory decoder."""

from __future__ import annotations

from pathlib import Path
import wave

import numpy as np
import numpy.typing as npt

from dagnam.data.loaders.system.column_store import Column, ColumnStore
from dagnam.data.loaders.system.decoders._helpers import extensions, spec_dict
from dagnam.data.loaders.system.decoders.base import DecodeError


def read_wav(path: Path) -> npt.NDArray[np.float32]:
    """Read mono/stereo PCM wav into float32 samples in roughly [-1, 1]."""
    with wave.open(str(path), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
        sample_width = wav.getsampwidth()
        channels = wav.getnchannels()
    if sample_width != 2:
        raise DecodeError(f"audio_folder: unsupported wav sample width {sample_width}")
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples


class AudioFolderDecoder:
    """Decode wav class subdirectories into audio and label columns."""

    def decode(self, artifact_dir: Path, layout: dict[str, object], split: str) -> ColumnStore:
        del split
        audio_spec = spec_dict(layout, "audio")
        audio_exts = extensions(audio_spec)
        root = artifact_dir / str(audio_spec.get("dir", "audio/"))
        if not root.exists():
            raise DecodeError(f"audio_folder: audio root does not exist: {root}")
        classes = sorted(item for item in root.iterdir() if item.is_dir())
        if not classes:
            raise DecodeError(f"audio_folder: no label subdirectories under {root}")

        paths: list[Path] = []
        labels: list[int] = []
        for label, class_dir in enumerate(classes):
            for wav_path in sorted(
                item for item in class_dir.iterdir() if item.suffix in audio_exts
            ):
                paths.append(wav_path)
                labels.append(label)
        if not paths:
            raise DecodeError(f"audio_folder: no audio files under {root}")
        return ColumnStore(
            {
                "audio": Column.lazy(paths, read_wav),
                "label": Column.eager(np.asarray(labels, dtype=np.int64)),
            }
        )
