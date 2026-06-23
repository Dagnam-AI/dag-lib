"""Audio class-subdirectory decoder."""

from __future__ import annotations

from pathlib import Path
import wave

import numpy as np
import numpy.typing as npt

from dagnam.data.loaders.system.column_store import Column, ColumnStore
from dagnam.data.loaders.system.decoders._helpers import (
    extensions,
    safe_extract_tar,
    spec_dict,
)
from dagnam.data.loaders.system.decoders.base import DecodeError


def _audio_root(artifact_dir: Path, configured_dir: str) -> Path:
    """Resolve the directory that holds the class subdirectories.

    Extracts a shipped tarball if present (mirrors ``image_mask_folder``), then
    honours an explicit layout ``dir`` when it exists, else falls back to the
    extracted root — Speech Commands, for example, places its class subdirectories
    at the archive root with no ``audio/`` wrapper.
    """
    base = artifact_dir
    tarballs = sorted(artifact_dir.glob("*.tar.gz"))
    if tarballs:
        unpacked = artifact_dir / "_unpacked_audio_folder"
        if not unpacked.exists():
            safe_extract_tar(tarballs[0], unpacked)
        roots = [item for item in unpacked.iterdir() if item.is_dir()]
        base = roots[0] if len(roots) == 1 else unpacked
    if configured_dir:
        candidate = base / configured_dir
        if candidate.exists():
            return candidate
    return base


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
        # _audio_root always resolves to an existing directory (the extracted root,
        # an honoured configured subdir, or the artifact dir itself).
        root = _audio_root(artifact_dir, str(audio_spec.get("dir", "")))
        classes = sorted(
            item for item in root.iterdir() if item.is_dir() and not item.name.startswith("_")
        )
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
