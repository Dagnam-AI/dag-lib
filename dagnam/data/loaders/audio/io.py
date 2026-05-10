"""Audio sample discovery and waveform loading."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

from dagnam.data.loaders.media import (
    AUDIO_EXTENSIONS,
    discover_class_folders,
    ensure_extracted,
    split_indices,
)

if TYPE_CHECKING:
    from dagnam.data.dataset import DagnamDataset


def _collect_audio_samples(
    dagnam_ds: DagnamDataset,
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
    classes = sorted(e.name for e in root.iterdir() if e.is_dir() and not e.name.startswith("."))
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
        new_len = round(len(waveform) * ratio)
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
