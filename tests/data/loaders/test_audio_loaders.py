"""Coverage for dagnam.data.loaders.audio.{dataset, io, transforms}.

torchaudio is broken on this Windows build and soundfile is not installed,
so we inject fakes into sys.modules and verify the loader code paths exhaustively.
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dagnam.data.dataset import DagnamDataset


def _build_audio_folder(root: Path, classes=("dog", "cat"), per_class=4) -> None:
    """Layout: root/{class}/*.wav (files only need to exist, fake loader ignores content)."""
    for cls in classes:
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            (d / f"{i}.wav").write_bytes(b"FAKE_WAV")


def _build_split_audio(root: Path) -> None:
    for split in ("train", "val", "test"):
        for cls in ("dog", "cat"):
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                (d / f"{i}.wav").write_bytes(b"X")


def _audio_meta(num_samples=8) -> dict:
    return {
        "id": "a1",
        "name": "audio",
        "format": "audio_folder",
        "dataset_type": "audio",
        "num_samples": num_samples,
        "num_classes": 2,
        "class_names": ["dog", "cat"],
        "audio": {"sample_rate": 16000, "n_mels": 64},
    }


# ---------------------------------------------------------------- io: _load_waveform_py


def test_load_waveform_via_soundfile(monkeypatch, tmp_path):
    from dagnam.data.loaders.audio import io as audio_io

    p = tmp_path / "x.wav"
    p.write_bytes(b"x")
    # 2-D stereo waveform — exercises the mono mean reduction.
    fake_sf = SimpleNamespace(
        read=lambda _path, dtype=None, always_2d=None: (np.ones((100, 2), dtype=np.float32), 8000)
    )
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)
    wav = audio_io._load_waveform_py(str(p), target_sr=16000, target_len=200)
    # Pad branch (100 < 200) and resample branch (8000 → 16000) both fire.
    assert len(wav) == 200


def test_load_waveform_truncates(monkeypatch, tmp_path):
    from dagnam.data.loaders.audio import io as audio_io

    p = tmp_path / "x.wav"
    p.write_bytes(b"x")
    fake_sf = SimpleNamespace(
        read=lambda _p, dtype=None, always_2d=None: (np.ones(1000, dtype=np.float32), 16000)
    )
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)
    wav = audio_io._load_waveform_py(str(p), target_sr=16000, target_len=100)
    assert len(wav) == 100


def test_load_waveform_falls_back_to_torchaudio(monkeypatch, tmp_path):
    from dagnam.data.loaders.audio import io as audio_io

    p = tmp_path / "x.wav"
    p.write_bytes(b"x")

    class _T:
        def __init__(self, data):
            self._data = data
            self.ndim = data.ndim

        def mean(self, dim=None):
            return _T(self._data.mean(axis=dim))

        def numpy(self):
            return self._data

    fake_torchaudio = SimpleNamespace(
        load=lambda _p: (_T(np.ones((2, 100), dtype=np.float32)), 16000)
    )

    # Make `import soundfile` raise so the fallback fires.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "soundfile":
            raise ImportError("missing")
        if name == "torchaudio":
            return fake_torchaudio
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    wav = audio_io._load_waveform_py(str(p), target_sr=16000, target_len=100)
    assert len(wav) == 100


def test_load_waveform_raises_when_no_backend(monkeypatch, tmp_path):
    from dagnam.data.loaders.audio import io as audio_io

    p = tmp_path / "x.wav"
    p.write_bytes(b"x")

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name in ("soundfile", "torchaudio"):
            raise ImportError(f"no {name}")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"soundfile.*torchaudio"):
        audio_io._load_waveform_py(str(p), target_sr=16000, target_len=100)


def test_load_waveform_no_resample_path(monkeypatch, tmp_path):
    """target_sr matches source_sr — skips the resample branch."""
    from dagnam.data.loaders.audio import io as audio_io

    p = tmp_path / "x.wav"
    p.write_bytes(b"x")
    fake_sf = SimpleNamespace(
        read=lambda _p, dtype=None, always_2d=None: (np.ones(100, dtype=np.float32), 16000)
    )
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)
    wav = audio_io._load_waveform_py(str(p), target_sr=16000, target_len=100)
    assert len(wav) == 100


# ---------------------------------------------------------------- io: _collect_audio_samples


def test_collect_audio_samples_unsplit(tmp_path):
    from dagnam.data.loaders.audio import io as audio_io

    _build_audio_folder(tmp_path, per_class=4)
    ds = DagnamDataset(_audio_meta(num_samples=8), tmp_path)
    samples, classes = audio_io._collect_audio_samples(
        ds, "train", val_ratio=0.2, test_ratio=0.2, seed=0
    )
    assert classes == ["cat", "dog"]
    assert len(samples) >= 1


def test_collect_audio_samples_split(tmp_path):
    from dagnam.data.loaders.audio import io as audio_io

    _build_split_audio(tmp_path)
    ds = DagnamDataset(_audio_meta(num_samples=12), tmp_path)
    samples, classes = audio_io._collect_audio_samples(
        ds, "val", val_ratio=0.0, test_ratio=0.0, seed=0
    )
    assert len(samples) >= 1


def test_resolve_audio_split_dir_aliases(tmp_path):
    from dagnam.data.loaders.audio import io as audio_io

    assert (
        audio_io._resolve_audio_split_dir(tmp_path, "val", ["train", "validation"])
        == tmp_path / "validation"
    )
    assert (
        audio_io._resolve_audio_split_dir(tmp_path, "validation", ["train", "val"])
        == tmp_path / "val"
    )
    assert audio_io._resolve_audio_split_dir(tmp_path, "test", ["dev"]) == tmp_path / "dev"
    assert audio_io._resolve_audio_split_dir(tmp_path, "val", ["train"]) == tmp_path / "train"


def test_resolve_audio_split_dir_direct(tmp_path):
    from dagnam.data.loaders.audio import io as audio_io

    assert (
        audio_io._resolve_audio_split_dir(tmp_path, "train", ["train", "val"]) == tmp_path / "train"
    )


def test_resolve_audio_split_dir_raises(tmp_path):
    from dagnam.data.loaders.audio import io as audio_io

    with pytest.raises(FileNotFoundError, match="No directory"):
        audio_io._resolve_audio_split_dir(tmp_path, "val", ["other"])


# ---------------------------------------------------------------- dataset


def _install_fake_torchaudio(monkeypatch):
    """Inject a minimal `torchaudio` module that the dataset code can exercise."""
    import torch

    class FakeResample:
        def __init__(self, src, target):
            self.src = src
            self.target = target

        def __call__(self, waveform):
            # Scale length by ratio for shape consistency.
            ratio = self.target / self.src
            new_len = max(1, int(waveform.shape[-1] * ratio))
            return torch.zeros((waveform.shape[0], new_len))

    class FakeMelSpectrogram:
        def __init__(self, sample_rate=None, n_mels=None, n_fft=None, hop_length=None):
            self.n_mels = n_mels or 64

        def __call__(self, waveform):
            # Return (channels=1, n_mels, frames) — squeezing(0) gives (n_mels, frames).
            frames = max(1, waveform.shape[-1] // 256)
            return torch.zeros((waveform.shape[0], self.n_mels, frames))

    fake_torchaudio = SimpleNamespace(
        load=lambda _p: (torch.zeros((1, 16000)), 16000),
        transforms=SimpleNamespace(Resample=FakeResample, MelSpectrogram=FakeMelSpectrogram),
    )
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)
    return fake_torchaudio


def test_audio_folder_dataset_basic(monkeypatch, tmp_path):
    _install_fake_torchaudio(monkeypatch)
    from dagnam.data.loaders.audio.dataset import AudioFolderDataset

    _build_audio_folder(tmp_path, per_class=2)
    files = list((tmp_path / "dog").glob("*.wav")) + list((tmp_path / "cat").glob("*.wav"))
    ds = AudioFolderDataset(
        file_paths=files,
        labels=[0, 0, 1, 1],
        target_sample_rate=16000,
        n_mels=8,
        max_duration_sec=1.0,
    )
    assert len(ds) == 4
    item, label = ds[0]
    assert item.ndim == 2  # (n_mels, frames)
    assert int(label) in (0, 1)


def test_audio_folder_dataset_with_transforms(monkeypatch, tmp_path):
    _install_fake_torchaudio(monkeypatch)
    import torch

    from dagnam.data.loaders.audio.dataset import AudioFolderDataset

    _build_audio_folder(tmp_path, per_class=1)
    files = [tmp_path / "dog" / "0.wav"]
    ds = AudioFolderDataset(
        file_paths=files,
        labels=[0],
        target_sample_rate=16000,
        n_mels=8,
        max_duration_sec=1.0,
        waveform_transform=lambda w: w,
        spectrogram_transform=lambda s: s,
        target_transform=lambda lbl: torch.tensor(lbl, dtype=torch.long),
    )
    item, label = ds[0]
    assert label.dtype == torch.long


def test_audio_folder_dataset_resamples_when_sr_differs(monkeypatch, tmp_path):
    """torchaudio.load returns 8000Hz but target is 16000Hz — triggers Resample."""
    import torch

    class FakeResample:
        def __init__(self, src, target):
            self.target = target

        def __call__(self, w):
            return torch.zeros((w.shape[0], self.target))

    class FakeMel:
        def __init__(self, **_kw):
            pass

        def __call__(self, w):
            return torch.zeros((w.shape[0], 8, max(1, w.shape[-1] // 256)))

    fake = SimpleNamespace(
        load=lambda _p: (torch.zeros((2, 8000)), 8000),  # stereo, 8kHz → triggers mono+resample
        transforms=SimpleNamespace(Resample=FakeResample, MelSpectrogram=FakeMel),
    )
    monkeypatch.setitem(sys.modules, "torchaudio", fake)

    from dagnam.data.loaders.audio.dataset import AudioFolderDataset

    _build_audio_folder(tmp_path, per_class=1)
    ds = AudioFolderDataset(
        file_paths=[tmp_path / "dog" / "0.wav"],
        labels=[0],
        target_sample_rate=16000,
        n_mels=8,
        max_duration_sec=1.0,
    )
    item, _ = ds[0]
    assert item.shape[-1] >= 1


def test_audio_folder_dataset_truncates_when_too_long(monkeypatch, tmp_path):
    import torch

    fake = SimpleNamespace(
        load=lambda _p: (torch.zeros((1, 100_000)), 16000),  # very long → truncates
        transforms=SimpleNamespace(
            Resample=lambda *_a, **_kw: lambda w: w,
            MelSpectrogram=lambda **_kw: lambda w: torch.zeros((w.shape[0], 8, w.shape[-1] // 256)),
        ),
    )
    monkeypatch.setitem(sys.modules, "torchaudio", fake)

    from dagnam.data.loaders.audio.dataset import AudioFolderDataset

    _build_audio_folder(tmp_path, per_class=1)
    ds = AudioFolderDataset(
        file_paths=[tmp_path / "dog" / "0.wav"],
        labels=[0],
        target_sample_rate=16000,
        n_mels=8,
        max_duration_sec=1.0,  # max=16000 samples
    )
    ds[0]


def test_collect_audio_files(tmp_path):
    from dagnam.data.loaders.audio.dataset import _collect_audio_files

    _build_audio_folder(tmp_path, per_class=2)
    # Add a hidden class + non-audio file
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "x.wav").write_bytes(b"x")
    (tmp_path / "dog" / "README.txt").write_text("not audio")
    files, labels, classes = _collect_audio_files(tmp_path)
    assert ".hidden" not in classes
    assert len(files) == 4  # 2 per class


def test_resolve_audio_split_dir_in_dataset(tmp_path):
    from dagnam.data.loaders.audio.dataset import _resolve_audio_split_dir

    assert _resolve_audio_split_dir(tmp_path, "train", ["train"]) == tmp_path / "train"
    assert _resolve_audio_split_dir(tmp_path, "val", ["validation"]) == tmp_path / "validation"
    assert _resolve_audio_split_dir(tmp_path, "test", ["dev"]) == tmp_path / "dev"
    assert _resolve_audio_split_dir(tmp_path, "val", ["train"]) == tmp_path / "train"

    with pytest.raises(FileNotFoundError):
        _resolve_audio_split_dir(tmp_path, "val", ["other"])


# ---------------------------------------------------------------- transforms (framework adapters)


def test_create_pytorch_loader_unsplit(monkeypatch, tmp_path):
    _install_fake_torchaudio(monkeypatch)
    from dagnam.data.loaders.audio.transforms import create_pytorch_loader

    _build_audio_folder(tmp_path, per_class=4)
    ds = DagnamDataset(_audio_meta(num_samples=8), tmp_path)
    loader = create_pytorch_loader(
        ds, split="train", batch_size=2, num_workers=0, val_ratio=0.2, test_ratio=0.2, seed=0
    )
    batch = next(iter(loader))
    assert batch[0].shape[0] >= 1


def test_create_pytorch_loader_split(monkeypatch, tmp_path):
    _install_fake_torchaudio(monkeypatch)
    from dagnam.data.loaders.audio.transforms import create_pytorch_loader

    _build_split_audio(tmp_path)
    ds = DagnamDataset(_audio_meta(num_samples=12), tmp_path)
    loader = create_pytorch_loader(ds, split="val", batch_size=2, num_workers=0)
    batch = next(iter(loader))
    assert batch[0].shape[0] >= 1


def test_create_pytorch_loader_uses_meta_audio_cfg(monkeypatch, tmp_path):
    _install_fake_torchaudio(monkeypatch)
    from dagnam.data.loaders.audio.transforms import create_pytorch_loader

    _build_audio_folder(tmp_path, per_class=2)
    ds = DagnamDataset(_audio_meta(num_samples=4), tmp_path)
    loader = create_pytorch_loader(
        ds, split="train", batch_size=1, num_workers=0, val_ratio=0.25, test_ratio=0.25
    )
    next(iter(loader))


def test_create_tensorflow_dataset(monkeypatch, tmp_path):
    pytest.importorskip("tensorflow")
    from dagnam.data.loaders.audio import io as audio_io
    from dagnam.data.loaders.audio.transforms import create_tensorflow_dataset

    # Stub out _load_waveform_py to skip needing soundfile/torchaudio.
    monkeypatch.setattr(
        audio_io,
        "_load_waveform_py",
        lambda path, target_sr, target_len: np.zeros(target_len, dtype=np.float32),
    )
    # Also patch the symbol the transforms module imported earlier.
    from dagnam.data.loaders.audio import transforms as audio_transforms

    monkeypatch.setattr(
        audio_transforms,
        "_load_waveform_py",
        lambda path, target_sr, target_len: np.zeros(target_len, dtype=np.float32),
    )

    _build_audio_folder(tmp_path, per_class=2)
    ds = DagnamDataset(_audio_meta(num_samples=4), tmp_path)
    tf_ds = create_tensorflow_dataset(
        ds, split="train", batch_size=2, shuffle=False, val_ratio=0.25, test_ratio=0.25, seed=0
    )
    next(iter(tf_ds))


def test_create_tensorflow_dataset_with_map_fns(monkeypatch, tmp_path):
    import tensorflow as tf

    from dagnam.data.loaders.audio import transforms as audio_transforms

    monkeypatch.setattr(
        audio_transforms,
        "_load_waveform_py",
        lambda path, target_sr, target_len: np.zeros(target_len, dtype=np.float32),
    )

    _build_audio_folder(tmp_path, per_class=2)
    ds = DagnamDataset(_audio_meta(num_samples=4), tmp_path)
    tf_ds = audio_transforms.create_tensorflow_dataset(
        ds,
        split="train",
        batch_size=2,
        shuffle=True,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=0,
        map_fn=lambda w, lbl: (tf.cast(w, tf.float32), lbl),
        batch_map_fn=lambda w, lbl: (w, lbl),
    )
    next(iter(tf_ds))


def test_create_flax_dataset(monkeypatch, tmp_path):
    pytest.importorskip("jax")
    from dagnam.data.loaders.audio import transforms as audio_transforms

    monkeypatch.setattr(
        audio_transforms,
        "_load_waveform_py",
        lambda path, target_sr, target_len: np.zeros(target_len, dtype=np.float32),
    )

    _build_audio_folder(tmp_path, per_class=2)
    ds = DagnamDataset(_audio_meta(num_samples=4), tmp_path)
    batches = audio_transforms.create_flax_dataset(
        ds, split="train", batch_size=2, shuffle=False, val_ratio=0.25, test_ratio=0.25, seed=0
    )
    assert batches


def test_create_flax_dataset_with_transforms(monkeypatch, tmp_path):
    pytest.importorskip("jax")
    from dagnam.data.loaders.audio import transforms as audio_transforms

    monkeypatch.setattr(
        audio_transforms,
        "_load_waveform_py",
        lambda path, target_sr, target_len: np.zeros(target_len, dtype=np.float32),
    )

    _build_audio_folder(tmp_path, per_class=2)
    ds = DagnamDataset(_audio_meta(num_samples=4), tmp_path)
    batches = audio_transforms.create_flax_dataset(
        ds,
        split="train",
        batch_size=2,
        shuffle=True,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=0,
        transform_fn=lambda w: w,
        batch_transform_fn=lambda f, lbl: (f, lbl),
    )
    assert batches
