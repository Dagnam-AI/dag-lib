"""Coverage for dagnam.data.loaders.audio.io."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from tests.data.loaders._audio_helpers import (
    WaveformArray,
    audio_meta,
    build_audio_folder,
    build_split_audio,
    read_long,
    read_short,
    read_stereo,
)

from dagnam.data.dataset import DagnamDataset

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch


# ---------------------------------------------------------------- io: load_waveform_py


def test_load_waveform_via_soundfile(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    from dagnam.data.loaders.audio import io as audio_io

    p = tmp_path / "x.wav"
    p.write_bytes(b"x")
    # 2-D stereo waveform — exercises the mono mean reduction.
    fake_sf = SimpleNamespace(read=read_stereo)
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)
    wav = audio_io.load_waveform_py(str(p), target_sr=16000, target_len=200)
    # Pad branch (100 < 200) and resample branch (8000 → 16000) both fire.
    assert len(wav) == 200


def test_load_waveform_truncates(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    from dagnam.data.loaders.audio import io as audio_io

    p = tmp_path / "x.wav"
    p.write_bytes(b"x")
    fake_sf = SimpleNamespace(read=read_long)
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)
    wav = audio_io.load_waveform_py(str(p), target_sr=16000, target_len=100)
    assert len(wav) == 100


def test_load_waveform_falls_back_to_torchaudio(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    from dagnam.data.loaders.audio import io as audio_io

    p = tmp_path / "x.wav"
    p.write_bytes(b"x")

    class _T:
        def __init__(self, data: WaveformArray) -> None:
            self._data = data
            self.ndim = data.ndim
            self.shape = tuple(int(dim) for dim in data.shape)

        def mean(self, dim: int | None = None) -> _T:
            return _T(self._data.mean(axis=dim))

        def numpy(self) -> WaveformArray:
            return self._data

    def load_fake_audio(_path: str) -> tuple[_T, int]:
        return _T(np.ones((2, 100), dtype=np.float32)), 16000

    fake_torchaudio = SimpleNamespace(load=load_fake_audio)

    # Make `import soundfile` raise so the fallback fires.
    real_import_module = audio_io.import_module  # pyright: ignore[reportPrivateImportUsage]

    def fake_import(name: str, package: str | None = None):
        if name == "soundfile":
            raise ImportError("missing")
        if name == "torchaudio":
            return fake_torchaudio
        return real_import_module(name, package)

    monkeypatch.setattr(audio_io, "import_module", fake_import)
    wav = audio_io.load_waveform_py(str(p), target_sr=16000, target_len=100)
    assert len(wav) == 100


def test_load_waveform_raises_when_no_backend(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    from dagnam.data.loaders.audio import io as audio_io

    p = tmp_path / "x.wav"
    p.write_bytes(b"x")

    real_import_module = audio_io.import_module  # pyright: ignore[reportPrivateImportUsage]

    def fake_import(name: str, package: str | None = None):
        if name in ("soundfile", "torchaudio"):
            raise ImportError(f"no {name}")
        return real_import_module(name, package)

    monkeypatch.setattr(audio_io, "import_module", fake_import)
    with pytest.raises(ImportError, match=r"soundfile.*torchaudio"):
        audio_io.load_waveform_py(str(p), target_sr=16000, target_len=100)


def test_load_waveform_no_resample_path(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    """target_sr matches source_sr — skips the resample branch."""
    from dagnam.data.loaders.audio import io as audio_io

    p = tmp_path / "x.wav"
    p.write_bytes(b"x")
    fake_sf = SimpleNamespace(read=read_short)
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)
    wav = audio_io.load_waveform_py(str(p), target_sr=16000, target_len=100)
    assert len(wav) == 100


def _read_mono_short(
    _path: str,
    *,
    dtype: str | None = None,
    always_2d: bool | None = None,
) -> tuple[WaveformArray, int]:
    """Mono waveform shorter than the requested target_len."""
    return np.ones(50, dtype=np.float32), 16000


def _read_single_sample(
    _path: str,
    *,
    dtype: str | None = None,
    always_2d: bool | None = None,
) -> tuple[WaveformArray, int]:
    """A 1-sample waveform so an extreme downsample yields new_len == 0."""
    return np.ones(1, dtype=np.float32), 16000


def test_load_waveform_pads_when_shorter_than_target(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    """A mono waveform shorter than target_len is zero-padded (no resample)."""
    from dagnam.data.loaders.audio import io as audio_io

    p = tmp_path / "x.wav"
    p.write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace(read=_read_mono_short))
    wav = audio_io.load_waveform_py(str(p), target_sr=16000, target_len=100)
    assert len(wav) == 100
    # The 50 padded tail samples must be zeros.
    assert np.all(wav[50:] == 0.0)
    assert np.all(wav[:50] == 1.0)


def test_load_waveform_skips_resample_when_new_len_zero(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    """An extreme downsample where round(len*ratio) == 0 skips interpolation."""
    from dagnam.data.loaders.audio import io as audio_io

    p = tmp_path / "x.wav"
    p.write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace(read=_read_single_sample))
    # target_sr (160) << source_sr (16000): ratio 0.01, new_len round(1*0.01)=0.
    wav = audio_io.load_waveform_py(str(p), target_sr=160, target_len=100)
    assert len(wav) == 100


def test_load_waveform_torchaudio_mono_no_channel_reduction(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    """A 1-D torchaudio tensor skips the multi-channel mean reduction."""
    from dagnam.data.loaders.audio import io as audio_io

    p = tmp_path / "x.wav"
    p.write_bytes(b"x")

    class _T:
        def __init__(self, data: WaveformArray) -> None:
            self._data = data
            self.shape = tuple(int(dim) for dim in data.shape)

        def numpy(self) -> WaveformArray:
            return self._data

    def load_fake_audio(_path: str) -> tuple[_T, int]:
        # 1-D tensor: len(shape) == 1 so the channel-mean branch is skipped.
        return _T(np.ones(100, dtype=np.float32)), 16000

    fake_torchaudio = SimpleNamespace(load=load_fake_audio)
    real_import_module = audio_io.import_module  # pyright: ignore[reportPrivateImportUsage]

    def fake_import(name: str, package: str | None = None):
        if name == "soundfile":
            raise ImportError("missing")
        if name == "torchaudio":
            return fake_torchaudio
        return real_import_module(name, package)

    monkeypatch.setattr(audio_io, "import_module", fake_import)
    wav = audio_io.load_waveform_py(str(p), target_sr=16000, target_len=100)
    assert len(wav) == 100
    assert np.all(wav == 1.0)


def test_enumerate_skips_non_audio_files(tmp_path: Path) -> None:
    """Non-audio files inside a class folder are skipped during enumeration."""
    from dagnam.data.loaders.audio import io as audio_io

    build_audio_folder(tmp_path, classes=("dog",), per_class=2)
    # A stray non-audio file (and a nested dir) must not be enumerated.
    (tmp_path / "dog" / "notes.txt").write_text("ignore me")
    (tmp_path / "dog" / "nested").mkdir()
    ds = DagnamDataset(audio_meta(num_samples=2), tmp_path)
    samples, classes = audio_io.collect_audio_samples(
        ds, "train", val_ratio=0.0, test_ratio=0.0, seed=0
    )
    assert classes == ["dog"]
    # Only the two .wav files are collected; the .txt and dir are excluded.
    assert all(p.suffix == ".wav" for p, _ in samples)
    assert len(samples) == 2


# ---------------------------------------------------------------- io: collect_audio_samples


def testcollect_audio_samples_unsplit(tmp_path: Path) -> None:
    from dagnam.data.loaders.audio import io as audio_io

    build_audio_folder(tmp_path, per_class=4)
    ds = DagnamDataset(audio_meta(num_samples=8), tmp_path)
    samples, classes = audio_io.collect_audio_samples(
        ds, "train", val_ratio=0.2, test_ratio=0.2, seed=0
    )
    assert classes == ["cat", "dog"]
    assert len(samples) >= 1


def testcollect_audio_samples_split(tmp_path: Path) -> None:
    from dagnam.data.loaders.audio import io as audio_io

    build_split_audio(tmp_path)
    ds = DagnamDataset(audio_meta(num_samples=12), tmp_path)
    samples, _classes = audio_io.collect_audio_samples(
        ds, "val", val_ratio=0.0, test_ratio=0.0, seed=0
    )
    assert len(samples) >= 1
