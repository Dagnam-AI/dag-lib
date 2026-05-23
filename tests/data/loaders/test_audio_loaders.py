"""Coverage for dagnam.data.loaders.audio.{dataset, io, transforms}.

torchaudio is broken on this Windows build and soundfile is not installed,
so we inject fakes into sys.modules and verify the loader code paths exhaustively.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Protocol, SupportsInt, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import pytest

from dagnam._types import JsonObject
from dagnam.data.dataset import DagnamDataset
from dagnam.data.loaders.audio.dataset import TorchTensor
from tests.typing_helpers import PytestMonkeyPatch

WaveformArray: TypeAlias = npt.NDArray[np.float32]


class TorchTestModule(Protocol):
    long: object

    def zeros(self, size: Sequence[int]) -> TorchTensor: ...

    def tensor(self, data: object, *, dtype: object) -> TorchTensor: ...


class LabelTensor(Protocol):
    dtype: object


def _torch() -> TorchTestModule:
    return cast(TorchTestModule, import_module("torch"))


def _identity_transform(value: object) -> object:
    return value


def _load_waveform_stub(_path: object, _target_sr: object, target_len: int) -> WaveformArray:
    return np.zeros(target_len, dtype=np.float32)


def _read_stereo(
    _path: str,
    *,
    dtype: str | None = None,
    always_2d: bool | None = None,
) -> tuple[WaveformArray, int]:
    return np.ones((100, 2), dtype=np.float32), 8000


def _read_long(
    _path: str,
    *,
    dtype: str | None = None,
    always_2d: bool | None = None,
) -> tuple[WaveformArray, int]:
    return np.ones(1000, dtype=np.float32), 16000


def _read_short(
    _path: str,
    *,
    dtype: str | None = None,
    always_2d: bool | None = None,
) -> tuple[WaveformArray, int]:
    return np.ones(100, dtype=np.float32), 16000


def _build_audio_folder(
    root: Path,
    classes: tuple[str, ...] = ("dog", "cat"),
    per_class: int = 4,
) -> None:
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


def _audio_meta(num_samples: int = 8) -> JsonObject:
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


# ---------------------------------------------------------------- io: load_waveform_py


def test_load_waveform_via_soundfile(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    from dagnam.data.loaders.audio import io as audio_io

    p = tmp_path / "x.wav"
    p.write_bytes(b"x")
    # 2-D stereo waveform — exercises the mono mean reduction.
    fake_sf = SimpleNamespace(read=_read_stereo)
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)
    wav = audio_io.load_waveform_py(str(p), target_sr=16000, target_len=200)
    # Pad branch (100 < 200) and resample branch (8000 → 16000) both fire.
    assert len(wav) == 200


def test_load_waveform_truncates(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    from dagnam.data.loaders.audio import io as audio_io

    p = tmp_path / "x.wav"
    p.write_bytes(b"x")
    fake_sf = SimpleNamespace(read=_read_long)
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)
    wav = audio_io.load_waveform_py(str(p), target_sr=16000, target_len=100)
    assert len(wav) == 100


def test_load_waveform_falls_back_to_torchaudio(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
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
    real_import_module = audio_io.import_module

    def fake_import(name: str, package: str | None = None):
        if name == "soundfile":
            raise ImportError("missing")
        if name == "torchaudio":
            return fake_torchaudio
        return real_import_module(name, package)

    monkeypatch.setattr(audio_io, "import_module", fake_import)
    wav = audio_io.load_waveform_py(str(p), target_sr=16000, target_len=100)
    assert len(wav) == 100


def test_load_waveform_raises_when_no_backend(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    from dagnam.data.loaders.audio import io as audio_io

    p = tmp_path / "x.wav"
    p.write_bytes(b"x")

    real_import_module = audio_io.import_module

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
    fake_sf = SimpleNamespace(read=_read_short)
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)
    wav = audio_io.load_waveform_py(str(p), target_sr=16000, target_len=100)
    assert len(wav) == 100


# ---------------------------------------------------------------- io: collect_audio_samples


def testcollect_audio_samples_unsplit(tmp_path: Path) -> None:
    from dagnam.data.loaders.audio import io as audio_io

    _build_audio_folder(tmp_path, per_class=4)
    ds = DagnamDataset(_audio_meta(num_samples=8), tmp_path)
    samples, classes = audio_io.collect_audio_samples(
        ds, "train", val_ratio=0.2, test_ratio=0.2, seed=0
    )
    assert classes == ["cat", "dog"]
    assert len(samples) >= 1


def testcollect_audio_samples_split(tmp_path: Path) -> None:
    from dagnam.data.loaders.audio import io as audio_io

    _build_split_audio(tmp_path)
    ds = DagnamDataset(_audio_meta(num_samples=12), tmp_path)
    samples, _classes = audio_io.collect_audio_samples(
        ds, "val", val_ratio=0.0, test_ratio=0.0, seed=0
    )
    assert len(samples) >= 1


def testresolve_audio_split_dir_aliases(tmp_path: Path) -> None:
    from dagnam.data.loaders.audio import io as audio_io

    assert (
        audio_io.resolve_audio_split_dir(tmp_path, "val", ["train", "validation"])
        == tmp_path / "validation"
    )
    assert (
        audio_io.resolve_audio_split_dir(tmp_path, "validation", ["train", "val"])
        == tmp_path / "val"
    )
    assert audio_io.resolve_audio_split_dir(tmp_path, "test", ["dev"]) == tmp_path / "dev"
    assert audio_io.resolve_audio_split_dir(tmp_path, "val", ["train"]) == tmp_path / "train"


def testresolve_audio_split_dir_direct(tmp_path: Path) -> None:
    from dagnam.data.loaders.audio import io as audio_io

    assert (
        audio_io.resolve_audio_split_dir(tmp_path, "train", ["train", "val"]) == tmp_path / "train"
    )


def testresolve_audio_split_dir_raises(tmp_path: Path) -> None:
    from dagnam.data.loaders.audio import io as audio_io

    with pytest.raises(FileNotFoundError, match="No directory"):
        audio_io.resolve_audio_split_dir(tmp_path, "val", ["other"])


# ---------------------------------------------------------------- dataset


def _install_fake_torchaudio(monkeypatch: PytestMonkeyPatch) -> SimpleNamespace:
    """Inject a minimal `torchaudio` module that the dataset code can exercise."""
    torch = _torch()

    class FakeResample:
        def __init__(self, src: int, target: int) -> None:
            self.src = src
            self.target = target

        def __call__(self, waveform: TorchTensor) -> TorchTensor:
            # Scale length by ratio for shape consistency.
            ratio = self.target / self.src
            new_len = max(1, int(waveform.shape[-1] * ratio))
            return torch.zeros((waveform.shape[0], new_len))

    class FakeMelSpectrogram:
        def __init__(
            self,
            sample_rate: int | None = None,
            n_mels: int | None = None,
            n_fft: int | None = None,
            hop_length: int | None = None,
        ) -> None:
            self.n_mels = n_mels or 64

        def __call__(self, waveform: TorchTensor) -> TorchTensor:
            # Return (channels=1, n_mels, frames) — squeezing(0) gives (n_mels, frames).
            frames = max(1, waveform.shape[-1] // 256)
            return torch.zeros((waveform.shape[0], self.n_mels, frames))

    def load_fake_audio(_path: str) -> tuple[TorchTensor, int]:
        return torch.zeros((1, 16000)), 16000

    fake_torchaudio = SimpleNamespace(
        load=load_fake_audio,
        transforms=SimpleNamespace(Resample=FakeResample, MelSpectrogram=FakeMelSpectrogram),
    )
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)
    return fake_torchaudio


def test_audio_folder_dataset_basic(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
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
    assert len(item.shape) == 2  # (n_mels, frames)
    assert int(cast(SupportsInt, label)) in (0, 1)


def test_audio_folder_dataset_with_transforms(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    _install_fake_torchaudio(monkeypatch)
    torch = _torch()

    from dagnam.data.loaders.audio.dataset import AudioFolderDataset

    _build_audio_folder(tmp_path, per_class=1)
    files = [tmp_path / "dog" / "0.wav"]
    ds = AudioFolderDataset(
        file_paths=files,
        labels=[0],
        target_sample_rate=16000,
        n_mels=8,
        max_duration_sec=1.0,
        waveform_transform=_identity_transform,
        spectrogram_transform=_identity_transform,
        target_transform=lambda lbl: torch.tensor(lbl, dtype=torch.long),
    )
    _item, label = ds[0]
    label_tensor = cast(LabelTensor, label)
    assert label_tensor.dtype == torch.long


def test_audio_folder_dataset_resamples_when_sr_differs(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    """torchaudio.load returns 8000Hz but target is 16000Hz — triggers Resample."""
    torch = _torch()

    class FakeResample:
        def __init__(self, src: int, target: int) -> None:
            self.target = target

        def __call__(self, w: TorchTensor) -> TorchTensor:
            return torch.zeros((w.shape[0], self.target))

    class FakeMel:
        def __init__(self, **_kw: object) -> None:
            pass

        def __call__(self, w: TorchTensor) -> TorchTensor:
            return torch.zeros((w.shape[0], 8, max(1, w.shape[-1] // 256)))

    def load_fake_audio(_path: str) -> tuple[TorchTensor, int]:
        return torch.zeros((2, 8000)), 8000

    fake = SimpleNamespace(
        load=load_fake_audio,
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


def test_audio_folder_dataset_truncates_when_too_long(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    torch = _torch()

    class IdentityResample:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __call__(self, waveform: TorchTensor) -> TorchTensor:
            return waveform

    class FakeMel:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __call__(self, waveform: TorchTensor) -> TorchTensor:
            return torch.zeros((waveform.shape[0], 8, waveform.shape[-1] // 256))

    def load_fake_audio(_path: str) -> tuple[TorchTensor, int]:
        return torch.zeros((1, 100_000)), 16000

    fake = SimpleNamespace(
        load=load_fake_audio,
        transforms=SimpleNamespace(Resample=IdentityResample, MelSpectrogram=FakeMel),
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


def testcollect_audio_files(tmp_path: Path) -> None:
    from dagnam.data.loaders.audio.dataset import collect_audio_files

    _build_audio_folder(tmp_path, per_class=2)
    # Add a hidden class + non-audio file
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "x.wav").write_bytes(b"x")
    (tmp_path / "dog" / "README.txt").write_text("not audio")
    files, _labels, classes = collect_audio_files(tmp_path)
    assert ".hidden" not in classes
    assert len(files) == 4  # 2 per class


def testresolve_audio_split_dir_in_dataset(tmp_path: Path) -> None:
    from dagnam.data.loaders.audio.dataset import resolve_audio_split_dir

    assert resolve_audio_split_dir(tmp_path, "train", ["train"]) == tmp_path / "train"
    assert resolve_audio_split_dir(tmp_path, "val", ["validation"]) == tmp_path / "validation"
    assert resolve_audio_split_dir(tmp_path, "test", ["dev"]) == tmp_path / "dev"
    assert resolve_audio_split_dir(tmp_path, "val", ["train"]) == tmp_path / "train"

    with pytest.raises(FileNotFoundError):
        resolve_audio_split_dir(tmp_path, "val", ["other"])


# ---------------------------------------------------------------- transforms (framework adapters)


def test_create_pytorch_loader_unsplit(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    _install_fake_torchaudio(monkeypatch)
    from dagnam.data.loaders.audio.transforms import create_pytorch_loader

    _build_audio_folder(tmp_path, per_class=4)
    ds = DagnamDataset(_audio_meta(num_samples=8), tmp_path)
    loader = create_pytorch_loader(
        ds, split="train", batch_size=2, num_workers=0, val_ratio=0.2, test_ratio=0.2, seed=0
    )
    batch = next(iter(loader))
    assert batch[0].shape[0] >= 1


def test_create_pytorch_loader_split(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    _install_fake_torchaudio(monkeypatch)
    from dagnam.data.loaders.audio.transforms import create_pytorch_loader

    _build_split_audio(tmp_path)
    ds = DagnamDataset(_audio_meta(num_samples=12), tmp_path)
    loader = create_pytorch_loader(ds, split="val", batch_size=2, num_workers=0)
    batch = next(iter(loader))
    assert batch[0].shape[0] >= 1


def test_create_pytorch_loader_uses_meta_audio_cfg(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    _install_fake_torchaudio(monkeypatch)
    from dagnam.data.loaders.audio.transforms import create_pytorch_loader

    _build_audio_folder(tmp_path, per_class=2)
    ds = DagnamDataset(_audio_meta(num_samples=4), tmp_path)
    loader = create_pytorch_loader(
        ds, split="train", batch_size=1, num_workers=0, val_ratio=0.25, test_ratio=0.25
    )
    next(iter(loader))


def test_create_tensorflow_dataset(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("tensorflow")
    from dagnam.data.loaders.audio import io as audio_io
    from dagnam.data.loaders.audio.transforms import create_tensorflow_dataset

    # Stub out load_waveform_py to skip needing soundfile/torchaudio.
    monkeypatch.setattr(
        audio_io,
        "load_waveform_py",
        _load_waveform_stub,
    )
    # Also patch the symbol the transforms module imported earlier.
    from dagnam.data.loaders.audio import transforms as audio_transforms

    monkeypatch.setattr(
        audio_transforms,
        "load_waveform_py",
        _load_waveform_stub,
    )

    _build_audio_folder(tmp_path, per_class=2)
    ds = DagnamDataset(_audio_meta(num_samples=4), tmp_path)
    tf_ds = create_tensorflow_dataset(
        ds, split="train", batch_size=2, shuffle=False, val_ratio=0.25, test_ratio=0.25, seed=0
    )
    next(iter(tf_ds))


def test_create_tensorflow_dataset_with_map_fns(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    from dagnam.data.loaders.audio import transforms as audio_transforms

    monkeypatch.setattr(
        audio_transforms,
        "load_waveform_py",
        _load_waveform_stub,
    )

    def map_sample(waveform: object, label: object) -> tuple[object, object]:
        return waveform, label

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
        map_fn=map_sample,
        batch_map_fn=map_sample,
    )
    next(iter(tf_ds))


def test_create_flax_dataset(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("jax")
    from dagnam.data.loaders.audio import transforms as audio_transforms

    monkeypatch.setattr(
        audio_transforms,
        "load_waveform_py",
        _load_waveform_stub,
    )

    _build_audio_folder(tmp_path, per_class=2)
    ds = DagnamDataset(_audio_meta(num_samples=4), tmp_path)
    batches = audio_transforms.create_flax_dataset(
        ds, split="train", batch_size=2, shuffle=False, val_ratio=0.25, test_ratio=0.25, seed=0
    )
    assert batches


def test_create_flax_dataset_with_transforms(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("jax")
    from dagnam.data.loaders.audio import transforms as audio_transforms

    monkeypatch.setattr(
        audio_transforms,
        "load_waveform_py",
        _load_waveform_stub,
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
