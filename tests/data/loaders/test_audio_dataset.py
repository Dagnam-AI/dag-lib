"""Coverage for dagnam.data.loaders.audio.dataset."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, SupportsInt, cast

import pytest
from tests.data.loaders._audio_helpers import (
    LabelTensor,
    build_audio_folder,
    identity_transform,
    install_fake_torchaudio,
    torch_module,
)

from dagnam.data.loaders.audio.dataset import TorchTensor

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch


# ---------------------------------------------------------------- dataset


def test_audio_folder_dataset_basic(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    install_fake_torchaudio(monkeypatch)
    from dagnam.data.loaders.audio.dataset import AudioFolderDataset

    build_audio_folder(tmp_path, per_class=2)
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
    assert int(cast("SupportsInt", label)) in (0, 1)


def test_audio_folder_dataset_with_transforms(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    install_fake_torchaudio(monkeypatch)
    torch = torch_module()

    from dagnam.data.loaders.audio.dataset import AudioFolderDataset

    build_audio_folder(tmp_path, per_class=1)
    files = [tmp_path / "dog" / "0.wav"]
    ds = AudioFolderDataset(
        file_paths=files,
        labels=[0],
        target_sample_rate=16000,
        n_mels=8,
        max_duration_sec=1.0,
        waveform_transform=identity_transform,
        spectrogram_transform=identity_transform,
        target_transform=lambda lbl: torch.tensor(lbl, dtype=torch.long),
    )
    _item, label = ds[0]
    label_tensor = cast("LabelTensor", label)
    assert label_tensor.dtype == torch.long


def test_audio_folder_dataset_resamples_when_sr_differs(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    """torchaudio.load returns 8000Hz but target is 16000Hz — triggers Resample."""
    torch = torch_module()

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

    build_audio_folder(tmp_path, per_class=1)
    ds = AudioFolderDataset(
        file_paths=[tmp_path / "dog" / "0.wav"],
        labels=[0],
        target_sample_rate=16000,
        n_mels=8,
        max_duration_sec=1.0,
    )
    item, _ = ds[0]
    assert item.shape[-1] >= 1


def test_audio_folder_dataset_truncates_when_too_long(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    torch = torch_module()

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

    build_audio_folder(tmp_path, per_class=1)
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

    build_audio_folder(tmp_path, per_class=2)
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
