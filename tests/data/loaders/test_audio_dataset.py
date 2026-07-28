"""Coverage for dagnam.data.loaders.audio.dataset."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, SupportsFloat, SupportsInt, cast
import wave

from tests.data.loaders._audio_helpers import (
    LabelTensor,
    build_audio_folder,
    identity_transform,
    install_fake_torchaudio,
    torch_module,
)

from dagnam.data.loaders.audio.dataset import (
    AudioFolderDataset,
    SampleTransform,
    TorchTensor,
)

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch


# ---------------------------------------------------------------- dataset


def test_bound_audio_folder_decodes_real_wav_as_raw_fixed_length_waveform(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    from dagnam.data.loaders.audio import dataset as dataset_module

    audio_path = tmp_path / "yes" / "clip.wav"
    audio_path.parent.mkdir()
    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00" * 800)

    def reject_torchaudio_decode() -> object:
        raise AssertionError("raw bound audio must not require torchaudio/TorchCodec decoding")

    monkeypatch.setattr(dataset_module, "_load_torchaudio", reject_torchaudio_decode)

    ds = dataset_module.AudioFolderDataset(
        file_paths=[audio_path],
        labels=[0],
        target_sample_rate=8000,
        target_length=800,
        max_duration_sec=5.0,
        return_waveform=True,
    )

    waveform, label = ds[0]
    assert tuple(waveform.shape) == (800,)
    assert int(cast("SupportsInt", label)) == 0


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


def _write_wav(tmp_path: Path) -> Path:
    audio_path = tmp_path / "yes" / "clip.wav"
    audio_path.parent.mkdir()
    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00" * 800)
    return audio_path


def _raw_dataset(
    audio_path: Path,
    labels: list[int],
    target_transform: SampleTransform | None = None,
) -> AudioFolderDataset:
    return AudioFolderDataset(
        file_paths=[audio_path],
        labels=labels,
        target_sample_rate=8000,
        target_length=800,
        max_duration_sec=5.0,
        return_waveform=True,
        target_transform=target_transform,
    )


def test_raw_waveform_mode_applies_the_target_transform(tmp_path: Path) -> None:
    """`return_waveform=True` has its own label path, separate from the mel one.

    The mel branch's target_transform was already covered; this one was not, so
    a label transform silently doing nothing in raw-waveform mode would not have
    been caught.
    """
    torch = torch_module()
    ds = _raw_dataset(
        _write_wav(tmp_path),
        [0],
        target_transform=lambda lbl: torch.tensor(
            int(cast("SupportsInt", lbl)) + 5, dtype=torch.long
        ),
    )

    waveform, label = ds[0]
    label_tensor = cast("LabelTensor", label)
    # 0 -> 5 proves the transform ran, not merely that a tensor came back.
    assert int(label_tensor.item()) == 5
    assert label_tensor.dtype == torch.long
    assert tuple(waveform.shape) == (800,)


def test_raw_waveform_mode_leaves_an_already_tensor_label_alone(tmp_path: Path) -> None:
    """A target_transform that already returns a tensor must not be re-wrapped.

    This is the branch that skips the `torch.tensor(...)` coercion entirely —
    re-wrapping would flatten a deliberately non-long dtype (here float32) back
    to long and silently corrupt a regression target.
    """
    torch = torch_module()
    ds = _raw_dataset(
        _write_wav(tmp_path),
        [1],
        target_transform=lambda lbl: torch.tensor(
            float(cast("SupportsFloat", lbl)), dtype=torch.float32
        ),
    )

    _waveform, label = ds[0]
    label_tensor = cast("LabelTensor", label)
    assert label_tensor.dtype == torch.float32
    assert float(label_tensor.item()) == 1.0
