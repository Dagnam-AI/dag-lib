"""Shared fakes/builders for the ``dagnam.data.loaders.audio.*`` tests.

torchaudio is broken on some platforms and soundfile is optional, so these
helpers inject fakes into ``sys.modules`` and build on-disk audio-folder
layouts. They are imported by the per-concern ``test_audio_{io,dataset,
transforms}`` modules.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import numpy.typing as npt

from dagnam.data.loaders.audio.dataset import TorchTensor

if TYPE_CHECKING:
    from pathlib import Path

    from tests.typing_helpers import PytestMonkeyPatch

    from dagnam._types import JsonObject

type WaveformArray = npt.NDArray[np.float32]


class TorchTestModule(Protocol):
    long: object
    # Raw-waveform datasets may carry a float target (a regression label), so the
    # stub has to describe float32 as well as long.
    float32: object

    def zeros(self, size: Sequence[int]) -> TorchTensor: ...

    def tensor(self, data: object, *, dtype: object) -> TorchTensor: ...


class LabelTensor(Protocol):
    dtype: object

    def item(self) -> float:
        """The scalar value, so a test can assert a transform actually ran
        rather than only that some tensor came back."""
        ...


def torch_module() -> TorchTestModule:
    return cast("TorchTestModule", import_module("torch"))


def identity_transform(value: object) -> object:
    return value


def load_waveform_stub(_path: object, _target_sr: object, target_len: int) -> WaveformArray:
    return np.zeros(target_len, dtype=np.float32)


def read_stereo(
    _path: str,
    *,
    dtype: str | None = None,
    always_2d: bool | None = None,
) -> tuple[WaveformArray, int]:
    return np.ones((100, 2), dtype=np.float32), 8000


def read_long(
    _path: str,
    *,
    dtype: str | None = None,
    always_2d: bool | None = None,
) -> tuple[WaveformArray, int]:
    return np.ones(1000, dtype=np.float32), 16000


def read_short(
    _path: str,
    *,
    dtype: str | None = None,
    always_2d: bool | None = None,
) -> tuple[WaveformArray, int]:
    return np.ones(100, dtype=np.float32), 16000


def build_audio_folder(
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


def build_split_audio(root: Path) -> None:
    for split in ("train", "val", "test"):
        for cls in ("dog", "cat"):
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                (d / f"{i}.wav").write_bytes(b"X")


def audio_meta(num_samples: int = 8) -> JsonObject:
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


def install_fake_torchaudio(monkeypatch: PytestMonkeyPatch) -> SimpleNamespace:
    """Inject a minimal `torchaudio` module that the dataset code can exercise."""
    torch = torch_module()

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
            # Return (channels=1, n_mels, frames) - squeezing(0) gives (n_mels, frames).
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
