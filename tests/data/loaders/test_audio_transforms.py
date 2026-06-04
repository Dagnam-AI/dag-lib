"""Coverage for dagnam.data.loaders.audio.transforms (framework adapters)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from tests.data.loaders._audio_helpers import (
    audio_meta,
    build_audio_folder,
    build_split_audio,
    install_fake_torchaudio,
    load_waveform_stub,
)

from dagnam.data.dataset import DagnamDataset

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch


# ---------------------------------------------------------------- transforms (framework adapters)


def test_create_pytorch_loader_unsplit(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    install_fake_torchaudio(monkeypatch)
    from dagnam.data.loaders.audio.transforms import create_pytorch_loader

    build_audio_folder(tmp_path, per_class=4)
    ds = DagnamDataset(audio_meta(num_samples=8), tmp_path)
    loader = create_pytorch_loader(
        ds, split="train", batch_size=2, num_workers=0, val_ratio=0.2, test_ratio=0.2, seed=0
    )
    batch = next(iter(loader))
    assert batch[0].shape[0] >= 1


def test_create_pytorch_loader_split(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    install_fake_torchaudio(monkeypatch)
    from dagnam.data.loaders.audio.transforms import create_pytorch_loader

    build_split_audio(tmp_path)
    ds = DagnamDataset(audio_meta(num_samples=12), tmp_path)
    loader = create_pytorch_loader(ds, split="val", batch_size=2, num_workers=0)
    batch = next(iter(loader))
    assert batch[0].shape[0] >= 1


def test_create_pytorch_loader_uses_meta_audio_cfg(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    install_fake_torchaudio(monkeypatch)
    from dagnam.data.loaders.audio.transforms import create_pytorch_loader

    build_audio_folder(tmp_path, per_class=2)
    ds = DagnamDataset(audio_meta(num_samples=4), tmp_path)
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
        load_waveform_stub,
    )
    # Also patch the symbol the transforms module imported earlier.
    from dagnam.data.loaders.audio import transforms as audio_transforms

    monkeypatch.setattr(
        audio_transforms,
        "load_waveform_py",
        load_waveform_stub,
    )

    build_audio_folder(tmp_path, per_class=2)
    ds = DagnamDataset(audio_meta(num_samples=4), tmp_path)
    tf_ds = create_tensorflow_dataset(
        ds, split="train", batch_size=2, shuffle=False, val_ratio=0.25, test_ratio=0.25, seed=0
    )
    next(iter(tf_ds))


def test_create_tensorflow_dataset_with_map_fns(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    from dagnam.data.loaders.audio import transforms as audio_transforms

    monkeypatch.setattr(
        audio_transforms,
        "load_waveform_py",
        load_waveform_stub,
    )

    def map_sample(waveform: object, label: object) -> tuple[object, object]:
        return waveform, label

    build_audio_folder(tmp_path, per_class=2)
    ds = DagnamDataset(audio_meta(num_samples=4), tmp_path)
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
        load_waveform_stub,
    )

    build_audio_folder(tmp_path, per_class=2)
    ds = DagnamDataset(audio_meta(num_samples=4), tmp_path)
    batches = audio_transforms.create_flax_dataset(
        ds, split="train", batch_size=2, shuffle=False, val_ratio=0.25, test_ratio=0.25, seed=0
    )
    assert batches


def test_create_flax_dataset_with_transforms(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("jax")
    from dagnam.data.loaders.audio import transforms as audio_transforms

    monkeypatch.setattr(
        audio_transforms,
        "load_waveform_py",
        load_waveform_stub,
    )

    build_audio_folder(tmp_path, per_class=2)
    ds = DagnamDataset(audio_meta(num_samples=4), tmp_path)
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
