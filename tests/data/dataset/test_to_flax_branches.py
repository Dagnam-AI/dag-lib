"""Branch coverage for the to_flax mixin error paths and upgrade dispatch.

These pin current behavior for the native-conversion guard rails (missing
native dataset, malformed samples), the FLAX-native re-split helper, and the
tensorflow_datasets upgrade path that promotes a PT-native system dataset to a
FLAX-native one.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import pytest

pytest.importorskip("jax")

from tests.data.dataset._native_helpers import make_native_numpy_ds
from tests.typing_helpers import JsonObject, PytestMonkeyPatch

from dagnam._types import NativeSplit
from dagnam.data.dataset import DagnamDataset
from dagnam.data.loaders.flax import FlaxBatch

if TYPE_CHECKING:
    import jax
    import numpy.typing as npt


class JaxNumpyModule(Protocol):
    def asarray(self, value: npt.ArrayLike) -> jax.Array: ...


def _flax_batch(features: npt.ArrayLike, labels: npt.ArrayLike) -> FlaxBatch:
    import jax.numpy as jnp

    jnp_mod = cast("JaxNumpyModule", jnp)
    return FlaxBatch(features=jnp_mod.asarray(features), labels=jnp_mod.asarray(labels))


def _system_meta(name: str) -> JsonObject:
    return {
        "id": "sys-flax",
        "name": name,
        "format": "native",
        "dataset_type": "image",
        "num_samples": 4,
        "num_classes": 2,
        "class_names": [],
        "source_type": "system",
    }


def _native_split(features: object, labels: object) -> NativeSplit:
    return cast("NativeSplit", (features, labels))


# ---------------------------------------------------------------- _native_to_flax guards


def test_native_to_flax_missing_native_raises() -> None:
    ds = DagnamDataset(_system_meta("none"), data_dir=None)
    ds.native_train = None
    with pytest.raises(ValueError, match="No native dataset"):
        ds._native_to_flax(split="train", batch_size=2, shuffle=False, val_ratio=0.1, seed=0)


def test_native_to_flax_tuple_test_without_tuple_test_uses_empty(tmp_path: Path) -> None:
    """native_train tuple, native_test None → x_test, y_test = (), () (line 55)."""
    ds = make_native_numpy_ds(tmp_path)
    ds.native_test = None
    batches = ds.to_flax_dataset(split="test", batch_size=2, shuffle=False)
    assert batches == []


def test_native_to_flax_sample_not_tuple_raises() -> None:
    class _BadDs:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, _i: int) -> object:
            return np.zeros((2, 2))  # not a tuple

    ds = DagnamDataset(_system_meta("bad"), data_dir=None)
    ds.native_train = cast("NativeSplit", _BadDs())
    ds.native_test = None
    with pytest.raises(TypeError, match="feature, label"):
        ds._native_to_flax(split="train", batch_size=1, shuffle=False, val_ratio=0.0, seed=0)


def test_native_to_flax_short_tuple_sample_raises() -> None:
    class _ShortDs:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, _i: int) -> object:
            return (np.zeros((2, 2)),)  # length 1

    ds = DagnamDataset(_system_meta("short"), data_dir=None)
    ds.native_train = cast("NativeSplit", _ShortDs())
    ds.native_test = None
    with pytest.raises(TypeError, match="feature, label"):
        ds._native_to_flax(split="train", batch_size=1, shuffle=False, val_ratio=0.0, seed=0)


def test_native_to_flax_non_int_label_raises() -> None:
    class _BadLabelDs:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, _i: int) -> tuple[object, object]:
            return np.zeros((2, 2), dtype=np.float32), "not-an-int"

    ds = DagnamDataset(_system_meta("badlabel"), data_dir=None)
    ds.native_train = cast("NativeSplit", _BadLabelDs())
    ds.native_test = None
    with pytest.raises(TypeError, match="integer-compatible"):
        ds._native_to_flax(split="train", batch_size=1, shuffle=False, val_ratio=0.0, seed=0)


# ---------------------------------------------------------------- _native_flax_dataset else


def test_native_flax_dataset_unknown_split_uses_train_fallback() -> None:
    """split not in {test, train, val} falls to `native_train_flax or []` (line 169)."""
    ds = DagnamDataset(_system_meta("fallback"), data_dir=None)
    ds.native_train_flax = [
        _flax_batch(np.zeros((2, 4), dtype=np.float32), np.zeros(2, dtype=np.int64))
    ]
    out = ds.native_flax_dataset(split="bogus", batch_size=2, shuffle=False)
    assert out


# ---------------------------------------------------------------- upgrade path


def test_try_upgrade_to_native_flax_already_native_returns_true() -> None:
    ds = DagnamDataset(_system_meta("already"), data_dir=None)
    ds.native_train_flax = [
        _flax_batch(np.zeros((2, 4), dtype=np.float32), np.zeros(2, dtype=np.int64))
    ]
    assert ds._try_upgrade_to_native_flax() is True


def test_try_upgrade_to_native_flax_non_system_returns_false() -> None:
    ds = DagnamDataset(_system_meta("nonsys"), data_dir=None)
    ds._raw_meta["source_type"] = "user"
    assert ds._try_upgrade_to_native_flax() is False


def test_try_upgrade_to_native_flax_no_tfds_returns_false(
    monkeypatch: PytestMonkeyPatch,
) -> None:
    ds = DagnamDataset(_system_meta("notfds"), data_dir=None)
    monkeypatch.setattr("dagnam.data.dataset.to_flax.find_spec", lambda _name: None)
    assert ds._try_upgrade_to_native_flax() is False


def test_try_upgrade_to_native_flax_unknown_name_returns_false(
    monkeypatch: PytestMonkeyPatch,
) -> None:
    ds = DagnamDataset(_system_meta("weird"), data_dir=None)
    monkeypatch.setattr("dagnam.data.dataset.to_flax.find_spec", lambda _name: object())
    monkeypatch.setattr("dagnam.data.loaders.system.resolve_tfds_name", lambda _meta: None)
    assert ds._try_upgrade_to_native_flax() is False


def test_try_upgrade_to_native_flax_upgrade_without_train_returns_false(
    monkeypatch: PytestMonkeyPatch,
) -> None:
    empty = DagnamDataset(_system_meta("empty"), data_dir=None)
    empty.native_train_flax = None
    monkeypatch.setattr("dagnam.data.dataset.to_flax.find_spec", lambda _name: object())
    monkeypatch.setattr("dagnam.data.loaders.system.resolve_tfds_name", lambda _meta: "mnist")
    monkeypatch.setattr(
        "dagnam.data.loaders.system.resolve_system_dataset_flax",
        lambda _meta: empty,
    )
    ds = DagnamDataset(_system_meta("empty"), data_dir=None)
    assert ds._try_upgrade_to_native_flax() is False


def test_to_flax_upgrades_to_native_flax(monkeypatch: PytestMonkeyPatch) -> None:
    """Drive the tfds-upgrade dispatch: find_spec + resolvers stubbed (lines 245-263, 338)."""
    upgraded = DagnamDataset(_system_meta("mnist"), data_dir=None)
    upgraded.native_train_flax = [
        _flax_batch(np.zeros((4, 4), dtype=np.float32), np.zeros(4, dtype=np.int64))
    ]
    upgraded.native_test_flax = [
        _flax_batch(np.zeros((4, 4), dtype=np.float32), np.zeros(4, dtype=np.int64))
    ]

    monkeypatch.setattr("dagnam.data.dataset.to_flax.find_spec", lambda _name: object())
    monkeypatch.setattr("dagnam.data.loaders.system.resolve_tfds_name", lambda _meta: "mnist")
    monkeypatch.setattr(
        "dagnam.data.loaders.system.resolve_system_dataset_flax",
        lambda _meta: upgraded,
    )

    ds = DagnamDataset(_system_meta("mnist"), data_dir=None)
    ds.native_train = _native_split(np.zeros((4, 4), dtype=np.float32), np.zeros(4, dtype=np.int64))
    out = ds.to_flax_dataset(split="train", batch_size=2, shuffle=False)
    assert out


# ---------------------------------------------------------------- audio dispatch


def test_to_flax_audio_folder_dispatches(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    """audio_folder format routes through the audio-transforms flax loader (382-386)."""

    def _fake_create_audio_flax(**kwargs: object) -> list[FlaxBatch]:
        assert kwargs["split"] == "train"
        return []

    monkeypatch.setattr(
        "dagnam.data.loaders.audio.transforms.create_flax_dataset",
        _fake_create_audio_flax,
    )
    ds = DagnamDataset(
        {
            "id": "aud",
            "name": "audio",
            "format": "audio_folder",
            "dataset_type": "audio",
            "num_samples": 2,
            "num_classes": 2,
            "class_names": [],
        },
        tmp_path,
    )
    result = ds.to_flax_dataset(split="train", batch_size=2, shuffle=False)
    assert result == []
