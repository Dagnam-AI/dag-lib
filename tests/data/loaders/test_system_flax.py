"""Coverage for the flax native system loader."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from tests.data.loaders._system_fakes import (
    FakeTfdsLoader,
    SplitTfdsLoader,
    as_numpy,
    fallback_resolve,
)

from dagnam.data.loaders.system.flax import resolve_system_dataset_flax

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch


# ---------------------------------------------------------------- flax system loader


def test_resolve_system_dataset_flax_unknown_falls_back(monkeypatch: PytestMonkeyPatch) -> None:
    from dagnam.data.loaders.system import flax as flax_mod

    monkeypatch.setattr(flax_mod, "resolve_system_dataset", fallback_resolve)
    assert resolve_system_dataset_flax({"name": "no-such-thing"}) == "FB"


def test_resolve_system_dataset_flax_falls_back_on_missing_tfds(
    monkeypatch: PytestMonkeyPatch,
) -> None:
    from dagnam.data.loaders.system import flax as flax_mod

    monkeypatch.setattr(flax_mod, "resolve_system_dataset", fallback_resolve)
    import builtins

    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> object:
        if name == "tensorflow_datasets":
            raise ImportError("nope")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert resolve_system_dataset_flax({"name": "mnist"}) == "FB"


def test_resolve_system_dataset_flax_image_path(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    """Numeric/uint8 images: scaled to [0,1] float32."""
    from dagnam.data.loaders.system import flax as flax_mod

    monkeypatch.setattr(flax_mod, "SYSTEM_CACHE_ROOT", tmp_path)
    pytest.importorskip("jax")

    # Build fake tfds module that yields uint8 image samples
    samples_train: list[tuple[object, int]] = [
        (np.zeros((4, 4, 1), dtype=np.uint8), 0) for _ in range(3)
    ]
    samples_test: list[tuple[object, int]] = [
        (np.ones((4, 4, 1), dtype=np.uint8) * 255, 1) for _ in range(2)
    ]

    fake_tfds = SimpleNamespace(
        load=SplitTfdsLoader(samples_train, samples_test),
        as_numpy=as_numpy,
    )
    import sys

    monkeypatch.setitem(sys.modules, "tensorflow_datasets", fake_tfds)
    ds = resolve_system_dataset_flax(
        {
            "name": "mnist",
            "id": "1",
            "format": "native",
            "dataset_type": "image",
            "num_classes": 2,
            "class_names": [],
            "num_samples": 5,
        }
    )
    assert ds.native_train_flax is not None
    assert ds.native_test_flax is not None


def test_resolve_system_dataset_flax_text_bytes(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    """Bytes samples: byte-encoded, padded."""
    from dagnam.data.loaders.system import flax as flax_mod

    monkeypatch.setattr(flax_mod, "SYSTEM_CACHE_ROOT", tmp_path)
    pytest.importorskip("jax")

    samples = [(b"hello", 0), (b"world!", 1)]

    fake_tfds = SimpleNamespace(load=FakeTfdsLoader(samples), as_numpy=as_numpy)
    import sys

    monkeypatch.setitem(sys.modules, "tensorflow_datasets", fake_tfds)
    ds = resolve_system_dataset_flax(
        {
            "name": "imdb",
            "id": "1",
            "format": "native",
            "dataset_type": "image",
            "num_classes": 2,
            "class_names": [],
            "num_samples": 2,
        }
    )
    assert ds.native_train_flax is not None


def test_resolve_system_dataset_flax_text_str(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    from dagnam.data.loaders.system import flax as flax_mod

    monkeypatch.setattr(flax_mod, "SYSTEM_CACHE_ROOT", tmp_path)
    pytest.importorskip("jax")

    samples = [("hello", 0), ("world!", 1)]
    fake_tfds = SimpleNamespace(load=FakeTfdsLoader(samples), as_numpy=as_numpy)
    import sys

    monkeypatch.setitem(sys.modules, "tensorflow_datasets", fake_tfds)
    ds = resolve_system_dataset_flax(
        {
            "name": "imdb",
            "id": "1",
            "format": "native",
            "dataset_type": "image",
            "num_classes": 2,
            "class_names": [],
            "num_samples": 2,
        }
    )
    assert ds.native_train_flax is not None


def test_resolve_system_dataset_flax_numeric_array(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    """Non-image numeric numpy arrays — cast to float32 without scaling."""
    from dagnam.data.loaders.system import flax as flax_mod

    monkeypatch.setattr(flax_mod, "SYSTEM_CACHE_ROOT", tmp_path)
    pytest.importorskip("jax")

    samples = [(np.zeros(4, dtype=np.float64), 0), (np.ones(4, dtype=np.float64), 1)]
    fake_tfds = SimpleNamespace(load=FakeTfdsLoader(samples), as_numpy=as_numpy)
    import sys

    monkeypatch.setitem(sys.modules, "tensorflow_datasets", fake_tfds)
    ds = resolve_system_dataset_flax(
        {
            "name": "mnist",
            "id": "1",
            "format": "native",
            "dataset_type": "image",
            "num_classes": 2,
            "class_names": [],
            "num_samples": 2,
        }
    )
    assert ds.native_train_flax is not None


def test_resolve_system_dataset_flax_fallback_for_misc_type(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    """Items that aren't ndarray/bytes/str hit the fallback `jnp.asarray(np.asarray(xs))`."""
    from dagnam.data.loaders.system import flax as flax_mod

    monkeypatch.setattr(flax_mod, "SYSTEM_CACHE_ROOT", tmp_path)
    pytest.importorskip("jax")

    samples = [(1.5, 0), (2.5, 1)]
    fake_tfds = SimpleNamespace(load=FakeTfdsLoader(samples), as_numpy=as_numpy)
    import sys

    monkeypatch.setitem(sys.modules, "tensorflow_datasets", fake_tfds)
    ds = resolve_system_dataset_flax(
        {
            "name": "mnist",
            "id": "1",
            "format": "native",
            "dataset_type": "image",
            "num_classes": 2,
            "class_names": [],
            "num_samples": 2,
        }
    )
    assert ds.native_train_flax is not None
