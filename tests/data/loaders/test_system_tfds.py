"""Coverage for the tensorflow_datasets native system loader."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from tests.data.loaders._system_fakes import (
    fallback_resolve,
)

from dagnam.data.loaders.system.tensorflow_datasets import (
    resolve_system_dataset_tf,
    resolve_tfds_name,
)

if TYPE_CHECKING:
    from tests.typing_helpers import JsonObject, PytestMonkeyPatch


# ---------------------------------------------------------------- tensorflow_datasets


def testresolve_tfds_name_exact_match() -> None:
    assert resolve_tfds_name({"name": "mnist"}) == "mnist"


def testresolve_tfds_name_substring() -> None:
    assert resolve_tfds_name({"name": "cifar-10-custom"}) == "cifar10"


def testresolve_tfds_name_returns_none_for_unknown() -> None:
    assert resolve_tfds_name({"name": "totally-unknown"}) is None


def test_resolve_system_dataset_tf_unknown_falls_back(monkeypatch: PytestMonkeyPatch) -> None:
    """When tfds name resolution returns None, fall back to PT native."""
    from dagnam.data.loaders.system import tensorflow_datasets as tfds_mod

    called: dict[str, bool] = {}

    def fake_resolve(_meta: JsonObject) -> str:
        called["called"] = True
        return "FB_DS"

    monkeypatch.setattr(tfds_mod, "resolve_system_dataset", fake_resolve)
    out = resolve_system_dataset_tf({"name": "no-such-dataset-name"})
    assert out == "FB_DS"


def test_resolve_system_dataset_tf_falls_back_on_missing_tfds(
    monkeypatch: PytestMonkeyPatch,
) -> None:
    from dagnam.data.loaders.system import tensorflow_datasets as tfds_mod

    monkeypatch.setattr(tfds_mod, "resolve_system_dataset", fallback_resolve)

    # Force import of tensorflow_datasets to fail
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
            raise ImportError("not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert resolve_system_dataset_tf({"name": "mnist"}) == "FB"


def test_resolve_system_dataset_tf_loads(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    """When tfds is available and the name resolves, build a native_tf dataset."""
    from dagnam.data.loaders.system import tensorflow_datasets as tfds_mod

    monkeypatch.setattr(tfds_mod, "SYSTEM_CACHE_ROOT", tmp_path)

    def fake_load(
        _name: str,
        split: str | None = None,
        _as_supervised: bool | None = None,
        _data_dir: Path | None = None,
    ) -> str:
        return f"TFDS:{split}"

    fake_tfds = SimpleNamespace(load=fake_load)
    import sys

    monkeypatch.setitem(sys.modules, "tensorflow_datasets", fake_tfds)
    out = resolve_system_dataset_tf(
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
    assert out.native_train_tf == "TFDS:train"
    assert out.native_test_tf == "TFDS:test"


def testresolve_tfds_name_non_string_returns_none() -> None:
    # A non-string ``name`` field can't be resolved to a tfds name.
    assert resolve_tfds_name({"name": 123}) is None


def test_load_supervised_split_uses_positional_fallback(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    """Older tfds whose load() rejects keyword args triggers the positional retry."""
    from dagnam.data.loaders.system import tensorflow_datasets as tfds_mod

    monkeypatch.setattr(tfds_mod, "SYSTEM_CACHE_ROOT", tmp_path)

    def fake_load(name: str, *args: object, **kwargs: object) -> str:
        if kwargs:
            raise TypeError("load() got an unexpected keyword argument 'as_supervised'")
        # Positional call: (name, split, True, cache)
        split = args[0]
        return f"POS:{split}"

    monkeypatch.setitem(sys.modules, "tensorflow_datasets", SimpleNamespace(load=fake_load))
    out = resolve_system_dataset_tf(
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
    assert out.native_train_tf == "POS:train"
    assert out.native_test_tf == "POS:test"


def test_load_supervised_split_reraises_unrelated_type_error(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    """A TypeError unrelated to keyword args is propagated, not swallowed."""
    from dagnam.data.loaders.system import tensorflow_datasets as tfds_mod

    monkeypatch.setattr(tfds_mod, "SYSTEM_CACHE_ROOT", tmp_path)

    def fake_load(_name: str, *_args: object, **_kwargs: object) -> str:
        raise TypeError("some other failure")

    monkeypatch.setitem(sys.modules, "tensorflow_datasets", SimpleNamespace(load=fake_load))
    with pytest.raises(TypeError, match="some other failure"):
        resolve_system_dataset_tf(
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
