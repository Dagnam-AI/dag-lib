"""Coverage for the tensorflow_datasets native system loader."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from tests.data.loaders._system_fakes import (
    fallback_resolve,
)

from dagnam.data.loaders.system.tensorflow_datasets import (
    ensure_system_trust,
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

    # Simulate tensorflow_datasets not being installed. The loader pulls it in
    # via ``importlib.import_module`` (not a bare ``import`` statement), so the
    # faithful way to force the missing-package branch is to make that call
    # raise ImportError.
    real_import_module = tfds_mod.import_module

    def fake_import_module(name: str, package: str | None = None) -> object:
        if name == "tensorflow_datasets":
            raise ImportError("not installed")
        return real_import_module(name, package)

    monkeypatch.setattr(tfds_mod, "import_module", fake_import_module)
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


def test_ensure_system_trust_injects_when_available(monkeypatch: PytestMonkeyPatch) -> None:
    """When ``truststore`` is importable, its OS-store SSL injection runs."""
    calls: dict[str, int] = {"inject": 0}

    def fake_inject() -> None:
        calls["inject"] += 1

    monkeypatch.setitem(sys.modules, "truststore", SimpleNamespace(inject_into_ssl=fake_inject))
    ensure_system_trust()
    assert calls["inject"] == 1


def test_ensure_system_trust_noop_when_truststore_missing(monkeypatch: PytestMonkeyPatch) -> None:
    """A missing ``truststore`` degrades to a no-op (clean networks don't need it)."""
    # ``None`` in sys.modules makes import_module raise ImportError deterministically,
    # regardless of whether truststore happens to be installed in the test env.
    monkeypatch.setitem(sys.modules, "truststore", None)
    ensure_system_trust()  # must not raise


class _FakeTensor:
    """Minimal tensor stub: division records the scaling the loader applies."""

    def __init__(self, value: object) -> None:
        self.value = value

    def __truediv__(self, other: float) -> tuple[str, object, float]:
        return ("scaled", self.value, other)


def _fake_tf() -> SimpleNamespace:
    return SimpleNamespace(
        uint8="u8",
        float32="f32",
        cast=lambda value, _dtype: _FakeTensor(value),
    )


def test_resolve_system_dataset_tf_normalizes_uint8_images(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    """uint8 image splits are mapped to float32/255 so Conv kernels accept them."""
    from dagnam.data.loaders.system import tensorflow_datasets as tfds_mod

    monkeypatch.setattr(tfds_mod, "SYSTEM_CACHE_ROOT", tmp_path)
    map_calls: list[tuple[str, object]] = []

    class FakeImageDS:
        def __init__(self, split: str) -> None:
            self.split = split
            self.element_spec = (
                SimpleNamespace(dtype=SimpleNamespace(name="uint8")),
                SimpleNamespace(dtype=SimpleNamespace(name="int64")),
            )

        def map(self, fn: object, num_parallel_calls: object = None) -> str:
            map_calls.append((self.split, fn))
            return f"NORM:{self.split}"

    def fake_load(
        _name: str, *, split: str, as_supervised: bool = True, data_dir: str = ""
    ) -> object:
        return FakeImageDS(split)

    monkeypatch.setitem(sys.modules, "tensorflow_datasets", SimpleNamespace(load=fake_load))
    monkeypatch.setitem(sys.modules, "tensorflow", _fake_tf())

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
    assert out.native_train_tf == "NORM:train"
    assert out.native_test_tf == "NORM:test"
    # The captured map fn scales the image by 255 and passes the label through.
    _split, fn = map_calls[0]
    normalize = cast("Callable[[object, object], object]", fn)
    assert normalize("IMG", "LBL") == (("scaled", "IMG", 255.0), "LBL")


def test_resolve_system_dataset_tf_skips_normalize_for_text(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    """Non-uint8 (e.g. text) features are passed through without an image map."""
    from dagnam.data.loaders.system import tensorflow_datasets as tfds_mod

    monkeypatch.setattr(tfds_mod, "SYSTEM_CACHE_ROOT", tmp_path)

    class FakeTextDS:
        def __init__(self, split: str) -> None:
            self.split = split
            self.element_spec = (
                SimpleNamespace(dtype=SimpleNamespace(name="string")),
                SimpleNamespace(dtype=SimpleNamespace(name="int64")),
            )

        def map(self, _fn: object, num_parallel_calls: object = None) -> str:
            raise AssertionError("text features must not be image-normalized")

    def fake_load(
        _name: str, *, split: str, as_supervised: bool = True, data_dir: str = ""
    ) -> object:
        return FakeTextDS(split)

    monkeypatch.setitem(sys.modules, "tensorflow_datasets", SimpleNamespace(load=fake_load))

    out = resolve_system_dataset_tf(
        {
            "name": "imdb",
            "id": "1",
            "format": "native",
            "dataset_type": "text",
            "num_classes": 2,
            "class_names": [],
            "num_samples": 2,
        }
    )
    assert isinstance(out.native_train_tf, FakeTextDS)
    assert out.native_train_tf.split == "train"


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
