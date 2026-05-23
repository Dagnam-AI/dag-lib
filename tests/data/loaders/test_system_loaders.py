"""Coverage for dagnam.data.loaders.system.* native loaders.

We avoid real network calls and heavy torchvision downloads by monkeypatching
the dataset constructors and checksum/download helpers.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast


import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from dagnam.data.loaders.audio.dataset import TorchTensor
from dagnam.data.loaders.system import torchvision as tv_mod
from dagnam.data.loaders.system.flax import resolve_system_dataset_flax
from dagnam.data.loaders.system.registry import resolve_system_dataset
from dagnam.data.loaders.system.tensorflow_datasets import (
    resolve_tfds_name,
    resolve_system_dataset_tf,
)
from tests.typing_helpers import JsonObject, ObjectTransform, PytestMonkeyPatch


class TorchTestModule(Protocol):
    def zeros(self, size: Sequence[int]) -> TorchTensor: ...


def _torch() -> TorchTestModule:
    return cast(TorchTestModule, import_module("torch"))


def _identity_transform(value: object) -> object:
    return value


def _expected_imdb_sha(_path: Path) -> str:
    return tv_mod.IMDB_SHA256


def _wrong_sha(_path: Path) -> str:
    return "wrong_hash"


def _fallback_resolve(_meta: JsonObject) -> str:
    return "FB"


def _as_numpy(value: object) -> object:
    return value


class FakeTfdsLoader:
    def __init__(self, samples: Sequence[tuple[object, int]]) -> None:
        self._samples = list(samples)

    def __call__(self, *_args: object, **_kwargs: object) -> list[tuple[object, int]]:
        return self._samples


class SplitTfdsLoader:
    def __init__(
        self,
        train_samples: list[tuple[object, int]],
        test_samples: list[tuple[object, int]],
    ) -> None:
        self._train_samples = train_samples
        self._test_samples = test_samples

    def __call__(
        self,
        _name: str,
        split: str | None = None,
        _as_supervised: bool | None = None,
        _data_dir: Path | None = None,
    ) -> list[tuple[object, int]]:
        return self._train_samples if split == "train" else self._test_samples

# ---------------------------------------------------------------- registry


def test_resolve_system_dataset_unknown_raises() -> None:
    from dagnam._core.exceptions import DatasetNotFoundError

    with pytest.raises(DatasetNotFoundError):
        resolve_system_dataset({"name": "absolutely-not-a-real-dataset-name"})


def test_resolve_system_dataset_exact_match(monkeypatch: PytestMonkeyPatch) -> None:
    called: dict[str, object] = {}

    def fake_load(meta: JsonObject, transform: ObjectTransform | None = None):
        called["meta"] = meta
        called["transform"] = transform
        return "FAKE_DS"

    monkeypatch.setitem(tv_mod.load_mnist.__globals__, "load_mnist", fake_load)  # no-op
    from dagnam.data.loaders.system import registry as reg

    monkeypatch.setitem(reg.NATIVE_LOADERS, "mnist", fake_load)
    result = resolve_system_dataset({"name": "MNIST"}, transform=_identity_transform)
    assert result == "FAKE_DS"
    assert called["transform"] is _identity_transform


def test_resolve_system_dataset_substring_match(monkeypatch: PytestMonkeyPatch) -> None:
    fake_called: list[object] = []

    def fake_load(meta: JsonObject, transform: ObjectTransform | None = None):
        fake_called.append(meta["name"])
        return "FAKE"

    from dagnam.data.loaders.system import registry as reg

    # Insert a unique key and trigger substring path
    monkeypatch.setitem(reg.NATIVE_LOADERS, "unique-prefix-xyz", fake_load)
    out = resolve_system_dataset({"name": "Unique-Prefix-Xyz-Dataset"})
    assert out == "FAKE"
    assert fake_called


# ---------------------------------------------------------------- torchvision loaders


def _stub_dataset(monkeypatch: PytestMonkeyPatch, attr_name: str) -> type[object]:
    """Replace `torchvision.datasets.{attr_name}` with a stub that returns a sentinel."""
    datasets = import_module("torchvision.datasets")
    torch = _torch()

    class _StubDataset:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

        def __len__(self) -> int:
            return 2

        def __getitem__(self, i: int) -> tuple[TorchTensor, int]:
            return torch.zeros((3, 4, 4)), i % 2

    monkeypatch.setattr(datasets, attr_name, _StubDataset)
    return _StubDataset


def testload_mnist(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(tv_mod, "SYSTEM_CACHE_ROOT", tmp_path)
    _stub_dataset(monkeypatch, "MNIST")
    ds = tv_mod.load_mnist(
        {
            "name": "MNIST",
            "id": "1",
            "format": "native",
            "dataset_type": "image",
            "num_classes": 2,
            "class_names": [],
            "num_samples": 2,
        }
    )
    assert ds.native_train is not None
    assert ds.native_test is not None


def testload_mnist_with_explicit_transform(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(tv_mod, "SYSTEM_CACHE_ROOT", tmp_path)
    _stub_dataset(monkeypatch, "MNIST")
    ds = tv_mod.load_mnist(
        {
            "name": "MNIST",
            "id": "1",
            "format": "native",
            "dataset_type": "image",
            "num_classes": 2,
            "class_names": [],
            "num_samples": 2,
        },
        transform=_identity_transform,
    )
    assert ds.native_train is not None


def testload_cifar10(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(tv_mod, "SYSTEM_CACHE_ROOT", tmp_path)
    _stub_dataset(monkeypatch, "CIFAR10")
    ds = tv_mod.load_cifar10(
        {
            "name": "cifar10",
            "id": "1",
            "format": "native",
            "dataset_type": "image",
            "num_classes": 2,
            "class_names": [],
            "num_samples": 2,
        }
    )
    assert ds.native_train is not None


def testload_cifar10_with_transform(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(tv_mod, "SYSTEM_CACHE_ROOT", tmp_path)
    _stub_dataset(monkeypatch, "CIFAR10")
    ds = tv_mod.load_cifar10(
        {
            "name": "cifar10",
            "id": "1",
            "format": "native",
            "dataset_type": "image",
            "num_classes": 2,
            "class_names": [],
            "num_samples": 2,
        },
        transform=_identity_transform,
    )
    assert ds


def testload_cifar100(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(tv_mod, "SYSTEM_CACHE_ROOT", tmp_path)
    _stub_dataset(monkeypatch, "CIFAR100")
    ds = tv_mod.load_cifar100(
        {
            "name": "cifar100",
            "id": "1",
            "format": "native",
            "dataset_type": "image",
            "num_classes": 2,
            "class_names": [],
            "num_samples": 2,
        }
    )
    assert ds


def testload_cifar100_with_transform(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(tv_mod, "SYSTEM_CACHE_ROOT", tmp_path)
    _stub_dataset(monkeypatch, "CIFAR100")
    ds = tv_mod.load_cifar100(
        {
            "name": "cifar100",
            "id": "1",
            "format": "native",
            "dataset_type": "image",
            "num_classes": 2,
            "class_names": [],
            "num_samples": 2,
        },
        transform=_identity_transform,
    )
    assert ds


def testload_fashion_mnist(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(tv_mod, "SYSTEM_CACHE_ROOT", tmp_path)
    _stub_dataset(monkeypatch, "FashionMNIST")
    ds = tv_mod.load_fashion_mnist(
        {
            "name": "fashion-mnist",
            "id": "1",
            "format": "native",
            "dataset_type": "image",
            "num_classes": 2,
            "class_names": [],
            "num_samples": 2,
        }
    )
    assert ds


def testload_fashion_mnist_with_transform(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(tv_mod, "SYSTEM_CACHE_ROOT", tmp_path)
    _stub_dataset(monkeypatch, "FashionMNIST")
    ds = tv_mod.load_fashion_mnist(
        {
            "name": "fashion-mnist",
            "id": "1",
            "format": "native",
            "dataset_type": "image",
            "num_classes": 2,
            "class_names": [],
            "num_samples": 2,
        },
        transform=_identity_transform,
    )
    assert ds


def testload_oxford_pets(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(tv_mod, "SYSTEM_CACHE_ROOT", tmp_path)
    _stub_dataset(monkeypatch, "OxfordIIITPet")
    ds = tv_mod.load_oxford_pets(
        {
            "name": "oxford pets",
            "id": "1",
            "format": "native",
            "dataset_type": "image",
            "num_classes": 2,
            "class_names": [],
            "num_samples": 2,
        }
    )
    assert ds.native_train is not None


def testload_oxford_pets_with_transform(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(tv_mod, "SYSTEM_CACHE_ROOT", tmp_path)
    _stub_dataset(monkeypatch, "OxfordIIITPet")
    ds = tv_mod.load_oxford_pets(
        {
            "name": "oxford pets",
            "id": "1",
            "format": "native",
            "dataset_type": "image",
            "num_classes": 2,
            "class_names": [],
            "num_samples": 2,
        },
        transform=_identity_transform,
    )
    assert ds


def testload_oxford_pets_falls_back_when_torchvision_lacks_it(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(tv_mod, "SYSTEM_CACHE_ROOT", tmp_path)
    datasets = import_module("torchvision.datasets")

    def _raises(*a: object, **kw: object) -> None:
        raise AttributeError("missing")

    monkeypatch.setattr(datasets, "OxfordIIITPet", _raises)
    ds = tv_mod.load_oxford_pets(
        {
            "name": "oxford pets",
            "id": "1",
            "format": "native",
            "dataset_type": "image",
            "num_classes": 2,
            "class_names": [],
            "num_samples": 0,
        }
    )
    # Falls back to file-based DagnamDataset
    assert ds is not None


def testload_speech_commands_fallback(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    """When torchaudio raises on import, the loader should fall back."""
    monkeypatch.setattr(tv_mod, "SYSTEM_CACHE_ROOT", tmp_path)

    real_import_module = tv_mod.import_module

    def fake_import(name: str, package: str | None = None):
        if name == "torchaudio":
            raise ImportError("torchaudio not available")
        return real_import_module(name, package)

    monkeypatch.setattr(tv_mod, "import_module", fake_import)
    ds = tv_mod.load_speech_commands(
        {
            "name": "speech commands",
            "id": "1",
            "format": "native",
            "dataset_type": "audio",
            "num_classes": 2,
            "class_names": [],
            "num_samples": 0,
        }
    )
    assert ds is not None


def testload_wikitext2_fallback(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(tv_mod, "SYSTEM_CACHE_ROOT", tmp_path)
    # If torchtext isn't installed the ImportError branch fires; verify it returns a file-based ds
    ds = tv_mod.load_wikitext2(
        {
            "name": "wikitext-2",
            "id": "1",
            "format": "native",
            "dataset_type": "image",
            "num_classes": 2,
            "class_names": [],
            "num_samples": 0,
        }
    )
    assert ds is not None


# ---------------------------------------------------------------- IMDB download path


def testload_imdb_uses_existing_verified_file(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(tv_mod, "SYSTEM_CACHE_ROOT", tmp_path)

    # Build a fake imdb.npz with the right hash so _ensure_verified_file accepts it
    cache_dir = tmp_path / "imdb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / "imdb.npz"
    np.savez(
        npz_path,
        x_train=np.array([[1, 2], [3, 4]]),
        y_train=np.array([0, 1]),
        x_test=np.array([[5, 6]]),
        y_test=np.array([1]),
    )

    # Patch checksum to return the expected hash so the verified-file path is taken.
    monkeypatch.setattr(tv_mod, "sha256", _expected_imdb_sha)

    ds = tv_mod.load_imdb(
        {
            "name": "IMDB",
            "id": "1",
            "format": "native",
            "dataset_type": "image",
            "num_classes": 2,
            "class_names": [],
            "num_samples": 4,
        }
    )
    assert ds.native_train is not None
    x_train, _y_train = cast(tuple[Sequence[object], Sequence[object]], ds.native_train)
    assert len(x_train) == 2


def testload_imdb_downloads_when_missing(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(tv_mod, "SYSTEM_CACHE_ROOT", tmp_path)

    # Synthesize the file contents we'd "download"
    sample_npz_path = tmp_path / "_sample.npz"
    np.savez(
        sample_npz_path,
        x_train=np.array([[1, 2]]),
        y_train=np.array([0]),
        x_test=np.array([[3, 4]]),
        y_test=np.array([1]),
    )
    sample_bytes = sample_npz_path.read_bytes()

    # Fake requests.get returning the bytes in chunks.
    class _FakeResp:
        def __init__(self, content: bytes) -> None:
            self._content = content

        def raise_for_status(self) -> None:
            pass

        def iter_content(self, chunk_size: int = 1) -> Iterator[bytes]:
            yield self._content

        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

    def fake_get(_url: str, **_kwargs: object) -> _FakeResp:
        return _FakeResp(sample_bytes)

    monkeypatch.setattr(tv_mod.requests, "get", fake_get)
    # Override checksum to match our expected hash so checksum passes.
    monkeypatch.setattr(tv_mod, "sha256", _expected_imdb_sha)

    ds = tv_mod.load_imdb(
        {
            "name": "IMDB",
            "id": "1",
            "format": "native",
            "dataset_type": "image",
            "num_classes": 2,
            "class_names": [],
            "num_samples": 2,
        }
    )
    assert ds.native_train is not None


def test_download_verified_file_rejects_non_https(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        tv_mod.download_verified_file("http://insecure.example.com/x", tmp_path / "x", "abc")


def test_download_verified_file_rejects_bad_checksum(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    class _FakeResp:
        def raise_for_status(self) -> None:
            pass

        def iter_content(self, chunk_size: int = 1) -> Iterator[bytes]:
            yield b"corrupt"

        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

    def fake_get(_url: str, **_kwargs: object) -> _FakeResp:
        return _FakeResp()

    monkeypatch.setattr(tv_mod.requests, "get", fake_get)
    monkeypatch.setattr(tv_mod, "sha256", _wrong_sha)
    dest = tmp_path / "f"
    with pytest.raises(ValueError, match="checksum mismatch"):
        tv_mod.download_verified_file("https://x/y", dest, "expected_hash")


def test_sha256_computes(tmp_path: Path) -> None:
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert tv_mod.sha256(p) == expected


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


def test_resolve_system_dataset_tf_falls_back_on_missing_tfds(monkeypatch: PytestMonkeyPatch) -> None:
    from dagnam.data.loaders.system import tensorflow_datasets as tfds_mod

    monkeypatch.setattr(tfds_mod, "resolve_system_dataset", _fallback_resolve)

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


# ---------------------------------------------------------------- flax system loader


def test_resolve_system_dataset_flax_unknown_falls_back(monkeypatch: PytestMonkeyPatch) -> None:
    from dagnam.data.loaders.system import flax as flax_mod

    monkeypatch.setattr(flax_mod, "resolve_system_dataset", _fallback_resolve)
    assert resolve_system_dataset_flax({"name": "no-such-thing"}) == "FB"


def test_resolve_system_dataset_flax_falls_back_on_missing_tfds(monkeypatch: PytestMonkeyPatch) -> None:
    from dagnam.data.loaders.system import flax as flax_mod

    monkeypatch.setattr(flax_mod, "resolve_system_dataset", _fallback_resolve)
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


def test_resolve_system_dataset_flax_image_path(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
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
        as_numpy=_as_numpy,
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


def test_resolve_system_dataset_flax_text_bytes(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    """Bytes samples: byte-encoded, padded."""
    from dagnam.data.loaders.system import flax as flax_mod

    monkeypatch.setattr(flax_mod, "SYSTEM_CACHE_ROOT", tmp_path)
    pytest.importorskip("jax")

    samples = [(b"hello", 0), (b"world!", 1)]

    fake_tfds = SimpleNamespace(load=FakeTfdsLoader(samples), as_numpy=_as_numpy)
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


def test_resolve_system_dataset_flax_text_str(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    from dagnam.data.loaders.system import flax as flax_mod

    monkeypatch.setattr(flax_mod, "SYSTEM_CACHE_ROOT", tmp_path)
    pytest.importorskip("jax")

    samples = [("hello", 0), ("world!", 1)]
    fake_tfds = SimpleNamespace(load=FakeTfdsLoader(samples), as_numpy=_as_numpy)
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


def test_resolve_system_dataset_flax_numeric_array(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    """Non-image numeric numpy arrays — cast to float32 without scaling."""
    from dagnam.data.loaders.system import flax as flax_mod

    monkeypatch.setattr(flax_mod, "SYSTEM_CACHE_ROOT", tmp_path)
    pytest.importorskip("jax")

    samples = [(np.zeros(4, dtype=np.float64), 0), (np.ones(4, dtype=np.float64), 1)]
    fake_tfds = SimpleNamespace(load=FakeTfdsLoader(samples), as_numpy=_as_numpy)
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


def test_resolve_system_dataset_flax_fallback_for_misc_type(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    """Items that aren't ndarray/bytes/str hit the fallback `jnp.asarray(np.asarray(xs))`."""
    from dagnam.data.loaders.system import flax as flax_mod

    monkeypatch.setattr(flax_mod, "SYSTEM_CACHE_ROOT", tmp_path)
    pytest.importorskip("jax")

    samples = [(1.5, 0), (2.5, 1)]
    fake_tfds = SimpleNamespace(load=FakeTfdsLoader(samples), as_numpy=_as_numpy)
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
