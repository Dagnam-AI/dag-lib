"""Coverage for dagnam.data.loaders.system.* native loaders.

We avoid real network calls and heavy torchvision downloads by monkeypatching
the dataset constructors and checksum/download helpers.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from dagnam.data.loaders.system import torchvision as tv_mod
from dagnam.data.loaders.system.flax import resolve_system_dataset_flax
from dagnam.data.loaders.system.registry import resolve_system_dataset
from dagnam.data.loaders.system.tensorflow_datasets import (
    _resolve_tfds_name,
    resolve_system_dataset_tf,
)

# ---------------------------------------------------------------- registry


def test_resolve_system_dataset_unknown_raises():
    from dagnam._core.exceptions import DatasetNotFoundError

    with pytest.raises(DatasetNotFoundError):
        resolve_system_dataset({"name": "absolutely-not-a-real-dataset-name"})


def test_resolve_system_dataset_exact_match(monkeypatch):
    called = {}

    def fake_load(meta, transform=None):
        called["meta"] = meta
        called["transform"] = transform
        return "FAKE_DS"

    monkeypatch.setitem(tv_mod._load_mnist.__globals__, "_load_mnist", fake_load)  # no-op
    from dagnam.data.loaders.system import registry as reg

    monkeypatch.setitem(reg._NATIVE_LOADERS, "mnist", fake_load)
    result = resolve_system_dataset({"name": "MNIST"}, transform="T")
    assert result == "FAKE_DS"
    assert called["transform"] == "T"


def test_resolve_system_dataset_substring_match(monkeypatch):
    fake_called = []

    def fake_load(meta, transform=None):
        fake_called.append(meta["name"])
        return "FAKE"

    from dagnam.data.loaders.system import registry as reg

    # Insert a unique key and trigger substring path
    monkeypatch.setitem(reg._NATIVE_LOADERS, "unique-prefix-xyz", fake_load)
    out = resolve_system_dataset({"name": "Unique-Prefix-Xyz-Dataset"})
    assert out == "FAKE"
    assert fake_called


# ---------------------------------------------------------------- torchvision loaders


def _stub_dataset(monkeypatch, attr_name):
    """Replace `torchvision.datasets.{attr_name}` with a stub that returns a sentinel."""
    from torchvision import datasets

    class _StubDataset:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __len__(self):
            return 2

        def __getitem__(self, i):
            import torch

            return torch.zeros(3, 4, 4), i % 2

    monkeypatch.setattr(datasets, attr_name, _StubDataset)
    return _StubDataset


def test_load_mnist(monkeypatch, tmp_path):
    monkeypatch.setattr(tv_mod, "_SYSTEM_CACHE_ROOT", tmp_path)
    _stub_dataset(monkeypatch, "MNIST")
    ds = tv_mod._load_mnist({"name": "MNIST", "id": "1", "format": "native", "dataset_type": "image", "num_classes": 2, "class_names": [], "num_samples": 2})
    assert ds._native_train is not None
    assert ds._native_test is not None


def test_load_mnist_with_explicit_transform(monkeypatch, tmp_path):
    monkeypatch.setattr(tv_mod, "_SYSTEM_CACHE_ROOT", tmp_path)
    _stub_dataset(monkeypatch, "MNIST")
    ds = tv_mod._load_mnist(
        {"name": "MNIST", "id": "1", "format": "native", "dataset_type": "image", "num_classes": 2, "class_names": [], "num_samples": 2}, transform="X"
    )
    assert ds._native_train is not None


def test_load_cifar10(monkeypatch, tmp_path):
    monkeypatch.setattr(tv_mod, "_SYSTEM_CACHE_ROOT", tmp_path)
    _stub_dataset(monkeypatch, "CIFAR10")
    ds = tv_mod._load_cifar10({"name": "cifar10", "id": "1", "format": "native", "dataset_type": "image", "num_classes": 2, "class_names": [], "num_samples": 2})
    assert ds._native_train is not None


def test_load_cifar10_with_transform(monkeypatch, tmp_path):
    monkeypatch.setattr(tv_mod, "_SYSTEM_CACHE_ROOT", tmp_path)
    _stub_dataset(monkeypatch, "CIFAR10")
    ds = tv_mod._load_cifar10(
        {"name": "cifar10", "id": "1", "format": "native", "dataset_type": "image", "num_classes": 2, "class_names": [], "num_samples": 2}, transform="Z"
    )
    assert ds


def test_load_cifar100(monkeypatch, tmp_path):
    monkeypatch.setattr(tv_mod, "_SYSTEM_CACHE_ROOT", tmp_path)
    _stub_dataset(monkeypatch, "CIFAR100")
    ds = tv_mod._load_cifar100({"name": "cifar100", "id": "1", "format": "native", "dataset_type": "image", "num_classes": 2, "class_names": [], "num_samples": 2})
    assert ds


def test_load_cifar100_with_transform(monkeypatch, tmp_path):
    monkeypatch.setattr(tv_mod, "_SYSTEM_CACHE_ROOT", tmp_path)
    _stub_dataset(monkeypatch, "CIFAR100")
    ds = tv_mod._load_cifar100(
        {"name": "cifar100", "id": "1", "format": "native", "dataset_type": "image", "num_classes": 2, "class_names": [], "num_samples": 2}, transform="Q"
    )
    assert ds


def test_load_fashion_mnist(monkeypatch, tmp_path):
    monkeypatch.setattr(tv_mod, "_SYSTEM_CACHE_ROOT", tmp_path)
    _stub_dataset(monkeypatch, "FashionMNIST")
    ds = tv_mod._load_fashion_mnist(
        {"name": "fashion-mnist", "id": "1", "format": "native", "dataset_type": "image", "num_classes": 2, "class_names": [], "num_samples": 2}
    )
    assert ds


def test_load_fashion_mnist_with_transform(monkeypatch, tmp_path):
    monkeypatch.setattr(tv_mod, "_SYSTEM_CACHE_ROOT", tmp_path)
    _stub_dataset(monkeypatch, "FashionMNIST")
    ds = tv_mod._load_fashion_mnist(
        {"name": "fashion-mnist", "id": "1", "format": "native", "dataset_type": "image", "num_classes": 2, "class_names": [], "num_samples": 2}, transform="T"
    )
    assert ds


def test_load_oxford_pets(monkeypatch, tmp_path):
    monkeypatch.setattr(tv_mod, "_SYSTEM_CACHE_ROOT", tmp_path)
    _stub_dataset(monkeypatch, "OxfordIIITPet")
    ds = tv_mod._load_oxford_pets(
        {"name": "oxford pets", "id": "1", "format": "native", "dataset_type": "image", "num_classes": 2, "class_names": [], "num_samples": 2}
    )
    assert ds._native_train is not None


def test_load_oxford_pets_with_transform(monkeypatch, tmp_path):
    monkeypatch.setattr(tv_mod, "_SYSTEM_CACHE_ROOT", tmp_path)
    _stub_dataset(monkeypatch, "OxfordIIITPet")
    ds = tv_mod._load_oxford_pets(
        {"name": "oxford pets", "id": "1", "format": "native", "dataset_type": "image", "num_classes": 2, "class_names": [], "num_samples": 2}, transform="T"
    )
    assert ds


def test_load_oxford_pets_falls_back_when_torchvision_lacks_it(monkeypatch, tmp_path):
    monkeypatch.setattr(tv_mod, "_SYSTEM_CACHE_ROOT", tmp_path)
    from torchvision import datasets

    def _raises(*a, **kw):
        raise AttributeError("missing")

    monkeypatch.setattr(datasets, "OxfordIIITPet", _raises)
    ds = tv_mod._load_oxford_pets(
        {"name": "oxford pets", "id": "1", "format": "native", "dataset_type": "image", "num_classes": 2, "class_names": [], "num_samples": 0}
    )
    # Falls back to file-based DagnamDataset
    assert ds is not None


def test_load_speech_commands_fallback(monkeypatch, tmp_path):
    """When torchaudio raises on import, the loader should fall back."""
    monkeypatch.setattr(tv_mod, "_SYSTEM_CACHE_ROOT", tmp_path)
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "torchaudio":
            raise ImportError("torchaudio not available")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    ds = tv_mod._load_speech_commands(
        {"name": "speech commands", "id": "1", "format": "native", "dataset_type": "audio", "num_classes": 2, "class_names": [], "num_samples": 0}
    )
    assert ds is not None


def test_load_wikitext2_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(tv_mod, "_SYSTEM_CACHE_ROOT", tmp_path)
    # If torchtext isn't installed the ImportError branch fires; verify it returns a file-based ds
    ds = tv_mod._load_wikitext2(
        {"name": "wikitext-2", "id": "1", "format": "native", "dataset_type": "image", "num_classes": 2, "class_names": [], "num_samples": 0}
    )
    assert ds is not None


# ---------------------------------------------------------------- IMDB download path


def test_load_imdb_uses_existing_verified_file(monkeypatch, tmp_path):
    monkeypatch.setattr(tv_mod, "_SYSTEM_CACHE_ROOT", tmp_path)

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

    # Patch _sha256 to return the expected hash so the verified-file path is taken.
    monkeypatch.setattr(tv_mod, "_sha256", lambda _p: tv_mod._IMDB_SHA256)

    ds = tv_mod._load_imdb({"name": "IMDB", "id": "1", "format": "native", "dataset_type": "image", "num_classes": 2, "class_names": [], "num_samples": 4})
    assert ds._native_train is not None
    x_train, y_train = ds._native_train
    assert len(x_train) == 2


def test_load_imdb_downloads_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(tv_mod, "_SYSTEM_CACHE_ROOT", tmp_path)

    expected_sha = tv_mod._IMDB_SHA256

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
        def __init__(self, content):
            self._content = content

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=1):
            yield self._content

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(tv_mod.requests, "get", lambda *a, **kw: _FakeResp(sample_bytes))
    # Override _sha256 to match our expected hash so checksum passes.
    monkeypatch.setattr(tv_mod, "_sha256", lambda _p: expected_sha)

    ds = tv_mod._load_imdb({"name": "IMDB", "id": "1", "format": "native", "dataset_type": "image", "num_classes": 2, "class_names": [], "num_samples": 2})
    assert ds._native_train is not None


def test_download_verified_file_rejects_non_https(tmp_path):
    with pytest.raises(ValueError, match="HTTPS"):
        tv_mod._download_verified_file("http://insecure.example.com/x", tmp_path / "x", "abc")


def test_download_verified_file_rejects_bad_checksum(monkeypatch, tmp_path):
    class _FakeResp:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=1):
            yield b"corrupt"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(tv_mod.requests, "get", lambda *a, **kw: _FakeResp())
    monkeypatch.setattr(tv_mod, "_sha256", lambda _p: "wrong_hash")
    dest = tmp_path / "f"
    with pytest.raises(ValueError, match="checksum mismatch"):
        tv_mod._download_verified_file("https://x/y", dest, "expected_hash")


def test_sha256_computes(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert tv_mod._sha256(p) == expected


# ---------------------------------------------------------------- tensorflow_datasets


def test_resolve_tfds_name_exact_match():
    assert _resolve_tfds_name({"name": "mnist"}) == "mnist"


def test_resolve_tfds_name_substring():
    assert _resolve_tfds_name({"name": "cifar-10-custom"}) == "cifar10"


def test_resolve_tfds_name_returns_none_for_unknown():
    assert _resolve_tfds_name({"name": "totally-unknown"}) is None


def test_resolve_system_dataset_tf_unknown_falls_back(monkeypatch):
    """When tfds name resolution returns None, fall back to PT native."""
    from dagnam.data.loaders.system import tensorflow_datasets as tfds_mod

    called = {}

    def fake_resolve(meta):
        called["called"] = True
        return "FB_DS"

    monkeypatch.setattr(tfds_mod, "resolve_system_dataset", fake_resolve)
    out = resolve_system_dataset_tf({"name": "no-such-dataset-name"})
    assert out == "FB_DS"


def test_resolve_system_dataset_tf_falls_back_on_missing_tfds(monkeypatch):
    from dagnam.data.loaders.system import tensorflow_datasets as tfds_mod

    monkeypatch.setattr(tfds_mod, "resolve_system_dataset", lambda meta: "FB")

    # Force import of tensorflow_datasets to fail
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "tensorflow_datasets":
            raise ImportError("not installed")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert resolve_system_dataset_tf({"name": "mnist"}) == "FB"


def test_resolve_system_dataset_tf_loads(monkeypatch, tmp_path):
    """When tfds is available and the name resolves, build a native_tf dataset."""
    from dagnam.data.loaders.system import tensorflow_datasets as tfds_mod

    monkeypatch.setattr(tfds_mod, "_SYSTEM_CACHE_ROOT", tmp_path)

    fake_tfds = SimpleNamespace(
        load=lambda name, split=None, as_supervised=None, data_dir=None: f"TFDS:{split}"
    )
    import sys

    monkeypatch.setitem(sys.modules, "tensorflow_datasets", fake_tfds)
    out = resolve_system_dataset_tf({"name": "mnist", "id": "1", "format": "native", "dataset_type": "image", "num_classes": 2, "class_names": [], "num_samples": 2})
    assert out._native_train_tf == "TFDS:train"
    assert out._native_test_tf == "TFDS:test"


# ---------------------------------------------------------------- flax system loader


def test_resolve_system_dataset_flax_unknown_falls_back(monkeypatch):
    from dagnam.data.loaders.system import flax as flax_mod

    monkeypatch.setattr(flax_mod, "resolve_system_dataset", lambda meta: "FB")
    assert resolve_system_dataset_flax({"name": "no-such-thing"}) == "FB"


def test_resolve_system_dataset_flax_falls_back_on_missing_tfds(monkeypatch):
    from dagnam.data.loaders.system import flax as flax_mod

    monkeypatch.setattr(flax_mod, "resolve_system_dataset", lambda meta: "FB")
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "tensorflow_datasets":
            raise ImportError("nope")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert resolve_system_dataset_flax({"name": "mnist"}) == "FB"


def test_resolve_system_dataset_flax_image_path(monkeypatch, tmp_path):
    """Numeric/uint8 images: scaled to [0,1] float32."""
    from dagnam.data.loaders.system import flax as flax_mod

    monkeypatch.setattr(flax_mod, "_SYSTEM_CACHE_ROOT", tmp_path)
    pytest.importorskip("jax")

    # Build fake tfds module that yields uint8 image samples
    samples_train = [(np.zeros((4, 4, 1), dtype=np.uint8), 0) for _ in range(3)]
    samples_test = [(np.ones((4, 4, 1), dtype=np.uint8) * 255, 1) for _ in range(2)]

    def fake_load(name, split=None, as_supervised=None, data_dir=None):
        return samples_train if split == "train" else samples_test

    fake_tfds = SimpleNamespace(load=fake_load, as_numpy=lambda x: x)
    import sys

    monkeypatch.setitem(sys.modules, "tensorflow_datasets", fake_tfds)
    ds = resolve_system_dataset_flax({"name": "mnist", "id": "1", "format": "native", "dataset_type": "image", "num_classes": 2, "class_names": [], "num_samples": 5})
    assert ds._native_train_flax is not None
    assert ds._native_test_flax is not None


def test_resolve_system_dataset_flax_text_bytes(monkeypatch, tmp_path):
    """Bytes samples: byte-encoded, padded."""
    from dagnam.data.loaders.system import flax as flax_mod

    monkeypatch.setattr(flax_mod, "_SYSTEM_CACHE_ROOT", tmp_path)
    pytest.importorskip("jax")

    samples = [(b"hello", 0), (b"world!", 1)]

    fake_tfds = SimpleNamespace(
        load=lambda *a, **kw: samples, as_numpy=lambda x: x
    )
    import sys

    monkeypatch.setitem(sys.modules, "tensorflow_datasets", fake_tfds)
    ds = resolve_system_dataset_flax(
        {"name": "imdb", "id": "1", "format": "native", "dataset_type": "image", "num_classes": 2, "class_names": [], "num_samples": 2}
    )
    assert ds._native_train_flax is not None


def test_resolve_system_dataset_flax_text_str(monkeypatch, tmp_path):
    from dagnam.data.loaders.system import flax as flax_mod

    monkeypatch.setattr(flax_mod, "_SYSTEM_CACHE_ROOT", tmp_path)
    pytest.importorskip("jax")

    samples = [("hello", 0), ("world!", 1)]
    fake_tfds = SimpleNamespace(load=lambda *a, **kw: samples, as_numpy=lambda x: x)
    import sys

    monkeypatch.setitem(sys.modules, "tensorflow_datasets", fake_tfds)
    ds = resolve_system_dataset_flax(
        {"name": "imdb", "id": "1", "format": "native", "dataset_type": "image", "num_classes": 2, "class_names": [], "num_samples": 2}
    )
    assert ds._native_train_flax is not None


def test_resolve_system_dataset_flax_numeric_array(monkeypatch, tmp_path):
    """Non-image numeric numpy arrays — cast to float32 without scaling."""
    from dagnam.data.loaders.system import flax as flax_mod

    monkeypatch.setattr(flax_mod, "_SYSTEM_CACHE_ROOT", tmp_path)
    pytest.importorskip("jax")

    samples = [(np.zeros(4, dtype=np.float64), 0), (np.ones(4, dtype=np.float64), 1)]
    fake_tfds = SimpleNamespace(load=lambda *a, **kw: samples, as_numpy=lambda x: x)
    import sys

    monkeypatch.setitem(sys.modules, "tensorflow_datasets", fake_tfds)
    ds = resolve_system_dataset_flax(
        {"name": "mnist", "id": "1", "format": "native", "dataset_type": "image", "num_classes": 2, "class_names": [], "num_samples": 2}
    )
    assert ds._native_train_flax is not None


def test_resolve_system_dataset_flax_fallback_for_misc_type(monkeypatch, tmp_path):
    """Items that aren't ndarray/bytes/str hit the fallback `jnp.asarray(np.asarray(xs))`."""
    from dagnam.data.loaders.system import flax as flax_mod

    monkeypatch.setattr(flax_mod, "_SYSTEM_CACHE_ROOT", tmp_path)
    pytest.importorskip("jax")

    samples = [(1.5, 0), (2.5, 1)]
    fake_tfds = SimpleNamespace(load=lambda *a, **kw: samples, as_numpy=lambda x: x)
    import sys

    monkeypatch.setitem(sys.modules, "tensorflow_datasets", fake_tfds)
    ds = resolve_system_dataset_flax(
        {"name": "mnist", "id": "1", "format": "native", "dataset_type": "image", "num_classes": 2, "class_names": [], "num_samples": 2}
    )
    assert ds._native_train_flax is not None
