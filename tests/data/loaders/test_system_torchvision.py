"""Coverage for torchvision native system loaders (incl. IMDB download)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import hashlib
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import pytest
from tests.data.loaders._system_fakes import (
    identity_transform,
)

from dagnam.data.loaders.audio.dataset import TorchTensor
from dagnam.data.loaders.system import torchvision as tv_mod

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch


class TorchTestModule(Protocol):
    def zeros(self, size: Sequence[int]) -> TorchTensor: ...


def _torch() -> TorchTestModule:
    return cast("TorchTestModule", import_module("torch"))


def _expected_imdb_sha(_path: Path) -> str:
    return tv_mod.IMDB_SHA256


def _wrong_sha(_path: Path) -> str:
    return "wrong_hash"


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
        transform=identity_transform,
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
        transform=identity_transform,
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
        transform=identity_transform,
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
        transform=identity_transform,
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
        transform=identity_transform,
    )
    assert ds


def testload_oxford_pets_falls_back_when_torchvision_lacks_it(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
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

    real_import_module = tv_mod.import_module  # pyright: ignore[reportPrivateImportUsage]

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


def testload_imdb_uses_existing_verified_file(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
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
    x_train, _y_train = cast("tuple[Sequence[object], Sequence[object]]", ds.native_train)
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

    monkeypatch.setattr(tv_mod.requests, "get", fake_get)  # pyright: ignore[reportPrivateImportUsage]
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


def test_download_verified_file_rejects_bad_checksum(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
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

    monkeypatch.setattr(tv_mod.requests, "get", fake_get)  # pyright: ignore[reportPrivateImportUsage]
    monkeypatch.setattr(tv_mod, "sha256", _wrong_sha)
    dest = tmp_path / "f"
    with pytest.raises(ValueError, match="checksum mismatch"):
        tv_mod.download_verified_file("https://x/y", dest, "expected_hash")


def test_sha256_computes(tmp_path: Path) -> None:
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert tv_mod.sha256(p) == expected


# ---------------------------------------------------------------- SPEECHCOMMANDS / WikiText2 success


def testload_speech_commands_success(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    """A working torchaudio yields native train/test SPEECHCOMMANDS datasets."""
    from types import SimpleNamespace

    monkeypatch.setattr(tv_mod, "SYSTEM_CACHE_ROOT", tmp_path)

    class _FakeSpeechCommands:
        def __init__(self, *, root: str, download: bool, subset: str) -> None:
            self.root = root
            self.download = download
            self.subset = subset

        def __len__(self) -> int:
            return 1

    fake_torchaudio = SimpleNamespace(datasets=SimpleNamespace(SPEECHCOMMANDS=_FakeSpeechCommands))

    real_import_module = tv_mod.import_module  # pyright: ignore[reportPrivateImportUsage]

    def fake_import(name: str, package: str | None = None):
        if name == "torchaudio":
            return fake_torchaudio
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
            "num_samples": 2,
        }
    )
    assert ds.native_train is not None
    assert ds.native_test is not None
    train = cast("_FakeSpeechCommands", ds.native_train)
    assert train.subset == "training"


def testload_wikitext2_success(monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    """A working torchtext yields native train/test WikiText2 iterables (materialized)."""
    from types import SimpleNamespace

    monkeypatch.setattr(tv_mod, "SYSTEM_CACHE_ROOT", tmp_path)

    def fake_wikitext2(*, root: str, split: str) -> Iterator[str]:
        return iter([f"{split}-line-1", f"{split}-line-2"])

    fake_torchtext_datasets = SimpleNamespace(WikiText2=fake_wikitext2)

    real_import_module = tv_mod.import_module  # pyright: ignore[reportPrivateImportUsage]

    def fake_import(name: str, package: str | None = None):
        if name == "torchtext.datasets":
            return fake_torchtext_datasets
        return real_import_module(name, package)

    monkeypatch.setattr(tv_mod, "import_module", fake_import)
    ds = tv_mod.load_wikitext2(
        {
            "name": "wikitext-2",
            "id": "1",
            "format": "native",
            "dataset_type": "text",
            "num_classes": 0,
            "class_names": [],
            "num_samples": 2,
        }
    )
    assert ds.native_train is not None
    assert ds.native_test is not None
    train = cast("list[str]", ds.native_train)
    assert train == ["train-line-1", "train-line-2"]


def test_download_verified_file_skips_empty_chunks(
    monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    """Empty keep-alive chunks from iter_content are skipped, not written."""
    payload = b"real-bytes"

    class _FakeResp:
        def raise_for_status(self) -> None:
            pass

        def iter_content(self, chunk_size: int = 1) -> Iterator[bytes]:
            # An empty chunk (keep-alive) must be skipped before real content.
            yield b""
            yield payload

        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

    def fake_get(_url: str, **_kwargs: object) -> _FakeResp:
        return _FakeResp()

    monkeypatch.setattr(tv_mod.requests, "get", fake_get)  # pyright: ignore[reportPrivateImportUsage]
    expected = hashlib.sha256(payload).hexdigest()
    dest = tmp_path / "out.bin"
    tv_mod.download_verified_file("https://x/y", dest, expected)
    assert dest.read_bytes() == payload
