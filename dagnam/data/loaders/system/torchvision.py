"""Native PyTorch/torchvision system dataset loaders."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import hashlib
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast
from urllib.parse import urlparse

import requests

from dagnam._types import IndexedDataset, JsonObject, NativeSplit
from dagnam.data.loaders.system.common import SYSTEM_CACHE_ROOT

if TYPE_CHECKING:
    from dagnam.data.dataset import DagnamDataset

_IMDB_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz"
_IMDB_SHA256 = "69664113be75683a8fe16e3ed0ab59fda8886cb3cd7ada244f7d9544e4676b9f"
IMDB_SHA256 = _IMDB_SHA256
_DOWNLOAD_TIMEOUT = (30, 60)
TransformFn = Callable[[object], object]


class _InterpolationModeEnum(Protocol):
    """The ``transforms.InterpolationMode`` enum surface we use (G087 mask resize)."""

    NEAREST: object


class TorchVisionTransformsModule(Protocol):
    """TorchVision transform constructors used by system datasets."""

    InterpolationMode: _InterpolationModeEnum

    def Compose(self, transforms: Sequence[TransformFn]) -> TransformFn: ...

    def ToTensor(self) -> TransformFn: ...

    def PILToTensor(self) -> TransformFn: ...  # noqa: N802 - torchvision API name

    def Normalize(self, mean: Sequence[float], std: Sequence[float]) -> TransformFn: ...

    def Resize(self, size: tuple[int, int], interpolation: object = ...) -> TransformFn: ...


class TorchVisionDatasetsModule(Protocol):
    """TorchVision dataset constructors used by system datasets."""

    def MNIST(
        self,
        *,
        root: str,
        train: bool,
        download: bool,
        transform: TransformFn | None,
    ) -> IndexedDataset: ...

    def CIFAR10(
        self,
        *,
        root: str,
        train: bool,
        download: bool,
        transform: TransformFn | None,
    ) -> IndexedDataset: ...

    def CIFAR100(
        self,
        *,
        root: str,
        train: bool,
        download: bool,
        transform: TransformFn | None,
    ) -> IndexedDataset: ...

    def FashionMNIST(
        self,
        *,
        root: str,
        train: bool,
        download: bool,
        transform: TransformFn | None,
    ) -> IndexedDataset: ...

    def OxfordIIITPet(
        self,
        *,
        root: str,
        split: str,
        download: bool,
        transform: TransformFn | None,
        target_types: str = ...,
        target_transform: TransformFn | None = ...,
    ) -> IndexedDataset: ...


class TorchaudioDatasetsModule(Protocol):
    """Torchaudio dataset constructors used by system datasets."""

    def SPEECHCOMMANDS(self, *, root: str, download: bool, subset: str) -> IndexedDataset: ...


class TorchaudioModule(Protocol):
    """Torchaudio module surface used by system datasets."""

    datasets: TorchaudioDatasetsModule


class TorchTextDatasetsModule(Protocol):
    """TorchText dataset constructors used by system datasets."""

    def WikiText2(self, *, root: str, split: str) -> Iterable[str]: ...


def _load_torchvision() -> tuple[TorchVisionDatasetsModule, TorchVisionTransformsModule]:
    return (
        cast("TorchVisionDatasetsModule", import_module("torchvision.datasets")),
        cast("TorchVisionTransformsModule", import_module("torchvision.transforms")),
    )


def _load_torchaudio() -> TorchaudioModule:
    return cast("TorchaudioModule", import_module("torchaudio"))


def _load_torchtext_datasets() -> TorchTextDatasetsModule:
    return cast("TorchTextDatasetsModule", import_module("torchtext.datasets"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified_file(url: str, dest: Path, expected_sha256: str) -> None:
    """Download an HTTPS file and atomically install it only if SHA-256 matches."""
    if urlparse(url).scheme != "https":
        raise ValueError("System dataset downloads must use HTTPS URLs")

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    try:
        with requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)

        actual = sha256(tmp)
        if actual != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise ValueError(
                f"Downloaded system dataset checksum mismatch: expected {expected_sha256}, got {actual}"
            )
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


_download_verified_file = download_verified_file


def ensure_verified_file(url: str, dest: Path, expected_sha256: str) -> None:
    if dest.exists() and sha256(dest) == expected_sha256:
        return
    dest.unlink(missing_ok=True)
    download_verified_file(url, dest, expected_sha256)


def load_mnist(
    meta: JsonObject,
    transform: TransformFn | None = None,
    binding: dict[str, Any] | None = None,
) -> DagnamDataset:
    del binding
    from dagnam.data.dataset import DagnamDataset

    datasets, transforms = _load_torchvision()
    cache = SYSTEM_CACHE_ROOT / "mnist"
    cache.mkdir(parents=True, exist_ok=True)

    # When the caller passes a custom transform, honor it as-is (caller owns normalization).
    # Otherwise, preserve the historical bundled default (ToTensor + dataset-specific Normalize).
    base_transform = (
        transform
        if transform is not None
        else transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    )

    train_ds = datasets.MNIST(root=str(cache), train=True, download=True, transform=base_transform)
    test_ds = datasets.MNIST(root=str(cache), train=False, download=True, transform=base_transform)

    return DagnamDataset(meta, cache, _native_train=train_ds, _native_test=test_ds)


def load_cifar10(
    meta: JsonObject,
    transform: TransformFn | None = None,
    binding: dict[str, Any] | None = None,
) -> DagnamDataset:
    del binding
    from dagnam.data.dataset import DagnamDataset

    datasets, transforms = _load_torchvision()
    cache = SYSTEM_CACHE_ROOT / "cifar10"
    cache.mkdir(parents=True, exist_ok=True)

    base_transform = (
        transform
        if transform is not None
        else transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
    )

    train_ds = datasets.CIFAR10(
        root=str(cache), train=True, download=True, transform=base_transform
    )
    test_ds = datasets.CIFAR10(
        root=str(cache), train=False, download=True, transform=base_transform
    )

    return DagnamDataset(meta, cache, _native_train=train_ds, _native_test=test_ds)


def load_cifar100(
    meta: JsonObject,
    transform: TransformFn | None = None,
    binding: dict[str, Any] | None = None,
) -> DagnamDataset:
    del binding
    from dagnam.data.dataset import DagnamDataset

    datasets, transforms = _load_torchvision()
    cache = SYSTEM_CACHE_ROOT / "cifar100"
    cache.mkdir(parents=True, exist_ok=True)

    base_transform = (
        transform
        if transform is not None
        else transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
    )

    train_ds = datasets.CIFAR100(
        root=str(cache), train=True, download=True, transform=base_transform
    )
    test_ds = datasets.CIFAR100(
        root=str(cache), train=False, download=True, transform=base_transform
    )

    return DagnamDataset(meta, cache, _native_train=train_ds, _native_test=test_ds)


def load_fashion_mnist(
    meta: JsonObject,
    transform: TransformFn | None = None,
    binding: dict[str, Any] | None = None,
) -> DagnamDataset:
    del binding
    from dagnam.data.dataset import DagnamDataset

    datasets, transforms = _load_torchvision()
    cache = SYSTEM_CACHE_ROOT / "fashion_mnist"
    cache.mkdir(parents=True, exist_ok=True)

    base_transform = (
        transform
        if transform is not None
        else transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
        )
    )

    train_ds = datasets.FashionMNIST(
        root=str(cache), train=True, download=True, transform=base_transform
    )
    test_ds = datasets.FashionMNIST(
        root=str(cache), train=False, download=True, transform=base_transform
    )

    return DagnamDataset(meta, cache, _native_train=train_ds, _native_test=test_ds)


def load_imdb(
    meta: JsonObject,
    transform: TransformFn | None = None,
    binding: dict[str, Any] | None = None,
) -> DagnamDataset:
    """Load IMDB via direct npz download (no TensorFlow dependency)."""
    del binding
    import numpy as np

    from dagnam.data.dataset import DagnamDataset

    cache = SYSTEM_CACHE_ROOT / "imdb"
    cache.mkdir(parents=True, exist_ok=True)
    npz_path = cache / "imdb.npz"

    ensure_verified_file(_IMDB_URL, npz_path, IMDB_SHA256)

    # Build a simple polars DataFrame so the existing to_pytorch_loader
    # file-based path can work.  However, we also set _native_train/test
    # as numpy arrays for direct use.
    # The upstream Keras IMDB npz stores ragged review sequences as object arrays,
    # so NumPy requires pickle support. The pinned SHA-256 check above prevents
    # network or cache tampering before this trusted file is deserialized.
    with np.load(str(npz_path), allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

    return DagnamDataset(
        meta,
        cache,
        _native_train=(x_train, y_train),
        _native_test=(x_test, y_test),
    )


def _oxford_pets_mask_target_transform(
    transforms: TorchVisionTransformsModule, size: tuple[int, int]
) -> TransformFn:
    """Build the trimap-mask target transform for Oxford-Pets segmentation (G087).

    The torchvision ``target_types='segmentation'`` target is a PIL trimap whose
    pixels are ``{1: foreground/pet, 2: background, 3: border}``. The supervised
    segmentation train step needs a ``[H, W]`` *integer class index* mask matching
    the resized image (``size`` is the architecture-derived ``(H, W)``, resolved
    from the binding — never a hardcoded resolution — with nearest-neighbour
    interpolation so labels are not blended) and zero-based class ids, so we remap
    ``{1, 2, 3} -> {0, 1, 2}``.
    """
    resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST)
    to_tensor = transforms.PILToTensor()  # uint8 [1, H, W], NOT scaled to [0, 1]

    def _transform(mask: object) -> object:
        tensor: Any = to_tensor(resize(mask))  # torch.Tensor [1, H, W] uint8 in {1, 2, 3}
        return (tensor.squeeze(0).long() - 1).clamp_(0, 2)  # [H, W] long in {0, 1, 2}

    return _transform


_OXFORD_TARGET_MAP: dict[str, str] = {
    "label": "category",
    "segmentation_mask": "segmentation",
}

# Last-resort image size when neither the binding nor the dataset metadata
# declares one (e.g. a no-binding local override). The binding's architecture-
# derived size always wins so any model's spatial dims are honored.
_DEFAULT_OXFORD_IMAGE_SIZE: tuple[int, int] = (224, 224)


def _coerce_hw(value: object) -> tuple[int, int] | None:
    """Coerce a ``[H, W]`` pair (a binding resize/size param) to a positive ``(H, W)``."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    height, width = value
    if (
        isinstance(height, int)
        and not isinstance(height, bool)
        and height > 0
        and isinstance(width, int)
        and not isinstance(width, bool)
        and width > 0
    ):
        return (height, width)
    return None


def _transform_params(binding: dict[str, Any] | None, key: str) -> dict[str, Any]:
    """Return ``binding[key]["params"]`` as a dict, defensively (``{}`` when absent)."""
    section = (binding or {}).get(key)
    params = section.get("params") if isinstance(section, dict) else None
    return params if isinstance(params, dict) else {}


def _oxford_image_size(binding: dict[str, Any] | None, meta: JsonObject) -> tuple[int, int]:
    """Resolve the input image ``(H, W)``: binding (arch input) -> metadata -> default."""
    return (
        _coerce_hw(_transform_params(binding, "input_transform").get("size"))
        or _coerce_hw(meta.get("image_size"))
        or _DEFAULT_OXFORD_IMAGE_SIZE
    )


def _oxford_mask_size(
    binding: dict[str, Any] | None, image_size: tuple[int, int]
) -> tuple[int, int]:
    """Resolve the mask ``(H, W)``: binding (arch output) -> the image size.

    Defaulting to ``image_size`` keeps the mask spatially aligned with the image
    for a spatial-preserving model when the binding omits an explicit target size.
    """
    return _coerce_hw(_transform_params(binding, "target_transform").get("resize")) or image_size


def load_oxford_pets(
    meta: JsonObject,
    transform: TransformFn | None = None,
    binding: dict[str, Any] | None = None,
) -> DagnamDataset:
    from dagnam.data.dataset import DagnamDataset

    datasets, transforms = _load_torchvision()
    cache = SYSTEM_CACHE_ROOT / "oxford_pets"
    cache.mkdir(parents=True, exist_ok=True)

    image_size = _oxford_image_size(binding, meta)
    base_transform = transform
    if base_transform is None:
        base_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )

    target_column = (binding or {}).get("target_column") or "label"
    target_types = _OXFORD_TARGET_MAP.get(str(target_column), "category")
    target_transform = (
        _oxford_pets_mask_target_transform(transforms, _oxford_mask_size(binding, image_size))
        if target_types == "segmentation"
        else None
    )

    try:
        train_ds = datasets.OxfordIIITPet(
            root=str(cache),
            split="trainval",
            target_types=target_types,
            download=True,
            transform=base_transform,
            target_transform=target_transform,
        )
        test_ds = datasets.OxfordIIITPet(
            root=str(cache),
            split="test",
            target_types=target_types,
            download=True,
            transform=base_transform,
            target_transform=target_transform,
        )
    except Exception:
        # Fallback: if torchvision doesn't have OxfordIIITPet, return file-based
        return DagnamDataset(meta, cache)

    return DagnamDataset(meta, cache, _native_train=train_ds, _native_test=test_ds)


def load_speech_commands(
    meta: JsonObject,
    transform: TransformFn | None = None,
    binding: dict[str, Any] | None = None,
) -> DagnamDataset:
    del binding
    from dagnam.data.dataset import DagnamDataset

    cache = SYSTEM_CACHE_ROOT / "speech_commands"
    cache.mkdir(parents=True, exist_ok=True)

    try:
        torchaudio = _load_torchaudio()
        train_ds = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(cache),
            download=True,
            subset="training",
        )
        test_ds = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(cache),
            download=True,
            subset="testing",
        )
        return DagnamDataset(meta, cache, _native_train=train_ds, _native_test=test_ds)
    except (ImportError, Exception):
        # torchaudio not installed or download failed — return file-based
        return DagnamDataset(meta, cache)


def load_wikitext2(
    meta: JsonObject,
    transform: TransformFn | None = None,
    binding: dict[str, Any] | None = None,
) -> DagnamDataset:
    del transform
    from dagnam.data.dataset import DagnamDataset

    cache = SYSTEM_CACHE_ROOT / "wikitext2"
    cache.mkdir(parents=True, exist_ok=True)

    self_supervised = (binding or {}).get("self_supervised")
    if isinstance(self_supervised, dict) and self_supervised.get("kind") == "next_token":
        from dagnam.data.loaders.text_lm import build_lm_sequences

        transform_config = (binding or {}).get("input_transform")
        params = transform_config.get("params", {}) if isinstance(transform_config, dict) else {}
        seq_len = params.get("sequence_length") if isinstance(params, dict) else None
        vocab_size = params.get("vocab_size") if isinstance(params, dict) else None
        raw_path = meta.get("file_path")
        corpus_path = Path(raw_path) if isinstance(raw_path, str) else cache / "wiki.train.tokens"
        text = corpus_path.read_text(encoding="utf-8")
        train = build_lm_sequences(
            text,
            seq_len=seq_len if isinstance(seq_len, int) and seq_len > 0 else 128,
            vocab_size=vocab_size if isinstance(vocab_size, int) and vocab_size > 1 else None,
        )
        native: NativeSplit = cast("NativeSplit", train)
        return DagnamDataset(meta, cache, _native_train=native, _native_test=native)

    try:
        torchtext_datasets = _load_torchtext_datasets()

        # torchtext returns iterators, not map-style datasets
        # Store as native for custom handling
        train_iter = torchtext_datasets.WikiText2(root=str(cache), split="train")
        test_iter = torchtext_datasets.WikiText2(root=str(cache), split="test")
        train_items: NativeSplit = cast("NativeSplit", list(train_iter))
        test_items: NativeSplit = cast("NativeSplit", list(test_iter))
        return DagnamDataset(
            meta,
            cache,
            _native_train=train_items,
            _native_test=test_items,
        )
    except (ImportError, Exception):
        return DagnamDataset(meta, cache)
