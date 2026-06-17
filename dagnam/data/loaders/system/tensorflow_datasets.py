"""TensorFlow Datasets-backed system dataset loaders."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from dagnam._types import JsonObject, TensorflowDataset
from dagnam.data.loaders.system.common import SYSTEM_CACHE_ROOT
from dagnam.data.loaders.system.registry import resolve_system_dataset

if TYPE_CHECKING:
    from dagnam.data.dataset import DagnamDataset

TFDS_NAME_MAP: dict[str, str] = {
    "mnist": "mnist",
    "mnist handwritten digits": "mnist",
    "cifar-10": "cifar10",
    "cifar10": "cifar10",
    "cifar-100": "cifar100",
    "cifar100": "cifar100",
    "fashion-mnist": "fashion_mnist",
    "fashionmnist": "fashion_mnist",
    "fashion_mnist": "fashion_mnist",
    "imdb": "imdb_reviews",
}


class TensorflowDatasetsModule(Protocol):
    """tensorflow_datasets surface used by this adapter."""

    def load(
        self,
        name: str,
        *,
        split: str,
        as_supervised: bool,
        data_dir: str,
    ) -> TensorflowDataset: ...


class _TruststoreModule(Protocol):
    """``truststore`` surface used to route tfds downloads through the OS store."""

    def inject_into_ssl(self) -> None: ...


class _TfTensor(Protocol):
    """A TensorFlow tensor surface supporting the scaling division."""

    def __truediv__(self, other: float) -> _TfTensor: ...


class _TfCastModule(Protocol):
    """``tensorflow`` surface used to cast/scale image features."""

    uint8: object
    float32: object

    def cast(self, value: object, dtype: object) -> _TfTensor: ...


def _normalize_image_features(dataset: TensorflowDataset) -> TensorflowDataset:
    """Cast uint8 image features to float32 in ``[0, 1]``.

    ``tfds`` ``as_supervised`` yields raw uint8 images, but the generated models
    have float32 convolution kernels (and the PyTorch and Flax loaders both
    normalize to ``[0, 1]``), so a uint8 forward pass raises an "Incompatible
    type conversion" error. Non-image features (e.g. text datasets such as
    ``imdb_reviews``) keep their original dtype and are returned unchanged.
    """
    spec = getattr(dataset, "element_spec", None)
    feature_spec = spec[0] if isinstance(spec, tuple) and len(spec) >= 1 else None
    dtype = getattr(feature_spec, "dtype", None)
    if getattr(dtype, "name", None) != "uint8":
        return dataset

    tf = cast("_TfCastModule", import_module("tensorflow"))

    def _to_float(features: object, label: object) -> tuple[_TfTensor, object]:
        return tf.cast(features, tf.float32) / 255.0, label

    return dataset.map(_to_float)


def ensure_system_trust() -> None:
    """Make ``tensorflow_datasets`` downloads trust the OS certificate store.

    tfds downloads with ``requests`` (the certifi CA bundle), so — unlike
    torchvision's urllib path, which uses the OS trust store — it rejects
    certificates injected by corporate TLS-inspection proxies with
    CERTIFICATE_VERIFY_FAILED, failing exactly where the PyTorch loader
    succeeds. ``truststore`` routes verification through the OS trust store,
    making tfds behave like torchvision. Best-effort: when ``truststore`` is not
    installed (a clean network does not need it) this is a no-op.
    """
    try:
        truststore = cast("_TruststoreModule", import_module("truststore"))
    except ImportError:
        return
    truststore.inject_into_ssl()


def _load_tfds() -> TensorflowDatasetsModule:
    return cast("TensorflowDatasetsModule", import_module("tensorflow_datasets"))


def _load_supervised_split(tfds: Any, name: str, split: str, cache: Path) -> TensorflowDataset:
    try:
        return cast(
            "TensorflowDataset",
            tfds.load(name, split=split, as_supervised=True, data_dir=str(cache)),
        )
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        # Positional fallback for older tensorflow_datasets whose `load`
        # signature predates keyword-only args; `True` is `as_supervised`.
        return cast("TensorflowDataset", tfds.load(name, split, True, cache))  # noqa: FBT003


def resolve_tfds_name(meta: JsonObject) -> str | None:
    """Return the tensorflow_datasets name for a system dataset, or None."""
    raw_name = meta.get("name", "")
    if not isinstance(raw_name, str):
        return None
    name = raw_name.lower()
    if name in TFDS_NAME_MAP:
        return TFDS_NAME_MAP[name]
    for key, tfds_name in TFDS_NAME_MAP.items():
        if key in name or name in key:
            return tfds_name
    return None


def resolve_system_dataset_tf(meta: JsonObject) -> DagnamDataset:
    """Load a system dataset as a native ``tf.data.Dataset`` via ``tensorflow_datasets``.

    Falls back to the PyTorch native loader (which is then converted
    in-memory by ``DagnamDataset._native_to_tensorflow``) if ``tfds`` is not
    installed or the dataset is not recognized.
    """
    from dagnam.data.dataset import DagnamDataset

    tfds_name = resolve_tfds_name(meta)
    if tfds_name is None:
        # Fall back to PT native + in-memory conversion.
        return resolve_system_dataset(meta)

    try:
        tfds = _load_tfds()
    except ImportError:
        # Fall back — caller uses _native_to_tensorflow on PT native.
        return resolve_system_dataset(meta)

    ensure_system_trust()
    cache = SYSTEM_CACHE_ROOT / tfds_name
    cache.mkdir(parents=True, exist_ok=True)

    train_ds = _normalize_image_features(_load_supervised_split(tfds, tfds_name, "train", cache))
    test_ds = _normalize_image_features(_load_supervised_split(tfds, tfds_name, "test", cache))

    return DagnamDataset(
        meta,
        cache,
        _native_train_tf=train_ds,
        _native_test_tf=test_ds,
    )
