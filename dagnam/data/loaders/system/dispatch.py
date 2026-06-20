"""Framework-aware system-dataset resolution.

A training job runs inside a single-framework virtual environment — only one
of torch / tensorflow / jax+flax is installed — so a system dataset must be
loaded with the library native to *that* environment:

* **PyTorch** resolves via ``torchvision``, populating ``_native_train``.
* **TensorFlow** and **Flax** resolve via ``tensorflow_datasets``, populating
  ``_native_train_tf`` / ``_native_train_flax`` (the slots their converters
  consume).

The historical loader was torchvision-only, so TF/Flax training venvs (which
have no torchvision) failed with a misleading ``FileNotFoundError`` after the
``ImportError`` was swallowed. This module dispatches to the right loader,
inferring the framework from the installed libraries when one isn't given.
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

from dagnam._types import JsonObject
from dagnam.data.loaders.system.flax import resolve_system_dataset_flax
from dagnam.data.loaders.system.registry import TransformFn, resolve_system_dataset
from dagnam.data.loaders.system.tensorflow_datasets import resolve_system_dataset_tf

if TYPE_CHECKING:
    from dagnam.data.dataset import DagnamDataset

# Framework identifiers used across dispatch.
PYTORCH = "pytorch"
TENSORFLOW = "tensorflow"
FLAX = "flax"


def detect_installed_framework() -> str:
    """Infer the deep-learning framework available in this environment.

    The detection order is deliberate:

    * ``torchvision`` present → :data:`PYTORCH`. It is the historical default
      and the only loader that populates the PyTorch-native slot.
    * ``jax`` present (with ``tensorflow_datasets``) → :data:`FLAX`. A Flax
      venv carries JAX *and* tfds; checking JAX first distinguishes it from a
      plain TensorFlow venv (both ship tfds).
    * ``tensorflow_datasets`` present → :data:`TENSORFLOW`.

    Falls back to :data:`PYTORCH` when nothing framework-specific is
    importable so the torchvision path raises a clear, dependency-naming
    ``ImportError`` instead of a misleading one.
    """
    if find_spec("torchvision") is not None:
        return PYTORCH
    if find_spec("jax") is not None and find_spec("tensorflow_datasets") is not None:
        return FLAX
    if find_spec("tensorflow_datasets") is not None:
        return TENSORFLOW
    return PYTORCH


def load_system_dataset(
    meta: JsonObject,
    *,
    framework: str | None = None,
    transform: TransformFn | None = None,
    binding: dict[str, object] | None = None,
) -> DagnamDataset:
    """Resolve a system dataset with the loader native to *framework*.

    When *framework* is ``None`` it is inferred from the installed libraries
    via :func:`detect_installed_framework`. The returned ``DagnamDataset`` has
    the framework-native split populated (``_native_train`` for PyTorch,
    ``_native_train_tf`` for TensorFlow, ``_native_train_flax`` for Flax), so
    the matching ``to_*`` converter takes its native fast path.

    ``transform`` is honored only by the PyTorch/torchvision loader; the
    tfds-backed TensorFlow and Flax loaders apply their own normalization.
    """
    resolved = framework or detect_installed_framework()
    if resolved == TENSORFLOW:
        return resolve_system_dataset_tf(meta)
    if resolved == FLAX:
        return resolve_system_dataset_flax(meta)
    if binding is None:
        return resolve_system_dataset(meta, transform)
    return resolve_system_dataset(meta, transform, binding)
