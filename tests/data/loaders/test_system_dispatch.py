"""Coverage for framework-aware system-dataset dispatch."""

from __future__ import annotations

from importlib.machinery import ModuleSpec
from typing import TYPE_CHECKING

from dagnam.data.loaders.system import dispatch

if TYPE_CHECKING:
    from tests.typing_helpers import JsonObject, PytestMonkeyPatch

_META: JsonObject = {"name": "MNIST"}


def _fake_find_spec(*present: str):
    """Return a ``find_spec`` stand-in that reports *present* modules as installed."""
    installed = set(present)

    def _find_spec(name: str):
        if name in installed:
            return ModuleSpec(name, loader=None)
        return None

    return _find_spec


# ---------------------------------------------------------- detection


def test_detect_pytorch_when_torchvision_present(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setattr(dispatch, "find_spec", _fake_find_spec("torchvision"))
    assert dispatch.detect_installed_framework() == dispatch.PYTORCH


def test_detect_flax_when_jax_and_tfds_present(monkeypatch: PytestMonkeyPatch) -> None:
    # A Flax venv carries JAX + tfds (and no torchvision); JAX disambiguates it
    # from a plain TensorFlow venv.
    monkeypatch.setattr(dispatch, "find_spec", _fake_find_spec("jax", "tensorflow_datasets"))
    assert dispatch.detect_installed_framework() == dispatch.FLAX


def test_detect_tensorflow_when_only_tfds_present(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setattr(dispatch, "find_spec", _fake_find_spec("tensorflow_datasets"))
    assert dispatch.detect_installed_framework() == dispatch.TENSORFLOW


def test_detect_jax_without_tfds_is_not_flax(monkeypatch: PytestMonkeyPatch) -> None:
    # JAX present but no tfds: cannot load system datasets via tfds, so it is not
    # treated as a Flax env and falls back to the torchvision (PyTorch) path.
    monkeypatch.setattr(dispatch, "find_spec", _fake_find_spec("jax"))
    assert dispatch.detect_installed_framework() == dispatch.PYTORCH


def test_detect_defaults_to_pytorch_when_nothing_installed(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setattr(dispatch, "find_spec", _fake_find_spec())
    assert dispatch.detect_installed_framework() == dispatch.PYTORCH


# ---------------------------------------------------------- dispatch


def test_load_routes_to_tensorflow(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setattr(dispatch, "resolve_system_dataset_tf", lambda meta: ("TF", meta))
    assert dispatch.load_system_dataset(_META, framework="tensorflow") == ("TF", _META)


def test_load_routes_to_flax(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setattr(dispatch, "resolve_system_dataset_flax", lambda meta: ("FLAX", meta))
    assert dispatch.load_system_dataset(_META, framework="flax") == ("FLAX", _META)


def test_load_routes_to_pytorch_with_transform(monkeypatch: PytestMonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_torch(meta: JsonObject, transform: object | None = None):
        captured["meta"] = meta
        captured["transform"] = transform
        return "TORCH"

    def identity(value: object) -> object:
        return value

    monkeypatch.setattr(dispatch, "resolve_system_dataset", fake_torch)
    result = dispatch.load_system_dataset(_META, framework="pytorch", transform=identity)
    assert result == "TORCH"
    assert captured["meta"] is _META
    assert captured["transform"] is identity


def test_load_infers_framework_when_unset(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setattr(dispatch, "detect_installed_framework", lambda: dispatch.TENSORFLOW)
    monkeypatch.setattr(dispatch, "resolve_system_dataset_tf", lambda meta: ("TF", meta))
    assert dispatch.load_system_dataset(_META) == ("TF", _META)


def test_load_routes_to_pytorch_with_binding(monkeypatch: PytestMonkeyPatch) -> None:
    """A non-None binding is threaded to the native resolver (pytorch path)."""
    captured: dict[str, object] = {}

    def fake_torch(
        meta: JsonObject,
        transform: object | None = None,
        binding: dict[str, object] | None = None,
    ):
        captured["binding"] = binding
        return "TORCH"

    monkeypatch.setattr(dispatch, "resolve_system_dataset", fake_torch)
    payload: dict[str, object] = {"target_column": "label"}
    result = dispatch.load_system_dataset(_META, framework="pytorch", binding=payload)
    assert result == "TORCH"
    assert captured["binding"] is payload
