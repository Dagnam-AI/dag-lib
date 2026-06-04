"""Coverage for the system-dataset registry resolver."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from tests.data.loaders._system_fakes import (
    identity_transform,
)

from dagnam.data.loaders.system import torchvision as tv_mod
from dagnam.data.loaders.system.registry import resolve_system_dataset

if TYPE_CHECKING:
    from tests.typing_helpers import JsonObject, ObjectTransform, PytestMonkeyPatch


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
    result = resolve_system_dataset({"name": "MNIST"}, transform=identity_transform)
    assert result == "FAKE_DS"
    assert called["transform"] is identity_transform


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
