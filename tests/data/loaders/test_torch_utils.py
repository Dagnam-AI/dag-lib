"""Coverage for dagnam.data.loaders.torch_utils."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from dagnam.data.loaders.torch_utils import should_pin_memory


def test_should_pin_memory_uses_accelerator_when_available() -> None:
    fake_torch = SimpleNamespace(accelerator=SimpleNamespace(is_available=lambda: True))
    with mock.patch.dict("sys.modules", {"torch": fake_torch}):
        assert should_pin_memory() is True


def test_should_pin_memory_swallows_accelerator_failure() -> None:
    def _raise() -> None:
        raise RuntimeError("boom")

    fake_torch = SimpleNamespace(
        accelerator=SimpleNamespace(is_available=_raise),
        cuda=SimpleNamespace(is_available=lambda: False),
        xpu=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    with mock.patch.dict("sys.modules", {"torch": fake_torch}):
        assert should_pin_memory() is False


def test_should_pin_memory_falls_back_to_cuda() -> None:
    fake_torch = SimpleNamespace(
        accelerator=None,
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    with mock.patch.dict("sys.modules", {"torch": fake_torch}):
        assert should_pin_memory() is True


def test_should_pin_memory_falls_back_to_xpu() -> None:
    fake_torch = SimpleNamespace(
        accelerator=None,
        cuda=SimpleNamespace(is_available=lambda: False),
        xpu=SimpleNamespace(is_available=lambda: True),
    )
    with mock.patch.dict("sys.modules", {"torch": fake_torch}):
        assert should_pin_memory() is True


def test_should_pin_memory_falls_back_to_mps() -> None:
    fake_torch = SimpleNamespace(
        accelerator=None,
        cuda=SimpleNamespace(is_available=lambda: False),
        xpu=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    with mock.patch.dict("sys.modules", {"torch": fake_torch}):
        assert should_pin_memory() is True


def test_should_pin_memory_returns_false_when_nothing_available() -> None:
    fake_torch = SimpleNamespace(
        accelerator=None,
        cuda=SimpleNamespace(is_available=lambda: False),
        xpu=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    with mock.patch.dict("sys.modules", {"torch": fake_torch}):
        assert should_pin_memory() is False


def test_should_pin_memory_handles_missing_attributes() -> None:
    """When the optional accelerator/cuda/xpu/mps attrs don't exist at all."""
    fake_torch = SimpleNamespace()
    with mock.patch.dict("sys.modules", {"torch": fake_torch}):
        assert should_pin_memory() is False
