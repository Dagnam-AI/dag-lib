"""Core plumbing: exceptions, config, auth, HTTP client, helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dagnam._core.client import DagnamClient

if TYPE_CHECKING:
    from dagnam._core.aio import AsyncDagnamClient

__all__ = ["AsyncDagnamClient", "DagnamClient"]

_LAZY_EXPORTS = {"AsyncDagnamClient": ("dagnam._core.aio", "AsyncDagnamClient")}


def __getattr__(name: str) -> Any:
    """Load the optional async client only when a caller requests it."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
