"""Core plumbing: exceptions, config, auth, HTTP client, helpers."""

from __future__ import annotations

try:
    from dagnam._core.aio import AsyncDagnamClient
except ImportError:
    AsyncDagnamClient = None  # type: ignore[assignment]

from dagnam._core.client import DagnamClient

__all__ = ["AsyncDagnamClient", "DagnamClient"]
