"""Async client facade for Dagnam.AI.

Requires ``pip install 'dagnam[aio]'``.
"""

from __future__ import annotations

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.sse import SSEEvent

__all__ = ["AsyncDagnamClient", "SSEEvent"]
