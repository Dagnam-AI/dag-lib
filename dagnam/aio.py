"""Async client facade for Dagnam.AI.

Requires ``pip install 'dagnam[aio]'``.
"""

from __future__ import annotations

from dagnam._core.aio import AsyncDagnamClient

__all__ = ["AsyncDagnamClient"]
