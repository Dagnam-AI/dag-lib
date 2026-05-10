"""Shared DagnamClient resolution used by inference/checkpoints/training modules.

Centralizes the ``client | api_key/api_url | auth-chain fallback`` pattern so
each public module does not reimplement it.
"""

from __future__ import annotations

from typing import Optional

from dagnam._core.auth import get_api_key, get_api_url
from dagnam._core.client import DagnamClient


def resolve_client(
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> DagnamClient:
    """Return an explicit client, or build one from the auth-resolution chain."""
    if client is not None:
        return client
    return DagnamClient(get_api_url(override=api_url), get_api_key(override=api_key))
