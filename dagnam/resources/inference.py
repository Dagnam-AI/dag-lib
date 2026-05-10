"""Inference client — call deployed models from Python.

Thin wrappers over the Dagnam.AI inference API that reuse the existing
``DagnamClient`` and auth-resolution chain (``DAGNAM_API_KEY`` env var,
config file, ``dagnam.configure()``, or explicit override).
"""

from __future__ import annotations

from typing import Optional

from dagnam._core.client import DagnamClient
from dagnam._core.resolver import resolve_client


def inference(
    deployment_id: str,
    inputs: dict,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    timeout: int = 30,
) -> dict:
    """Call a deployed model's /predict endpoint.

    >>> result = dagnam.inference("dep_abc123", {"text": "hello"})
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.predict(deployment_id, inputs, timeout=timeout)


def inference_batch(
    deployment_id: str,
    inputs: list,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    timeout: int = 30,
) -> list:
    """Batch-predict against a deployed model.

    >>> results = dagnam.inference_batch("dep_abc123", [{"x": 1}, {"x": 2}])
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.predict_batch(deployment_id, inputs, timeout=timeout)


def deployment_health(
    deployment_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> dict:
    """Check a deployment's health status.

    >>> health = dagnam.deployment_health("dep_abc123")
    >>> health["status"]
    'healthy'
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.deployment_health(deployment_id)
