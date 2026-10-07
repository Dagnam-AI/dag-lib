"""Inference client — call deployed models from Python.

Thin wrappers over the Dagnam.AI inference API that reuse the existing
``DagnamClient`` and auth-resolution chain (``DAGNAM_API_KEY`` env var,
config file, ``dagnam.configure()``, or explicit override).

``inference``, ``inference_batch``, ``deployment_health`` and
``inference_schema`` authenticate with the deployment's own key (the
``api_key`` a deployment returns when it is created or its key is rotated), so
pass ``api_key=<deployment key>``; an account key is refused. ``inference_stream``
uses the account key.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Optional

from dagnam._core.client import DagnamClient
from dagnam._core.client.base import DEFAULT_PREDICT_TIMEOUT
from dagnam._core.resolver import resolve_client
from dagnam._core.sse import TERMINAL_INFERENCE_EVENTS, SSEEvent, iter_sse_once
from dagnam._types import JsonArray, JsonObject

if TYPE_CHECKING:
    import requests


def inference(
    deployment_id: str,
    inputs: JsonObject,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    timeout: int = DEFAULT_PREDICT_TIMEOUT,
) -> JsonObject:
    """Call a deployed model's /predict endpoint with ``inputs`` as the model input.

    A first call after idle can wait out a cold start of several minutes, so
    ``timeout`` defaults to 600 seconds; lower it to fail fast.

    >>> result = dagnam.inference("dep_abc123", {"text": "hello"}, api_key=deployment_key)
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.predict(deployment_id, inputs, timeout=timeout)


def inference_batch(
    deployment_id: str,
    inputs: JsonArray,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    timeout: int = DEFAULT_PREDICT_TIMEOUT,
) -> JsonArray:
    """Batch-predict against a deployed model.

    >>> results = dagnam.inference_batch(
    ...     "dep_abc123", [{"text": "a"}, {"text": "b"}], api_key=deployment_key
    ... )
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.predict_batch(deployment_id, inputs, timeout=timeout)


def deployment_health(
    deployment_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Check a deployment's health status.

    >>> health = dagnam.deployment_health("dep_abc123", api_key=deployment_key)
    >>> health["health_status"]
    'healthy'
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.deployment_health(deployment_id)


def inference_schema(
    deployment_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Return the input/output schema for a deployment's inference endpoint.

    >>> schema = dagnam.inference_schema("dep_abc123", api_key=deployment_key)
    >>> schema["input_schema"]
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.schema(deployment_id)


def inference_stream(
    deployment_id: str,
    inputs: JsonObject,
    *,
    include_heartbeats: bool = False,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> Iterator[SSEEvent]:
    """Stream a prediction token-by-token from a text/LLM deployment.

    Yields ``SSEEvent``s: ``token`` frames, then a terminal ``complete`` (or
    ``error``). Single-shot: a dropped connection raises ``StreamError``
    rather than reconnecting, because replaying generation would duplicate
    output. Auth uses a short-lived scoped stream token minted per
    connection — the API key never appears in a URL.

    >>> for ev in dagnam.inference_stream("dep_abc123", {"text": "hello"}):
    ...     if ev.event == "token":
    ...         print(ev.data["token"], end="", flush=True)
    """
    resolved = resolve_client(client, api_key, api_url)

    def _open() -> requests.Response:
        return resolved.open_inference_stream(deployment_id, inputs)

    return iter_sse_once(
        _open,
        terminal_events=TERMINAL_INFERENCE_EVENTS,
        include_heartbeats=include_heartbeats,
        resource_label=f"Inference stream for {deployment_id}",
    )
