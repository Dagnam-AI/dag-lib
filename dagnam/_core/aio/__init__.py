"""Async HTTP client implementation."""

from __future__ import annotations

from dagnam._core.aio.account import AsyncAccountMixin
from dagnam._core.aio.base import BaseAsyncDagnamClient
from dagnam._core.aio.checkpoints import AsyncCheckpointsMixin
from dagnam._core.aio.codegen import AsyncCodegenMixin
from dagnam._core.aio.datasets import AsyncDatasetsMixin
from dagnam._core.aio.deployments import AsyncDeploymentsMixin
from dagnam._core.aio.foundation import AsyncFoundationMixin
from dagnam._core.aio.hub import AsyncHubMixin
from dagnam._core.aio.inference import AsyncInferenceMixin
from dagnam._core.aio.models import AsyncModelsMixin
from dagnam._core.aio.projects import AsyncProjectsMixin
from dagnam._core.aio.training import AsyncTrainingMixin


class AsyncDagnamClient(
    AsyncAccountMixin,
    AsyncDatasetsMixin,
    AsyncInferenceMixin,
    AsyncCheckpointsMixin,
    AsyncDeploymentsMixin,
    AsyncHubMixin,
    AsyncModelsMixin,
    AsyncProjectsMixin,
    AsyncCodegenMixin,
    AsyncTrainingMixin,
    AsyncFoundationMixin,
    BaseAsyncDagnamClient,
):
    """Async wrapper around the Dagnam.AI REST API using ``httpx``."""


__all__ = ["AsyncDagnamClient"]
