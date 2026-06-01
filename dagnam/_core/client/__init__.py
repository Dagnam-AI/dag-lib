"""Synchronous HTTP client."""

from __future__ import annotations

from dagnam._core.client.account import AccountClientMixin
from dagnam._core.client.base import BaseDagnamClient
from dagnam._core.client.checkpoints import CheckpointsClientMixin
from dagnam._core.client.codegen import CodegenClientMixin
from dagnam._core.client.datasets import DatasetsClientMixin
from dagnam._core.client.deployments import DeploymentsClientMixin
from dagnam._core.client.hub import HubClientMixin
from dagnam._core.client.inference import InferenceClientMixin
from dagnam._core.client.projects import ProjectsClientMixin
from dagnam._core.client.training import TrainingClientMixin


class DagnamClient(
    AccountClientMixin,
    DatasetsClientMixin,
    InferenceClientMixin,
    CheckpointsClientMixin,
    DeploymentsClientMixin,
    HubClientMixin,
    ProjectsClientMixin,
    CodegenClientMixin,
    TrainingClientMixin,
    BaseDagnamClient,
):
    """Thin wrapper around the Dagnam.AI REST API."""


__all__ = ["DagnamClient"]
