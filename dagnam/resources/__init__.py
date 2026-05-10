"""API resource modules (hub, projects, deployments, codegen, etc.)."""

from __future__ import annotations

from dagnam.resources import (
    checkpoints,
    codegen,
    datasets,
    deployments,
    hub,
    inference,
    projects,
    training,
)

datasets_upload = datasets

__all__ = [
    "checkpoints",
    "codegen",
    "datasets",
    "datasets_upload",
    "deployments",
    "hub",
    "inference",
    "projects",
    "training",
]
