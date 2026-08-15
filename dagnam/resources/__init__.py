"""API resource modules (hub, projects, deployments, codegen, etc.)."""

from __future__ import annotations

from dagnam.resources import (
    account,
    checkpoints,
    codegen,
    datasets,
    deployments,
    foundation,
    hub,
    inference,
    models,
    projects,
    studio,
    training,
)

datasets_upload = datasets

__all__ = [
    "account",
    "checkpoints",
    "codegen",
    "datasets",
    "datasets_upload",
    "deployments",
    "foundation",
    "hub",
    "inference",
    "models",
    "projects",
    "studio",
    "training",
]
