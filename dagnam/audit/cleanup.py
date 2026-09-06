"""``audit delete``: remove every platform artifact the state recorded, with a receipt (spec section 10).

Each recorded id goes through the same three moves -- delete, re-read
expecting not-found, record -- driven by :data:`KINDS`, never by a branch per
artifact. A delete that answers not-found is ``already_absent``, so running
the command twice is safe. Local workload files and the secret store go last,
after the platform has confirmed every id is gone.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
from typing import Any, Protocol

from dagnam._core.exceptions import (
    DagnamError,
    DatasetNotFoundError,
    DeploymentNotFoundError,
    ModelNotFoundError,
    ProjectNotFoundError,
)
from dagnam._types import JsonObject
from dagnam.audit.secrets import SecretStore
from dagnam.audit.state import AuditState, load_state
from dagnam.audit.workspace import write_atomic

SCHEMA = "dagnam.audit.deleted/1"
DELETED_FILE = "deleted.json"


class CleanupClient(Protocol):
    """The ``DagnamClient`` methods deletion drives; a fake implements these and no more."""

    def get_deployment(self, deployment_id: str) -> JsonObject:
        """``GET /api/v1/deployments/{id}``; raises ``DeploymentNotFoundError``."""
        ...

    def delete_deployment(self, deployment_id: str) -> JsonObject | None:
        """``DELETE /api/v1/deployments/{id}``."""
        ...

    def get_model_version(self, version_id: str) -> JsonObject:
        """``GET /api/v1/model-versions/{id}``; raises ``ModelNotFoundError``."""
        ...

    def delete_model_entry(self, model_id: str) -> None:
        """``DELETE /api/v1/models/{id}`` (every version with it)."""
        ...

    def get_dataset_meta(self, dataset_id: str, version: str | None = None) -> JsonObject:
        """``GET /api/v1/datasets/{id}/meta``; raises ``DatasetNotFoundError``."""
        ...

    def delete_dataset(self, dataset_id: str) -> None:
        """``DELETE /api/v1/datasets/{id}``."""
        ...

    def get_project(self, project_id: str) -> JsonObject:
        """``GET /api/v1/projects/{id}``; raises ``ProjectNotFoundError``."""
        ...

    def delete_project(self, project_id: str) -> None:
        """``DELETE /api/v1/projects/{id}``."""
        ...


def _delete_model_version(client: CleanupClient, version_id: str) -> None:
    """A version is deleted through its entry (the registry has no per-version delete)."""
    entry_id = client.get_model_version(version_id)["entry_id"]
    client.delete_model_entry(str(entry_id))


type Delete = Callable[[CleanupClient, str], object]
type Get = Callable[[CleanupClient, str], object]

KINDS: tuple[tuple[str, Delete, Get, type[DagnamError]], ...] = (
    (
        "deployment",
        lambda c, i: c.delete_deployment(i),
        lambda c, i: c.get_deployment(i),
        DeploymentNotFoundError,
    ),
    (
        "model_version",
        _delete_model_version,
        lambda c, i: c.get_model_version(i),
        ModelNotFoundError,
    ),
    (
        "dataset",
        lambda c, i: c.delete_dataset(i),
        lambda c, i: c.get_dataset_meta(i),
        DatasetNotFoundError,
    ),
    (
        "project",
        lambda c, i: c.delete_project(i),
        lambda c, i: c.get_project(i),
        ProjectNotFoundError,
    ),
)
"""Deletion order (children before the project): kind, delete, re-read, its not-found error."""


def recorded_ids(state: AuditState) -> dict[str, list[str]]:
    """Every platform id in the state by kind, in deletion order, without duplicates."""
    ids: dict[str, list[str]] = {kind: [] for kind, *_ in KINDS}
    for steps in state.workloads.values():
        for step in steps.values():
            for kind, value in (
                ("deployment", step.deployment_id),
                ("model_version", step.model_version_id),
                ("dataset", step.dataset_id),
            ):
                if value is not None and value not in ids[kind]:
                    ids[kind].append(value)
    if state.project_id is not None:
        ids["project"].append(state.project_id)
    return ids


def _delete_one(
    client: CleanupClient,
    kind: str,
    delete: Delete,
    get: Get,
    absent: type[DagnamError],
    item_id: str,
) -> str:
    try:
        delete(client, item_id)
    except absent:
        return "already_absent"
    try:
        get(client, item_id)
    except absent:
        return "deleted"
    raise RuntimeError(f"{kind} {item_id} still exists after delete")


def delete_audit(audit_dir: Path, client: CleanupClient) -> dict[str, Any]:
    """Delete every recorded artifact, write ``deleted.json``, then drop local rows and secrets.

    Returns the receipt. Raises ``RuntimeError`` when a deleted id can still
    be read back; the receipt is only written once every id is confirmed gone.
    """
    state = load_state(audit_dir)
    items: list[dict[str, Any]] = []
    for kind, delete, get, absent in KINDS:
        for item_id in recorded_ids(state)[kind]:
            status = _delete_one(client, kind, delete, get, absent, item_id)
            items.append({"kind": kind, "id": item_id, "status": status})
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "deleted_at": datetime.now(UTC).isoformat(),
        "items": items,
    }
    write_atomic(audit_dir / DELETED_FILE, json.dumps(receipt, indent=2))
    secrets = SecretStore(audit_dir)
    for steps in state.workloads.values():
        for step in steps.values():
            if step.key_ref is not None:
                secrets.forget(step.key_ref)
    shutil.rmtree(audit_dir / "workloads", ignore_errors=True)
    return receipt


__all__ = ["DELETED_FILE", "KINDS", "SCHEMA", "CleanupClient", "delete_audit", "recorded_ids"]
