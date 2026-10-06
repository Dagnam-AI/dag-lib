"""Receipt rows exactly as the platform writes them: ``kind``, ``id``, ``status``, ``code``, ``reason``.

Each constant is a row (or its reason) the platform's delete and cancel produce
today, one per ``code`` its receipt documents. A test that feeds one of these
through the client is testing the shape that ships, not a stand-in for it.
"""

from __future__ import annotations

from dagnam._types import JsonObject, JsonValue

# The reasons the platform words its decisions with; a client never matches on them.
NOT_OURS = "project was not created by this audit"
NOT_ONLY_OURS = "project holds artifacts this audit did not record"
PROJECT_HELD = "the project was kept because something in it was kept or not removed"
ENTRY_SHARED = "registry entry holds versions this audit did not record"
WEIGHTS_SERVED = "a live deployment serves these weights; delete it, then run the delete again"
NOT_IN_PROJECT = "not in this audit's project; it was left alone"
WEIGHTS_KEPT = "the trained weights could not be removed from storage; run the delete again"
HAS_SERVED = "the deployment has served before and is rolling out again; it was left as it is"
DATASET_HELD = "Dataset is referenced by a training run and cannot be deleted"

SCHEMA_DELETED = "dagnam.audit.deleted/1"
SCHEMA_CANCELLED = "dagnam.audit.cancelled/1"


def row(
    kind: str, item_id: str, status: str, code: str | None = None, reason: str | None = None
) -> JsonObject:
    """One receipt entry; ``code`` defaults to the status for the three plain outcomes."""
    plain = status in ("deleted", "already_absent", "stopped")
    return {
        "kind": kind,
        "id": item_id,
        "status": status,
        "code": code if code is not None else (status if plain else None),
        "reason": reason,
    }


def legacy(entry: JsonObject) -> JsonObject:
    """The same entry as a platform that predates ``code`` writes it."""
    return {key: value for key, value in entry.items() if key != "code"}


def receipt(*entries: JsonObject, schema: str = SCHEMA_DELETED) -> JsonObject:
    """A whole receipt, in the platform's wire shape."""
    rows: list[JsonValue] = list(entries)
    return {"schema": schema, "deleted_at": "2026-10-03T21:00:00Z", "entries": rows}


# One row per code of the platform's table: (the entry, what its documented meaning is).
DELETED_ROW = row("dataset", "ds-1", "deleted")
ALREADY_ABSENT_ROW = row("deployment", "dep-1", "already_absent")
STOPPED_ROW = row("training_job", "job-1", "stopped")
PROJECT_NOT_OURS_ROW = row("project", "proj-1", "kept", "project_not_ours", NOT_OURS)
PROJECT_SHARED_ROW = row("project", "proj-1", "kept", "project_shared", NOT_ONLY_OURS)
PROJECT_HELD_ROW = row("project", "proj-1", "kept", "project_held", PROJECT_HELD)
ENTRY_SHARED_ROW = row("model_entry", "entry-1", "kept", "entry_shared", ENTRY_SHARED)
WEIGHTS_SERVED_ROW = row("model_version", "mv-1", "kept", "weights_served", WEIGHTS_SERVED)
NOT_IN_PROJECT_ROW = row("deployment", "dep-9", "kept", "not_in_project", NOT_IN_PROJECT)
WEIGHTS_NOT_REMOVED_ROW = row(
    "model_version", "mv-1", "blocked", "weights_not_removed", WEIGHTS_KEPT
)
ALREADY_STOPPED_ROW = row("training_job", "job-1", "stopped", "already_stopped")
IN_USE_ELSEWHERE_ROW = row(
    "dataset", "ds-1", "kept", "in_use_elsewhere", "another training run still uses it"
)
NOT_CREATED_HERE_ROW = row("dataset", "ds-9", "kept", "not_created_here", "recorded, not tagged")
FAILED_ROW = row("dataset", "ds-1", "blocked", "failed", "an internal error in the step")
HAS_SERVED_ROW = row("deployment", "dep-1", "blocked", "has_served", HAS_SERVED)
REFUSED_ROW = row("dataset", "ds-1", "blocked", "refused", DATASET_HELD)


def designed(
    *entries: JsonObject, status: str = "deleted", schema: str = SCHEMA_DELETED
) -> JsonObject:
    """A whole receipt in the designed platform's shape: ``audit_status`` and ``attempt`` as well."""
    return {**receipt(*entries, schema=schema), "audit_status": status, "attempt": 1}


def receipt_rows_of(whole: JsonObject) -> list[JsonObject]:
    """The rows of a receipt built by :func:`receipt`."""
    rows = whole["entries"]
    assert isinstance(rows, list)
    return [row for row in rows if isinstance(row, dict)]
