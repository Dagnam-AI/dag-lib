"""Data steps: upload the derived rows, read the sniffed format, split, and check the server's PII scan.

Everything here runs once per candidate that trains; the hosted floor never
reaches these steps. What is uploaded is exactly ``workloads/<id>/dataset.jsonl``
-- redacted, deduplicated rows Task 6 wrote -- and ``split.json`` verbatim,
with the holdout named ``validation`` because that is the split name the
classifier recipe's emitted script reads (G0 finding, spec section 2).
"""

from __future__ import annotations

import json

from dagnam._types import JsonObject
from dagnam.audit.state import AuditState
from dagnam.audit.steps import StepContext, required, string_field, wait_for, wait_task

DATASET_TYPE = "text"
FILE_FORMAT = "json"
"""The upload ``format`` is the FILE format; the row format is sniffed into the version."""
SPLIT_NAMES = {"train": "train", "eval_holdout": "validation"}
"""Task 6's split names -> the platform's."""


def upload(state: AuditState, ctx: StepContext) -> AuditState:
    """Create the dataset from ``dataset.jsonl``; done once ``dataset_id`` is set."""
    step = ctx.step(state)
    if step.dataset_id is not None:
        return state
    created = ctx.client.upload_dataset(
        file_path=ctx.workload_dir / "dataset.jsonl",
        name=f"audit-{ctx.workload_id}-{ctx.spec.kind.value}",
        dataset_type=DATASET_TYPE,
        format=FILE_FORMAT,
        description=f"workload audit {ctx.label}: derived, redacted rows",
    )
    step.dataset_id = str(created["id"])
    return state


def _newest_version(ctx: StepContext, dataset_id: str) -> JsonObject:
    """The highest-numbered version as a pollable payload: ``ready`` once sniffed and counted."""
    versions = ctx.client.list_dataset_versions(dataset_id)
    if not versions:
        return {"status": "pending"}
    newest = max(versions, key=lambda v: int(str(v.get("version_number") or 0)))
    ready = bool(newest.get("num_samples")) and bool(newest.get("data_format"))
    return {**newest, "status": "ready" if ready else "pending"}


def resolve_version(state: AuditState, ctx: StepContext) -> AuditState:
    """Wait for the upload's version to be sniffed; it must be the recipe's row format."""
    step = ctx.step(state)
    if step.version_id is not None:
        return state
    dataset_id = required(step.dataset_id, "dataset_id")
    version = wait_for(
        ctx,
        lambda: _newest_version(ctx, dataset_id),
        success={"ready"},
        failure=set(),
        timeout=ctx.task_timeout,
        name=f"dataset {dataset_id} sniff",
    )
    expected = str(ctx.meta()["stats"]["format_key"])
    sniffed = string_field(version, "data_format")
    if sniffed != expected:
        step.error = f"format_mismatch: platform sniffed {sniffed!r}, the recipe needs {expected!r}"
        return state
    step.version_id = str(version["id"])
    return state


def split(state: AuditState, ctx: StepContext) -> AuditState:
    """Submit ``split.json`` as an explicit split; done once the task is enqueued."""
    step = ctx.step(state)
    if step.split_task_id is not None:
        return state
    body = json.loads((ctx.workload_dir / "split.json").read_text(encoding="utf-8"))
    memberships: dict[str, list[int]] = {}
    for raw_name, indices in body["member_row_indices"].items():
        name = str(raw_name)
        memberships[SPLIT_NAMES.get(name, name)] = [int(i) for i in indices]
    task = ctx.client.create_explicit_splits(
        required(step.dataset_id, "dataset_id"),
        required(step.version_id, "version_id"),
        memberships,
    )
    step.split_task_id = str(task["task_id"])
    return state


def wait_split(state: AuditState, ctx: StepContext) -> AuditState:
    """Wait for the split; the NEW version it materializes is the one the run trains on."""
    step = ctx.step(state)
    if step.split_done:
        return state
    result = wait_task(ctx, required(step.split_task_id, "split_task_id"))
    if result.get("status") == "rejected":
        step.error = f"split_rejected: {result.get('reason')}"
        return state
    new_version = string_field(result, "new_version_id")
    if new_version is not None:
        step.version_id = new_version
    step.split_done = True
    return state


def pii_scan(state: AuditState, ctx: StepContext) -> AuditState:
    """Enqueue the server's report-only PII scan of the split version."""
    step = ctx.step(state)
    if step.pii_task_id is not None:
        return state
    task = ctx.client.scan_pii(
        required(step.dataset_id, "dataset_id"), required(step.version_id, "version_id")
    )
    step.pii_task_id = str(task["task_id"])
    return state


def wait_pii(state: AuditState, ctx: StepContext) -> AuditState:
    """The two implementations must agree (spec section 9); a disagreement stops the workload.

    The rows were redacted client-side before upload, so agreement means the
    server found no residual finding in any class and looked for every class
    the client's ``meta.json`` pass list names.
    """
    step = ctx.step(state)
    if step.pii_agrees is not None:
        return state
    result = wait_task(ctx, required(step.pii_task_id, "pii_task_id"))
    counts = result.get("counts_by_code")
    found = (
        {str(code): int(str(n)) for code, n in counts.items()} if isinstance(counts, dict) else {}
    )
    residual = {code: n for code, n in found.items() if n > 0}
    server_pass = result.get("pass_list")
    looked_for = {str(c) for c in server_pass} if isinstance(server_pass, list) else set()
    local_pass = {str(c) for c in ctx.meta()["stats"]["redact"]["pass_list"]}
    missing = sorted(local_pass - looked_for)
    step.pii_agrees = not residual and not missing
    if not step.pii_agrees:
        step.error = (
            f"pii_disagreement: server found {residual} after client redaction; "
            f"classes the server did not scan: {missing}"
        )
    return state


__all__ = [
    "DATASET_TYPE",
    "FILE_FORMAT",
    "SPLIT_NAMES",
    "pii_scan",
    "resolve_version",
    "split",
    "upload",
    "wait_pii",
    "wait_split",
]
