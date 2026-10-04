"""Data steps: upload the derived rows, read the sniffed format, split, and check the server's PII scan.

Everything here runs once per candidate that trains; the hosted floor never
reaches these steps. What is uploaded is exactly ``workloads/<id>/dataset.jsonl``
-- the redacted, deduplicated rows the scan derived -- and ``split.json``
verbatim, with the holdout named ``validation`` because that is the split name
the classifier recipe's emitted script reads.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from dagnam.audit.preflight import UPGRADE, installed_contract, version_gap
from dagnam.audit.steps import (
    StepContext,
    answer_field,
    regular_path,
    required,
    string_field,
    wait_for,
    wait_task,
)
from dagnam.audit.workspace import read_regular

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping
    from pathlib import Path

    from dagnam._types import JsonObject
    from dagnam.audit.state import AuditState

LOG = logging.getLogger("dagnam.audit")
DATASET_TYPE = "text"
FILE_FORMAT = "json"
"""The upload ``format`` is the FILE format; the row format is sniffed into the version."""
SPLIT_NAMES = {"train": "train", "eval_holdout": "validation"}
"""The scan's split names -> the platform's."""
PROCESSING_FAILED = "failed"
"""``analysis_status`` once the platform has given up on processing the upload."""


def resume_key(state: AuditState, ctx: StepContext) -> str | None:
    """What names an UNPUBLISHED candidate's upload across runs: the directory's nonce and the candidate.

    Not the file, the run's flags or the version of this SDK -- nothing a rerun may change.
    ``None`` for a published audit, whose uploads are tagged with the audit instead (the
    description then carries no key), and for a state with no nonce.
    """
    if state.audit_id is not None or state.project_nonce is None:
        return None
    return f"{state.project_nonce}/{ctx.label}"


def _same_rows(found: JsonObject, path: Path) -> bool:
    """Whether a listed dataset is the very file ``path``: the integrity fields the listing carries.

    A dataset resource exposes no content digest, so what it does expose is compared: the stored
    size in bytes must equal the file's, and, once the platform has analysed the upload, its
    sample count must equal the file's rows. A row with no size is never the same.
    """
    size = found.get("size_bytes")
    if isinstance(size, bool) or size != path.stat().st_size:
        return False
    samples = found.get("num_samples")
    if found.get("analysis_status") == "completed" and isinstance(samples, int) and samples > 0:
        return samples == sum(1 for line in read_regular(path).splitlines() if line.strip())
    return True


def _adopt_tagged(state: AuditState, ctx: StepContext, name: str) -> str | None:
    """The dataset a published audit's lost upload made, if the platform can PROVE it is that one.

    Nothing the listing is asked for is trusted: a platform may ignore the ``audit_id`` filter and
    list every dataset the caller can see, another account's public ones included. A row is
    adopted only when it ITSELF says it was made for this audit (``audit_id``), by this audit's
    owner (``owner_id`` of the audit), under this name, and is the file this run would upload.
    A platform that returns no ``audit_id`` or ``owner_id`` on its rows never adopts: the rows are
    uploaded afresh, which costs one duplicate the audit's own delete removes by its tag.
    """
    audit_id = str(state.audit_id)
    owner = ctx.client.get_audit(audit_id).get("owner_id")
    if not isinstance(owner, str):
        return None
    path = regular_path(ctx.workload_dir / "dataset.jsonl")
    known = {step.dataset_id for step in state.all_steps()}
    for found in ctx.client.list_datasets(audit_id=audit_id):
        if (
            found.get("name") == name
            and found.get("audit_id") == audit_id
            and found.get("owner_id") == owner
            and str(found.get("id")) not in known
            and _same_rows(found, path)
        ):
            return str(found["id"])
    return None


def _project_owner(state: AuditState, ctx: StepContext) -> str | None:
    """The owner of this directory's own project, or ``None`` when it cannot be read.

    An unpublished upload is found by a key in its description, a value any copy of the
    description carries; the dataset must also belong to the owner of the project this directory
    made. A platform that names no owner, or a directory with no project, never adopts.
    """
    if state.project_id is None:
        return None
    owner = ctx.client.get_project(state.project_id).get("owner_id")
    return owner if isinstance(owner, str) else None


def _adopt(state: AuditState, ctx: StepContext, name: str, key: str | None) -> str | None:
    """The id of the dataset an earlier ask of this upload made, if one is there.

    A multipart upload cannot be replayed by the platform, so a create the process never heard
    the answer to is looked for instead of repeated. A published audit's is found by what the
    platform records on it (:func:`_adopt_tagged`); an unpublished one's carries the directory's
    key in its description, a value only this directory holds, and belongs to the owner of this
    directory's project. A dataset this state already
    records (any candidate, retired ones included) is never adopted: a forced rescan's new rows
    must not meet the old candidate's upload. Both are adopted only when the listing's own
    size and sample count say it is the file this run would upload (:func:`_same_rows`).
    """
    if state.audit_id is not None:
        return _adopt_tagged(state, ctx, name)
    if key is None:
        return None
    owner = _project_owner(state, ctx)
    if owner is None:
        return None
    known = {step.dataset_id for step in state.all_steps()}
    path = regular_path(ctx.workload_dir / "dataset.jsonl")
    for found in ctx.client.list_datasets(search=key):
        if (
            found.get("name") == name
            and found.get("owner_id") == owner
            and f"[{key}]" in str(found.get("description") or "")
            and str(found["id"]) not in known
        ):
            if _same_rows(found, path):
                return str(found["id"])
            LOG.warning(
                "dataset %s, an earlier upload of %s, holds other rows than this run's; uploading"
                " these afresh (the old one stays in your account: delete it in the Studio)",
                found["id"],
                ctx.label,
            )
    return None


def upload(state: AuditState, ctx: StepContext) -> AuditState:
    """Create the dataset from ``dataset.jsonl``; done once ``dataset_id`` is set.

    A published audit's upload carries the audit's id, so the platform tags the dataset at
    creation and the audit's own delete finds it whatever any step later says. An upload an
    earlier run made and lost the answer to (Ctrl+C, a timeout) is adopted rather than made
    again: a second dataset would hold the customer's rows with no record in ``state.json``.
    """
    step = ctx.step(state)
    if step.dataset_id is not None:
        return state
    name = f"audit-{ctx.workload_id}-{ctx.spec.kind.value}"
    key = resume_key(state, ctx)
    adopted = _adopt(state, ctx, name, key)
    if adopted is not None:
        step.dataset_id = adopted
        return state
    tag = "" if key is None else f" [{key}]"
    created = ctx.client.upload_dataset(
        file_path=regular_path(ctx.workload_dir / "dataset.jsonl"),
        name=name,
        dataset_type=DATASET_TYPE,
        format=FILE_FORMAT,
        description=f"workload audit {ctx.label}: derived, redacted rows{tag}",
        audit_id=state.audit_id,
    )
    step.dataset_id = answer_field(created, "id", "the dataset upload")
    return state


def _newest_version(ctx: StepContext, dataset_id: str) -> JsonObject:
    """The highest-numbered version as a pollable payload: ``ready`` once sniffed and counted.

    ``failed`` when the platform's own processing run for the upload gave up:
    that is terminal on the dataset row, so waiting for a version it will
    never produce would burn the whole task timeout with no cause.
    """
    dataset = ctx.client.get_dataset(dataset_id)
    if dataset.get("analysis_status") == PROCESSING_FAILED:
        return {"status": PROCESSING_FAILED, "analysis_error": dataset.get("analysis_error")}
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
        # `failed` is terminal, not an LRO failure: the audit records it as
        # this candidate's step error below, which is what halts the run with
        # the platform's own reason instead of raising out of the step.
        lambda: _newest_version(ctx, dataset_id),
        success={"ready", PROCESSING_FAILED},
        failure=set(),
        timeout=ctx.task_timeout,
        name=f"dataset {dataset_id} sniff",
    )
    if version["status"] == PROCESSING_FAILED:
        reason = string_field(version, "analysis_error") or "the platform gave no reason"
        step.error = f"upload_failed: {reason}"
        return state
    expected = str(ctx.meta()["stats"]["format_key"])
    sniffed = string_field(version, "data_format")
    if sniffed != expected:
        step.error = f"format_mismatch: platform sniffed {sniffed!r}, the recipe needs {expected!r}"
        return state
    step.version_id = answer_field(version, "id", "the dataset version read")
    return state


def split(state: AuditState, ctx: StepContext) -> AuditState:
    """Submit ``split.json`` as an explicit split; done once the task is enqueued."""
    step = ctx.step(state)
    if step.split_task_id is not None:
        return state
    body = json.loads(read_regular(ctx.workload_dir / "split.json"))
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


LISTED = 3
"""How many class names a stop message lists before it says how many more there are."""
REMOVES = " `dagnam audit delete <audit-dir>` removes the uploaded rows and the whole audit."
"""What every PII stop ends with: the rows are on the platform, and this takes them off.

Said as it is: ``audit delete`` has no narrower form, so it also removes every other
workload's run and endpoint. ``<audit-dir>`` is not filled in -- the message is stored
in the account, which is never told where on this machine the audit lives.
"""


def _listing(names: Collection[str]) -> str:
    """The first few class names, and how many more there are."""
    shown = sorted(names)[:LISTED]
    more = len(names) - len(shown)
    return ", ".join(shown) + (f" and {more} more" if more else "")


def pii_stop(
    residual: Mapping[str, int],
    looked_for: Collection[str],
    local_pass: Collection[str],
    ctx: StepContext,
) -> str:
    """Why the two PII passes disagree, what to do about it, and that the rows are already up.

    Three causes look alike in the counts and need different cures, so each is
    named. A finding in a class this SDK redacts is a real disagreement and is
    never explained away as a version gap, whatever else differs -- though a
    platform whose contract differs from this install's says so first, since a
    patch changes what is found inside the same classes. Findings only in classes
    these rows were never scanned for mean the platform runs a newer privacy
    contract than the scan did. No finding, with classes the platform did not look
    for, means it runs an older one. Whichever it is, the dataset went up before
    the scan could run, so the message ends with what removes it.

    The message is a candidate's ``error``, which the platform stores up to 500
    characters of: the cause comes first, class lists are cut, and no path of this
    machine is in it.
    """
    missing = sorted(set(local_pass) - set(looked_for))
    known = sorted(code for code in residual if code in local_pass)
    gap = version_gap(ctx.platform_contracts, installed_contract())
    unscanned = f" Not scanned: {_listing(missing)}." if missing and residual else ""
    found = _listing([f"{code} ({n})" for code, n in sorted(residual.items())])
    if known:
        return (
            f"the platform found {found} in classes this SDK redacts: the detectors disagree, or"
            f" the rows changed since the scan. Scan again; report it if it repeats.{gap}{unscanned}"
            f"{REMOVES}"
        )
    if residual:
        cure = gap or f" `{UPGRADE}` (or `pip install -U dagnam`), then scan again."
        return (
            "the platform runs a newer privacy contract than these rows were redacted with: it"
            f" found {found} in classes the scan did not look for.{cure}{unscanned}{REMOVES}"
        )
    if looked_for:
        return (
            "the platform runs an older privacy contract than this SDK: its scan does not cover"
            f" {_listing(missing)} (it found nothing in the classes it does cover). Run again once"
            f" it is upgraded.{gap}{REMOVES}"
        )
    return (
        "the platform's scan did not say which classes it looked for, so it cannot confirm the"
        f" redaction. Scan and run again.{gap}{REMOVES}"
    )


def wait_pii(state: AuditState, ctx: StepContext) -> AuditState:
    """The two implementations must agree; a disagreement stops the workload.

    The rows were redacted client-side before upload, so agreement means the
    server found no residual finding in any class and looked for every class
    the client's ``meta.json`` pass list names. The stop is a privacy control
    and is never relaxed; :func:`pii_stop` only makes it say which of its
    causes it is.
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
    step.pii_agrees = not residual and local_pass <= looked_for
    if not step.pii_agrees:
        stop = pii_stop(residual, looked_for, local_pass, ctx)
        step.error = f"pii_disagreement: {stop}"
    return state


__all__ = [
    "DATASET_TYPE",
    "FILE_FORMAT",
    "PROCESSING_FAILED",
    "SPLIT_NAMES",
    "pii_scan",
    "pii_stop",
    "resolve_version",
    "split",
    "upload",
    "wait_pii",
    "wait_split",
]
