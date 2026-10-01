"""The scan report turned into publish bodies: one workload row, each step's status, and what fits.

:mod:`dagnam.audit.publish` owns the conversation with the server; this module
owns the arithmetic underneath it -- the per-call means the server stores
instead of the scan's totals, the redaction counts Task 6 left on disk, the
status each step leaves a candidate in, and the two rules about
``AuditCreate.workloads``' own length cap. Every function here is pure and
total: it is called inside the publisher's guard, and a number the scan does
not carry becomes ``None``, never an exception.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
import json
import logging
from pathlib import Path
from typing import Any

from dagnam_contracts.audit.serving import SERVING_RATES, serving_cost_usd_month

from dagnam._types import JsonObject
from dagnam.audit.candidates import StudentKind
from dagnam.audit.state import StepState
from dagnam.audit.steps import error_code
from dagnam.audit.steps_serve import DEPLOY_RUNNING
from dagnam.audit.steps_train import RUN_COMPLETED

_LOGGER = logging.getLogger("dagnam.audit.publish")
"""The publisher's logger: what this module warns about is a publish decision."""

MAX_WORKLOADS = 200
"""``AuditCreate.workloads``' own cap; a larger scan publishes the ones that matter.

More *selected* than this is the one shape that cannot be published at all --
see :func:`too_many_selected`.
"""

EXCERPT_MAX = 200
REASON_MAX = 300
ID_MAX = 64


STATUS_BY_STEP: Mapping[str, str] = {
    "upload": "uploading",
    "resolve_version": "uploading",
    "split": "splitting",
    "wait_split": "splitting",
    "pii_scan": "pii_check",
    "wait_pii": "pii_check",
    "submit": "submitting",
    "wait_run": "training",
    "resolve_model_version": "training",
    "create_deployment": "deploying",
    "create_revision": "deploying",
    "wait_active": "deploying",
    "replay_and_score": "scored",
}
"""Every step of ``orchestrate.STEPS`` -> the candidate status it leaves behind."""

DONE_BY_STEP: Mapping[str, Callable[[StepState], bool]] = {
    "upload": lambda s: s.dataset_id is not None,
    "resolve_version": lambda s: s.version_id is not None,
    "split": lambda s: s.split_task_id is not None,
    "wait_split": lambda s: bool(s.split_done),
    "pii_scan": lambda s: s.pii_task_id is not None,
    "wait_pii": lambda s: s.pii_agrees is not None,
    "submit": lambda s: s.run_id is not None,
    "wait_run": lambda s: s.run_status == RUN_COMPLETED,
    "resolve_model_version": lambda s: s.model_version_id is not None,
    "create_deployment": lambda s: s.deployment_id is not None,
    "create_revision": lambda s: s.deploy_status is not None,
    # ``or s.scored`` mirrors the step's own guard: a candidate that already
    # scored was live once, and a cancel since may have paused that endpoint.
    "wait_active": lambda s: s.deploy_status == DEPLOY_RUNNING or bool(s.scored),
    "replay_and_score": lambda s: bool(s.scored),
}
"""Each step's own "already done" guard, mirrored from ``steps_*``, in ``STEPS`` order.

Only :meth:`Publisher.backfill` reads it: a run whose first ``create_audit``
failed reaches the account with candidates already part-finished, and this is
what says which of their steps to publish before the run carries on.
"""

REPLAY_STEP = "replay"
"""Published just before ``replay_and_score``: the server has no ``deploying -> scored`` edge."""
PII_DISAGREEMENT = "pii_disagreement"
FAILED = "failed"


def status_for(step_name: str, step: StepState) -> str:
    """The status a step leaves the candidate in; a recorded error wins over the map."""
    if step.error is not None and not step.scored:
        return PII_DISAGREEMENT if error_code(step) == PII_DISAGREEMENT else FAILED
    return STATUS_BY_STEP[step_name]


SCORED_BY = "cli"
MEASURED_FROM = "client"
"""The replay's percentiles are the client's own round trips, not the server's."""
_ERROR_MAX = 500
_NAME_MAX = 120


def patch_body(
    step_name: str, step: StepState, status: str | None, serving_cost: float | None
) -> JsonObject:
    """The ``CandidatePatch`` body for one step: its status plus whatever the step produced.

    ``status`` overrides what :func:`status_for` reads off the step (the
    back-fill of a step that succeeded on a candidate whose later error is
    recorded); ``serving_cost`` goes out only once the candidate scored.
    """
    resolved = status_for(step_name, step) if status is None else status
    body: JsonObject = {"step": step_name, "status": resolved}
    if step.error is not None and status is None:
        # The error belongs to the step that recorded it, not to the ones a
        # back-fill replays behind it -- those are the ones given a status.
        body["error"] = step.error[:_ERROR_MAX]
    for key, value in (
        ("dataset_version_id", step.version_id),
        ("training_job_id", step.training_job_id),
        ("deployment_id", step.deployment_id),
        ("base_display_name", None if step.base is None else step.base[:_NAME_MAX]),
        ("training_cost_credits", step.training_cost_credits),
        ("replay_cost_credits", step.replay_cost_credits),
    ):
        if value is not None:
            body[key] = value
    if step.agreement is not None:
        body["agreement"] = dict(step.agreement)
    if step.latency is not None:
        body["latency"] = {**step.latency, "measured_from": MEASURED_FROM}
    if step.scored:
        body["scored_by"] = SCORED_BY
        if serving_cost is not None:
            body["serving_cost_usd_month"] = serving_cost
    return body


def number(value: object) -> float | None:
    """``value`` as a float when it is a real number, else ``None``."""
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def mean(total: object, calls: object) -> int | None:
    """A per-call mean from the scan's totals; the server stores means, not totals."""
    amount, over = number(total), number(calls)
    return None if amount is None or not over else round(amount / over)


def sub(entry: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = entry.get(key)
    return value if isinstance(value, Mapping) else {}


def student_cost(kind: StudentKind, entry: Mapping[str, Any]) -> float | None:
    """What serving ``kind`` costs a month at one scanned workload's volume, or ``None``.

    A student priced per output token has no price for a workload whose export
    carried no completion tokens: ``$0`` would win every frontier and inflate
    every saving, so the cost is unknown instead.
    """
    tokens = int(number(sub(entry, "tokens").get("completion")) or 0)
    if tokens <= 0 and "usd_per_m_output_tokens" in SERVING_RATES[kind]:
        return None
    return serving_cost_usd_month(
        kind,
        calls_per_day=number(entry.get("calls_per_day")) or 0.0,
        completion_tokens=tokens,
        calls=int(number(entry.get("calls")) or 1) or 1,
    )


def pii_counts(audit_dir: Path, workload_id: str) -> dict[str, int]:
    """The redaction counts Task 6 wrote for this workload; empty when it derived none."""
    meta = audit_dir / "workloads" / workload_id / "meta.json"
    if not meta.is_file():
        return {}
    counts = sub(sub(json.loads(meta.read_text(encoding="utf-8")), "stats"), "redact").get("counts")
    return {str(code): int(n) for code, n in counts.items()} if isinstance(counts, Mapping) else {}


def capped(entries: list[Mapping[str, Any]], selected: Collection[str]) -> list[Mapping[str, Any]]:
    """At most :data:`MAX_WORKLOADS` entries: the run's own first, then by spend.

    The server refuses a longer list outright, which would make the whole
    publish a dropped 422 -- so a scan that found more workloads than the audit
    page holds publishes the ones the run took plus the biggest spenders, and
    says in the log what it left out. Selected workloads sort first, so the
    ones this run will open candidates against are never what the cap drops;
    :func:`too_many_selected` is what guarantees they all fit.
    """
    if len(entries) <= MAX_WORKLOADS:
        return entries
    ranked = sorted(
        entries,
        key=lambda e: (str(e["id"]) not in selected, -(number(e.get("cost_usd_month")) or 0.0)),
    )
    _LOGGER.warning(
        "audit publish: the scan found %d workloads and the account holds %d;"
        " publishing the %d this run took plus the highest-spending others",
        len(entries),
        MAX_WORKLOADS,
        sum(1 for e in entries if str(e["id"]) in selected),
    )
    return ranked[:MAX_WORKLOADS]


def too_many_selected(count: int) -> str | None:
    """Why a run of ``count`` workloads cannot be published, or ``None`` when it can.

    A candidate is opened against a workload the audit carries, so publishing a
    list the cap truncated would 404 every candidate of every row it dropped.
    There is no honest body to send for a run this wide: it stays local, with
    every number still in ``audit-report.json``.
    """
    if count <= MAX_WORKLOADS:
        return None
    return (
        f"not published: this run took {count} workloads and an audit page holds"
        f" {MAX_WORKLOADS}; every number stays in the local report."
        " Run fewer at a time (--workloads) to publish it."
    )


def workload_body(
    entry: Mapping[str, Any], *, selected: bool, pii_counts: Mapping[str, int]
) -> JsonObject:
    """One ``scan-report.json`` workload as the publish body's ``WorkloadPublish``."""
    verdict = sub(entry, "verdict")
    tokens = sub(entry, "tokens")
    return {
        "workload_id": str(entry["id"])[:ID_MAX],
        "structure_class": str(entry["structure_class"]),
        "template_excerpt": str(entry.get("template_excerpt") or "")[:EXCERPT_MAX],
        "calls_per_day": number(entry.get("calls_per_day")) or 0.0,
        "mean_prompt_tokens": mean(tokens.get("prompt"), entry.get("calls")),
        "mean_completion_tokens": mean(tokens.get("completion"), entry.get("calls")),
        "spend_usd_month": number(entry.get("cost_usd_month")),
        "export_p50_ms": number(sub(entry, "latency_ms").get("p50")),
        "verdict": str(verdict.get("status") or "unknown_cost"),
        "verdict_reason": str(verdict.get("reason") or "")[:REASON_MAX],
        "ratio": number(verdict.get("ratio")),
        "pii_counts": dict(pii_counts),
        "selected": selected,
        # A scan from before the field, or a value the page does not know, is text.
        "response_mode": "tool_call" if entry.get("response_mode") == "tool_call" else "text",
    }


__all__ = [
    "DONE_BY_STEP",
    "EXCERPT_MAX",
    "FAILED",
    "ID_MAX",
    "MAX_WORKLOADS",
    "MEASURED_FROM",
    "PII_DISAGREEMENT",
    "REASON_MAX",
    "REPLAY_STEP",
    "SCORED_BY",
    "STATUS_BY_STEP",
    "capped",
    "mean",
    "number",
    "patch_body",
    "pii_counts",
    "status_for",
    "student_cost",
    "sub",
    "too_many_selected",
    "workload_body",
]
