"""The scan report turned into publish bodies: one workload row, and what fits.

:mod:`dagnam.audit.publish` owns the conversation with the server; this module
owns the arithmetic underneath it -- the per-call means the server stores
instead of the scan's totals, the redaction counts Task 6 left on disk, and the
two rules about ``AuditCreate.workloads``' own length cap. Every function here
is pure and total: it is called inside the publisher's guard, and a number the
scan does not carry becomes ``None``, never an exception.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
import json
import logging
from pathlib import Path
from typing import Any

from dagnam._types import JsonObject

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
    }


__all__ = [
    "EXCERPT_MAX",
    "ID_MAX",
    "MAX_WORKLOADS",
    "REASON_MAX",
    "capped",
    "mean",
    "number",
    "pii_counts",
    "sub",
    "too_many_selected",
    "workload_body",
]
