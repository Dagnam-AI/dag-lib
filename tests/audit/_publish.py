"""Shared scaffolding for the publisher tests: one scan report and one ``start`` call."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from dagnam.audit.publish import Publisher

AUDIT_ID = "audit-1"


def entry(workload_id: str, cls: str, status: str) -> dict[str, Any]:
    """One ``scan-report.json`` workload with every number the publish body reads."""
    return {
        "id": workload_id,
        "structure_class": cls,
        "template_excerpt": "Classify the ticket",
        "calls": 3_000,
        "calls_per_day": 100.0,
        "tokens": {"prompt": 300_000, "completion": 6_000},
        "cost_usd_month": 900.0,
        "latency_ms": {"p50": 400.0, "p95": 900.0},
        "verdict": {"status": status, "ratio": 12.0, "reason": "12.0x over the floor"},
    }


SCAN: dict[str, Any] = {
    "generated_at": "2026-09-06T10:00:00+00:00",
    "price_table_version": "2026-09",
    "workloads": [
        entry("w1", "enum_label", "candidate"),
        entry("w2", "json_object", "marginal"),
    ],
}


def start(publisher: Publisher, audit_dir: Path, **overrides: Any) -> None:
    """``Publisher.start`` over :data:`SCAN` with ``w1`` selected, unless overridden."""
    settings: dict[str, Any] = {
        "floor": 0.97,
        "max_credits": 500,
        "sdk_version": "9.9.9",
        **overrides,
    }
    publisher.start(audit_dir, SCAN, ["w1"], **settings)


__all__ = ["AUDIT_ID", "SCAN", "entry", "start"]
