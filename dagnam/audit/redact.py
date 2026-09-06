"""Redact every PII class the shared contract detects before a row touches disk (spec §10)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dagnam_contracts.hygiene import PII_CODES, PiiAction, apply_pii_policy, scan_rows

# Spec U3: ``apply_pii_policy(rows, {every class: "redact"})`` -- the classes
# come from the contract, so a detector added there is redacted here.
PII_POLICY: dict[str, PiiAction] = dict.fromkeys(PII_CODES, "redact")


@dataclass(frozen=True, slots=True)
class RedactStats:
    """Findings per class, the classes looked for, and how many rows changed."""

    counts: dict[str, int]
    pass_list: tuple[str, ...]
    rows_changed: int


def redact_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], RedactStats]:
    """Return redacted copies of ``rows`` (never dropped) with the per-class counts."""
    scan = scan_rows(rows, max_issues=0)
    redacted, rows_changed, _removed = apply_pii_policy(rows, PII_POLICY)
    return redacted, RedactStats(
        counts=scan.counts_by_code, pass_list=scan.pass_list, rows_changed=rows_changed
    )


__all__ = ["PII_POLICY", "RedactStats", "redact_rows"]
