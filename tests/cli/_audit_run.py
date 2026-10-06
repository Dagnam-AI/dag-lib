"""What the ``audit run`` CLI tests share: a scanned directory, and the platform behind the client."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.audit._chat import json_row, label_row

from dagnam._core.exceptions import APIError
from dagnam._types import JsonObject
from dagnam.audit.workspace import write_workload

PASS_LIST = ["PII_EMAIL", "PII_PHONE", "PII_PAYMENT_CARD", "PII_NATIONAL_ID"]


SPLIT = {"train": list(range(16)), "eval_holdout": [16, 17, 18, 19]}


DERIVED: dict[str, Any] = {
    "rows": 20,
    "dedup_removed": 0,
    "redactions": 0,
    "truncated": 0,
    "truths_redacted": 0,
    "train_rows_capped": 0,
    "train_rows_over_budget": 0,
    "split": {"train": 16, "eval_holdout": 4},
}


"""The ``dataset`` object a scan writes for a workload it derived rows for."""


def workload(workload_id: str, cls: str, status: str, *, derived: bool = True) -> dict[str, Any]:
    return {
        "id": workload_id,
        "template_hash": workload_id,
        "template_excerpt": "Classify",
        "structure_class": cls,
        "confidence": "high",
        "calls": 3_000,
        "calls_per_day": 100.0,
        "tokens": {"prompt": 300_000, "completion": 6_000},
        "cost_usd_month": 900.0,
        "cost_source": "export",
        "latency_ms": {"p50": 400.0, "p95": 900.0},
        "distinct_outputs": 2,
        "entropy": 1.0,
        "models": ["gpt-4o-mini"],
        "verdict": {"status": status, "ratio": 12.0, "savings_usd_month": 800.0, "reason": "x"},
        "dataset": dict(DERIVED) if derived else None,
    }


SCAN: dict[str, Any] = {
    "schema": "dagnam.audit.scan/1",
    "generated_at": "2026-09-06T10:00:00+00:00",
    "source": "langfuse",
    "window": {
        "start": "2026-08-01T00:00:00+00:00",
        "end": "2026-08-31T00:00:00+00:00",
        "days": 30,
    },
    "price_table_version": "2026-09",
    "totals": {
        "calls": 6_000,
        "cost_usd_month": 1_800.0,
        "prompt_tokens": 1,
        "completion_tokens": 1,
    },
    "pii": {"pass_list": PASS_LIST, "counts": {"PII_EMAIL": 2}},
    "workloads": [
        workload("w1", "enum_label", "candidate"),
        workload("w2", "json_object", "marginal"),
        workload("w3", "free_text", "not_audited", derived=False),
    ],
    "warnings": [],
}


def stats(format_key: str, counts: dict[str, int]) -> dict[str, Any]:
    return {
        "format_key": format_key,
        "redact": {"counts": counts, "pass_list": PASS_LIST, "rows_changed": sum(counts.values())},
    }


SDK_CONTRACT = "0.4.2"


"""The contract version the SDK under test reports as installed."""


class Build:
    """What the fake platform's ``GET /health/build`` answers, and how often it was read."""

    def __init__(self) -> None:
        self.answer: JsonObject | APIError = {
            "revision": "abc",
            "version": "1",
            "contracts": SDK_CONTRACT,
        }
        self.reads = 0

    def __call__(self) -> JsonObject:
        self.reads += 1
        if isinstance(self.answer, APIError):
            raise self.answer
        return dict(self.answer)


def write_prepared(root: Path) -> Path:
    """A scanned audit directory: two derived workloads and the scan report."""
    root.mkdir()
    write_workload(
        root,
        "w1",
        [label_row(i) for i in range(20)],
        SPLIT,
        stats("labeled-example", {"PII_EMAIL": 2}),
    )
    write_workload(root, "w2", [json_row(i) for i in range(20)], SPLIT, stats("chat-messages", {}))
    (root / "scan-report.json").write_text(json.dumps(SCAN), encoding="utf-8")
    return root
