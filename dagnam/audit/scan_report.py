"""The scan report: ``scan-report.json`` (the contract) and ``scan-report.md`` (a view of it).

:func:`build_scan_report` folds discovered workloads, their verdicts and the
price table into the design's JSON shape; :func:`write_scan_report` writes the
JSON and renders the markdown from that JSON object, never from live values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from dagnam.audit.discover import Workload
from dagnam.audit.economics import Verdict, customer_verdict, price_workload, replaceability
from dagnam.audit.prices import PriceTable
from dagnam.audit.thresholds import MAX_UNSTRUCTURED_SHARE, MIN_HOLDOUT, PRICE_TABLE_STALE_DAYS

SCHEMA = "dagnam.audit.scan/1"


@dataclass(frozen=True, slots=True)
class Window:
    """The export window the rates were computed over."""

    start: datetime
    end: datetime
    days: float


@dataclass(frozen=True)
class ScanReport:
    """The scan report as the design's JSON contract, one field per top-level key."""

    generated_at: str
    source: str
    window: dict[str, Any]
    price_table_version: str
    totals: dict[str, Any]
    pii: dict[str, Any]
    workloads: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]
    schema: str = SCHEMA

    def to_json(self) -> dict[str, Any]:
        """The ``scan-report.json`` object, ``schema`` first."""
        return {
            "schema": self.schema,
            "generated_at": self.generated_at,
            "source": self.source,
            "window": self.window,
            "price_table_version": self.price_table_version,
            "totals": self.totals,
            "pii": self.pii,
            "workloads": list(self.workloads),
            "warnings": list(self.warnings),
        }


def build_scan_report(
    workloads: Sequence[Workload],
    *,
    source: str,
    window: Window,
    price_table: PriceTable,
    pii_pass_list: Sequence[str],
    pii_counts: Mapping[str, int],
    datasets: Mapping[str, Mapping[str, Any]] | None = None,
) -> ScanReport:
    """Price what the export did not, judge every workload, and assemble the report.

    ``datasets`` carries the derived dataset's numbers per workload id (the
    contract's ``dataset`` object); a holdout under :data:`MIN_HOLDOUT` turns
    that workload's verdict into ``too_few_samples``.
    """
    now = datetime.now(UTC)
    priced = [price_workload(w, price_table) for w in workloads]
    entries = tuple(
        {**w.to_json(), "verdict": asdict(_verdict(w, dataset)), "dataset": dataset}
        for w in priced
        for dataset in [(datasets or {}).get(w.id)]
    )
    warnings: list[str] = []
    age = price_table.age_days(now.date())
    if age > PRICE_TABLE_STALE_DAYS:
        warnings.append(
            f"price table {price_table.version} is {age} days old (stale after"
            f" {PRICE_TABLE_STALE_DAYS}); pass --price-table with a newer one"
        )
    calls = sum(w.calls for w in priced)
    low_confidence = sum(w.calls for w in priced if w.confidence == "low")
    if calls and low_confidence / calls > MAX_UNSTRUCTURED_SHARE:
        warnings.append(
            f"{low_confidence / calls:.0%} of traces have no system prompt (over"
            f" {MAX_UNSTRUCTURED_SHARE:.0%}); discovery is low-confidence for this export"
        )
    return ScanReport(
        generated_at=now.isoformat(),
        source=source,
        window={
            "start": window.start.isoformat(),
            "end": window.end.isoformat(),
            "days": window.days,
        },
        price_table_version=price_table.version,
        totals={
            "calls": calls,
            "cost_usd_month": sum(w.cost_usd_month or 0.0 for w in priced),
            "prompt_tokens": sum(w.prompt_tokens for w in priced),
            "completion_tokens": sum(w.completion_tokens for w in priced),
        },
        pii={"pass_list": list(pii_pass_list), "counts": dict(pii_counts)},
        workloads=entries,
        warnings=tuple(warnings),
    )


def _verdict(w: Workload, dataset: Mapping[str, Any] | None) -> Verdict:
    verdict = replaceability(w)
    holdout = dataset["split"].get("eval_holdout", 0) if dataset is not None else None
    if holdout is not None and holdout < MIN_HOLDOUT:
        reason = f"{holdout} holdout rows after the split; {MIN_HOLDOUT} needed"
        return Verdict("too_few_samples", None, None, reason)
    return verdict


def write_scan_report(report: ScanReport, out_dir: Path) -> None:
    """Write ``scan-report.json`` and ``scan-report.md`` (rendered from the JSON) into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report.to_json(), indent=2)
    (out_dir / "scan-report.json").write_text(text, encoding="utf-8")
    (out_dir / "scan-report.md").write_text(render_markdown(json.loads(text)), encoding="utf-8")


def _usd(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.2f}"


def _row(w: dict[str, Any]) -> str:
    verdict = w["verdict"]
    ratio = "-" if verdict["ratio"] is None else f"{verdict['ratio']:.1f}x"
    savings = (
        _usd(verdict["savings_usd_month"]) if verdict["savings_usd_month"] is not None else "-"
    )
    label = customer_verdict(verdict["status"], winner=False)  # a scan has no winner yet
    return (
        f"| {w['id']} | {w['structure_class']} | {w['calls_per_day']:,.1f} | {_usd(w['cost_usd_month'])}"
        f" | {ratio} | {savings} | {label} | {verdict['status']}: {verdict['reason']} |"
    )


_TABLE_HEAD = (
    "| workload | class | calls/day | $/month | ratio | savings $/month | verdict | why |",
    "|---|---|---|---|---|---|---|---|",
)


def render_markdown(js: Mapping[str, Any]) -> str:
    """Render ``scan-report.md`` from the ``scan-report.json`` object; KEEP rows come first."""
    keep = [
        w
        for w in js["workloads"]
        if customer_verdict(w["verdict"]["status"], winner=False) == "KEEP"
    ]
    candidates = [w for w in js["workloads"] if w not in keep]
    totals, window = js["totals"], js["window"]
    lines = [
        "# Workload scan",
        "",
        f"- generated: {js['generated_at']}",
        f"- source: {js['source']}",
        f"- window: {window['start']} to {window['end']} ({window['days']:g} days)",
        f"- price table: {js['price_table_version']}",
        f"- totals: {totals['calls']:,} calls, ${_usd(totals['cost_usd_month'])}/month,"
        f" {totals['prompt_tokens']:,} prompt + {totals['completion_tokens']:,} completion tokens",
        "",
    ]
    if js["warnings"]:
        lines += ["## Warnings", "", *(f"- {w}" for w in js["warnings"]), ""]
    lines += ["## Not worth replacing", "", *_TABLE_HEAD, *(_row(w) for w in keep), ""]
    lines += ["## Candidates", "", *_TABLE_HEAD, *(_row(w) for w in candidates), ""]
    counts = ", ".join(f"{k}: {v}" for k, v in js["pii"]["counts"].items()) or "none found"
    lines += [
        "## PII",
        "",
        f"- scanned for: {', '.join(js['pii']['pass_list'])}",
        f"- found: {counts}",
        "",
    ]
    return "\n".join(lines)
