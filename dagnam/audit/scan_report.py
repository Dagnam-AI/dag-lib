"""The scan report: ``scan-report.json`` (the contract) and ``scan-report.md`` (a view of it).

:func:`build_scan_report` folds discovered workloads, their verdicts and the
price table into the design's JSON shape; :func:`write_scan_report` writes the
JSON and renders the markdown from that JSON object, never from live values.
:func:`write_scan` writes a whole scan -- every workload's rows, then the report.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from dagnam._core.exceptions import DagnamError
from dagnam.audit.derive import WorkloadDataset
from dagnam.audit.discover import Workload
from dagnam.audit.economics import Verdict, customer_verdict, price_workload, replaceability
from dagnam.audit.prices import PriceTable
from dagnam.audit.readers.messages import TOOL_CALL_NOTE
from dagnam.audit.redact import PII_POLICY
from dagnam.audit.state import AuditState, load_state, save_state
from dagnam.audit.steps_train import credits_spent
from dagnam.audit.thresholds import (
    MAX_UNSTRUCTURED_SHARE,
    MIN_HOLDOUT,
    PRICE_TABLE_STALE_DAYS,
    SFT_MAX_TOKENS,
    SFT_MIN_TRAIN_ROWS,
)
from dagnam.audit.workspace import write_workload

SCHEMA = "dagnam.audit.scan/1"
_SECONDS_PER_DAY = 86_400


@dataclass(frozen=True, slots=True)
class Window:
    """The export window the rates were computed over."""

    start: datetime
    end: datetime
    days: float
    sample_rate: float = 1.0
    """The share of the traffic the export holds (``--sample-rate``)."""


def scan_window(workloads: Sequence[Workload], *, window_days: int, sample_rate: float) -> Window:
    """The window the workloads' calls span, at least ``window_days`` long."""
    first = min(w.first_ts for w in workloads if w.first_ts is not None)
    last = max(w.last_ts for w in workloads if w.last_ts is not None)
    span = (last - first).total_seconds() / _SECONDS_PER_DAY
    return Window(first, last, max(float(window_days), span), sample_rate)


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
    contract's ``dataset`` object); a holdout under :data:`MIN_HOLDOUT`, or
    fewer than :data:`SFT_MIN_TRAIN_ROWS` training rows once the ones over the
    student's context are left out, turns that workload's verdict into
    ``too_few_samples``.
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
    warnings.extend(
        f"workload {w.id}: {TOOL_CALL_NOTE}" for w in priced if w.response_mode == "tool_call"
    )
    warnings.extend(
        f"workload {workload_id}: redaction rewrote {dataset['truths_redacted']:,} of"
        f" {dataset['rows']:,} training targets; the holdout is scored against the redacted"
        " values, so agreement can overstate how well a replacement reproduces them"
        for workload_id, dataset in (datasets or {}).items()
        if dataset.get("truths_redacted")
    )
    warnings.extend(
        f"workload {workload_id} trains on a {dataset['split']['train']:,}-row sample of its"
        f" {dataset['split']['train'] + dataset['train_rows_capped']:,} training rows, so a"
        " candidate finishes inside its recipe's 1-hour ceiling"
        for workload_id, dataset in (datasets or {}).items()
        if dataset.get("train_rows_capped")
    )
    warnings.extend(
        f"workload {workload_id}: {dataset['train_rows_over_budget']:,} training rows exceed the"
        f" student's {SFT_MAX_TOKENS:,}-token context (by an estimate) and are left"
        " out of its training; the holdout keeps them"
        for workload_id, dataset in (datasets or {}).items()
        if dataset.get("train_rows_over_budget")
    )
    return ScanReport(
        generated_at=now.isoformat(),
        source=source,
        window={
            "start": window.start.isoformat(),
            "end": window.end.isoformat(),
            "days": window.days,
            "sample_rate": window.sample_rate,
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


def dataset_entry(dataset: WorkloadDataset) -> dict[str, Any]:
    """The contract's ``dataset`` object for one derived workload."""
    stats = dataset.stats
    return {
        "rows": len(dataset.rows),
        "dedup_removed": stats["dedup"]["removed"],
        "redactions": sum(stats["redact"]["counts"].values()),
        "truncated": stats["derive"]["truncated"],
        "truths_redacted": stats["redact"]["truths_changed"],
        "train_rows_capped": stats["cap"]["dropped"],
        "train_rows_over_budget": stats["budget"]["dropped"],
        "split": {name: len(rows) for name, rows in dataset.split.items()},
    }


def _verdict(w: Workload, dataset: Mapping[str, Any] | None) -> Verdict:
    verdict = replaceability(w)
    if dataset is None:
        return verdict
    holdout = dataset["split"].get("eval_holdout", 0)
    if holdout < MIN_HOLDOUT:
        reason = f"{holdout} holdout rows after the split; {MIN_HOLDOUT} needed"
        return Verdict("too_few_samples", None, None, reason)
    over = dataset.get("train_rows_over_budget", 0)
    if over and dataset["split"]["train"] < SFT_MIN_TRAIN_ROWS:
        # B2-3: the recipe drops these rows at train time, after the credits are spent.
        reason = f"{over:,} rows exceed the student's {SFT_MAX_TOKENS:,}-token context"
        return Verdict("too_few_samples", None, None, reason)
    return verdict


def superseded_workloads(out_dir: Path, datasets: Mapping[str, WorkloadDataset]) -> list[str]:
    """The workloads a run in ``out_dir`` started whose rows this scan would change.

    ``state.json`` records every workload a run trained, and a resumed run
    replays that workload's ``eval_holdout`` against the model it trained. A
    scan of another export that rewrote the rows under it would score the
    model on data it never saw and pair one export's spend with another's
    agreement. A published scan also protects every derived dataset before
    the first candidate starts. Re-scanning unchanged rows and splits passes.
    """
    state = load_state(out_dir)
    protected = dict.fromkeys(state.workloads)
    added: set[str] = set()
    if state.audit_id is not None:
        scan_path = out_dir / "scan-report.json"
        if scan_path.exists():
            scan = json.loads(scan_path.read_text(encoding="utf-8"))
            published = {
                entry["id"] for entry in scan["workloads"] if entry.get("dataset") is not None
            }
        else:
            published = {
                path.parent.name for path in (out_dir / "workloads").glob("*/dataset.jsonl")
            }
        protected.update(dict.fromkeys(sorted(published)))
        added = set(datasets) - published
        protected.update(dict.fromkeys(datasets))
    changed: list[str] = []
    for workload_id in protected:
        folder = out_dir / "workloads" / workload_id
        dataset = datasets.get(workload_id)
        try:
            lines = (folder / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
            split = json.loads((folder / "split.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            lines, split = [], {}
        same = (
            workload_id not in added
            and dataset is not None
            and (
                [json.loads(line) for line in lines] == dataset.rows
                and split.get("member_row_indices") == dataset.split
            )
        )
        if not same:
            changed.append(workload_id)
    return changed


class SupersededRunError(DagnamError):
    """``out`` holds a run whose training rows this scan would rewrite (see :func:`superseded_workloads`)."""

    def __init__(
        self, out_dir: Path, workload_ids: Sequence[str], *, published: bool = False
    ) -> None:
        self.workload_ids = tuple(workload_ids)
        action = "Scan into a new --out" if published else "Scan into a new --out, or pass --force"
        super().__init__(
            f"{out_dir} holds a run of another scan: this export would change"
            f" {', '.join(workload_ids)}. {action}."
        )


def write_scan(
    out_dir: Path,
    workloads: Sequence[Workload],
    datasets: Mapping[str, WorkloadDataset],
    *,
    source: str,
    window: Window,
    price_table: PriceTable,
    force: bool = False,
) -> ScanReport:
    """Write every workload's rows, then ``scan-report.{json,md}``, into ``out_dir``.

    Refuses (:class:`SupersededRunError`) to rewrite the rows of a run's
    workloads unless ``force``; forced, the answers a rewritten holdout's replay
    had already kept are stale and are removed. Old candidates are retired for
    cleanup; the next run starts fresh. Published runs require a new ``--out``
    even with ``force``. The caller holds ``out_dir``'s lock.
    """
    changed = superseded_workloads(out_dir, datasets)
    state = load_state(out_dir)
    if changed and (not force or state.audit_id is not None):
        raise SupersededRunError(out_dir, changed, published=state.audit_id is not None)
    if changed:
        superseded = AuditState(workloads={key: state.workloads[key] for key in changed})
        state.retired_cost_credits += credits_spent(superseded, out_dir)
        for workload_id in changed:
            state.retired.extend(state.workloads.pop(workload_id).values())
        state.halted = None
        save_state(out_dir, state)
        for suffix in ("json", "md"):
            (out_dir / f"audit-report.{suffix}").unlink(missing_ok=True)
    for workload_id in changed:
        for stale in (out_dir / "workloads" / workload_id).glob("replay-*.jsonl"):
            stale.unlink()
    pii_counts: Counter[str] = Counter()
    for workload_id, dataset in datasets.items():
        write_workload(out_dir, workload_id, dataset.rows, dataset.split, dataset.stats)
        pii_counts.update(dataset.stats["redact"]["counts"])
    report = build_scan_report(
        workloads,
        source=source,
        window=window,
        price_table=price_table,
        pii_pass_list=list(PII_POLICY),
        pii_counts=dict(pii_counts),
        datasets={workload_id: dataset_entry(d) for workload_id, d in datasets.items()},
    )
    write_scan_report(report, out_dir)
    return report


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
        f"- window: {window['start']} to {window['end']} ({window['days']:g} days)"
        + (
            f", sampled at {window['sample_rate']:.0%} and scaled up"
            if window.get("sample_rate", 1.0) < 1
            else ""
        ),
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
