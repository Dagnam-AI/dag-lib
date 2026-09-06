"""The audit report: ``audit-report.json`` (the contract) and ``audit-report.md`` (a view of it).

:func:`build_audit_report` is the scan report plus, per audited workload, the
candidates the run produced, the frontier's winner and the switch values
(spec section 7). The verdict a customer reads -- REPLACE, NOT YET, KEEP
(section 7a) -- is a view over ``verdict.status`` and ``winner`` that the
markdown renders from the JSON, never a second computation. A deployment key
never appears: the switch names a ``key_ref``.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from dagnam.audit.candidates import CANDIDATES, CandidateKind, CandidateSpec
from dagnam.audit.economics import customer_verdict, serving_cost_usd_month
from dagnam.audit.frontier import CandidateResult, frontier
from dagnam.audit.orchestrate import FLOOR_BY_STRUCTURE
from dagnam.audit.prices import PriceTable
from dagnam.audit.state import AuditState, StepState
from dagnam.audit.steps import error_code
from dagnam.audit.structure import StructureClass
from dagnam.audit.thresholds import (
    DAYS_PER_MONTH,
    FLOOR_JSON,
    FLOOR_LABEL,
    MAINTENANCE_USD_MONTH,
    MIN_HOLDOUT,
    MIN_TRACES_PER_WORKLOAD,
    RATIO_CANDIDATE,
    RATIO_NOT_WORTH_IT,
)

SCHEMA = "dagnam.audit.report/1"
DEFAULT_BASE_URL = "https://api.dagnam.ai/v1"
"""The OpenAI-compatible root a switched client points at."""


def _hosted_floor(entry: Mapping[str, Any], table: PriceTable) -> tuple[str | None, float | None]:
    """The cheaper hosted variant of the workload's one model and its monthly cost."""
    models = entry.get("models") or []
    row = table.rows.get(str(models[0])) if len(models) == 1 else None
    variant = row.cheaper_variant if row is not None else None
    if variant is None:
        return None, None
    tokens = entry["tokens"]
    window_cost = table.cost(variant, int(tokens["prompt"]), int(tokens["completion"]))
    if window_cost is None:
        return variant, None
    window_days = entry["calls"] / entry["calls_per_day"]
    return variant, window_cost * DAYS_PER_MONTH / window_days


def candidate_status(spec: CandidateSpec, step: StepState) -> str:
    """One word for where a candidate got to: its error code, else its furthest step."""
    if spec.recipe_key is None:
        return "untested"
    code = error_code(step)
    if code is not None:
        return code
    if step.scored:
        return "scored"
    if step.deploy_status is not None:
        return "running" if step.deploy_status == "running" else "deploying"
    if step.run_status is not None:
        return step.run_status
    return "uploaded" if step.dataset_id is not None else "pending"


def _candidate(
    spec: CandidateSpec, step: StepState, entry: Mapping[str, Any], table: PriceTable
) -> dict[str, Any]:
    if spec.serving_rate_key is None:  # the hosted floor: priced from the table, never run
        base, cost = _hosted_floor(entry, table)
    else:
        base = step.base
        cost = serving_cost_usd_month(
            spec.serving_rate_key,
            calls_per_day=float(entry["calls_per_day"]),
            completion_tokens=int(entry["tokens"]["completion"]),
            calls=int(entry["calls"]),
        )
    latency = step.latency or {}
    return {
        "kind": spec.kind.value,
        "base": base,
        "run_id": step.run_id,
        "model_version_id": step.model_version_id,
        "deployment_id": step.deployment_id,
        "agreement": step.agreement,
        "latency_ms": {
            "p50": latency.get("p50"),
            "p95": latency.get("p95"),
            "source": "measured" if step.latency else "untested",
        },
        "serving_cost_usd_month": {"value": cost, "basis": "estimated"},
        "training_cost_credits": step.training_cost_credits,
        "status": candidate_status(spec, step),
    }


def _winner(candidates: list[dict[str, Any]], default_floor: float) -> dict[str, Any] | None:
    points: list[CandidateResult] = []
    floor = default_floor
    for c in candidates:
        agreement, cost = c["agreement"], c["serving_cost_usd_month"]["value"]
        if agreement is None or cost is None:
            continue
        floor = float(agreement.get("floor", floor))
        points.append(
            CandidateResult(
                kind=CandidateKind(c["kind"]),
                ci=(float(agreement["ci95"][0]), float(agreement["ci95"][1])),
                cost_usd_month=float(cost),
                p95_ms=c["latency_ms"]["p95"],
            )
        )
    winner = frontier(points, floor=floor)
    if winner is None:
        return None
    won = next(c for c in candidates if c["kind"] == winner.kind.value)
    return {
        "kind": winner.kind.value,
        "cost_usd_month": winner.cost_usd_month,
        "agreement_lo": winner.agreement_lo,
        "deployment_id": won["deployment_id"],
    }


def build_audit_report(
    state: AuditState,
    scan: Mapping[str, Any],
    *,
    price_table: PriceTable,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    """The ``audit-report.json`` object: ``scan`` plus candidates, winner and switch per workload."""
    workloads: list[dict[str, Any]] = []
    for entry in scan["workloads"]:
        steps = state.workloads.get(str(entry["id"]))
        if steps is None:
            workloads.append({**entry, "candidates": [], "winner": None, "switch": None})
            continue
        cls = StructureClass(str(entry["structure_class"]))
        candidates = [
            _candidate(spec, steps.get(spec.kind, StepState()), entry, price_table)
            for spec in CANDIDATES[cls]
        ]
        winner = _winner(candidates, FLOOR_BY_STRUCTURE[cls])
        key_ref = None
        if winner is not None:
            key_ref = steps[CandidateKind(winner["kind"])].key_ref
        switch = (
            {"base_url": base_url, "model": winner["deployment_id"], "key_ref": key_ref}
            if winner is not None
            else None
        )
        workloads.append({**entry, "candidates": candidates, "winner": winner, "switch": switch})
    return {
        **scan,
        "schema": SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(),
        "scan_generated_at": scan["generated_at"],
        "project_id": state.project_id,
        "halted": state.halted,
        "workloads": workloads,
    }


def write_audit_report(report: Mapping[str, Any], out_dir: Path) -> None:
    """Write ``audit-report.json`` and ``audit-report.md`` (rendered from the JSON) into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2)
    (out_dir / "audit-report.json").write_text(text, encoding="utf-8")
    (out_dir / "audit-report.md").write_text(render_markdown(json.loads(text)), encoding="utf-8")


def render_switch_snippet(
    deployment_id: str, key_ref: str, *, base_url: str = DEFAULT_BASE_URL
) -> str:
    """The stock OpenAI-client snippet that switches a workload; the key stays a ``key_ref``."""
    return "\n".join(
        [
            "from openai import OpenAI",
            "",
            "client = OpenAI(",
            f'    base_url="{base_url}",',
            f"    api_key=DEPLOYMENT_KEY,  # key_ref {key_ref}: read it from your keyring or"
            " audit secrets.json; never paste it here",
            ")",
            f'client.chat.completions.create(model="{deployment_id}", messages=[...])',
        ]
    )


def _usd(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.2f}"


def _label(w: Mapping[str, Any]) -> str:
    return customer_verdict(w["verdict"]["status"], winner=w["winner"] is not None)


def _keep_row(w: Mapping[str, Any]) -> str:
    verdict = w["verdict"]
    ratio = "-" if verdict["ratio"] is None else f"{verdict['ratio']:.1f}x"
    return (
        f"| {w['id']} | {w['structure_class']} | {w['calls_per_day']:,.1f}"
        f" | {_usd(w['cost_usd_month'])} | {ratio} | {verdict['status']}: {verdict['reason']} |"
    )


def _agreement(c: Mapping[str, Any]) -> str:
    a = c["agreement"]
    if a is None:
        return "-"
    return f"{a['value']:.3f} [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}] n={a['n']}"


def _candidate_row(c: Mapping[str, Any], winner: Mapping[str, Any] | None) -> str:
    lat = c["latency_ms"]
    p95 = "-" if lat["p95"] is None else f"{lat['p95']:.0f} ({lat['source']})"
    credits = "-" if c["training_cost_credits"] is None else f"{c['training_cost_credits']:g}"
    cost = c["serving_cost_usd_month"]
    priced = "-" if cost["value"] is None else f"{cost['value']:.2f} ({cost['basis']})"
    mark = " (winner)" if winner is not None and winner["kind"] == c["kind"] else ""
    return (
        f"| {c['kind']}{mark} | {c['base'] or '-'} | {_agreement(c)} | {p95} | {priced}"
        f" | {credits} | {c['status']} |"
    )


_CANDIDATE_HEAD = (
    "| candidate | base | agreement [ci95] | p95 ms | serving $/month | training credits | status |",
    "|---|---|---|---|---|---|---|",
)


def _candidate_section(w: Mapping[str, Any]) -> list[str]:
    lines = [f"### {w['id']} - {_label(w)}", "", f"- {w['verdict']['reason']}"]
    if w["dataset"] is not None:
        d = w["dataset"]
        lines.append(
            f"- dataset: {d['rows']} rows, {d['redactions']} redactions,"
            f" {d['dedup_removed']} duplicates removed, {d['truncated']} truncated;"
            f" split {d['split']}"
        )
    lines += ["", *_CANDIDATE_HEAD, *(_candidate_row(c, w["winner"]) for c in w["candidates"]), ""]
    if w["winner"] is None:
        lines += ["No candidate cleared the floor in this audit; retested on the next one.", ""]
    return lines


def _artifacts(w: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for c in w["candidates"]:
        if c["run_id"] is None:
            continue
        lines.append(
            f"- {w['id']}/{c['kind']}: run {c['run_id']}, model version"
            f" {c['model_version_id'] or '-'}, deployment {c['deployment_id'] or '-'}"
        )
        if c["model_version_id"] is not None:
            lines.append(f"  - export: `dagnam models download {c['model_version_id']} <artifact>`")
    return lines


def render_markdown(js: Mapping[str, Any]) -> str:
    """Render ``audit-report.md`` from the ``audit-report.json`` object; KEEP rows come first."""
    keep = [w for w in js["workloads"] if _label(w) == "KEEP"]
    audited = [w for w in js["workloads"] if w not in keep]
    lines = [
        "# Workload audit",
        "",
        f"- generated: {js['generated_at']} (scan {js['scan_generated_at']})",
        f"- source: {js['source']}",
        f"- price table: {js['price_table_version']}",
        f"- project: {js['project_id'] or '-'}",
    ]
    if js["halted"] is not None:
        lines.append(f"- halted: {js['halted']}")
    lines += ["", "## Not worth replacing", ""]
    lines += [
        "| workload | class | calls/day | $/month | ratio | why |",
        "|---|---|---|---|---|---|",
    ]
    lines += [*(_keep_row(w) for w in keep), "", "## Candidates", ""]
    for w in audited:
        lines += _candidate_section(w)
    lines += ["## Switch", ""]
    for w in audited:
        if w["switch"] is not None:
            snippet = render_switch_snippet(
                w["switch"]["model"],
                w["switch"]["key_ref"] or "-",
                base_url=w["switch"]["base_url"],
            )
            lines += [f"### {w['id']}", "", "```python", snippet, "```", ""]
    lines += ["## Artifacts", ""]
    for w in audited:
        lines += _artifacts(w)
    lines += ["- datasets: `dagnam dataset download <dataset id>`", ""]
    window = js["window"]
    counts = ", ".join(f"{k}: {v}" for k, v in js["pii"]["counts"].items()) or "none found"
    lines += [
        "## Method",
        "",
        f"- window: {window['start']} to {window['end']} ({window['days']:g} days)",
        f"- thresholds: ratio < {RATIO_NOT_WORTH_IT:g}x is not worth it, >= {RATIO_CANDIDATE:g}x is"
        f" a candidate; maintenance ${MAINTENANCE_USD_MONTH:g}/month; floors"
        f" {FLOOR_LABEL} (labels) / {FLOOR_JSON} (JSON) on the agreement lower bound;"
        f" at least {MIN_TRACES_PER_WORKLOAD:,} traces and {MIN_HOLDOUT} holdout rows",
        f"- PII scanned for: {', '.join(js['pii']['pass_list'])}; found: {counts}",
        "- serving costs are estimated from the platform's rate card; latency is measured"
        " through each candidate's endpoint; the hosted floor is untested",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "DEFAULT_BASE_URL",
    "SCHEMA",
    "build_audit_report",
    "candidate_status",
    "render_markdown",
    "render_switch_snippet",
    "write_audit_report",
]
