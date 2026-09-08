"""``dagnam audit``: scan, status, cancel, delete (``run`` lives in :mod:`dagnam.cli.audit_run`).

``scan`` never opens a socket: it reads the export, discovers and prices the
workloads, derives the redacted rows for the ones worth auditing, and writes
``scan-report.{json,md}`` plus ``workloads/<id>/`` under ``--out``. The other
commands drive the platform through ``DagnamClient`` with the stored API key.
The ``dagnam.audit`` package is imported inside each handler so ``dagnam
--help`` stays import-light.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dagnam._core.exceptions import DeploymentNotFoundError
from dagnam.cli.audit_run import client_from_env, cmd_audit_run, fail
from dagnam.cli.common import confirm_or_abort, print_next_step
from dagnam.cli.presentation import Column, emit_result, render_table

if TYPE_CHECKING:
    from dagnam._core.client import DagnamClient
    from dagnam.audit.derive import WorkloadDataset
    from dagnam.audit.state import AuditState
    from dagnam.cli.common import SubParsersAction

SOURCES = ("langfuse", "langsmith", "openai", "jsonl", "csv")
DEFAULT_OUT = "./audit"
METRICS_RANGE = "7d"
"""The longest range the metrics endpoint serves; a 30-day view waits on the backend."""


def parse_window(value: str) -> int:
    """``30d`` or ``30`` -> 30; anything else is a usage error."""
    digits = value[:-1] if value.endswith("d") else value
    if not (digits.isascii() and digits.isdigit()) or int(digits) < 1:
        raise argparse.ArgumentTypeError(
            f"--window expects a number of days like 30d, not {value!r}"
        )
    return int(digits)


def parse_floor(value: str) -> float:
    """A quality floor in ``(0, 1]``; anything else is a usage error.

    The platform holds the same bound, so catching it here turns a value that
    would make every publish a dropped 422 into an argument error before the
    run starts.
    """
    try:
        floor = float(value)
    except ValueError:
        floor = float("nan")
    if not 0 < floor <= 1:
        raise argparse.ArgumentTypeError(
            f"--floor expects an agreement lower bound above 0 and at most 1, not {value!r}"
        )
    return floor


def parse_credits(value: str) -> int:
    """A credit ceiling of zero or more; anything else is a usage error.

    ``str.isdigit`` is true of ``"\u00b2"`` and every other non-ASCII digit
    form, none of which ``int()`` accepts -- so the ASCII check is what keeps
    this an argument error rather than a traceback.
    """
    if not (value.isascii() and value.isdigit()):
        raise argparse.ArgumentTypeError(
            f"--max-credits expects a whole number of credits, not {value!r}"
        )
    return int(value)


def parse_map(pairs: Sequence[str] | None) -> dict[str, str] | None:
    """``["field=column", ...]`` -> ``{"field": "column"}``; ``None`` when no pair was given."""
    if not pairs:
        return None
    mapping: dict[str, str] = {}
    for pair in pairs:
        field, sep, column = pair.partition("=")
        if not sep or not field or not column:
            raise argparse.ArgumentTypeError(f"--map expects field=column, not {pair!r}")
        mapping[field] = column
    return mapping


def _dataset_entry(dataset: WorkloadDataset) -> dict[str, Any]:
    """The contract's ``dataset`` object for one derived workload."""
    stats = dataset.stats
    return {
        "rows": len(dataset.rows),
        "dedup_removed": stats["dedup"]["removed"],
        "redactions": sum(stats["redact"]["counts"].values()),
        "truncated": stats["derive"]["truncated"],
        "split": {name: len(rows) for name, rows in dataset.split.items()},
    }


def _render_scan(result: object) -> str:
    js: Mapping[str, Any] = result if isinstance(result, dict) else {}
    verdicts = Counter(str(w["verdict"]["status"]) for w in js["workloads"])
    rows = [
        {
            "id": w["id"],
            "class": w["structure_class"],
            "calls_per_day": f"{w['calls_per_day']:,.1f}",
            "cost": "unknown" if w["cost_usd_month"] is None else f"{w['cost_usd_month']:,.2f}",
            "verdict": w["verdict"]["status"],
        }
        for w in js["workloads"]
    ]
    table = render_table(
        (
            Column("Workload", "id", 20),
            Column("Class", "class", 12),
            Column("Calls/day", "calls_per_day", 12, "right"),
            Column("$/month", "cost", 12, "right"),
            Column("Verdict", "verdict", 16),
        ),
        rows,
    )
    summary = ", ".join(f"{n} {status}" for status, n in sorted(verdicts.items()))
    lines = [table, "", f"{len(rows)} workloads: {summary or 'none'}"]
    lines.extend(f"warning: {w}" for w in js["warnings"])
    return "\n".join(lines)


def cmd_audit_scan(args: argparse.Namespace) -> None:
    """Read the export, discover and price workloads, derive rows; no network."""
    from dagnam.audit import (
        build_dataset,
        build_scan_report,
        discover_workloads,
        read_traces,
        replaceability,
        write_scan_report,
        write_workload,
    )
    from dagnam.audit.economics import price_workload
    from dagnam.audit.orchestrate import AUDITED_VERDICTS
    from dagnam.audit.prices import PriceTable
    from dagnam.audit.redact import PII_POLICY
    from dagnam.audit.scan_report import Window
    from dagnam.audit.thresholds import MAX_SEQ_LENGTH

    out = Path(args.out)
    stream, _stats = read_traces(
        Path(args.export), source=args.source, column_map=parse_map(args.map)
    )
    records = list(stream)
    if not records:
        fail(args, f"{args.export}: no traces to audit")
    table = PriceTable.load(Path(args.price_table) if args.price_table else None)
    workloads = discover_workloads(records, window_days=args.window)
    datasets: dict[str, dict[str, Any]] = {}
    pii_counts: Counter[str] = Counter()
    for w in workloads:
        if replaceability(price_workload(w, table)).status not in AUDITED_VERDICTS:
            continue
        dataset = build_dataset(
            [records[i] for i in w.record_indices],
            structure_class=w.structure_class.value,
            max_seq_length=MAX_SEQ_LENGTH,
        )
        write_workload(out, w.id, dataset.rows, dataset.split, dataset.stats)
        datasets[w.id] = _dataset_entry(dataset)
        pii_counts.update(dataset.stats["redact"]["counts"])
    stamps = [r.ts for r in records]
    span = (max(stamps) - min(stamps)).total_seconds() / 86_400
    report = build_scan_report(
        workloads,
        source=args.source,
        window=Window(min(stamps), max(stamps), max(float(args.window), span)),
        price_table=table,
        pii_pass_list=list(PII_POLICY),
        pii_counts=dict(pii_counts),
        datasets=datasets,
    )
    write_scan_report(report, out)
    emit_result(report.to_json(), output=None, json_stdout=args.json, render_human=_render_scan)
    if datasets:
        print_next_step(f"dagnam audit run {out}")


def status_rows(state: AuditState, client: DagnamClient | None) -> list[dict[str, Any]]:
    """One row per workload x candidate; ``requests_7d`` from the metrics endpoint when a client is given."""
    from dagnam.audit.candidates import CANDIDATES
    from dagnam.audit.report import candidate_status

    specs = {spec.kind: spec for specs in CANDIDATES.values() for spec in specs}
    rows: list[dict[str, Any]] = []
    for workload_id, candidates in state.workloads.items():
        for kind, step in candidates.items():
            agreement = step.agreement
            requests: int | None = None
            if client is not None and step.deployment_id is not None:
                try:
                    metrics = client.get_deployment_metrics(
                        step.deployment_id, time_range=METRICS_RANGE
                    )
                except DeploymentNotFoundError:
                    metrics = {}  # deleted out from under the state file; `audit delete` reconciles
                count = metrics.get("requests_count")
                requests = int(count) if isinstance(count, int | float) else None
            rows.append(
                {
                    "workload": workload_id,
                    "candidate": kind.value,
                    "status": candidate_status(specs[kind], step),
                    "run_status": step.run_status,
                    "training_job_id": step.training_job_id,
                    "deployment_id": step.deployment_id,
                    "agreement": None if agreement is None else agreement["value"],
                    "credits": step.training_cost_credits,
                    "requests_7d": requests,
                }
            )
    return rows


def _render_status(result: object) -> str:
    js: Mapping[str, Any] = result if isinstance(result, dict) else {}
    rows: list[dict[str, Any]] = [
        {
            **{k: "-" if v is None else v for k, v in row.items()},
            "agreement": "-" if row["agreement"] is None else f"{row['agreement']:.3f}",
        }
        for row in js["rows"]
    ]
    if not rows:
        return "No candidates yet: run `dagnam audit run <audit-dir>`."
    table = render_table(
        (
            Column("Workload", "workload", 20),
            Column("Candidate", "candidate", 12),
            Column("Status", "status", 18),
            Column("Job", "training_job_id", 36),
            Column("Deployment", "deployment_id", 36),
            Column("Agreement", "agreement", 10, "right"),
            Column("Credits", "credits", 8, "right"),
            Column("Req/7d", "requests_7d", 8, "right"),
        ),
        rows,
    )
    lines = [table]
    if js["audit_id"] is not None:
        lines.append(f"\naudit {js['audit_id']}: {js['url']}")
    if js["halted"] is not None:
        lines.append(f"\nhalted: {js['halted']}")
    return "\n".join(lines)


def cmd_audit_status(args: argparse.Namespace) -> None:
    """The state table, one row per workload x candidate, with each endpoint's 7-day requests."""
    from dagnam._core.auth import get_api_url
    from dagnam.audit.publish import audit_url
    from dagnam.audit.state import load_state

    state = load_state(Path(args.audit_dir))
    needs_client = any(
        step.deployment_id is not None for c in state.workloads.values() for step in c.values()
    )
    rows = status_rows(state, client_from_env() if needs_client else None)
    audit_id = state.audit_id
    result = {
        "project_id": state.project_id,
        "audit_id": audit_id,
        "url": None if audit_id is None else audit_url(get_api_url(), audit_id),
        "halted": state.halted,
        "rows": rows,
    }
    emit_result(result, output=None, json_stdout=args.json, render_human=_render_status)


def _render_receipt(receipt: Mapping[str, Any], path: Path) -> str:
    """One line per artifact the server touched, then where the receipt was written.

    Every field is read with a default: a row the server spells differently
    from this client is worth showing as ``?``, never worth a ``KeyError`` that
    hides the whole receipt.
    """
    from dagnam.audit.cleanup import receipt_rows

    return "\n".join(
        [
            *(
                f"{row.get('kind', '?')} {row.get('id', '?')}: {row.get('status', '?')}"
                + (f" ({row['reason']})" if row.get("reason") else "")
                for row in receipt_rows(receipt)
            ),
            f"Receipt: {path}",
        ]
    )


def cmd_audit_cancel(args: argparse.Namespace) -> None:
    """Cancel every in-flight run and pause every deployment the state records; delete nothing."""
    from dagnam._core.exceptions import DeploymentStateError
    from dagnam.audit.cleanup import (
        CANCELLED_FILE,
        DEPLOY_PAUSED,
        TERMINAL_RUN,
        mark_cancelled,
        receipt_rows,
        write_receipt,
    )
    from dagnam.audit.state import load_state, save_state

    audit_dir = Path(args.audit_dir)
    state = load_state(audit_dir)
    client = client_from_env()
    if state.audit_id is not None:
        # The run published: the account knows every artifact it created, so the
        # server stops them all in one call and answers with the receipt.
        receipt = client.cancel_audit(state.audit_id)
        path = write_receipt(audit_dir, receipt, CANCELLED_FILE)
        # A deployment the server could not pause is not paused here either:
        # the two cancel paths must leave the same state for the same situation.
        mark_cancelled(
            state,
            [
                str(row["id"])
                for row in receipt_rows(receipt)
                if row.get("kind") == "deployment"
                and row.get("status") == "blocked"
                and row.get("id")
            ],
        )
        save_state(audit_dir, state)
        emit_result(
            receipt,
            output=None,
            json_stdout=args.json,
            render_human=lambda _: _render_receipt(receipt, path),
        )
        return
    actions: list[dict[str, str]] = []
    unpaused: list[str] = []
    for workload_id, candidates in state.workloads.items():
        for kind, step in candidates.items():
            label = f"{workload_id}/{kind.value}"
            job = step.training_job_id
            if job is not None and step.run_status not in TERMINAL_RUN:
                client.cancel_training_job(job)
                actions.append({"candidate": label, "action": "cancelled_job", "id": job})
            if step.deployment_id is not None and step.deploy_status != DEPLOY_PAUSED:
                # A deployment whose revision never activated sits in ``not_provisioned``
                # and the platform refuses the transition; record it and carry on.
                try:
                    client.pause_deployment(step.deployment_id)
                except DeploymentStateError as exc:
                    unpaused.append(step.deployment_id)
                    actions.append(
                        {
                            "candidate": label,
                            "action": "pause_refused",
                            "id": step.deployment_id,
                            "reason": str(exc),
                        }
                    )
                else:
                    actions.append(
                        {
                            "candidate": label,
                            "action": "paused_deployment",
                            "id": step.deployment_id,
                        }
                    )
    mark_cancelled(state, unpaused)
    save_state(audit_dir, state)
    emit_result(
        {"halted": state.halted, "actions": actions},
        output=None,
        json_stdout=args.json,
        render_human=lambda _: "\n".join(
            [*(f"{a['action']} {a['id']} ({a['candidate']})" for a in actions), "Cancelled."]
        ),
    )


def cmd_audit_delete(args: argparse.Namespace) -> None:
    """Delete every platform artifact the run created and write ``deleted.json``."""
    from dagnam.audit.cleanup import (
        DELETED_FILE,
        delete_audit,
        forget_locally,
        receipt_rows,
        recorded_ids,
        write_receipt,
    )
    from dagnam.audit.state import load_state

    audit_dir = Path(args.audit_dir)
    state = load_state(audit_dir)
    ids = recorded_ids(state)
    listing = "\n".join(f"  {kind}: {', '.join(found)}" for kind, found in ids.items() if found)
    if state.audit_id is not None:
        listing = f"{listing}\n  audit: {state.audit_id} (and its published report)".lstrip("\n")
    confirm_or_abort(
        f"This deletes from your account:\n{listing or '  (nothing recorded)'}", assume_yes=args.yes
    )
    client = client_from_env()
    if state.audit_id is None:
        receipt = delete_audit(audit_dir, client)
    else:
        # The published audit owns the same artifacts; the server walks them and
        # answers with the receipt, and only the local rows are left to drop.
        receipt = client.delete_audit(state.audit_id)
        write_receipt(audit_dir, receipt)
        if not any(row.get("status") == "blocked" for row in receipt_rows(receipt)):
            forget_locally(audit_dir, state)
    emit_result(
        receipt,
        output=None,
        json_stdout=args.json,
        render_human=lambda _: _render_receipt(receipt, audit_dir / DELETED_FILE),
    )


def register_audit(subparsers: SubParsersAction) -> None:
    """Register the ``audit`` command group on the top-level subparsers."""
    audit = subparsers.add_parser(
        "audit",
        help="Audit exported LLM traces: find replaceable workloads, train and serve candidates.",
        description="Workload audit: scan traces locally, then train, serve and score candidates.",
    )
    sub = audit.add_subparsers(dest="audit_command", required=True)

    scan = sub.add_parser(
        "scan",
        help="Discover and price workloads from a trace export (no network).",
        description="Read a trace export, discover workloads, price them, derive redacted rows.",
    )
    scan.add_argument("export", help="Path to the trace export file.")
    scan.add_argument("--source", required=True, choices=SOURCES, help="Export format.")
    scan.add_argument(
        "--map",
        action="append",
        metavar="FIELD=COLUMN",
        help="Map a trace field to one of your columns (jsonl/csv sources); repeatable.",
    )
    scan.add_argument(
        "--window",
        type=parse_window,
        default=30,
        metavar="30d",
        help="Export window the per-day rates assume when the timestamps span less.",
    )
    scan.add_argument(
        "--out", default=DEFAULT_OUT, help=f"Audit directory (default {DEFAULT_OUT})."
    )
    scan.add_argument("--price-table", help="Override the bundled vendor price table (JSON).")
    scan.add_argument("--json", action="store_true", help="Print scan-report.json to stdout.")
    scan.set_defaults(func=cmd_audit_scan)

    run = sub.add_parser(
        "run",
        help="Train, deploy and score candidates for the workloads worth auditing.",
        description="Upload the derived rows (after confirmation), then run the candidate frontier.",
    )
    run.add_argument("audit_dir", help="The directory `audit scan --out` wrote.")
    run.add_argument(
        "--workloads", help="Comma-separated workload ids to run (default: all audited)."
    )
    run.add_argument(
        "--floor", type=parse_floor, help="Quality floor on the agreement lower bound (0 < f <= 1)."
    )
    run.add_argument(
        "--max-credits",
        type=parse_credits,
        default=None,
        help="Credit ceiling for training plus the metered holdout replay; stop before"
        " exceeding it.",
    )
    run.add_argument(
        "--local-only",
        action="store_true",
        help="Do not publish this run to your account; everything stays in the audit directory.",
    )
    run.add_argument("--yes", action="store_true", help="Skip the upload confirmation.")
    run.add_argument(
        "--no-wait", action="store_true", help="Return after submitting; resume later."
    )
    run.add_argument("--json", action="store_true", help="Print audit-report.json to stdout.")
    run.set_defaults(func=cmd_audit_run)

    status = sub.add_parser("status", help="Show every job and endpoint the run created.")
    status.add_argument("audit_dir", help="The audit directory.")
    status.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    status.set_defaults(func=cmd_audit_status)

    cancel = sub.add_parser("cancel", help="Cancel running jobs and pause deployments.")
    cancel.add_argument("audit_dir", help="The audit directory.")
    cancel.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    cancel.set_defaults(func=cmd_audit_cancel)

    delete = sub.add_parser(
        "delete", help="Delete every artifact the run created; write a receipt."
    )
    delete.add_argument("audit_dir", help="The audit directory.")
    delete.add_argument("--yes", action="store_true", help="Skip the typed confirmation.")
    delete.add_argument("--json", action="store_true", help="Print the receipt as JSON.")
    delete.set_defaults(func=cmd_audit_delete)
