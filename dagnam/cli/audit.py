"""``dagnam audit``: status, and the command group itself.

``scan`` lives in :mod:`dagnam.cli.audit_scan`, ``run`` in
:mod:`dagnam.cli.audit_run` and ``cancel`` / ``delete`` in
:mod:`dagnam.cli.audit_cleanup`. The commands here drive the platform through
``DagnamClient`` with the stored API key. The ``dagnam.audit`` package is
imported inside each handler so ``dagnam --help`` stays import-light.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dagnam._core.exceptions import APIError, DeploymentNotFoundError
from dagnam.cli.audit_cleanup import cmd_audit_cancel, cmd_audit_delete
from dagnam.cli.audit_run import client_from_env, cmd_audit_run, resolve_audit_dir
from dagnam.cli.audit_scan import (
    cmd_audit_scan,
    parse_sample_rate,
    parse_window,
    retired_live,
    retired_notice,
)
from dagnam.cli.presentation import Column, emit_result, render_table

if TYPE_CHECKING:
    from dagnam._core.client import DagnamClient
    from dagnam.audit.state import AuditState, StepState
    from dagnam.cli.common import SubParsersAction

SOURCES = ("langfuse", "langsmith", "openai", "jsonl", "csv")
DEFAULT_OUT = "./audit"
RETIRED = "retired"
"""The status of a candidate a forced rescan took out of the audit; its artifacts may live on."""
METRICS_RANGE = "7d"
"""The longest range the metrics endpoint serves; a 30-day view waits on the backend."""


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


def _status_row(
    workload: str | None,
    candidate: str | None,
    status: str,
    step: StepState,
    client: DagnamClient | None,
) -> dict[str, Any]:
    """One candidate's row; ``requests_7d`` from the metrics endpoint when a client is given."""
    agreement = step.agreement
    requests: int | None = None
    if client is not None and step.deployment_id is not None:
        try:
            metrics = client.get_deployment_metrics(step.deployment_id, time_range=METRICS_RANGE)
        except (APIError, DeploymentNotFoundError, TypeError):
            metrics = {}  # deleted, or a read that failed: the count is a nicety, not the status
        count = metrics.get("requests_count")
        requests = int(count) if isinstance(count, int | float) else None
    return {
        "workload": workload,
        "candidate": candidate,
        "status": status,
        "run_status": step.run_status,
        "training_job_id": step.training_job_id,
        "deployment_id": step.deployment_id,
        "agreement": None if agreement is None else agreement["value"],
        "credits": step.training_cost_credits,
        "requests_7d": requests,
    }


def status_rows(state: AuditState, client: DagnamClient | None) -> list[dict[str, Any]]:
    """One row per workload x candidate, then one per candidate a forced rescan retired.

    A retired candidate has left its workload -- the rescan rewrote the rows it
    was trained on -- but its run and its endpoint are still on the platform
    until a cancel or a delete, so they stay in the table, as :data:`RETIRED`,
    under the workload and candidate they had been (unnamed when an older
    state retired them without recording it). One that never reached the
    platform (the hosted floor, say) left nothing there to show.
    """
    from dagnam.audit.candidates import CANDIDATES
    from dagnam.audit.report import candidate_status

    specs = {spec.kind: spec for specs in CANDIDATES.values() for spec in specs}
    rows = [
        _status_row(workload_id, kind.value, candidate_status(specs[kind], step), step, client)
        for workload_id, candidates in state.workloads.items()
        for kind, step in candidates.items()
    ]
    rows.extend(
        _status_row(step.workload_id, step.kind and step.kind.value, RETIRED, step, client)
        for step in state.retired
        if step.dataset_id or step.training_job_id or step.model_version_id or step.deployment_id
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
    live = [(str(row["kind"]), str(row["id"])) for row in js["retired_live"]]
    if live:
        lines.extend(["", *retired_notice(live, Path(js["audit_dir"]))])
    return "\n".join(lines)


def cmd_audit_status(args: argparse.Namespace) -> None:
    """The state table, one row per candidate, with each endpoint's 7-day requests.

    Candidates a forced rescan retired are listed too, and whatever of theirs
    is still going is named with the command that stops it.
    """
    from dagnam._core.auth import get_api_url
    from dagnam.audit.publish import audit_url
    from dagnam.audit.state import load_state

    audit_dir = resolve_audit_dir(args.audit_dir)
    state = load_state(audit_dir)
    needs_client = any(step.deployment_id is not None for step in state.all_steps())
    rows = status_rows(state, client_from_env() if needs_client else None)
    audit_id = state.audit_id
    result = {
        "audit_dir": args.audit_dir,
        "project_id": state.project_id,
        "audit_id": audit_id,
        "url": None if audit_id is None else audit_url(get_api_url(), audit_id),
        "halted": state.halted,
        "rows": rows,
        "retired_live": [{"kind": kind, "id": item_id} for kind, item_id in retired_live(state)],
    }
    emit_result(result, output=None, json_stdout=args.json, render_human=_render_status)


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
        "--sample-rate",
        type=parse_sample_rate,
        default=1.0,
        metavar="0.1",
        help="The share of traffic a sampled export holds; volume and spend are scaled up.",
    )
    scan.add_argument(
        "--out", default=DEFAULT_OUT, help=f"Audit directory (default {DEFAULT_OUT})."
    )
    scan.add_argument("--price-table", help="Override the bundled vendor price table (JSON).")
    scan.add_argument("--json", action="store_true", help="Print scan-report.json to stdout.")
    scan.add_argument("--force", action="store_true", help="Rewrite rows a run in --out used.")
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
        help="Credit ceiling for training plus the metered holdout replay; nothing that could"
        " pass it is started (default: the plan's estimate, rounded up to 100).",
    )
    run.add_argument(
        "--local-only",
        action="store_true",
        help="Do not mirror this run's progress and report into your account (the rows are"
        " still uploaded to, and trained on, the platform).",
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
    delete.add_argument(
        "--already-deleted",
        action="store_true",
        help=(
            "The platform answers 404 for this audit and you know it is deleted: remove this"
            " directory's local rows and deployment keys instead of keeping them. Irreversible:"
            " check `dagnam whoami` first."
        ),
    )
    delete.add_argument(
        "--include-endpoints",
        action="store_true",
        help=(
            "Also delete endpoints that are still serving; apps calling them will start"
            " getting errors. Without it the delete stops before deleting anything while one"
            " is serving."
        ),
    )
    delete.add_argument("--json", action="store_true", help="Print the receipt as JSON.")
    delete.set_defaults(func=cmd_audit_delete)
