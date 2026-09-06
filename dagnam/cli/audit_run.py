"""``dagnam audit run``: say exactly what will leave the machine, get a yes, then run the frontier.

Nothing is uploaded before the listing: which workloads, how many redacted
rows each, the redaction counts per class, which project they go to and the
credit ceiling. Without ``--yes`` the command asks once on a terminal and
refuses outright when stdin is not one (spec section 10).
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, NoReturn

from dagnam.cli.common import confirm_or_abort, error, print_next_step
from dagnam.cli.presentation import emit_result

if TYPE_CHECKING:
    from dagnam._core.client import DagnamClient
    from dagnam.audit.state import AuditState
    from dagnam.audit.structure import StructureClass


def fail(args: argparse.Namespace, message: str, *, hint: str | None = None) -> NoReturn:
    """Exit 1 with a one-line reason; under ``--json`` the reason is a JSON object on stdout."""
    if getattr(args, "json", False):
        print(json.dumps({"error": message, "hint": hint}))
        sys.exit(1)
    error(message, hint=hint)


def client_from_env() -> DagnamClient:
    """The SDK client on the stored credentials (never a CLI flag)."""
    from dagnam._core.auth import get_api_key, get_api_url
    from dagnam._core.client import DagnamClient

    return DagnamClient(get_api_url(), get_api_key())


def upload_listing(
    audit_dir: Path,
    selected: Sequence[tuple[str, StructureClass]],
    state: AuditState,
    max_credits: int,
) -> list[str]:
    """The lines the user confirms: every workload, its rows and redactions, the project, the ceiling."""
    lines = ["About to upload (redacted, derived rows only; raw traces stay here):"]
    for workload_id, structure_class in selected:
        meta = json.loads(
            (audit_dir / "workloads" / workload_id / "meta.json").read_text(encoding="utf-8")
        )
        redact = meta["stats"]["redact"]
        counts = ", ".join(f"{k}: {v}" for k, v in redact["counts"].items()) or "none found"
        lines.append(
            f"  {workload_id} ({structure_class.value}): {meta['rows']} rows"
            f" (split {meta['splits']}); redactions: {counts}"
            f" in {redact['rows_changed']} rows; scanned for {', '.join(redact['pass_list'])}"
        )
    project = (
        f"existing project {state.project_id}"
        if state.project_id is not None
        else f"a new private project 'workload-audit-{audit_dir.resolve().name}'"
    )
    lines.append(f"  to: {project}")
    lines.append(f"  credit ceiling: {max_credits}")
    return lines


def _render_run(audit_dir: Path) -> Any:
    from dagnam.audit.economics import customer_verdict

    def render(result: object) -> str:
        js: Mapping[str, Any] = result if isinstance(result, dict) else {}
        lines: list[str] = []
        for w in js["workloads"]:
            if not w["candidates"]:
                continue
            winner = w["winner"]
            label = customer_verdict(w["verdict"]["status"], winner=winner is not None)
            detail = (
                f" - {winner['kind']} at ${winner['cost_usd_month']:.2f}/month,"
                f" agreement >= {winner['agreement_lo']:.3f}; switch model {winner['deployment_id']}"
                if winner is not None
                else ""
            )
            lines.append(f"{w['id']}: {label}{detail}")
        lines.append(f"Report: {audit_dir / 'audit-report.md'}")
        return "\n".join(lines)

    return render


def cmd_audit_run(args: argparse.Namespace) -> None:
    """List what will be uploaded, confirm, run (or resume) the frontier, write the report."""
    from dagnam.audit.orchestrate import (
        DEFAULT_MAX_CREDITS,
        SCAN_REPORT,
        run_audit,
        select_workloads,
    )
    from dagnam.audit.prices import PriceTable
    from dagnam.audit.report import build_audit_report, write_audit_report
    from dagnam.audit.state import load_state

    audit_dir = Path(args.audit_dir)
    workloads = args.workloads.split(",") if args.workloads else None
    max_credits = DEFAULT_MAX_CREDITS if args.max_credits is None else args.max_credits
    try:
        selected = select_workloads(audit_dir, workloads)
    except (FileNotFoundError, ValueError) as exc:
        fail(args, str(exc))
    if not selected:
        fail(args, "nothing to run: the scan found no workload worth auditing")
    state = load_state(audit_dir)
    print("\n".join(upload_listing(audit_dir, selected, state, max_credits)))
    if not args.yes and not sys.stdin.isatty():
        fail(
            args,
            "refusing to upload without confirmation on a non-interactive terminal",
            hint=f"dagnam audit run {audit_dir} --yes",
        )
    confirm_or_abort("Upload the rows listed above?", assume_yes=args.yes)

    client = client_from_env()
    state = run_audit(
        audit_dir,
        floor=args.floor,
        workloads=workloads,
        max_credits=max_credits,
        wait=not args.no_wait,
        client=client,
    )
    scan = json.loads((audit_dir / SCAN_REPORT).read_text(encoding="utf-8"))
    # ponytail: the report prices the hosted floor from the bundled table; a
    # scan run with --price-table gets its version noted but not its rows.
    report = build_audit_report(
        state, scan, price_table=PriceTable.load(None), base_url=f"{client.api_url}/v1"
    )
    write_audit_report(report, audit_dir)
    if state.halted is not None:
        fail(
            args,
            f"audit halted: {state.halted['reason']} ({audit_dir / 'audit-report.md'} written)",
            hint=f"dagnam audit status {audit_dir}",
        )
    emit_result(report, output=None, json_stdout=args.json, render_human=_render_run(audit_dir))
    if args.no_wait:
        print_next_step(f"dagnam audit run {audit_dir} --yes")
