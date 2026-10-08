"""``dagnam audit run``: say exactly what will leave the machine, get a yes, then run the frontier.

Nothing is uploaded before the listing: which workloads, how many redacted
rows each, the redaction counts per class, which project they go to, the
credit ceiling, and -- unless ``--local-only`` was passed -- that the run's
progress and report are mirrored into the account. Without ``--yes`` the
command asks once on a terminal and refuses outright when stdin is not one.

Between the yes and the first upload the platform is asked which contract it
runs (:func:`dagnam.audit.preflight.check_platform`, called by the run itself):
an SDK ahead of its platform stops there, with both versions, rather than after
the rows have gone up.
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


NOT_FOUND = 404


def fail(args: argparse.Namespace, message: str, *, hint: str | None = None) -> NoReturn:
    """Exit 1 with a one-line reason; under ``--json`` the reason is a JSON object on stdout."""
    if getattr(args, "json", False):
        print(json.dumps({"error": message, "hint": hint}))
        sys.exit(1)
    error(message, hint=hint)


def resolve_audit_dir(value: str) -> Path:
    """An audit directory argument, resolved once: a link NAMING the directory is the user's choice.

    A temp or home directory is often one, so the real path is used from then on, while a link
    INSIDE the directory (a workload folder, a file the audit writes or reads) is still refused.
    """
    return Path(value).resolve()


def client_from_env() -> DagnamClient:
    """The SDK client on the stored credentials (never a CLI flag)."""
    from dagnam._core.auth import get_api_key, get_api_url
    from dagnam._core.client import DagnamClient

    return DagnamClient(get_api_url(), get_api_key())


def announce(line: str) -> None:
    """Print one publisher line to stderr, so stdout stays the report ``--json`` pipes."""
    print(line, file=sys.stderr)


PUBLISH_LINE = (
    "  published to your account: progress and the report (workload ids, verdicts, spend,"
    " masked excerpts, the audit directory's name; never rows or keys);"
    " 'audit delete' removes them; --local-only keeps them here"
)
"""What ``audit run`` says it mirrors into the account, in the listing the user confirms."""


def upload_listing(
    audit_dir: Path,
    selected: Sequence[tuple[str, StructureClass]],
    state: AuditState,
    max_credits: int,
    *,
    local_only: bool = False,
    planned: bool = False,
) -> list[str]:
    """The lines the user confirms: every workload, its rows and redactions, the project, the ceiling.

    ``planned`` says the ceiling is the plan's own estimate (no ``--max-credits``).
    """
    from dagnam.audit.steps import workload_meta

    lines = ["About to upload (redacted, derived rows only; raw traces stay here):"]
    for workload_id, structure_class in selected:
        meta = workload_meta(audit_dir, workload_id)
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
    basis = (
        "the plan's estimate, rounded up to 100; --max-credits sets your own"
        if planned
        else "training and the metered holdout replay"
    )
    lines.append(f"  credit ceiling: {max_credits} ({basis})")
    if not local_only:
        lines.append(PUBLISH_LINE)
    return lines


def _render_run(audit_dir: Path) -> Any:
    """The terminal's summary: a line per audited workload, and under it why any candidate stopped.

    A candidate's status is one word (its error code); the reason behind that
    word -- which names the cause and what to do -- is the report's ``error``,
    printed here so that it is on the terminal and not only in the files.
    """
    from dagnam.audit.economics import customer_verdict
    from dagnam.audit.readers.messages import TOOL_CALL_NOTE
    from dagnam.audit.report import TOOL_CALL, no_winner_reason

    def render(result: object) -> str:
        js: Mapping[str, Any] = result if isinstance(result, dict) else {}
        lines: list[str] = []
        for w in js["workloads"]:
            if not w["candidates"]:
                continue
            winner = w["winner"]
            label = customer_verdict(w["verdict"]["status"], winner=winner is not None)
            if winner is not None:
                detail = (
                    f" - {winner['kind']} at ${winner['cost_usd_month']:.2f}/month,"
                    f" agreement >= {winner['agreement_lo']:.3f};"
                    f" switch model {winner['deployment_id']}"
                )
            else:
                # Said, not left to the status column: a failed deploy or a run
                # still training measured nothing, which is not missing the floor.
                detail = f" - {no_winner_reason(w)}" if label == "NOT YET" else ""
            lines.append(f"{w['id']}: {label}{detail}")
            lines.extend(f"  {c['kind']}: {c['error']}" for c in w["candidates"] if c.get("error"))
            if winner is not None and w.get("response_mode") == TOOL_CALL:
                lines.append(f"  {TOOL_CALL_NOTE}")
        lines.append(f"Report: {audit_dir / 'audit-report.md'}")
        return "\n".join(lines)

    return render


def _halt_line(audit_dir: Path, halted: Mapping[str, Any]) -> str:
    """Why the run stopped, with its cause; a run that never got its audit says nothing was spent."""
    reason, detail = halted["reason"], halted.get("detail")
    line = f"audit halted: {reason}" + (f": {detail}" if detail else "")
    if reason == "publish_failed":
        line += (
            "; nothing was uploaded or spent; once the cause is fixed, run `dagnam audit run` again"
        )
    return f"{line} ({audit_dir / 'audit-report.md'} written)"


def cmd_audit_run(args: argparse.Namespace) -> None:
    """List what will be uploaded, confirm, run (or resume) the frontier, write the report."""
    from dagnam.audit.preflight import PlatformTooOldError
    from dagnam.audit.state import (
        DELETED_STATE,
        AuditBusyError,
        AuditDeletedError,
        load_state,
        lock_audit,
    )
    from dagnam.audit.workspace import UnsafeWorkloadsError

    audit_dir = resolve_audit_dir(args.audit_dir)
    try:
        # Held from before the listing to the end of the run: a scan in between
        # would replace the rows after they were listed, and a second run would
        # pass the same prompt and submit the same work again.
        with lock_audit(audit_dir):
            if load_state(audit_dir).halted == DELETED_STATE:  # before it lists anything to upload
                raise AuditDeletedError(f"{audit_dir} is a deleted audit: there is nothing to run")
            state, report = _confirm_and_run(args, audit_dir)
    except (
        AuditBusyError,
        AuditDeletedError,
        FileNotFoundError,
        PlatformTooOldError,
        UnsafeWorkloadsError,
    ) as exc:
        fail(args, str(exc))
    if state.halted is not None:
        fail(args, _halt_line(audit_dir, state.halted), hint=f"dagnam audit status {audit_dir}")
    emit_result(report, output=None, json_stdout=args.json, render_human=_render_run(audit_dir))
    if args.no_wait:
        print_next_step(f"dagnam audit run {audit_dir} --yes")


def _confirm_and_run(
    args: argparse.Namespace, audit_dir: Path
) -> tuple[AuditState, dict[str, Any]]:
    """The part of ``audit run`` that needs ``audit_dir`` held: listing, prompt, run, report."""
    from dagnam.audit.orchestrate import (
        SCAN_REPORT,
        plan_credits,
        run_audit_held,
        select_workloads,
    )
    from dagnam.audit.prices import PriceTable
    from dagnam.audit.publish import Publisher
    from dagnam.audit.report import build_audit_report, write_audit_report
    from dagnam.audit.state import load_state
    from dagnam.audit.workspace import UnsafeWorkloadsError, read_regular

    workloads = args.workloads.split(",") if args.workloads else None
    try:
        selected = select_workloads(audit_dir, workloads)
    except (FileNotFoundError, ValueError, UnsafeWorkloadsError) as exc:
        fail(args, str(exc))
    if not selected:
        fail(args, "nothing to run: the scan found no workload worth auditing")
    state = load_state(audit_dir)
    planned = args.max_credits is None
    max_credits = plan_credits(audit_dir, selected, state) if planned else args.max_credits
    listing = upload_listing(
        audit_dir, selected, state, max_credits, local_only=args.local_only, planned=planned
    )
    print("\n".join(listing))
    if not args.yes and not sys.stdin.isatty():
        fail(
            args,
            "refusing to upload without confirmation on a non-interactive terminal",
            hint=f"dagnam audit run {audit_dir} --yes",
        )
    confirm_or_abort("Upload the rows listed above?", assume_yes=args.yes)

    client = client_from_env()
    state = run_audit_held(
        audit_dir,
        floor=args.floor,
        workloads=workloads,
        max_credits=max_credits,
        wait=not args.no_wait,
        client=client,
        publisher=None if args.local_only else Publisher(client, state, announce),
        notice=announce,
    )
    scan = json.loads(read_regular(audit_dir / SCAN_REPORT))
    # ponytail: the report prices the hosted floor from the bundled table; a
    # scan run with --price-table gets its version noted but not its rows.
    report = build_audit_report(
        state, scan, price_table=PriceTable.load(None), base_url=f"{client.api_url}/v1"
    )
    write_audit_report(report, audit_dir)
    return state, report
