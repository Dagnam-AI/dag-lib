"""``dagnam audit scan``: read the export, discover and price workloads, derive the rows.

``scan`` never opens a socket: it reads the export, discovers and prices the
workloads, derives the redacted rows for the ones worth auditing, and writes
``scan-report.{json,md}`` plus ``workloads/<id>/`` under ``--out``. The
``dagnam.audit`` package is imported inside each handler so ``dagnam --help``
stays import-light.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

from dagnam.cli.audit_run import fail, resolve_audit_dir
from dagnam.cli.common import print_next_step
from dagnam.cli.presentation import Column, emit_result, render_table

if TYPE_CHECKING:
    from dagnam.audit.record import TraceRecord
    from dagnam.audit.state import AuditState


def parse_window(value: str) -> int:
    """``30d`` or ``30`` -> 30; anything else is a usage error."""
    digits = value[:-1] if value.endswith("d") else value
    if not (digits.isascii() and digits.isdigit()) or int(digits) < 1:
        raise argparse.ArgumentTypeError(
            f"--window expects a number of days like 30d, not {value!r}"
        )
    return int(digits)


def parse_sample_rate(value: str) -> float:
    """The share of the traffic the export holds, ``0.1`` or ``10%``: in ``(0, 1]``."""
    try:
        rate = float(value[:-1]) / 100 if value.endswith("%") else float(value)
    except ValueError:
        rate = float("nan")
    if not 0 < rate <= 1:
        raise argparse.ArgumentTypeError(
            f"--sample-rate expects the share of traffic the export holds, like 0.1 or 10%,"
            f" not {value!r}"
        )
    return rate


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


def retired_live(state: AuditState) -> list[tuple[str, str]]:
    """``(kind, id)`` of every run and endpoint a forced rescan retired while it was still going.

    Retiring a candidate takes it out of the audit, not off the platform: a
    run among them trains -- and bills -- to its recipe's time bound, and an
    endpoint stays up, until ``dagnam audit cancel`` stops them.
    """
    from dagnam.audit.cleanup import live

    return list(dict.fromkeys(pair for step in state.retired for pair in live(step)))


def retired_notice(pairs: Sequence[tuple[str, str]], audit_dir: Path) -> list[str]:
    """What is still going among the retired candidates, and the command that stops it."""
    if not pairs:
        return []
    return [
        f"warning: {len(pairs)} retired run(s) or endpoint(s) are still live and still billed:",
        *(f"  {kind} {item_id}" for kind, item_id in pairs),
        f"Stop them: dagnam audit cancel {audit_dir}",
    ]


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
    """Read the export, discover and price workloads, derive rows; no network.

    The export is streamed, never held whole: once to discover the workloads,
    then twice more to plan and derive the rows of the ones worth auditing.

    A forced rescan retires the candidates of the workloads it rewrites; any
    run or endpoint among them that is still going is named on stderr with the
    command that stops it. A scan never opens a socket, so it cannot stop them
    itself.
    """
    from dagnam.audit import discover_workloads, read_traces, replaceability
    from dagnam.audit.derive import derive_workloads
    from dagnam.audit.discover import by_spend
    from dagnam.audit.economics import price_workload
    from dagnam.audit.orchestrate import AUDITED_VERDICTS
    from dagnam.audit.prices import PriceTable
    from dagnam.audit.scan_report import SupersededRunError, scan_window, write_scan
    from dagnam.audit.state import AuditBusyError, load_state, lock_audit
    from dagnam.audit.thresholds import MAX_SEQ_LENGTH
    from dagnam.audit.workspace import UnsafeWorkloadsError

    # An audit directory named through a link (a temp or home directory often is one) is the
    # user's own choice: it is resolved once here and the real path is used from then on, while
    # a link INSIDE it (in a workload folder, at a file a scan writes) is still refused.
    out, export, column_map = resolve_audit_dir(args.out), Path(args.export), parse_map(args.map)

    def records() -> Iterator[TraceRecord]:
        return read_traces(export, source=args.source, column_map=column_map)[0]

    found = discover_workloads(records(), window_days=args.window, sample_rate=args.sample_rate)
    if not found:
        fail(args, f"{args.export}: no traces to audit")
    table = PriceTable.load(Path(args.price_table) if args.price_table else None)
    workloads = by_spend(price_workload(w, table) for w in found)  # sorted once priced
    audited = [w for w in workloads if replaceability(w).status in AUDITED_VERDICTS]
    datasets = derive_workloads(records, audited, max_seq_length=MAX_SEQ_LENGTH)
    out.mkdir(parents=True, exist_ok=True)
    try:
        # Not under a live run: a workload it has not reached yet is in no
        # `state.json` to protect, and its rows would change mid-run.
        with lock_audit(out):
            report = write_scan(
                out,
                workloads,
                datasets,
                source=args.source,
                window=scan_window(
                    workloads, window_days=args.window, sample_rate=args.sample_rate
                ),
                price_table=table,
                force=args.force,
            )
    except (AuditBusyError, SupersededRunError, UnsafeWorkloadsError) as exc:
        fail(args, str(exc))
    emit_result(report.to_json(), output=None, json_stdout=args.json, render_human=_render_scan)
    for line in retired_notice(retired_live(load_state(out)), out):
        print(line, file=sys.stderr)
    if datasets:
        print_next_step(f"dagnam audit run {out}")


__all__ = [
    "cmd_audit_scan",
    "parse_map",
    "parse_sample_rate",
    "parse_window",
    "retired_live",
    "retired_notice",
]
