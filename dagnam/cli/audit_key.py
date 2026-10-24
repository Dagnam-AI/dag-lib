"""``dagnam audit key``: print a workload's winning deployment key, for ``export KEY=$(...)``.

Stdout carries the key (or, under ``--json``, the object that holds it) and nothing
else; every sentence goes to stderr. The key comes from the audit directory's
:class:`~dagnam.audit.secrets.SecretStore`, is never logged, and is never part of
an error message. Exit 2 means the audit is readable but holds no key to give.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import NoReturn

from dagnam.cli.audit_run import fail, resolve_audit_dir
from dagnam.cli.errors import render_message

NO_KEY_HINT = (
    "Rotate the deployment key in the Studio, then use the new key directly:"
    " rotating invalidates the copy this machine holds."
)


def _refuse(message: str, *, hint: str | None = None) -> NoReturn:
    """Exit 2: the audit was read, and it has no such key to print."""
    print(render_message(message, hint=hint), file=sys.stderr)
    sys.exit(2)


def cmd_audit_key(args: argparse.Namespace) -> None:
    """Print the key of ``args.workload``'s winning deployment; exit 2 when there is none."""
    from dagnam.audit.orchestrate import SCAN_REPORT
    from dagnam.audit.prices import PriceTable
    from dagnam.audit.report import build_audit_report
    from dagnam.audit.secrets import SecretStore
    from dagnam.audit.state import load_state
    from dagnam.audit.workspace import UnsafeWorkloadsError, read_regular

    audit_dir = resolve_audit_dir(args.audit_dir)
    try:
        scan = json.loads(read_regular(audit_dir / SCAN_REPORT))
        state = load_state(audit_dir)
    except (FileNotFoundError, ValueError, UnsafeWorkloadsError) as exc:
        fail(args, str(exc))
    # The report is the one place the winner is decided, so this names the same
    # deployment `audit run` printed and the audit page shows.
    report = build_audit_report(state, scan, price_table=PriceTable.load(None))
    entries = {str(entry["id"]): entry for entry in report["workloads"]}
    if args.workload not in entries:
        _refuse(
            f"no workload {args.workload!r} in {audit_dir}",
            hint="known workloads: " + (", ".join(entries) or "none"),
        )
    switch = entries[args.workload]["switch"]
    if switch is None:
        _refuse(
            f"workload {args.workload!r} has no winner yet",
            hint=f"dagnam audit run {audit_dir}, or `dagnam audit status {audit_dir}`",
        )
    key_ref = switch["key_ref"]
    key = None if key_ref is None else SecretStore(audit_dir).load(key_ref)
    if key is None:
        _refuse(f"no deployment key is stored for workload {args.workload!r}", hint=NO_KEY_HINT)
    if args.json:
        result = {
            "workload": args.workload,
            "deployment_id": switch["model"],
            "key_ref": key_ref,
            "key": key,
        }
        print(json.dumps(result, indent=2))
    else:
        print(key)
