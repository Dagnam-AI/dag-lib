"""``dagnam audit cancel`` and ``dagnam audit delete``: ask the platform once, show, record.

For a published audit the platform's own routes are the only thing that touches its
resources: this command calls one, prints the receipt, records the decisions, removes this
machine's files, and exits. It never finishes a job from here. When the platform does not
answer (a 5xx, a timeout, a refusal), or does not know the audit (404), it issues no destructive
call at all: the receipt says so (``audit blocked [not_answered]``; or nothing to do) and the
exit status is 1 for the first, 0 for the second. An unpublished audit (no ``audit_id``) has no
platform record, so the ids this directory recorded as created are walked directly.

The exit status is one function (:func:`~dagnam.audit.receipt_rows.exit_status`): 1 iff
something of the audit's own is left; a row the platform kept never fails a command.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

from dagnam.cli.audit_run import client_from_env, fail, resolve_audit_dir
from dagnam.cli.common import confirm_or_abort, error, mask_key
from dagnam.cli.presentation import emit_result

if TYPE_CHECKING:
    from dagnam._core.client import DagnamClient


def _wrong_account_note() -> None:
    """Say which account a 404 was asked of, so a key from another account is seen at once."""
    from dagnam._core.auth import get_api_key, get_api_url
    from dagnam._core.exceptions import DagnamError

    try:
        key = mask_key(get_api_key())
    except DagnamError:
        key = "none stored"
    print(
        f"The platform at {get_api_url()} (key {key}) says it has no such"
        " audit for this key: the key may belong to another account, or the host be another one"
        " (see `dagnam whoami`), or the audit may be deleted. Nothing was changed on the platform"
        " or on this machine: the rows, the deployment keys and state.json are all kept. Once you"
        " know it is deleted, `dagnam audit delete <dir> --already-deleted` clears this directory.",
        file=sys.stderr,
    )


def _render_receipt(receipt: Mapping[str, Any], path: Path) -> str:
    """One line per artifact the receipt names, then where it was written.

    Every field is read with a default: a row the platform spells differently from this client
    is worth showing as ``?``, never worth a ``KeyError`` that hides the whole receipt. A
    ``code`` that says something the status does not is shown beside it, as sent.
    """
    from dagnam.audit.cleanup import receipt_rows

    def line(row: Mapping[str, Any]) -> str:
        code = row.get("code")
        tag = f" [{code}]" if code and code != row.get("status") else ""
        reason = f" ({row['reason']})" if row.get("reason") else ""
        return f"{row.get('kind', '?')} {row.get('id', '?')}: {row.get('status', '?')}{tag}{reason}"

    return "\n".join([*map(line, receipt_rows(receipt)), f"Receipt: {path}"])


def _finish(receipt: Mapping[str, Any], verb: str) -> tuple[int, int, int, int]:
    """``(exit status, left, kept, unreadable)`` of a receipt, the status from the one deciding function."""
    from dagnam.audit.cleanup import receipt_rows
    from dagnam.audit.receipt_rows import Verdict, decide, exit_status

    rows = receipt_rows(receipt)
    verdicts = [decide(row).verdict for row in rows]
    status = exit_status(rows, receipt.get("audit_status"), verb=verb)
    return (
        status,
        verdicts.count(Verdict.LEFT),
        verdicts.count(Verdict.KEPT),
        verdicts.count(Verdict.UNKNOWN),
    )


def _unknown_note(unknown: int) -> None:
    """Say, on stderr, that some rows used a status this version does not know."""
    if unknown:
        print(
            f"{unknown} receipt {'row has' if unknown == 1 else 'rows have'} a status this"
            " version of dagnam does not know (shown above); nothing was done to"
            f" {'it' if unknown == 1 else 'them'}, and it counts as not finished.",
            file=sys.stderr,
        )


def _cancel_under_live_run(
    args: argparse.Namespace, audit_dir: Path, client: DagnamClient, busy: str
) -> Mapping[str, Any]:
    """The platform's cancel of a published audit whose run is live; the run owns ``state.json``.

    The platform halts the audit, so the run stops at its next publish and records what it
    sees then; nothing is recorded or written here. An unpublished audit has nothing the
    platform could stop, and a platform that does not answer (or has no such audit) is an error
    that says what to do.
    """
    from dagnam.audit.cleanup import ask_platform
    from dagnam.audit.state import load_state

    state = load_state(audit_dir)
    if state.audit_id is None:
        fail(args, f"{busy}: stop that `dagnam audit run` (Ctrl+C) first, then cancel")
    answer = ask_platform(client.cancel_audit, state.audit_id)
    if answer.receipt is None:
        why = f"the platform did not answer the cancel ({answer.failure})"
        fail(args, f"{busy}: {why}; stop that `dagnam audit run` (Ctrl+C), then cancel again")
    return answer.receipt


def cmd_audit_cancel(args: argparse.Namespace) -> None:
    """Cancel an audit: the platform stops what is the audit's own; an unpublished one is stopped here.

    Exits 1 when a run or an endpoint may still be running or serving (a ``blocked`` row), or
    the platform did not answer. A run the platform found already finished (``already_stopped``)
    is left as it was, so it stays resumable.
    """
    from dagnam.audit.cleanup import CANCELLED_FILE, AuditDeletedError, cancel_audit
    from dagnam.audit.state import AuditBusyError, lock_audit
    from dagnam.audit.workspace import UnsafeWorkloadsError

    audit_dir = resolve_audit_dir(args.audit_dir)
    client = client_from_env()
    try:
        # A live run owns `state.json` and the lock: it is stopped (Ctrl+C) before it is cancelled.
        with lock_audit(audit_dir):
            receipt = cancel_audit(audit_dir, client, on_missing=_wrong_account_note)
    except AuditBusyError as exc:
        receipt = _cancel_under_live_run(args, audit_dir, client, str(exc))
    except (AuditDeletedError, FileNotFoundError, UnsafeWorkloadsError) as exc:
        fail(args, str(exc))
    path = audit_dir / CANCELLED_FILE
    emit_result(
        receipt,
        output=None,
        json_stdout=args.json,
        render_human=lambda _: _render_receipt(receipt, path),
    )
    status, _, _, unknown = _finish(receipt, "cancel")
    _unknown_note(unknown)
    if status:
        error(
            f"something may still be running or serving, or the platform did not answer (see"
            f" above); the receipt is {path}",
            hint=f"dagnam audit cancel {audit_dir}",
        )


def _listing(audit_dir: Path) -> str:
    """What the delete is about to remove, from the state, for the confirmation prompt."""
    from dagnam.audit.cleanup import recorded_ids
    from dagnam.audit.state import load_state

    state = load_state(audit_dir)
    lines = [f"  {kind}: {', '.join(ids)}" for kind, ids in recorded_ids(state).items() if ids]
    if state.audit_id is not None:
        lines.append(f"  audit: {state.audit_id} (and its published report)")
    return "\n".join(lines) or "  (nothing recorded)"


def _asks(prompt: str, *, assume_yes: bool) -> bool:
    """A yes/no question that, unlike ``confirm_or_abort``, lets the command go on when declined."""
    if assume_yes:
        return True
    print(prompt)
    try:
        return input("Type 'yes' to confirm: ").strip() == "yes"
    except EOFError:
        return False


def _offer_claim(args: argparse.Namespace, audit_dir: Path, client: DagnamClient) -> None:
    """For an audit an older dagnam published: offer to claim what it recorded, before the delete.

    The platform tags what a client names at creation; an older client named nothing, so the
    platform keeps what it cannot prove the audit created. Claiming them (while the audit still
    exists) is what lets the delete remove them. One confirmation; declined, the delete goes on
    and says what it kept.
    """
    from dagnam.audit.claims import CLAIMABLE, ClaimError, claim_recorded
    from dagnam.audit.cleanup import recorded_ids
    from dagnam.audit.state import load_state, save_state

    state = load_state(audit_dir)
    ids = recorded_ids(state)
    named = [
        f"  {kind}: {', '.join(ids[kind])}" for kind in CLAIMABLE if kind != "project" and ids[kind]
    ]
    if state.audit_id is None or state.tagged or not named:
        return
    prompt = (
        "This audit was published by an older dagnam, before the platform recorded which"
        " resources an audit created, so the platform keeps what it cannot prove is the audit's."
        " These are recorded in this directory:\n"
        + "\n".join(named)
        + "\nClaim them for the audit so that the delete removes them too?"
    )
    if not _asks(prompt, assume_yes=args.yes):
        return
    try:
        claim_recorded(client, state, lambda line: print(line, file=sys.stderr))
    except ClaimError as exc:
        fail(args, f"the claim failed ({exc}); nothing was deleted: run the delete again")
    state.tagged = True
    save_state(audit_dir, state)


def _kept_untagged_note(audit_dir: Path, receipt: Mapping[str, Any]) -> None:
    """Say plainly which of this directory's own resources the platform kept as not created by the audit."""
    from dagnam.audit.cleanup import receipt_rows
    from dagnam.audit.state import load_state

    own = {
        i
        for step in load_state(audit_dir).all_steps()
        for i in (step.dataset_id, step.training_job_id, step.deployment_id)
        if i
    }
    named = [
        f"{row.get('kind')} {row.get('id')}"
        for row in receipt_rows(receipt)
        if row.get("code") == "not_created_here" and str(row.get("id")) in own
    ]
    if named:
        print(
            f"The platform kept {', '.join(named)}: this directory recorded "
            f"{'it' if len(named) == 1 else 'them'}, but the platform cannot prove the audit "
            "created "
            f"{'it' if len(named) == 1 else 'them'} (an audit published before it recorded "
            "provenance). They are yours: a claim makes them the audit's while it exists, and "
            "now that it is deleted they have to be removed in the Studio.",
            file=sys.stderr,
        )


def cmd_audit_delete(args: argparse.Namespace) -> None:
    """Delete an audit: the platform deletes what is the audit's own; an unpublished one is deleted here.

    The delete always prints its receipt. Its exit status says whether it finished: 1 when
    anything that belongs to the audit is left (a ``blocked`` row, a platform that answers that
    it did not delete the audit, one that did not answer, a local file behind a link). A row the
    platform kept on purpose is shown and does not fail it. When the platform did not delete the
    audit, NOTHING local is removed and the next ``audit delete`` finishes it.
    """
    from dagnam.audit.cleanup import DELETED_FILE, delete_audit
    from dagnam.audit.state import AuditBusyError, lock_audit
    from dagnam.audit.workspace import UnsafeWorkloadsError

    audit_dir = resolve_audit_dir(args.audit_dir)
    try:
        # Held from before the listing: a live run would go on creating what this deletes.
        with lock_audit(audit_dir):
            confirm_or_abort(
                f"This deletes from your account:\n{_listing(audit_dir)}", assume_yes=args.yes
            )
            client = client_from_env()
            _offer_claim(args, audit_dir, client)
            receipt = delete_audit(
                audit_dir,
                client,
                on_missing=_wrong_account_note,
                assume_gone=args.already_deleted,
            )
    except (AuditBusyError, FileNotFoundError, UnsafeWorkloadsError) as exc:
        fail(args, str(exc))
    path = audit_dir / DELETED_FILE
    emit_result(
        receipt,
        output=None,
        json_stdout=args.json,
        render_human=lambda _: _render_receipt(receipt, path),
    )
    status, left, kept, unknown = _finish(receipt, "delete")
    # The last words, on stderr in both modes so under --json stdout is the receipt alone.
    _unknown_note(unknown)
    _kept_untagged_note(audit_dir, receipt)
    if status:
        still = "artifact is" if left == 1 else "artifacts are"
        what = (
            f"{left} {still} still there (blocked above)"
            if left
            else "the platform did not finish the delete"
        )
        more = f"; {kept} more {'was' if kept == 1 else 'were'} kept on purpose" if kept else ""
        error(
            f"{what}{more}; nothing local was removed unless the platform deleted the audit;"
            f" the receipt is {path}",
            hint=f"dagnam audit delete {audit_dir} --yes",
        )
    if kept:
        print(
            f"Everything the audit created is deleted, except {kept}"
            f" {'item' if kept == 1 else 'items'} kept on purpose because"
            f" {'it is' if kept == 1 else 'they are'} not the audit's to remove (kept above).",
            file=sys.stderr,
        )
    else:
        print("Everything the audit created is deleted.", file=sys.stderr)


__all__ = ["cmd_audit_cancel", "cmd_audit_delete"]
