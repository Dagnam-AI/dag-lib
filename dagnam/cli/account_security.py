"""Handlers for ``dagnam account change-password`` and ``dagnam account sessions``.

Split out of ``dagnam.cli.account`` (which retains version/whoami/usage/
logout/config) to keep every module under the repo's ~500-line file-size cap.
Holds the security-sensitive account surface: an interactive, getpass-driven
password change and session listing/revocation. The password value is only
ever read via ``getpass`` and sent as a request body field; it is never
printed, logged, or included in any command's output.
"""

from __future__ import annotations

import argparse
import getpass
from typing import TYPE_CHECKING

from dagnam.cli.common import confirm_or_abort, error, format_local
from dagnam.cli.presentation import Column, emit_result, render_table

if TYPE_CHECKING:
    from dagnam.cli.common import SubParsersAction

# Sessions revoke-all is destructive (logs out every active session at once),
# so it is gated behind confirm_or_abort rather than a bare y/N prompt.
_REVOKE_ALL_PROMPT = "This will revoke ALL active sessions for your account."


def cmd_change_password(args: argparse.Namespace) -> None:
    """Prompt for the current and a confirmed new password, then change it.

    Every value is read via ``getpass`` (never echoed to the terminal, never
    an argv/help value) and only the current/new password fields are sent in
    the request body. Nothing about the change is printed except a static
    confirmation message - the backend's response body is intentionally not
    echoed to keep this command's output free of anything secret-adjacent.
    """
    current_password = getpass.getpass("Current password: ")
    if not current_password:
        error("Current password cannot be empty.")

    new_password = getpass.getpass("New password: ")
    if not new_password:
        error("New password cannot be empty.")

    confirm_password = getpass.getpass("Confirm new password: ")
    if new_password != confirm_password:
        error("New password and confirmation do not match.")

    import dagnam

    dagnam.account.change_password(current_password, new_password)
    print("Password changed successfully.")


def _format_device_info(value: object) -> str:
    data = value if isinstance(value, dict) else {}
    parts = [str(data[key]) for key in ("browser", "os", "device") if data.get(key)]
    return ", ".join(parts) if parts else "-"


def _render_sessions_table(payload: object) -> str:
    items = payload if isinstance(payload, list) else []
    rows: list[dict[str, object]] = []
    for item in items:
        session = item if isinstance(item, dict) else {}
        rows.append(
            {
                "id": session.get("id", "-"),
                "device": _format_device_info(session.get("device_info")),
                "last_active": format_local(session.get("last_active_at")),
                "created": format_local(session.get("created_at")),
                "current": "yes" if session.get("is_current") else "",
            }
        )
    if not rows:
        return "No active sessions."
    return render_table(
        (
            Column("Session ID", "id", 36),
            Column("Device", "device", 24),
            Column("Last active", "last_active", 12),
            Column("Created", "created", 12),
            Column("Current", "current", 7),
        ),
        rows,
    )


def _render_revoke_all_result(payload: object) -> str:
    data = payload if isinstance(payload, dict) else {}
    message = data.get("message")
    return str(message) if message is not None else "All sessions revoked."


def cmd_sessions_list(args: argparse.Namespace) -> None:
    """Print the caller's active sessions."""
    import dagnam

    result = dagnam.account.list_sessions()
    emit_result(
        result, output=args.output, json_stdout=args.json, render_human=_render_sessions_table
    )


def cmd_sessions_revoke(args: argparse.Namespace) -> None:
    """Revoke a single session by id."""
    session_id = args.session_id
    if not session_id.strip():
        error("Session id cannot be empty.")

    import dagnam

    dagnam.account.revoke_session(session_id)
    print(f"Revoked session {session_id}.")


def cmd_sessions_revoke_all(args: argparse.Namespace) -> None:
    """Revoke every active session, after a typed confirmation unless --yes."""
    confirm_or_abort(_REVOKE_ALL_PROMPT, assume_yes=args.yes)

    import dagnam

    result = dagnam.account.revoke_all_sessions()
    emit_result(
        result,
        output=args.output,
        json_stdout=args.json,
        render_human=_render_revoke_all_result,
    )


def _add_output_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="Print raw JSON.")
    parser.add_argument("--output", help="Write the raw JSON to this path.")


def register_account_security(account_sub: SubParsersAction) -> None:
    """Register ``dagnam account change-password`` and ``... sessions [...]``."""
    change_password_cmd = account_sub.add_parser(
        "change-password",
        help="Change your account password.",
        description=(
            "Change your password via interactive prompts (current, new, "
            "confirm). Values are read with getpass and never echoed."
        ),
    )
    change_password_cmd.set_defaults(func=cmd_change_password)

    sessions_cmd = account_sub.add_parser("sessions", help="List or revoke active sessions.")
    sessions_sub = sessions_cmd.add_subparsers(dest="sessions_command", required=True)

    sessions_list = sessions_sub.add_parser("list", help="List active sessions.")
    _add_output_flags(sessions_list)
    sessions_list.set_defaults(func=cmd_sessions_list)

    sessions_revoke = sessions_sub.add_parser("revoke", help="Revoke one session by id.")
    sessions_revoke.add_argument("session_id", help="Session id to revoke.")
    sessions_revoke.set_defaults(func=cmd_sessions_revoke)

    sessions_revoke_all = sessions_sub.add_parser(
        "revoke-all", help="Revoke every active session for your account."
    )
    sessions_revoke_all.add_argument(
        "--yes", action="store_true", help="Skip the typed confirmation prompt."
    )
    _add_output_flags(sessions_revoke_all)
    sessions_revoke_all.set_defaults(func=cmd_sessions_revoke_all)
