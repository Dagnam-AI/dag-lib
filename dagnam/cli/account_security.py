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

# Disabling 2FA removes a security factor, so it is gated the same way
# revoke-all is: a typed confirmation, not a y/N keypress.
_DISABLE_2FA_PROMPT = "This will REMOVE two-factor authentication from your account."


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


def cmd_2fa_status(args: argparse.Namespace) -> None:
    """Report whether two-factor authentication is active."""
    import dagnam

    enabled = dagnam.account.two_factor_enabled()
    emit_result(
        {"two_factor_enabled": enabled},
        output=args.output,
        json_stdout=args.json,
        render_human=lambda _payload: (
            "Two-factor authentication is ENABLED."
            if enabled
            else "Two-factor authentication is DISABLED."
        ),
    )


def _render_enrollment(payload: object) -> str:
    """Format the one-time enrollment material.

    Unlike ``change-password``, this command MUST print the response body: the
    secret and backup codes are the whole point of the call and the server will
    never return them again. The warning is part of the output, not a docstring,
    because a caller who scrolls past it has lost their recovery codes.
    """
    if not isinstance(payload, dict):
        return str(payload)
    lines = [
        "Two-factor enrollment started. NOT yet active -- run `dagnam account 2fa verify`.",
        "",
        "Save these now. They are shown ONCE and cannot be retrieved again:",
        "",
        f"  Secret:  {payload.get('secret', '')}",
    ]
    uri = payload.get("qr_code_uri")
    if uri:
        lines.append(f"  QR URI:  {uri}")
    codes = payload.get("backup_codes")
    if isinstance(codes, list) and codes:
        lines.append("")
        lines.append("  Backup codes (each usable once):")
        lines.extend(f"    {code}" for code in codes)
    return "\n".join(lines)


def cmd_2fa_enable(args: argparse.Namespace) -> None:
    """Begin enrollment and print the one-time secret and backup codes.

    The password is read via ``getpass`` rather than taken as an argument:
    an argv value lands in shell history and in every process listing on the
    machine.
    """
    password = getpass.getpass("Current password: ")
    if not password:
        error("Password cannot be empty.")

    import dagnam

    emit_result(
        dagnam.account.enable_two_factor(password),
        output=args.output,
        json_stdout=args.json,
        render_human=_render_enrollment,
    )


def cmd_2fa_verify(args: argparse.Namespace) -> None:
    """Confirm enrollment with a TOTP code, activating 2FA.

    The code MAY be passed as an argument: it is single-use and expires within
    seconds, so it is not a durable secret the way the password is, and
    accepting it in argv is what makes the command scriptable.
    """
    code = args.code or input("Authentication code: ").strip()
    if not code:
        error("Authentication code cannot be empty.")

    import dagnam

    dagnam.account.verify_two_factor(code)
    print("Two-factor authentication is now enabled.")


def cmd_2fa_disable(args: argparse.Namespace) -> None:
    """Turn 2FA off, behind a typed confirmation and the account password."""
    confirm_or_abort(_DISABLE_2FA_PROMPT, assume_yes=args.yes)

    password = getpass.getpass("Current password: ")
    if not password:
        error("Password cannot be empty.")

    import dagnam

    dagnam.account.disable_two_factor(password)
    print("Two-factor authentication disabled.")


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

    twofa_cmd = account_sub.add_parser(
        "2fa",
        help="Inspect and manage two-factor authentication.",
        description=(
            "Enrollment is two steps: `enable` returns the secret and backup "
            "codes, and `verify` activates 2FA with a code from the "
            "authenticator you just enrolled. Enrollment that is never verified "
            "leaves 2FA inactive, so a mistyped authenticator cannot lock you out."
        ),
    )
    twofa_sub = twofa_cmd.add_subparsers(dest="twofa_command", required=True)

    twofa_status = twofa_sub.add_parser("status", help="Report whether 2FA is active.")
    _add_output_flags(twofa_status)
    twofa_status.set_defaults(func=cmd_2fa_status)

    twofa_enable = twofa_sub.add_parser(
        "enable",
        help="Start enrollment; prints the one-time secret and backup codes.",
        description=(
            "Prompts for your account password. The secret and backup codes are "
            "shown ONCE and cannot be retrieved again. 2FA is not active until "
            "`dagnam account 2fa verify` succeeds."
        ),
    )
    _add_output_flags(twofa_enable)
    twofa_enable.set_defaults(func=cmd_2fa_enable)

    twofa_verify = twofa_sub.add_parser(
        "verify", help="Activate 2FA with a code from your authenticator."
    )
    twofa_verify.add_argument("code", nargs="?", help="6-digit code. Prompted for when omitted.")
    twofa_verify.set_defaults(func=cmd_2fa_verify)

    twofa_disable = twofa_sub.add_parser("disable", help="Turn 2FA off.")
    twofa_disable.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    twofa_disable.set_defaults(func=cmd_2fa_disable)
