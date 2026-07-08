"""Handlers for ``dagnam account export`` and ``dagnam account delete``.

Split out of ``dagnam.cli.account`` (which retains version/whoami/usage/
logout/config) to keep every module under the repo's ~500-line file-size cap,
mirroring the sibling ``account_security``/``account_profile`` modules.

``export`` requests a data export and downloads the resulting archive in one
shot. ``delete`` permanently removes the caller's account behind a getpass
password prompt and a typed confirmation - the password is only ever read via
``getpass`` and sent as a request body field; it is never printed, logged, or
included in any command's output.
"""

from __future__ import annotations

import argparse
import getpass
from typing import TYPE_CHECKING

from dagnam.cli.common import confirm_or_abort, error

if TYPE_CHECKING:
    from dagnam.cli.common import SubParsersAction

# Deletion is irreversible, so it is gated behind confirm_or_abort rather than
# a bare y/N prompt.
_DELETE_PROMPT = "This will PERMANENTLY delete your account and cannot be undone."


def cmd_account_export(args: argparse.Namespace) -> None:
    """Request a data export and download the resulting archive.

    One-shot request + download: requests an export, then immediately
    downloads it by the returned ``export_id``. ``--out`` names the
    destination directory (default: current directory); the saved filename is
    server-provided and sanitized by the SDK.
    """
    import dagnam

    meta = dagnam.account.export_data()
    export_id = str(meta["export_id"])
    dest = args.out or "."
    path = dagnam.account.download_export(export_id, out=dest)
    print(f"Saved export to {path}.")


def cmd_account_delete(args: argparse.Namespace) -> None:
    """Delete the caller's account after a password prompt and confirmation."""
    password = getpass.getpass("Current password: ")
    if not password:
        error("Password cannot be empty.")

    confirm_or_abort(_DELETE_PROMPT, assume_yes=args.yes)

    import dagnam

    dagnam.account.delete_account(password)
    print("Account deleted.")


def register_account_data(account_sub: SubParsersAction) -> None:
    """Register ``dagnam account export`` and ``dagnam account delete``."""
    export_cmd = account_sub.add_parser(
        "export",
        help="Request and download a data export of your account.",
        description="Request a data export and download the resulting archive.",
    )
    export_cmd.add_argument(
        "--out",
        help="Destination directory for the downloaded archive (default: current directory).",
    )
    export_cmd.set_defaults(func=cmd_account_export)

    delete_cmd = account_sub.add_parser(
        "delete",
        help="Permanently delete your account.",
        description=(
            "Delete your account after entering your current password and "
            "confirming. This cannot be undone."
        ),
    )
    delete_cmd.add_argument(
        "--yes", action="store_true", help="Skip the typed confirmation prompt."
    )
    delete_cmd.set_defaults(func=cmd_account_delete)
