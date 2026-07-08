"""Register command handler: terminal-only account onboarding."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import getpass
from typing import TYPE_CHECKING

from dagnam.cli.common import error, format_ascii_art, mask_key, print_next_step

if TYPE_CHECKING:
    from dagnam.cli.common import SubParsersAction

_DEFAULT_API_URL = "https://api.dagnam.ai"


def cmd_register(
    args: argparse.Namespace,
    getpass_func: Callable[[str], str] | None = None,
    input_func: Callable[[str], str] | None = None,
) -> None:
    """Create a new account, bootstrap a session, and persist only the API key.

    Prompts for an email and a password (confirmed twice, never echoed), then
    delegates the register -> log in -> mint-API-key sequence to
    ``dagnam.account.register``. The session token minted along the way never
    reaches this function - only the resulting API key is written to
    ``~/.dagnam/config.json``, with 0o600 permissions via
    :func:`dagnam._core.config.save_config`.
    """
    import dagnam
    from dagnam._core import config as _cfg
    from dagnam._core.exceptions import APIError, DagnamError

    api_url = getattr(args, "api_url", None)

    print(f"{format_ascii_art()}\n")
    print("Create a Dagnam.AI account.\n")

    email = (input_func or input)("Email: ").strip()
    if not email:
        error("Email cannot be empty.")

    pw = (getpass_func or getpass.getpass)("Password: ")
    if not pw:
        error("Password cannot be empty.")
    pw_confirm = (getpass_func or getpass.getpass)("Confirm password: ")
    if pw != pw_confirm:
        error("Passwords do not match.")

    try:
        key_obj = dagnam.account.register(email, pw, api_url=api_url)
    except APIError as exc:
        # Forward-compat hook: once the backend gains an email-verification
        # step, a fresh account's first request returns 403 with a body that
        # names verification. Only THEN show the inbox guidance - a plain 403
        # today means "account suspended/deleted" (from the login step), whose
        # real server reason must be surfaced verbatim, not masked by a
        # misleading "check your inbox" message.
        if exc.status_code == 403 and "verif" in exc.message.lower():
            error(
                "Your account needs email verification - check your inbox, then run `dagnam login`."
            )
        error(f"Registration failed: {exc}")
    except DagnamError as exc:
        error(f"Registration failed: {exc}")

    key = key_obj.get("key")
    if not isinstance(key, str):
        error("Registration succeeded, but no API key was returned.")

    config = _cfg.load_config()
    config["api_key"] = key
    if api_url and api_url != _DEFAULT_API_URL:
        config["api_url"] = api_url
    _cfg.save_config(config)

    print("Account created and API key saved.")
    print(f"API key: {mask_key(key)}")
    print_next_step("dagnam projects list")


def _register(args: argparse.Namespace) -> None:
    cmd_register(args, getpass.getpass, input)


def register_register(subparsers: SubParsersAction) -> None:
    """Register the ``register`` command on the top-level subparsers."""
    register_cmd = subparsers.add_parser(
        "register",
        help="Create an account and store an API key.",
        description=(
            "Create a Dagnam.AI account, mint a fresh API key, and save it "
            "to ~/.dagnam/config.json."
        ),
    )
    register_cmd.add_argument("--api-url", help="API base URL (default: https://api.dagnam.ai).")
    register_cmd.set_defaults(func=_register)
