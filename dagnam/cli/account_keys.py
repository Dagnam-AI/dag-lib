"""Handlers for the top-level ``dagnam keys`` command group.

A NEW top-level group (like ``dagnam profile``), not a subcommand of
``account`` - it manages the caller's API keys: create, list, revoke. The
plaintext secret returned by create is printed to stdout exactly once, with a
store-it-now notice; it is never logged, never persisted by the SDK, and never
re-printed by ``keys list``.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from dagnam.cli.common import error, format_local
from dagnam.cli.presentation import Column, emit_result, render_table

if TYPE_CHECKING:
    from dagnam.cli.common import SubParsersAction


def _render_created_key(payload: object) -> str:
    data: dict[str, object] = payload if isinstance(payload, dict) else {}
    lines: list[str] = []
    key = data.get("key")
    if key is not None:
        lines.extend(
            [
                "API key created. Store this secret now - it will not be shown again:",
                "",
                f"  {key}",
                "",
            ]
        )
    rows: list[dict[str, object]] = [
        {"field": field, "value": str(value)} for field, value in data.items() if field != "key"
    ]
    if rows:
        lines.append(
            render_table((Column("Field", "field", 28), Column("Value", "value", 48)), rows)
        )
    elif not lines:
        lines.append("No data returned.")
    return "\n".join(lines)


def _render_keys_table(payload: object) -> str:
    items = payload if isinstance(payload, list) else []
    rows: list[dict[str, object]] = []
    for item in items:
        keydata = item if isinstance(item, dict) else {}
        permissions = keydata.get("permissions")
        permissions_list = permissions if isinstance(permissions, list) else []
        rows.append(
            {
                "name": keydata.get("name", "-"),
                "id": keydata.get("id", "-"),
                "prefix": keydata.get("key_prefix", "-"),
                "permissions": ",".join(str(p) for p in permissions_list),
                "uses": keydata.get("usage_count", 0),
                "last_used": format_local(keydata.get("last_used_at")),
                "expires": format_local(keydata.get("expires_at")),
            }
        )
    if not rows:
        return "No API keys."
    return render_table(
        (
            Column("Name", "name", 20),
            Column("ID", "id", 36),
            Column("Prefix", "prefix", 12),
            Column("Permissions", "permissions", 20),
            Column("Uses", "uses", 6, "right"),
            Column("Last used", "last_used", 12),
            Column("Expires", "expires", 12),
        ),
        rows,
    )


def cmd_keys_create(args: argparse.Namespace) -> None:
    """Create an API key and print the plaintext secret exactly once."""
    if not args.name.strip():
        error("Key name cannot be empty.")

    import dagnam

    result = dagnam.account.create_api_key(args.name, args.scope, args.expires_in_days)
    emit_result(result, output=args.output, json_stdout=args.json, render_human=_render_created_key)


def cmd_keys_list(args: argparse.Namespace) -> None:
    """Print the caller's API keys (secrets never included)."""
    import dagnam

    result = dagnam.account.list_api_keys()
    emit_result(result, output=args.output, json_stdout=args.json, render_human=_render_keys_table)


def cmd_keys_revoke(args: argparse.Namespace) -> None:
    """Revoke a single API key by id."""
    key_id = args.key_id
    if not key_id.strip():
        error("Key id cannot be empty.")

    import dagnam

    dagnam.account.revoke_api_key(key_id)
    print(f"Revoked API key {key_id}.")


def _add_output_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="Print raw JSON.")
    parser.add_argument("--output", help="Write the raw JSON to this path.")


def register_keys(subparsers: SubParsersAction) -> None:
    """Register the top-level ``dagnam keys [create|list|revoke]`` command."""
    keys_cmd = subparsers.add_parser("keys", help="Create, list, and revoke API keys.")
    keys_sub = keys_cmd.add_subparsers(dest="keys_command", required=True)

    keys_create = keys_sub.add_parser("create", help="Create a new API key.")
    keys_create.add_argument("--name", required=True, help="A name for the key.")
    keys_create.add_argument(
        "--scope",
        action="append",
        metavar="SCOPE",
        help="Permission scope; repeat for multiple, e.g. --scope read --scope write.",
    )
    keys_create.add_argument(
        "--expires-in-days",
        type=int,
        metavar="DAYS",
        help="Optional expiry in days (1-365).",
    )
    _add_output_flags(keys_create)
    keys_create.set_defaults(func=cmd_keys_create)

    keys_list = keys_sub.add_parser("list", help="List your API keys.")
    _add_output_flags(keys_list)
    keys_list.set_defaults(func=cmd_keys_list)

    keys_revoke = keys_sub.add_parser("revoke", help="Revoke one API key by id.")
    keys_revoke.add_argument("key_id", help="API key id to revoke.")
    keys_revoke.set_defaults(func=cmd_keys_revoke)
