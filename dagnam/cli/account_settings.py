"""Handlers for the ``dagnam account settings`` and ``dagnam account notifications`` groups.

Split out of ``dagnam.cli.account`` (which retains version/whoami/usage/
logout/config) to keep each module under the repo's ~500-line file-size cap.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from dagnam._types import JsonObject
from dagnam.cli.common import error
from dagnam.cli.presentation import Column, emit_result, render_table

if TYPE_CHECKING:
    from dagnam.cli.common import SubParsersAction

_NOTIFICATION_BOOL_FIELDS = frozenset(
    {
        "email_enabled",
        "training_alerts",
        "deployment_alerts",
        "security_alerts",
        "marketing_emails",
        "weekly_digest",
    }
)
_SETTINGS_BOOL_FIELDS = frozenset({"canvas_grid_enabled"})
_SETTINGS_INT_FIELDS = frozenset({"auto_save_interval"})
# Names the resource layer's `update_settings`/`update_notification_preferences`
# reserve for client resolution (see dagnam.resources.account). No real backend
# settings/notification field uses these, but the patch dict is later spread as
# `**patch` into that call, so a field literally named one of these would shadow
# the resolver kwarg instead of landing in the JSON body. Reject it up front.
_RESERVED_PATCH_KEYS = frozenset({"client", "api_key", "api_url"})


def _coerce_bool(key: str, raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"false", "0", "no", "off"}:
        return False
    error(f"Invalid boolean value for {key}: {raw!r} (expected true/false)")


def _parse_patch(
    pairs: list[str], *, bool_fields: frozenset[str], int_fields: frozenset[str] = frozenset()
) -> JsonObject:
    """Parse ``KEY=VALUE`` CLI args into a JSON patch dict with light type coercion."""
    patch: JsonObject = {}
    for pair in pairs:
        key, sep, raw_value = pair.partition("=")
        if not sep or not key:
            error(f"Invalid KEY=VALUE argument: {pair!r}")
        if key in _RESERVED_PATCH_KEYS:
            error(f"Unsupported field name: {key!r}")
        if key in bool_fields:
            patch[key] = _coerce_bool(key, raw_value)
        elif key in int_fields:
            try:
                patch[key] = int(raw_value)
            except ValueError:
                error(f"Invalid integer value for {key}: {raw_value!r}")
        else:
            patch[key] = raw_value
    return patch


def _render_kv_table(payload: object) -> str:
    data: dict[str, object] = payload if isinstance(payload, dict) else {}
    rows: list[dict[str, object]] = [
        {"field": key, "value": str(value)}
        for key, value in data.items()
        if key not in {"id", "user_id"}
    ]
    if not rows:
        return "No data returned."
    return render_table((Column("Field", "field", 28), Column("Value", "value", 40)), rows)


def cmd_settings_get(args: argparse.Namespace) -> None:
    """Print the caller's current UI/editor settings."""
    import dagnam

    result = dagnam.account.get_settings()
    emit_result(result, output=args.output, json_stdout=args.json, render_human=_render_kv_table)


def cmd_settings_set(args: argparse.Namespace) -> None:
    """Patch one or more settings fields from ``KEY=VALUE`` arguments."""
    import dagnam

    patch = _parse_patch(
        args.pairs, bool_fields=_SETTINGS_BOOL_FIELDS, int_fields=_SETTINGS_INT_FIELDS
    )
    # _parse_patch rejects _RESERVED_PATCH_KEYS, so this dict can never shadow
    # update_settings' own client/api_key/api_url kwargs at runtime; pyright
    # cannot see that invariant through a plain dict[str, JsonValue] spread.
    result = dagnam.account.update_settings(**patch)  # pyright: ignore[reportArgumentType]
    emit_result(result, output=args.output, json_stdout=args.json, render_human=_render_kv_table)


def cmd_settings_reset(args: argparse.Namespace) -> None:
    """Reset settings to their defaults."""
    import dagnam

    result = dagnam.account.reset_settings()
    emit_result(result, output=args.output, json_stdout=args.json, render_human=_render_kv_table)


def cmd_notifications_get(args: argparse.Namespace) -> None:
    """Print the caller's current notification preferences."""
    import dagnam

    result = dagnam.account.notification_preferences()
    emit_result(result, output=args.output, json_stdout=args.json, render_human=_render_kv_table)


def cmd_notifications_set(args: argparse.Namespace) -> None:
    """Patch one or more notification-preference fields from ``KEY=VALUE`` arguments."""
    import dagnam

    patch = _parse_patch(args.pairs, bool_fields=_NOTIFICATION_BOOL_FIELDS)
    # See the matching comment in cmd_settings_set: _parse_patch already rejects
    # _RESERVED_PATCH_KEYS, so this spread cannot shadow the resolver kwargs.
    result = dagnam.account.update_notification_preferences(
        **patch  # pyright: ignore[reportArgumentType]
    )
    emit_result(result, output=args.output, json_stdout=args.json, render_human=_render_kv_table)


def _add_output_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="Print raw JSON.")
    parser.add_argument("--output", help="Write the raw JSON to this path.")


def register_settings(account_sub: SubParsersAction) -> None:
    settings_cmd = account_sub.add_parser(
        "settings", help="Get, update, or reset UI/editor settings."
    )
    settings_sub = settings_cmd.add_subparsers(dest="settings_command", required=True)

    settings_get = settings_sub.add_parser("get", help="Print current settings.")
    _add_output_flags(settings_get)
    settings_get.set_defaults(func=cmd_settings_get)

    settings_set = settings_sub.add_parser(
        "set",
        help="Update one or more settings fields.",
        description="Update settings via KEY=VALUE pairs, e.g. theme=dark auto_save_interval=250.",
    )
    settings_set.add_argument("pairs", nargs="+", metavar="KEY=VALUE", help="Field(s) to update.")
    _add_output_flags(settings_set)
    settings_set.set_defaults(func=cmd_settings_set)

    settings_reset = settings_sub.add_parser("reset", help="Reset settings to defaults.")
    _add_output_flags(settings_reset)
    settings_reset.set_defaults(func=cmd_settings_reset)


def register_notifications(account_sub: SubParsersAction) -> None:
    notifications_cmd = account_sub.add_parser(
        "notifications", help="Get or update notification preferences."
    )
    notifications_sub = notifications_cmd.add_subparsers(
        dest="notifications_command", required=True
    )

    notifications_get = notifications_sub.add_parser(
        "get", help="Print current notification preferences."
    )
    _add_output_flags(notifications_get)
    notifications_get.set_defaults(func=cmd_notifications_get)

    notifications_set = notifications_sub.add_parser(
        "set",
        help="Update one or more notification fields.",
        description="Update preferences via KEY=VALUE pairs, e.g. training_alerts=false.",
    )
    notifications_set.add_argument(
        "pairs", nargs="+", metavar="KEY=VALUE", help="Field(s) to update."
    )
    _add_output_flags(notifications_set)
    notifications_set.set_defaults(func=cmd_notifications_set)
