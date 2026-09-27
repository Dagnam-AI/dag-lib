"""Handler for the top-level ``dagnam profile show <username>`` command.

The public, read-only profile route needs no credentials. The caller's own
profile, settings and security live in the web app: those routes accept only a
browser session, so the CLI no longer wraps them.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from dagnam.cli.presentation import Column, emit_result, render_table

if TYPE_CHECKING:
    from dagnam.cli.common import SubParsersAction


def _render_public_profile(payload: object) -> str:
    data: dict[str, object] = payload if isinstance(payload, dict) else {}
    lines = [f"Display name: {data.get('display_name', '-')}"]
    bio = data.get("bio")
    if bio:
        lines.append(f"Bio: {bio}")
    lines.append(f"Role: {data.get('role', 'user')}")
    avatar_url = data.get("avatar_url")
    if avatar_url:
        lines.append(f"Avatar: {avatar_url}")
    join_date = data.get("join_date")
    if join_date:
        lines.append(f"Joined: {join_date}")

    stats = data.get("stats")
    stats = stats if isinstance(stats, dict) else {}
    lines.append(
        f"Models: {stats.get('models_published', 0)}  "
        f"Stars: {stats.get('stars_received', 0)}  "
        f"Downloads: {stats.get('total_downloads', 0)}"
    )

    models = data.get("models")
    models = models if isinstance(models, list) else []
    rows: list[dict[str, object]] = [
        {
            "name": model.get("name", "-"),
            "stars": model.get("stars_count", 0),
            "downloads": model.get("downloads_count", 0),
        }
        for model in models
        if isinstance(model, dict)
    ]
    if rows:
        table = render_table(
            (
                Column("Model", "name", 32),
                Column("Stars", "stars", 8, "right"),
                Column("Downloads", "downloads", 10, "right"),
            ),
            rows,
        )
        lines.extend(["", table])
    return "\n".join(lines)


def cmd_profile_show(args: argparse.Namespace) -> None:
    """Print a user's public profile by username."""
    import dagnam

    result = dagnam.account.get_public_profile(args.username)
    emit_result(
        result, output=args.output, json_stdout=args.json, render_human=_render_public_profile
    )


def _add_output_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="Print raw JSON.")
    parser.add_argument("--output", help="Write the raw JSON to this path.")


def register_profile(subparsers: SubParsersAction) -> None:
    """Register the top-level ``dagnam profile show <username>`` command."""
    profile_cmd = subparsers.add_parser(
        "profile",
        help="View a user's public profile.",
        description="View a user's publicly visible profile and published models.",
    )
    profile_sub = profile_cmd.add_subparsers(dest="profile_top_command", required=True)

    profile_show = profile_sub.add_parser("show", help="Print a user's public profile.")
    profile_show.add_argument("username", help="Username to look up.")
    _add_output_flags(profile_show)
    profile_show.set_defaults(func=cmd_profile_show)
