"""Handlers for ``dagnam account profile`` and the top-level ``dagnam profile`` group.

Split out of ``dagnam.cli.account`` (which retains version/whoami/usage/
logout/config) to keep every module under the repo's ~500-line file-size cap.
Holds both profile CLI surfaces: the authenticated ``dagnam account profile
[get|set|photo]`` subcommands and the public, read-only ``dagnam profile show
<username>`` top-level command.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from dagnam._types import JsonObject
from dagnam.cli.common import error
from dagnam.cli.presentation import Column, emit_result, render_table

if TYPE_CHECKING:
    from dagnam.cli.common import SubParsersAction

# Names the resource layer's `update_profile` reserves for client resolution
# (see dagnam.resources.account). No real backend profile field uses these,
# but the patch dict is later spread as `**patch` into that call, so a field
# literally named one of these would shadow the resolver kwarg instead of
# landing in the JSON body. Reject it up front.
_RESERVED_PATCH_KEYS = frozenset({"client", "api_key", "api_url"})


def parse_profile_patch(pairs: list[str]) -> JsonObject:
    """Parse ``KEY=VALUE`` CLI args into a JSON profile-patch dict.

    Every profile field (``first_name``, ``bio``, ``website``, ...) is a plain
    string on the wire, so no type coercion is needed beyond the reserved-key
    guard shared with the other account patch commands.
    """
    patch: JsonObject = {}
    for pair in pairs:
        key, sep, raw_value = pair.partition("=")
        if not sep or not key:
            error(f"Invalid KEY=VALUE argument: {pair!r}")
        if key in _RESERVED_PATCH_KEYS:
            error(f"Unsupported field name: {key!r}")
        patch[key] = raw_value
    return patch


def _render_profile_table(payload: object) -> str:
    data: dict[str, object] = payload if isinstance(payload, dict) else {}
    rows: list[dict[str, object]] = [
        {"field": key, "value": str(value)} for key, value in data.items() if key != "id"
    ]
    if not rows:
        return "No data returned."
    return render_table((Column("Field", "field", 28), Column("Value", "value", 48)), rows)


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


def _render_photo_result(payload: object) -> str:
    data: dict[str, object] = payload if isinstance(payload, dict) else {}
    url = data.get("profile_photo_url")
    return str(url) if url is not None else "Photo uploaded."


def cmd_profile_get(args: argparse.Namespace) -> None:
    """Print the caller's current profile."""
    import dagnam

    result = dagnam.account.get_profile()
    emit_result(
        result, output=args.output, json_stdout=args.json, render_human=_render_profile_table
    )


def cmd_profile_set(args: argparse.Namespace) -> None:
    """Patch one or more profile fields from ``KEY=VALUE`` arguments."""
    import dagnam

    patch = parse_profile_patch(args.pairs)
    # parse_profile_patch already rejects _RESERVED_PATCH_KEYS, so this spread
    # cannot shadow update_profile's own client/api_key/api_url kwargs at
    # runtime; pyright cannot see that invariant through a plain dict spread.
    result = dagnam.account.update_profile(**patch)  # pyright: ignore[reportArgumentType]
    emit_result(
        result, output=args.output, json_stdout=args.json, render_human=_render_profile_table
    )


def cmd_profile_photo(args: argparse.Namespace) -> None:
    """Upload a local image file as the caller's profile photo."""
    path = Path(args.path)
    if not path.is_file():
        error(f"No such file: {path}")

    import dagnam

    result = dagnam.account.upload_profile_photo(str(path))
    emit_result(
        result, output=args.output, json_stdout=args.json, render_human=_render_photo_result
    )


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


def register_account_profile(account_sub: SubParsersAction) -> None:
    """Register ``dagnam account profile [get|set|photo]`` under the ``account`` group."""
    profile_cmd = account_sub.add_parser(
        "profile", help="Get, update, or upload the caller's profile."
    )
    profile_sub = profile_cmd.add_subparsers(dest="profile_command", required=True)

    profile_get = profile_sub.add_parser("get", help="Print the current profile.")
    _add_output_flags(profile_get)
    profile_get.set_defaults(func=cmd_profile_get)

    profile_set = profile_sub.add_parser(
        "set",
        help="Update one or more profile fields.",
        description="Update the profile via KEY=VALUE pairs, e.g. bio='Building things.'",
    )
    profile_set.add_argument("pairs", nargs="+", metavar="KEY=VALUE", help="Field(s) to update.")
    _add_output_flags(profile_set)
    profile_set.set_defaults(func=cmd_profile_set)

    profile_photo = profile_sub.add_parser(
        "photo",
        help="Upload a profile photo.",
        description="Upload a local PNG/JPG/JPEG/WEBP image as the profile photo.",
    )
    profile_photo.add_argument("path", help="Path to a local image file.")
    _add_output_flags(profile_photo)
    profile_photo.set_defaults(func=cmd_profile_photo)


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
