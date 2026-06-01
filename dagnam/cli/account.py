"""Handlers for CLI version, identity, and configuration commands."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

from dagnam.cli.common import error, mask_key, resolve_version
from dagnam.cli.presentation import Column, emit_result, render_table

_MUTABLE_CONFIG_KEYS = {"training_metrics_path"}


def cmd_version(args: argparse.Namespace) -> None:
    """Print version plus interpreter/platform info for bug reports."""
    payload = {
        "dagnam": resolve_version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    print(f"dagnam {payload['dagnam']}")
    print(f"Python {payload['python']}")
    print(payload["platform"])


def cmd_whoami(args: argparse.Namespace) -> None:
    """Print the resolved API URL, masked key, and credential source."""
    from dagnam._core.auth import get_api_key, get_api_url
    from dagnam._core.exceptions import AuthError

    try:
        api_key = get_api_key()
    except AuthError:
        print("Not logged in. Run 'dagnam login'.", file=sys.stderr)
        sys.exit(1)

    source = "DAGNAM_API_KEY environment variable"
    if os.environ.get("DAGNAM_API_KEY") is None:
        source = "config file"

    print(f"API URL: {get_api_url()}")
    print(f"API key: {mask_key(api_key)}")
    print(f"Source: {source}")


def _read_config_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        error(f"Could not read {path}: {exc}")
    if not isinstance(data, dict):
        error(f"Could not read {path}: expected a JSON object")
    return data


def _write_config_securely(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(config, indent=2))
        fh.write("\n")
    if sys.platform != "win32":
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass


def cmd_logout(args: argparse.Namespace) -> None:
    """Remove the stored API key while preserving other config values."""
    from dagnam._core import config as _cfg

    config = _read_config_file(_cfg.CONFIG_FILE)
    if config is None or "api_key" not in config:
        print("Not logged in.")
        return

    del config["api_key"]
    _write_config_securely(_cfg.CONFIG_FILE, config)
    print("Logged out.")

    if os.environ.get("DAGNAM_API_KEY") is not None:
        print(
            "Note: DAGNAM_API_KEY is still set in your environment and will "
            "continue to authenticate requests.",
            file=sys.stderr,
        )


def _masked_config(config: dict[str, Any]) -> dict[str, Any]:
    masked = dict(config)
    api_key = masked.get("api_key")
    if isinstance(api_key, str):
        masked["api_key"] = mask_key(api_key)
    return masked


def cmd_config_list(args: argparse.Namespace) -> None:
    """Print saved config values, masking secrets."""
    from dagnam._core import config as _cfg

    config = _read_config_file(_cfg.CONFIG_FILE)
    print(json.dumps(_masked_config(config or {}), indent=2))


def cmd_config_get(args: argparse.Namespace) -> None:
    """Print a single saved config value, masking secrets."""
    from dagnam._core import config as _cfg

    config = _read_config_file(_cfg.CONFIG_FILE) or {}
    if args.key not in config:
        if args.key in _MUTABLE_CONFIG_KEYS:
            print(f"{args.key} is not configured")
            return
        error(f"Config key not found: {args.key}")

    value = config[args.key]
    if args.key == "api_key" and isinstance(value, str):
        value = mask_key(value)
    print(value)


def cmd_config_set(args: argparse.Namespace) -> None:
    """Set a supported saved config value while preserving other keys."""
    from dagnam._core import config as _cfg

    if args.key not in _MUTABLE_CONFIG_KEYS:
        error(f"Unsupported config key for set: {args.key}")

    config = _read_config_file(_cfg.CONFIG_FILE) or {}
    config[args.key] = args.value
    _write_config_securely(_cfg.CONFIG_FILE, config)
    print(f"Set {args.key} = {args.value}")


def cmd_config_unset(args: argparse.Namespace) -> None:
    """Unset a supported saved config value while preserving other keys."""
    from dagnam._core import config as _cfg

    if args.key not in _MUTABLE_CONFIG_KEYS:
        error(f"Unsupported config key for unset: {args.key}")

    config = _read_config_file(_cfg.CONFIG_FILE) or {}
    config.pop(args.key, None)
    _write_config_securely(_cfg.CONFIG_FILE, config)
    print(f"Unset {args.key}")


def _usage_value(entry: dict[str, Any], key: str) -> Any:
    value = entry.get(key)
    return value if value is not None else "-"


def _render_usage(snapshot: object) -> str:
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    plan = snapshot.get("plan") if isinstance(snapshot, dict) else None
    plan = plan if isinstance(plan, dict) else {}
    plan_name = plan.get("display_name") or plan.get("code") or "-"
    lines = [f"Plan: {plan_name}"]
    if snapshot.get("read_only_grace"):
        lines.append("Status: READ-ONLY GRACE (usage limit reached)")
    if snapshot.get("pending_plan"):
        lines.append(f"Pending plan: {snapshot['pending_plan']}")

    limits = snapshot.get("limits") if isinstance(snapshot, dict) else None
    limits = limits if isinstance(limits, list) else []
    if not limits:
        return "\n".join([*lines, "No limit information returned."])
    rows: list[dict[str, object]] = []
    for item in limits:
        entry = item if isinstance(item, dict) else {}
        current = entry.get("current")
        limit = entry.get("limit")
        if isinstance(current, int | float) and isinstance(limit, int | float):
            remaining: Any = max(limit - current, 0)
        else:
            remaining = "unlimited" if limit is None else "-"
        limit_display = "unlimited" if limit is None else limit
        rows.append(
            {
                "key": entry.get("key", "-"),
                "current": _usage_value(entry, "current"),
                "limit": limit_display,
                "remaining": remaining,
            }
        )
    table = render_table(
        (
            Column("Limit", "key", 32),
            Column("Used", "current", 12, "right"),
            Column("Limit", "limit", 12, "right"),
            Column("Remaining", "remaining", 12, "right"),
        ),
        rows,
    )
    return "\n".join([*lines, "", table])


def cmd_usage(args: argparse.Namespace) -> None:
    """Print the caller's plan and real-time usage against plan limits."""
    import dagnam
    from dagnam._core.exceptions import DagnamError

    try:
        snapshot = dagnam.account.entitlements()
    except DagnamError as exc:
        error(str(exc))

    emit_result(
        snapshot,
        output=args.output,
        json_stdout=args.json,
        render_human=_render_usage,
    )
