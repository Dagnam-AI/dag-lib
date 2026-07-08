"""Handlers for CLI version, identity, and configuration commands."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
from typing import TYPE_CHECKING, Any, TypeGuard

from dagnam.cli.account_data import register_account_data
from dagnam.cli.account_profile import register_account_profile
from dagnam.cli.account_security import register_account_security
from dagnam.cli.account_settings import register_notifications, register_settings
from dagnam.cli.common import error, mask_key, resolve_version
from dagnam.cli.presentation import Column, emit_result, render_table

if TYPE_CHECKING:
    from dagnam.cli.common import SubParsersAction

_MUTABLE_CONFIG_KEYS = {"training_metrics_path"}
_LIMIT_LABELS = {
    "storage.bytes": "storage",
    "storage.max_upload_bytes": "max upload size",
    "training.minutes_per_period": "training minutes",
    "training.max_duration_minutes": "max training duration",
    "training.concurrent_jobs": "concurrent training jobs",
    "models.max_parameters": "max model parameters",
    "projects.count": "projects",
    "projects.private_count": "private projects",
    "hub.private_model_count": "private hub models",
    "deployments.count": "deployments",
    "api_keys.count": "API keys",
    "projects.version_retention": "project versions retained",
}
# Short unit suffix per non-byte metric so every usage row carries a unit (byte
# rows already render B/KB/MB/GB via _format_bytes). Kept terse to fit the
# numeric columns; unknown keys fall back to no suffix.
_LIMIT_UNITS = {
    "training.minutes_per_period": "min",
    "training.max_duration_minutes": "min",
    "training.concurrent_jobs": "jobs",
    "models.max_parameters": "params",
    "projects.count": "projects",
    "projects.private_count": "projects",
    "hub.private_model_count": "models",
    "deployments.count": "deploys",
    "api_keys.count": "keys",
    "projects.version_retention": "versions",
}


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

    api_key = get_api_key()

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


def cmd_logout(args: argparse.Namespace) -> None:
    """Remove the stored API key while preserving other config values."""
    from dagnam._core import config as _cfg

    config = _read_config_file(_cfg.CONFIG_FILE)
    if config is None or "api_key" not in config:
        print("Not logged in.")
        return

    del config["api_key"]
    _cfg.save_config(config)
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
    _cfg.save_config(config)
    print(f"Set {args.key} = {args.value}")


def cmd_config_unset(args: argparse.Namespace) -> None:
    """Unset a supported saved config value while preserving other keys."""
    from dagnam._core import config as _cfg

    if args.key not in _MUTABLE_CONFIG_KEYS:
        error(f"Unsupported config key for unset: {args.key}")

    config = _read_config_file(_cfg.CONFIG_FILE) or {}
    config.pop(args.key, None)
    _cfg.save_config(config)
    print(f"Unset {args.key}")


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, int | float)


def _is_byte_limit(key: object) -> bool:
    if not isinstance(key, str):
        return False
    return key.endswith(".bytes") or key.endswith("_bytes")


def _trim_decimal(value: float) -> str:
    text = f"{value:.1f}"
    return text[:-2] if text.endswith(".0") else text


def _format_bytes(value: int | float) -> str:
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    amount = float(value)
    for unit in units:
        if abs(amount) < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(amount)} B"
            return f"{amount:.1f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable byte unit loop")  # pragma: no cover - final unit returns


def _format_count(value: int | float) -> str:
    amount = float(value)
    for threshold, suffix in (
        (1_000_000_000_000, "T"),
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    ):
        if abs(amount) >= threshold:
            return f"{_trim_decimal(amount / threshold)}{suffix}"
    return str(int(amount)) if amount.is_integer() else _trim_decimal(amount)


def _limit_unit(key: object) -> str:
    if not isinstance(key, str):
        return ""
    return _LIMIT_UNITS.get(key, "")


def _format_usage_value(limit_key: object, value: object) -> str:
    if value is None:
        return "unlimited"
    if not _is_number(value):
        return "-"
    if _is_byte_limit(limit_key):
        return _format_bytes(value)
    count = _format_count(value)
    unit = _limit_unit(limit_key)
    return f"{count} {unit}" if unit else count


def _limit_label(key: object) -> str:
    if not isinstance(key, str):
        return "-"
    return _LIMIT_LABELS.get(key, key.replace(".", " ").replace("_", " "))


def _remaining_ratio(current: object, limit: object) -> float | None:
    if not _is_number(current) or not _is_number(limit):
        return None
    if limit <= 0:
        return 0.0
    return max(min((limit - current) / limit, 1.0), 0.0)


def _remaining_bar(current: object, limit: object, *, width: int = 10) -> str:
    ratio = _remaining_ratio(current, limit)
    if ratio is None:
        return "unlimited" if limit is None else "-"
    filled = round(ratio * width)
    return f"{'#' * filled}{'-' * (width - filled)}"


def _remaining_percent(current: object, limit: object) -> str:
    ratio = _remaining_ratio(current, limit)
    if ratio is None:
        return "unlimited" if limit is None else "-"
    return f"{round(ratio * 100)}%"


def _render_usage(snapshot: object) -> str:
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    plan = snapshot.get("plan")
    plan = plan if isinstance(plan, dict) else {}
    plan_name = plan.get("display_name") or plan.get("code") or "-"
    lines = [f"Plan: {plan_name}"]
    if snapshot.get("read_only_grace"):
        lines.append("Status: READ-ONLY GRACE (usage limit reached)")
    if snapshot.get("pending_plan"):
        lines.append(f"Pending plan: {snapshot['pending_plan']}")

    limits = snapshot.get("limits")
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
            remaining = None if limit is None else "-"
        rows.append(
            {
                "key": _limit_label(entry.get("key")),
                "current": _format_usage_value(entry.get("key"), current),
                "limit": _format_usage_value(entry.get("key"), limit),
                "remaining": _format_usage_value(entry.get("key"), remaining),
                "bar": _remaining_bar(current, limit),
                "percent": _remaining_percent(current, limit),
            }
        )
    table = render_table(
        (
            Column("Limit type", "key", 32),
            Column("Used", "current", 12, "right"),
            Column("Limit", "limit", 12, "right"),
            Column("Remaining", "remaining", 12, "right"),
            Column("Meter", "bar", 10),
            Column("Available %", "percent", 11, "right"),
        ),
        rows,
    )
    return "\n".join([*lines, "", table])


def cmd_usage(args: argparse.Namespace) -> None:
    """Print the caller's plan and real-time usage against plan limits."""
    import dagnam

    snapshot = dagnam.account.entitlements()

    emit_result(
        snapshot,
        output=args.output,
        json_stdout=args.json,
        render_human=_render_usage,
    )


def register_account(subparsers: SubParsersAction) -> None:
    """Register the ``version``, ``whoami``, ``usage``, ``logout``, and ``config`` commands."""
    version_cmd = subparsers.add_parser(
        "version",
        help="Show version and environment info.",
        description="Print the dagnam version, Python version, and platform.",
    )
    version_cmd.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    version_cmd.set_defaults(func=cmd_version)

    whoami = subparsers.add_parser(
        "whoami",
        help="Show the current authenticated identity.",
        description="Show the resolved API URL, masked API key, and its source.",
    )
    whoami.set_defaults(func=cmd_whoami)

    usage = subparsers.add_parser(
        "usage",
        help="Show plan, usage, and remaining limits.",
        description="Show your plan and real-time usage against plan limits.",
    )
    usage.add_argument("--json", action="store_true", help="Print the full entitlement snapshot.")
    usage.add_argument("--output", help="Write the full entitlement snapshot to this path.")
    usage.set_defaults(func=cmd_usage)

    logout = subparsers.add_parser(
        "logout",
        help="Remove stored credentials.",
        description="Remove the stored API key from ~/.dagnam/config.json.",
    )
    logout.set_defaults(func=cmd_logout)

    config_cmd = subparsers.add_parser(
        "config",
        help="Inspect and update saved configuration.",
        description="Read and update supported values in ~/.dagnam/config.json.",
    )
    config_sub = config_cmd.add_subparsers(dest="config_command", required=True)
    config_list = config_sub.add_parser(
        "list", help="Print all config values.", description="Print config (api_key masked)."
    )
    config_list.set_defaults(func=cmd_config_list)
    config_get = config_sub.add_parser(
        "get", help="Print one config value.", description="Print a single config value."
    )
    config_get.add_argument("key", help="Config key to read, e.g. api_url.")
    config_get.set_defaults(func=cmd_config_get)
    config_set = config_sub.add_parser(
        "set",
        help="Set a config value.",
        description="Set a supported config value such as training_metrics_path.",
    )
    config_set.add_argument("key", help="Config key to set, e.g. training_metrics_path.")
    config_set.add_argument("value", help="Value to save.")
    config_set.set_defaults(func=cmd_config_set)
    config_unset = config_sub.add_parser(
        "unset",
        help="Unset a config value.",
        description="Unset a supported config value such as training_metrics_path.",
    )
    config_unset.add_argument("key", help="Config key to unset, e.g. training_metrics_path.")
    config_unset.set_defaults(func=cmd_config_unset)

    account_cmd = subparsers.add_parser(
        "account",
        help="Manage settings and notification preferences.",
        description="Get, update, or reset the caller's settings and notification preferences.",
    )
    account_sub = account_cmd.add_subparsers(dest="account_command", required=True)
    register_settings(account_sub)
    register_notifications(account_sub)
    register_account_profile(account_sub)
    register_account_security(account_sub)
    register_account_data(account_sub)
