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

_MUTABLE_CONFIG_KEYS = {"training_metrics_path"}


def cmd_version(args: argparse.Namespace) -> None:
    """Print version plus interpreter/platform info for bug reports."""
    print(f"dagnam {resolve_version()}")
    print(f"Python {platform.python_version()}")
    print(platform.platform())


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
    print(f"Source:  {source}")


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
