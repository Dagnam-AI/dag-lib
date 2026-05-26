"""Login command handler."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import getpass
import json
import os
from pathlib import Path
import sys

from dagnam._types import JsonObject, ensure_json_object
from dagnam.cli.common import error


def _web_url_from_api_url(api_url: str) -> str:
    """Best-effort guess at the matching frontend URL (empty if unknown)."""
    normalized = api_url.rstrip("/")
    if normalized == "https://api.dagnam.ai":
        return "https://dagnam.ai"
    if normalized.startswith("http://localhost:") or normalized.startswith("http://127.0.0.1:"):
        return "http://localhost:5173"
    return ""


def _lock_down_config_path(config_dir: Path, config_file: Path) -> None:
    """Verify and restrict POSIX config path ownership and permissions."""
    if sys.platform == "win32":
        return

    current_uid = os.getuid() if hasattr(os, "getuid") else None
    try:
        dir_stat = os.stat(config_dir)
    except OSError as exc:
        error(f"Could not inspect config directory permissions: {exc}")

    if current_uid is not None and dir_stat.st_uid != current_uid:
        error(f"Refusing to use config directory not owned by current user: {config_dir}")

    if dir_stat.st_mode & 0o077:
        try:
            os.chmod(config_dir, 0o700)
        except OSError as exc:
            error(f"Could not restrict config directory permissions: {exc}")

    if config_file.exists():
        try:
            file_stat = os.stat(config_file)
        except OSError as exc:
            error(f"Could not inspect config file permissions: {exc}")
        if current_uid is not None and file_stat.st_uid != current_uid:
            error(f"Refusing to overwrite config file not owned by current user: {config_file}")


def cmd_login(args: argparse.Namespace, getpass_func: Callable[[str], str] | None = None) -> None:
    from dagnam._core import config as _cfg
    from dagnam._core.client import DagnamClient
    from dagnam._core.exceptions import DagnamError

    api_url = getattr(args, "api_url", None) or "https://api.dagnam.ai"
    web_url = _web_url_from_api_url(api_url)

    print("Log in to Dagnam.\n")
    print("Don't have an API key yet?")
    if web_url:
        print(f"  1. Sign up or log in at  {web_url}")
        print("  2. Go to  Settings > Security > Personal API Keys")
        print('  3. Click "Create API Key", copy the sk_... value')
        print("  4. Paste it below.\n")
    else:
        print(f"  1. Open the frontend that talks to {api_url} and sign up there.")
        print("  2. Settings > Security > Personal API Keys > Create, copy the sk_... value")
        print("  3. Paste it below.\n")

    api_key = (getpass_func or getpass.getpass)("API key: ")
    if not api_key.strip():
        error("API key cannot be empty.")

    # Validate by hitting a normal authenticated endpoint.
    client = DagnamClient(api_url, api_key)
    try:
        client.list_datasets()
    except DagnamError as exc:
        settings_url = f"{web_url}/settings?tab=security" if web_url else api_url
        signup_url = web_url or api_url
        error(
            f"Authentication failed: {exc}\n\n"
            "The key wasn't accepted. Possible causes:\n"
            f"  - You haven't created a key yet     -> {settings_url}\n"
            f"  - You don't have an account yet     -> {signup_url}\n"
            "  - You're pointed at the wrong server (try `dagnam login --api-url <url>`)"
        )

    # Persist. On POSIX, lock down the directory (0700) and file (0600) so the
    # API key isn't world-readable by other local users. chmod is a no-op on
    # Windows (where filesystem ACLs handle this) so we guard the call.
    _cfg.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _lock_down_config_path(_cfg.CONFIG_DIR, _cfg.CONFIG_FILE)

    config: JsonObject = {}
    if _cfg.CONFIG_FILE.exists():
        try:
            config = ensure_json_object(json.loads(_cfg.CONFIG_FILE.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            config = {}

    config["api_key"] = api_key
    if api_url != "https://api.dagnam.ai":
        config["api_url"] = api_url
    training_metrics_path = getattr(args, "training_metrics_path", None)
    if training_metrics_path:
        config["training_metrics_path"] = training_metrics_path

    # Write atomically with restrictive permissions from the start: create the
    # file via os.open with O_CREAT | O_WRONLY | O_TRUNC and mode 0o600 so the
    # key never sits on disk with default umask-derived perms.
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(_cfg.CONFIG_FILE, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(config, indent=2))
    except BaseException:
        # fdopen took ownership of fd on success; on failure before fdopen
        # returns we'd already have leaked fd, but os.fdopen either succeeds
        # or raises after wrapping, so a generic guard is enough here.
        raise
    if sys.platform != "win32":
        try:
            os.chmod(_cfg.CONFIG_FILE, 0o600)
        except OSError:
            pass
    print(f"Credentials saved to {_cfg.CONFIG_FILE}")
    if not config.get("training_metrics_path"):
        print(
            "Local training metrics path is not configured. To view local training "
            "progress in Dagnam, run: dagnam config set training_metrics_path "
            "./dagnam_metrics.jsonl"
        )
