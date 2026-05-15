"""Login command handler."""

from __future__ import annotations

import getpass
import json
import os
from pathlib import Path
import sys

from dagnam.cli.common import _error


def _lock_down_config_path(config_dir: Path, config_file: Path) -> None:
    """Verify and restrict POSIX config path ownership and permissions."""
    if sys.platform == "win32":
        return

    current_uid = os.getuid() if hasattr(os, "getuid") else None
    try:
        dir_stat = os.stat(config_dir)
    except OSError as exc:
        _error(f"Could not inspect config directory permissions: {exc}")

    if current_uid is not None and dir_stat.st_uid != current_uid:
        _error(f"Refusing to use config directory not owned by current user: {config_dir}")

    if dir_stat.st_mode & 0o077:
        try:
            os.chmod(config_dir, 0o700)
        except OSError as exc:
            _error(f"Could not restrict config directory permissions: {exc}")

    if config_file.exists():
        try:
            file_stat = os.stat(config_file)
        except OSError as exc:
            _error(f"Could not inspect config file permissions: {exc}")
        if current_uid is not None and file_stat.st_uid != current_uid:
            _error(f"Refusing to overwrite config file not owned by current user: {config_file}")


def _cmd_login(args, getpass_func=None) -> None:
    from dagnam._core import config as _cfg
    from dagnam._core.client import DagnamClient
    from dagnam._core.exceptions import DagnamError

    api_key = (getpass_func or getpass.getpass)("API key: ")
    if not api_key.strip():
        _error("API key cannot be empty.")

    api_url = getattr(args, "api_url", None) or "https://api.dagnam.ai"

    # Validate by hitting the meta endpoint for a quick test
    client = DagnamClient(api_url, api_key)
    try:
        client.list_datasets()
    except DagnamError as exc:
        _error(f"Authentication failed: {exc}")

    # Persist. On POSIX, lock down the directory (0700) and file (0600) so the
    # API key isn't world-readable by other local users. chmod is a no-op on
    # Windows (where filesystem ACLs handle this) so we guard the call.
    _cfg.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _lock_down_config_path(_cfg.CONFIG_DIR, _cfg.CONFIG_FILE)

    config: dict = {}
    if _cfg.CONFIG_FILE.exists():
        try:
            config = json.loads(_cfg.CONFIG_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            config = {}

    config["api_key"] = api_key
    if api_url != "https://api.dagnam.ai":
        config["api_url"] = api_url

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
