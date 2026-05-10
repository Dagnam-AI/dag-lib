"""Login command handler."""

from __future__ import annotations

import getpass
import json

from dagnam.cli.common import _error


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

    # Persist
    _cfg.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config: dict = {}
    if _cfg.CONFIG_FILE.exists():
        try:
            config = json.loads(_cfg.CONFIG_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            config = {}

    config["api_key"] = api_key
    if api_url != "https://api.dagnam.ai":
        config["api_url"] = api_url

    _cfg.CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Credentials saved to {_cfg.CONFIG_FILE}")
