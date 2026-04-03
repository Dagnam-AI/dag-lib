"""Configuration management for the dagnam library.

Reads persistent configuration from ~/.dagnam/config.json.
"""

import json
from pathlib import Path
from typing import Any

CONFIG_DIR: Path = Path.home() / ".dagnam"
CONFIG_FILE: Path = CONFIG_DIR / "config.json"


def load_config() -> dict:
    """Read ~/.dagnam/config.json. Returns empty dict if file doesn't exist or is malformed."""
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a single value from the config file."""
    return load_config().get(key, default)
