"""Configuration management for the dagnam library.

Reads persistent configuration from ~/.dagnam/config.json.
"""

import json
import logging
import os
from pathlib import Path
import sys

from dagnam._types import JsonObject, JsonValue, ensure_json_object

CONFIG_DIR: Path = Path.home() / ".dagnam"
CONFIG_FILE: Path = CONFIG_DIR / "config.json"

_logger = logging.getLogger(__name__)


def load_config() -> JsonObject:
    """Read ~/.dagnam/config.json. Returns empty dict if file doesn't exist or is malformed."""
    try:
        return ensure_json_object(json.loads(CONFIG_FILE.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        # Malformed config is a real user-visible problem (e.g. a partial
        # write from a crashed `dagnam login`), so surface it at debug level
        # rather than silently falling back to defaults.
        _logger.debug("Failed to parse %s as JSON: %s", CONFIG_FILE, exc)
        return {}
    except OSError as exc:
        _logger.debug("Failed to read %s: %s", CONFIG_FILE, exc)
        return {}


def get_config_value(key: str, default: JsonValue = None) -> JsonValue:
    """Get a single value from the config file."""
    return load_config().get(key, default)


def save_config(config: JsonObject) -> None:
    """Write ``config`` to ``CONFIG_FILE`` with owner-only (0o600) permissions.

    Creates ``CONFIG_DIR`` if needed, then writes via ``os.open`` with
    ``O_CREAT | O_TRUNC | O_WRONLY`` (plus ``O_NOFOLLOW`` where supported, to
    reject a symlink swap). The ``0o600`` mode passed to ``os.open`` is applied
    only when the file is newly created, so a fresh config file is never
    world-readable; when overwriting a pre-existing file the content is
    truncated in place and the trailing POSIX-only ``chmod`` re-tightens the
    mode. That ``chmod`` is best-effort - a failure (e.g. on an unusual
    filesystem) must not abort a config write, and Windows relies on its own
    filesystem ACLs instead, so the call is skipped there entirely. This is
    the single writer every CLI command that persists config (login, logout,
    config set/unset, register) shares, so a secure-write fix only needs to
    land in one place.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(CONFIG_FILE, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(config, indent=2))
        fh.write("\n")
    if sys.platform != "win32":
        try:
            os.chmod(CONFIG_FILE, 0o600)
        except OSError:
            pass
