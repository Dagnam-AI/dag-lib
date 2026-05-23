"""Configuration management for the dagnam library.

Reads persistent configuration from ~/.dagnam/config.json.
"""

import json
import logging
from pathlib import Path
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
