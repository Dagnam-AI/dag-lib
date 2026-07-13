"""Configuration management for the dagnam library.

Reads persistent configuration from ~/.dagnam/config.json.
"""

import json
import logging
import os
from pathlib import Path
import sys

from dagnam._core.exceptions import DagnamError
from dagnam._types import JsonObject, JsonValue, ensure_json_object

CONFIG_DIR: Path = Path.home() / ".dagnam"
CONFIG_FILE: Path = CONFIG_DIR / "config.json"

_logger = logging.getLogger(__name__)

# Warn-once-per-process guard for the config-file permission check. Process-
# global by design: the check is warn-only and must not spam every request.
_config_perms_warned = False


def _warn_if_config_insecure() -> None:
    """Warn once per process if the config file is foreign-owned or exposed.

    The file may hold the API key, so any group/world access bit — or a
    foreign owner, which is a tampering vector for ``api_url`` redirection —
    warrants a loud, one-time warning. Reads are never blocked: locking a
    user out over permissions would be worse than the exposure. POSIX only;
    Windows relies on filesystem ACLs (mirrors ``save_config``'s chmod skip).
    """
    global _config_perms_warned
    if _config_perms_warned or sys.platform == "win32":
        return
    try:
        st = os.stat(CONFIG_FILE)
    except OSError:
        return
    foreign = st.st_uid != os.getuid()
    exposed = bool(st.st_mode & 0o077)
    if foreign or exposed:
        _config_perms_warned = True
        _logger.warning(
            "Dagnam config file %s has insecure permissions (foreign-owner=%s, "
            "group/world-accessible=%s); it may hold your API key and could be "
            "read or tampered with by other users. Restrict it with: chmod 600 %s",
            CONFIG_FILE,
            foreign,
            exposed,
            CONFIG_FILE,
        )


def load_config() -> JsonObject:
    """Read ~/.dagnam/config.json. Returns empty dict if file doesn't exist or is malformed."""
    try:
        parsed = ensure_json_object(json.loads(CONFIG_FILE.read_text(encoding="utf-8")))
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
    _warn_if_config_insecure()
    return parsed


def get_config_value(key: str, default: JsonValue = None) -> JsonValue:
    """Get a single value from the config file."""
    return load_config().get(key, default)


def _ensure_secure_config_location() -> None:
    """POSIX: refuse to write into a foreign-owned config dir/file; tighten dir mode.

    Shared by every writer (login, register, logout, config set/unset) so they all
    get the protection ``cli.login`` used to enforce alone: in a world-writable
    home where an attacker pre-created ``~/.dagnam`` or ``config.json``, the
    freshly-minted API key must not be written into a location the attacker owns
    or can read. Windows relies on filesystem ACLs and is skipped. A missing dir
    is fine — ``save_config`` creates it with a tight mode.
    """
    if sys.platform == "win32":
        return
    current_uid = os.getuid() if hasattr(os, "getuid") else None
    try:
        dir_stat = os.stat(CONFIG_DIR)
    except OSError:
        return
    if current_uid is not None and dir_stat.st_uid != current_uid:
        raise DagnamError(
            f"Refusing to use config directory not owned by current user: {CONFIG_DIR}"
        )
    if dir_stat.st_mode & 0o077:
        os.chmod(CONFIG_DIR, 0o700)
    if CONFIG_FILE.exists():
        file_stat = os.stat(CONFIG_FILE)
        if current_uid is not None and file_stat.st_uid != current_uid:
            raise DagnamError(
                f"Refusing to overwrite config file not owned by current user: {CONFIG_FILE}"
            )


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
    _ensure_secure_config_location()
    CONFIG_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
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
