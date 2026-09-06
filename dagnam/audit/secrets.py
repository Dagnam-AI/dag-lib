"""Where deployment keys live: the OS keyring, else a 0600 file under the audit dir (spec section 10).

The state file records a ``key_ref``; the key itself is never in ``state.json``,
never in a report, never printed. ``keyring`` is the optional ``dagnam[audit]``
extra, so it is imported lazily and its absence (or a machine with no
keyring backend, e.g. a headless server) falls back to the file with a warning.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from types import ModuleType

SERVICE = "dagnam-audit"
SECRETS_FILE = "secrets.json"
_LOGGER = logging.getLogger("dagnam.audit")


def _keyring() -> tuple[ModuleType, type[Exception]] | None:
    """The ``keyring`` module and its error base class, or ``None`` without the extra."""
    try:
        import keyring
        from keyring.errors import KeyringError
    except ImportError:
        return None
    return keyring, KeyringError


class SecretStore:
    """Store and read back one audit directory's deployment keys by ``key_ref``."""

    def __init__(self, audit_dir: Path) -> None:
        self._dir = audit_dir
        self.mode: str | None = None
        """``"keyring"`` or ``"file"`` once a key has been stored; ``None`` before."""

    def _user(self, key_ref: str) -> str:
        return f"{self._dir.resolve()}::{key_ref}"

    def _path(self) -> Path:
        return self._dir / SECRETS_FILE

    def _read_file(self) -> dict[str, str]:
        path = self._path()
        if not path.exists():
            return {}
        return {str(k): str(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}

    def _write_file(self, secrets: dict[str, str]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._path()
        # Created 0600 from the first byte; chmod covers a file that already existed wider.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(secrets, handle, indent=2)
        os.chmod(path, 0o600)

    def store(self, key_ref: str, value: str) -> str:
        """Store ``value`` under ``key_ref`` and return the mode used (``keyring`` or ``file``)."""
        backend = _keyring()
        if backend is not None:
            module, error = backend
            try:
                module.set_password(SERVICE, self._user(key_ref), value)
            except error as exc:
                _LOGGER.warning(
                    "keyring unavailable (%s); storing %s in %s", exc, key_ref, SECRETS_FILE
                )
            else:
                self.mode = "keyring"
                return self.mode
        else:
            _LOGGER.warning(
                "keyring is not installed (pip install 'dagnam[audit]'); storing %s in %s",
                key_ref,
                SECRETS_FILE,
            )
        self._write_file({**self._read_file(), key_ref: value})
        self.mode = "file"
        return self.mode

    def load(self, key_ref: str) -> str | None:
        """The key stored under ``key_ref`` from either backend, or ``None``."""
        backend = _keyring()
        if backend is not None:
            module, error = backend
            try:
                found = module.get_password(SERVICE, self._user(key_ref))
            except error:
                found = None
            if isinstance(found, str):
                return found
        return self._read_file().get(key_ref)


__all__ = ["SECRETS_FILE", "SERVICE", "SecretStore"]
