"""The trace directory: made private, read for its saved status and measured.

Permissions: the directory is created ``0700`` (its parents get the process defaults). An
existing directory must be a real directory (not a symlink) owned by this user. If it is open to
group or others it is tightened to ``0700`` only when it is empty or already holds a recorder
file; a shared directory with other files in it (``/var/log/app``, a home directory) is refused,
because taking its group's access away would break whoever reads it. The directory is opened
with ``O_DIRECTORY|O_NOFOLLOW`` and checked and changed through that descriptor, so it cannot be
swapped between the check and the change. Where ``O_DIRECTORY`` does not exist (Windows) a
directory is only required to be a real one.
"""

from __future__ import annotations

import errno
import json
import os
import re
import stat
from typing import Any, Callable, Dict, List

STATUS_FILE = ".status.json"
DISABLED_FILE = ".disabled"
"""A file of this name in the directory switches recording off (see ``Recorder.paused``)."""
STATUS_MAX = 1_000_000
"""The most of a ``.status.json`` that is read: a larger one is treated as unreadable."""
NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
NONBLOCK = getattr(os, "O_NONBLOCK", 0)
DIRECTORY = getattr(os, "O_DIRECTORY", 0)
fchmod: Callable[[int, int], None] = getattr(os, "fchmod", lambda descriptor, mode: None)
_OWNER = os.getuid() if hasattr(os, "getuid") else None
_DATA_NAME = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\.jsonl")


def _is_ours(names: List[str]) -> bool:
    """Whether a directory with these entries is, or may become, the recorder's: empty, or has its files."""
    return not names or any(name == STATUS_FILE or _DATA_NAME.fullmatch(name) for name in names)


def ensure_private_dir(directory: str) -> None:
    """Make sure ``directory`` is a private directory of ours, or raise ``OSError``."""
    os.makedirs(os.path.dirname(directory) or ".", exist_ok=True)
    try:
        os.mkdir(directory, 0o700)
        return
    except FileExistsError:
        pass
    if not DIRECTORY:  # no way to open a directory: all that can be asked is that it is one
        if os.path.islink(directory) or not os.path.isdir(directory):
            raise NotADirectoryError(errno.ENOTDIR, "not a real directory", directory)
        return
    # A symlink or a file fails the open; the owner and mode are read from the open descriptor,
    # and changed through it, so the path cannot be swapped for another in between.
    descriptor = os.open(directory, os.O_RDONLY | DIRECTORY | NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if _OWNER is not None and info.st_uid != _OWNER:
            raise PermissionError(errno.EPERM, "the trace directory belongs to someone else")
        if info.st_mode & 0o077:
            if not _is_ours(os.listdir(descriptor)):
                raise PermissionError(
                    errno.EPERM, "the trace directory is shared and holds other files; not changed"
                )
            fchmod(descriptor, 0o700)
    finally:
        os.close(descriptor)
    if not os.access(directory, os.W_OK | os.X_OK):
        raise PermissionError(errno.EACCES, "the trace directory is not writable", directory)


def recorded_bytes(directory: str) -> int:
    """The size of the dated data files in ``directory``; 0 if it cannot be listed."""
    total = 0
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if _DATA_NAME.fullmatch(entry.name) and entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
    except OSError:
        return 0
    return total


def read_status(directory: str) -> Dict[str, Any]:
    """The saved status, or ``{}`` for anything but a readable regular file holding an object.

    The open cannot wait (a FIFO would never deliver a writer) and does not follow a symlink, and
    a file of more than ``STATUS_MAX`` characters counts as unreadable.
    """
    try:
        descriptor = os.open(
            os.path.join(directory, STATUS_FILE), os.O_RDONLY | NOFOLLOW | NONBLOCK
        )
    except OSError:
        return {}
    with os.fdopen(descriptor, encoding="utf-8") as handle:
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                return {}
            text = handle.read(STATUS_MAX + 1)
            data = json.loads(text) if len(text) <= STATUS_MAX else None
        except (OSError, ValueError):
            return {}
    return data if isinstance(data, dict) else {}


def as_count(value: object) -> int:
    """A saved counter: a non-negative integer, else 0."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0
