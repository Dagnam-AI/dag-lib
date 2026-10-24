"""The recorder's files on disk: find them, prune the old ones, summarize them, purge them.

A recorder file is exactly one of two things in the trace directory: a daily data file named
``YYYY-MM-DD.jsonl`` or ``.status.json``, and in both cases a regular file. Everything else (other
names, directories, symlinks, FIFOs) is not the recorder's: it is never read, counted or deleted,
and a directory that is itself a symlink is refused. Nothing here prints or returns trace content;
the only text taken from a data file is a first and last timestamp, and only if it has the exact
shape of one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import errno
import json
import os
import re
import stat
from typing import Dict, List, Optional, Tuple

from dagnam_record._writer import COUNT_KEYS, STATUS_FILE, as_count, read_status

DEFAULT_DIR = "~/.dagnam/traces"
_DATA_NAME = re.compile(r"(\d{4}-\d{2}-\d{2})\.jsonl")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z")
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)


class NotRecorderDirectoryError(ValueError):
    """The directory holds no recorder file, so it is not touched."""


@dataclass(frozen=True)
class Found:
    """The recorder files of a directory: data files in date order, and the status file."""

    data: List[str]
    status: Optional[str]


@dataclass(frozen=True)
class Purged:
    """The names ``purge`` deleted and the names it could not delete."""

    deleted: List[str]
    failed: List[str]


@dataclass(frozen=True)
class Summary:
    """What ``python -m dagnam_record status`` reports."""

    directory: str
    files: int
    lines: int
    first: Optional[str]
    last: Optional[str]
    counts: Optional[Dict[str, int]]
    started: Optional[str]
    expired: Optional[bool]


def resolve(directory: str) -> str:
    """The absolute path of ``directory`` (``~`` expanded; symlinks are not followed here)."""
    return os.path.abspath(os.path.expanduser(directory))


def find(directory: str) -> Found:
    """The recorder files in ``directory``; ``OSError`` if it is missing, unreadable or a symlink."""
    if os.path.islink(directory):
        raise NotADirectoryError(errno.ENOTDIR, "the trace directory is a symlink", directory)
    data: List[str] = []
    status: Optional[str] = None
    with os.scandir(directory) as entries:
        for entry in entries:
            if not entry.is_file(follow_symlinks=False):  # not a symlink, FIFO or directory
                continue
            if _DATA_NAME.fullmatch(entry.name):
                data.append(entry.path)
            elif entry.name == STATUS_FILE:
                status = entry.path
    return Found(sorted(data), status)


def _remove_if_before(path: str, cutoff: date) -> bool:
    """Delete the data file at ``path`` if the date in its name is before ``cutoff``."""
    try:
        if date.fromisoformat(os.path.basename(path)[:-6]) >= cutoff:
            return False
        os.unlink(path)
    except (OSError, ValueError):  # not a real date, or not deletable: leave it
        return False
    return True


def prune(directory: str, window_days: int, now: float) -> List[str]:
    """Delete data files dated more than ``window_days`` days before ``now`` (UTC); return them.

    A file dated exactly ``window_days`` days ago is kept. Only recorder data files are
    candidates (never ``.status.json`` and never anything else), by the date in the name. A
    missing or unusable directory prunes nothing.
    """
    cutoff = datetime.fromtimestamp(now, timezone.utc).date() - timedelta(days=window_days)
    try:
        found = find(directory)
    except OSError:
        return []
    return [path for path in found.data if _remove_if_before(path, cutoff)]


def _stamp(raw: bytes) -> Optional[str]:
    """The ``ts`` of one line, only if it is a well-formed timestamp."""
    try:
        parsed = json.loads(raw)
    except ValueError:
        return None
    stamp = parsed.get("ts") if isinstance(parsed, dict) else None
    return stamp if isinstance(stamp, str) and _TIMESTAMP.fullmatch(stamp) else None


def _scan(path: str) -> Tuple[int, Optional[bytes], Optional[bytes]]:
    """The number of non-blank lines of a data file and its first and last one (as bytes)."""
    try:
        descriptor = os.open(path, os.O_RDONLY | _NOFOLLOW | _NONBLOCK)
    except OSError:
        return 0, None, None
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):  # swapped since the listing
        os.close(descriptor)
        return 0, None, None
    first: Optional[bytes] = None
    last: Optional[bytes] = None
    count = 0
    with os.fdopen(descriptor, "rb") as handle:
        for raw in handle:
            if raw.strip():
                count += 1
                first = raw if first is None else first
                last = raw
    return count, first, last


def summarize(directory: str) -> Summary:
    """Count the recorder's files and lines in ``directory`` and read its saved counts.

    Raises ``OSError`` for a directory that is missing, unreadable or a symlink, and
    ``NotRecorderDirectoryError`` when it holds no recorder file.
    """
    found = find(directory)
    if not found.data and found.status is None:
        raise NotRecorderDirectoryError(f"no recorder files in {directory}")
    lines = 0
    first: Optional[str] = None
    last: Optional[str] = None
    for path in found.data:
        count, head, tail = _scan(path)
        lines += count
        if head is not None and tail is not None:
            first = first or _stamp(head)
            last = _stamp(tail) or last
    saved = read_status(directory) if found.status is not None else {}
    counts = saved.get("counts")
    started = saved.get("started")
    expired = saved.get("expired")
    return Summary(
        directory=directory,
        files=len(found.data),
        lines=lines,
        first=first,
        last=last,
        counts=(
            {key: as_count(counts.get(key)) for key in COUNT_KEYS}
            if isinstance(counts, dict)
            else None
        ),
        started=(
            datetime.fromtimestamp(started, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if isinstance(started, (int, float)) and not isinstance(started, bool)
            else None
        ),
        expired=expired if isinstance(expired, bool) else None,
    )


def _unlink(path: str) -> Optional[bool]:
    """Delete ``path`` (never through a symlink): True, False if it cannot be, None if already gone."""
    try:
        os.unlink(path)
    except FileNotFoundError:  # removed by someone else since we looked: gone is gone
        return None
    except OSError:
        return False
    return True


def purge(directory: str) -> Purged:
    """Delete the recorder's files in ``directory`` and report which went and which could not.

    Raises ``NotRecorderDirectoryError`` when the directory holds no recorder file (nothing is
    deleted then) and ``OSError`` when it is missing, unreadable or a symlink.
    """
    found = find(directory)
    paths = found.data + ([found.status] if found.status is not None else [])
    if not paths:
        raise NotRecorderDirectoryError(f"no recorder files in {directory}")
    outcomes = {path: _unlink(path) for path in paths}
    deleted = [os.path.basename(path) for path, outcome in outcomes.items() if outcome is True]
    failed = [os.path.basename(path) for path, outcome in outcomes.items() if outcome is False]
    return Purged(deleted, failed)
