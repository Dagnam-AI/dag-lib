"""The recorder's files on disk: find them, prune the old ones, summarize them, purge them.

A recorder file is exactly one of two things in the trace directory: a daily data file named
``YYYY-MM-DD.jsonl`` (a real date) or ``.status.json``, and in both cases a regular file. Everything
else (other names, directories, symlinks, FIFOs) is not the recorder's: it is never read, counted
or deleted, and a directory that is itself a symlink is refused.

A name alone is not enough, because another program may use the same names. A daily file is the
recorder's only if its own first bytes are those of an envelope line (``{"v":1,"trace_id"``, which
the recorder always writes first); nothing else vouches for it, so an empty file is left alone.
``.status.json`` is the recorder's if it parses as an object with a ``counts`` object, or if some
daily file passed the test. A directory in which no file qualifies holds no recorder file.

Nothing here prints or returns trace content; the only text taken from a data file is a first and
last timestamp, and only if it has the exact shape of one. Data files are read in bounded pieces,
so a huge line in a foreign file costs no more memory than a short one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import errno
import os
import re
import stat
from typing import Dict, List, Optional, Tuple, Union

from dagnam_record._private_dir import STATUS_FILE, as_count, read_status
from dagnam_record._writer import COUNT_KEYS

DEFAULT_DIR = "~/.dagnam/traces"
_DATA_NAME = re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2})\.jsonl")
_PREFIX = b'{"v":1,"trace_id"'
_STAMPED = re.compile(
    rb'\{"v":1,"trace_id":"[^"]{1,128}","ts":"'
    rb'([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z)"'
)
_CHUNK = 65536  # the most of a line read at once
_HEAD = 512  # the start of each line kept to find its timestamp
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
    """The names ``purge`` deleted (or, for a dry run, would delete) and those it could not, with why."""

    deleted: List[str]
    failed: List[Tuple[str, str]]


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


def _real_date(name: str) -> bool:
    try:
        date.fromisoformat(name[:-6])
    except ValueError:
        return False
    return True


def starts_like_a_line(path: str) -> bool:
    """Whether the first bytes of the regular file at ``path`` are those of an envelope line."""
    try:
        descriptor = os.open(path, os.O_RDONLY | _NOFOLLOW | _NONBLOCK)
    except OSError:
        return False
    try:
        return stat.S_ISREG(os.fstat(descriptor).st_mode) and (
            os.read(descriptor, len(_PREFIX)) == _PREFIX
        )
    finally:
        os.close(descriptor)


def find(directory: str) -> Found:
    """The recorder files in ``directory``; ``OSError`` if it is missing, unreadable or a symlink."""
    if os.path.islink(directory):
        raise NotADirectoryError(errno.ENOTDIR, "the trace directory is a symlink", directory)
    named: List[str] = []
    status: Optional[str] = None
    with os.scandir(directory) as entries:
        for entry in entries:
            if not entry.is_file(follow_symlinks=False):  # not a symlink, FIFO or directory
                continue
            if _DATA_NAME.fullmatch(entry.name) and _real_date(entry.name):
                named.append(entry.path)
            elif entry.name == STATUS_FILE:
                status = entry.path
    data = sorted(path for path in named if starts_like_a_line(path))
    ours = status is not None and isinstance(read_status(directory).get("counts"), dict)
    return Found(data, status if ours or data else None)


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


def _stamp(head: bytes) -> Optional[str]:
    """The ``ts`` of a line, read from its first bytes, only if the line starts like an envelope."""
    found = _STAMPED.match(head)
    return found.group(1).decode("ascii") if found else None


def _scan(path: str) -> Tuple[int, Optional[str], Optional[str]]:
    """The non-blank lines of a data file, and the first and last timestamps in it."""
    try:
        descriptor = os.open(path, os.O_RDONLY | _NOFOLLOW | _NONBLOCK)
    except OSError:
        return 0, None, None
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):  # swapped since the listing
        os.close(descriptor)
        return 0, None, None
    first: Optional[str] = None
    last: Optional[str] = None
    count = 0
    head = b""
    at_start = True
    with os.fdopen(descriptor, "rb") as handle:
        while True:
            chunk = handle.readline(_CHUNK)  # never more than _CHUNK bytes, however long the line
            if not chunk:
                break
            if at_start:
                head = b""
                count += bool(chunk.strip())
            head += chunk[: _HEAD - len(head)]
            if chunk.endswith(b"\n"):
                stamp = _stamp(head)
                first = first or stamp
                last = stamp or last
            at_start = chunk.endswith(b"\n")
        if not at_start:  # the last line has no newline
            stamp = _stamp(head)
            first = first or stamp
            last = stamp or last
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
        count, earliest, latest = _scan(path)
        lines += count
        first = first or earliest
        last = latest or last
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


def _unlink(path: str) -> Union[None, bool, str]:
    """Delete ``path`` (never through a symlink): True; None if already gone; else the reason."""
    try:
        os.unlink(path)
    except FileNotFoundError:  # removed by someone else since we looked: gone is gone
        return None
    except OSError as exc:
        return exc.strerror or type(exc).__name__
    return True


def purge(directory: str, *, dry_run: bool = False) -> Purged:
    """Delete the recorder's files in ``directory`` and report which went and which could not.

    With ``dry_run`` nothing is deleted and ``deleted`` lists what would be. Raises
    ``NotRecorderDirectoryError`` when the directory holds no recorder file (nothing is deleted
    then) and ``OSError`` when it is missing, unreadable or a symlink.
    """
    found = find(directory)
    paths = found.data + ([found.status] if found.status is not None else [])
    if not paths:
        raise NotRecorderDirectoryError(f"no recorder files in {directory}")
    if dry_run:
        return Purged([os.path.basename(path) for path in paths], [])
    outcomes = {path: _unlink(path) for path in paths}
    deleted = [os.path.basename(path) for path, outcome in outcomes.items() if outcome is True]
    failed = [
        (os.path.basename(path), outcome)
        for path, outcome in outcomes.items()
        if isinstance(outcome, str)
    ]
    return Purged(deleted, failed)
