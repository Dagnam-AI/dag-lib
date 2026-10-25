"""The writer: a bounded queue, one daemon thread and a daily, private JSON-lines file.

``submit`` never blocks and never raises: a line that cannot be kept is counted and dropped, so
the live call it describes is never touched. Counts are saved to ``.status.json`` in the
directory so that ``python -m dagnam_record status`` can read them without the process.

``.status.json`` holds ``directory``, ``started`` (the first run, which the recording window is
counted from), ``updated``, ``expired``, ``sdk_versions`` (the version of each SDK that was
patched, see ``_compat``) and ``counts``: one number per key of ``COUNT_KEYS``.
``dropped_error`` also counts a line submitted after ``close()``. ``dropped_partial`` counts a
call the recorder saw only part of (a stream abandoned or cut off before its end, a call
cancelled or interrupted): it is never written, because a partial answer is no training row.

Permissions: the directory is created ``0700`` (its parents get the process defaults). An
existing directory must be a real directory (not a symlink) owned by this user, and is tightened
to ``0700`` if it is looser; it is opened with ``O_DIRECTORY|O_NOFOLLOW`` and checked and changed
through that descriptor, so it cannot be swapped between the check and the change. Files are
opened with ``O_NOFOLLOW|O_NONBLOCK``, must be regular files (a FIFO is refused, never waited
on) and are set to ``0600``. Where ``O_DIRECTORY`` does not exist (Windows) a directory is only
required to be a real one.
"""

from __future__ import annotations

import atexit
from datetime import datetime, timezone
import errno
import json
import logging
import os
import queue
import stat
import threading
import time
from typing import Any, Callable, Dict, Optional
import weakref

from dagnam_record._stream import deliver_pending

LOGGER = logging.getLogger("dagnam_record")
STATUS_FILE = ".status.json"
STATUS_MAX = 1_000_000
"""The most of a ``.status.json`` that is read: a larger one is treated as unreadable."""
COUNT_KEYS = (
    "written",
    "dropped_full",
    "dropped_large",
    "dropped_error",
    "dropped_expired",
    "dropped_partial",
)
_SAVE_EVERY = 30.0
_DAY = 86400
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_fchmod: Callable[[int, int], None] = getattr(os, "fchmod", lambda descriptor, mode: None)
_OWNER = os.getuid() if hasattr(os, "getuid") else None
_WRITERS: weakref.WeakSet[Writer] = weakref.WeakSet()


def _ensure_dir(directory: str) -> None:
    """Make sure ``directory`` is a private directory of ours, or raise ``OSError``."""
    os.makedirs(os.path.dirname(directory) or ".", exist_ok=True)
    try:
        os.mkdir(directory, 0o700)
        return
    except FileExistsError:
        pass
    if not _DIRECTORY:  # no way to open a directory: all that can be asked is that it is one
        if os.path.islink(directory) or not os.path.isdir(directory):
            raise NotADirectoryError(errno.ENOTDIR, "not a real directory", directory)
        return
    # A symlink or a file fails the open; the owner and mode are read from the open descriptor,
    # and changed through it, so the path cannot be swapped for another in between.
    descriptor = os.open(directory, os.O_RDONLY | _DIRECTORY | _NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if _OWNER is not None and info.st_uid != _OWNER:
            raise PermissionError(errno.EPERM, "the trace directory belongs to someone else")
        if info.st_mode & 0o077:
            _fchmod(descriptor, 0o700)
    finally:
        os.close(descriptor)
    if not os.access(directory, os.W_OK | os.X_OK):
        raise PermissionError(errno.EACCES, "the trace directory is not writable", directory)


def read_status(directory: str) -> Dict[str, Any]:
    """The saved status, or ``{}`` for anything but a readable regular file holding an object.

    The open cannot wait (a FIFO would never deliver a writer) and does not follow a symlink, and
    a file of more than ``STATUS_MAX`` characters counts as unreadable.
    """
    try:
        descriptor = os.open(
            os.path.join(directory, STATUS_FILE), os.O_RDONLY | _NOFOLLOW | _NONBLOCK
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
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def reinit_all_after_fork() -> None:
    """Reset every live writer in a forked child (registered with ``os.register_at_fork``)."""
    for writer in list(_WRITERS):
        writer.reinit_after_fork()


class Writer:
    """Append envelope lines to ``<directory>/<UTC date>.jsonl`` from one background thread."""

    def __init__(
        self,
        directory: str,
        *,
        window_days: int = 14,
        max_queue: int = 1000,
        max_line_bytes: int = 1_000_000,
        clock: Callable[[], float] = time.time,
        start: bool = True,
    ) -> None:
        """``start=False`` leaves the thread unstarted so a test can drive the loop itself."""
        self._dir = os.path.abspath(os.path.expanduser(directory))
        self._window = window_days * _DAY
        self._max_line_bytes = max_line_bytes
        self._max_queue = max_queue
        self._clock = clock
        self._queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=max_queue)
        self._lock = threading.Lock()
        self._gate = threading.Lock()
        try:
            _ensure_dir(self._dir)
            previous = read_status(self._dir)
            refused = False
        except OSError:
            previous, refused = {}, True
        saved = previous.get("counts")
        saved = saved if isinstance(saved, dict) else {}
        # ponytail: counts carry over one process at a time; several processes sharing a
        # directory overwrite each other's file. Add per-process files if that matters.
        self.counts: Dict[str, int] = {key: as_count(saved.get(key)) for key in COUNT_KEYS}
        self.counts["dropped_error"] += refused  # the directory is not one we may write to
        self._loaded = dict(self.counts)  # what was already on record when this writer began
        self._successor: Optional[Writer] = None  # takes lines that arrive after close()
        self._moved: Optional[Writer] = None  # takes counts that arrive after hand_over()
        self._retired = False  # handed over and saved for good: the successor owns the file now
        self.usable = not refused  # False: the directory was refused, every line will be dropped
        self.sdk_versions: Dict[str, str] = {}
        started = previous.get("started")
        self._dirty = True  # a first run has a start time to save
        self._started = clock()
        if isinstance(started, (int, float)) and not isinstance(started, bool):
            self._started, self._dirty = float(started), False
        if self._started > clock():  # a clock set back, or a start time from the future
            self._started, self._dirty = clock(), True
        self.expired = False
        self._is_expired(clock())  # a window that ended before this process began says so now
        self._saved_at = clock()
        self._fd: Optional[int] = None
        self._day = ""
        self._closed = False
        self._thread: Optional[threading.Thread] = None
        _WRITERS.add(self)
        atexit.register(self.close)
        if start:
            self._spawn()

    def submit(self, line: str) -> None:
        """Queue one line. Never blocks, never raises: what cannot be kept is counted.

        A writer that was handed over passes the line on to its successor, however long the
        chain of successors has grown (a loop, not recursion).
        """
        try:
            writer: Optional[Writer] = self
            while writer is not None:
                writer = writer._offer(line)
        except Exception:  # whatever went wrong, the caller is never told
            self._bump("dropped_error")

    def _offer(self, line: str) -> Optional[Writer]:
        """Queue ``line`` or count why not; the writer to try instead, if this one was handed over."""
        try:
            with self._gate:  # close() takes it to stop intake, so no line lands after the stop
                if self._closed:
                    successor = self._successor
                    if successor is not None:
                        return self._flatten_successor(successor)
                    key = "dropped_error"
                elif len(line) > self._max_line_bytes:  # characters never outnumber bytes
                    key = "dropped_large"
                else:
                    self._queue.put_nowait(line)
                    return None
        except queue.Full:
            key = "dropped_full"
        except Exception:  # not a string, or anything else
            key = "dropped_error"
        self._bump(key)
        return None

    def _flatten_successor(self, target: Writer) -> Writer:
        while target._successor is not None:
            target = target._successor
        self._successor = target  # the next line takes one step, not the whole chain
        return target

    def _flatten_moved(self, target: Writer) -> Writer:
        while target._moved is not None:
            target = target._moved
        self._moved = target
        return target

    def set_sdk_versions(self, versions: Dict[str, str]) -> None:
        """Note the version of each patched SDK; it is saved with the counts."""
        if versions != self.sdk_versions:
            self.sdk_versions = dict(versions)
            self._dirty = True

    def count(self, key: str) -> None:
        """Count a line that was dropped before it reached ``submit`` (a key of ``COUNT_KEYS``)."""
        self._bump(key)

    def close(self, timeout: float = 2.0) -> None:
        """Write what is queued (waiting up to ``timeout``), save the counts and stop."""
        self._halt(timeout)
        self._persist()

    def _halt(self, timeout: float) -> None:
        """Shut intake, let the thread finish the queue (up to ``timeout``) and drop the file."""
        deliver_pending()  # streams that were let go: their lines must be in before intake stops
        with self._gate:
            self._closed = True
        atexit.unregister(self.close)
        thread = self._thread
        if thread is not None:
            try:
                self._queue.put_nowait(None)  # wakes an idle thread; a busy one sees _closed
            except queue.Full:
                pass
            thread.join(timeout)
        if thread is None or not thread.is_alive():
            self._drain()
            self._drop_fd()

    def hand_over(self, successor: Writer, timeout: float = 2.0) -> None:
        """Stop in favour of ``successor``, losing no line and no count.

        What is queued is written, then the counts are frozen and saved. A line or a count that
        reaches this writer afterwards (a call that began with the old recorder, or a thread that
        did not finish within ``timeout``) is passed to ``successor``, which counts it. In the
        same directory the successor also takes over the counts and the window start of this
        writer (its own ``window_days`` then decides whether the window has ended).
        """
        self._successor = successor
        self._halt(timeout)
        with self._lock:
            final = dict(self.counts)
            self._moved = successor  # from here on every count goes to the successor
        self._persist()  # the counts are frozen now: this is exactly what they were
        self._retired = True
        if successor._dir == self._dir:
            successor._take_over(final, self._started)

    def _take_over(self, final: Dict[str, int], started: float) -> None:
        with self._lock:
            for key in COUNT_KEYS:
                self.counts[key] += max(final[key] - self._loaded[key], 0)
            self._started = started  # whether the window has ended is judged afresh from it
            self._dirty = True
            self.expired = False
        self._is_expired(self._clock())

    def reinit_after_fork(self) -> None:
        """Start over in a forked child.

        The child has none of the parent's threads, and a lock another thread held at the fork
        would stay locked for good, so the queue, lock, thread and counts are made anew. The
        parent's pending lines stay the parent's (it writes them), and its counts are not the
        child's. The window start is kept.
        """
        self._queue = queue.Queue(maxsize=self._max_queue)
        self._lock = threading.Lock()
        self._gate = threading.Lock()
        self.counts = dict.fromkeys(COUNT_KEYS, 0)
        self._loaded = dict.fromkeys(COUNT_KEYS, 0)
        self._dirty = False
        self._saved_at = self._clock()
        self._drop_fd()  # closes this process's copy only
        restart = self._thread is not None and not self._closed
        self._thread = None
        if restart:
            self._spawn()

    def _spawn(self) -> None:
        self._thread = threading.Thread(
            target=self._run, args=(self._queue,), name="dagnam-record", daemon=True
        )
        self._thread.start()

    def _drain(self) -> None:
        """Run the loop body for everything queued, in the calling thread."""
        deliver_pending()
        while True:
            try:
                line = self._queue.get_nowait()
            except queue.Empty:
                return
            if line is not None:
                self._write(line)
                self._persist_if_due()

    def _run(self, source: queue.Queue[Optional[str]]) -> None:
        while True:
            try:
                # Once closed, finish what is queued and stop, even if the stop marker found no room.
                line = source.get_nowait() if self._closed else source.get(timeout=_SAVE_EVERY)
            except queue.Empty:
                deliver_pending()  # a safe point: this thread holds no writer lock here
                self._persist_if_due()
                if self._closed:
                    return
                continue
            if line is None:
                return
            self._write(line)
            self._persist_if_due()
            deliver_pending()

    def _bump(self, key: str) -> None:
        writer = self
        while True:
            with writer._lock:
                moved = writer._moved
                if moved is None:
                    writer.counts[key] += 1
                    writer._dirty = True
                    return
            writer = writer._flatten_moved(moved)

    def check_expired(self) -> bool:
        """Whether the recording window has ended (the first to see it logs it, once)."""
        return self.expired or self._is_expired(self._clock())

    def _is_expired(self, now: float) -> bool:
        if self.expired:
            return True
        if now - self._started < self._window:
            return False
        # ponytail: two threads that see the end together may both log it; one INFO line twice.
        self.expired = self._dirty = True
        LOGGER.info("dagnam_record: the recording window has ended; no more calls are recorded")
        return True

    def _write(self, line: str) -> None:
        """Append one whole line. Any failure is counted, never raised."""
        try:
            now = self._clock()
            if self._is_expired(now):
                self._drop_fd()
                self._bump("dropped_expired")
                return
            data = line.encode("utf-8") + b"\n"
            if len(data) - 1 > self._max_line_bytes:
                self._bump("dropped_large")
                return
            # One os.write on an O_APPEND descriptor: a reader never sees half a line.
            if os.write(self._descriptor(now), data) != len(data):
                raise OSError(errno.ENOSPC, "short write")
            self._bump("written")
        except Exception:  # fail open: the recorder must never hurt the call it describes
            self._drop_fd()
            self._bump("dropped_error")

    def _descriptor(self, now: float) -> int:
        day = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d")
        if self._fd is not None and os.fstat(self._fd).st_nlink == 0:
            self._drop_fd()  # the file was deleted (a purge) while we held it: start a new one
        if self._fd is None or day != self._day:
            self._drop_fd()
            _ensure_dir(self._dir)
            path = os.path.join(self._dir, day + ".jsonl")
            # O_NONBLOCK: a FIFO here fails (ENXIO) or is refused below, and never stalls the thread.
            flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | _NOFOLLOW | _NONBLOCK
            self._fd = os.open(path, flags, 0o600)
            if not stat.S_ISREG(os.fstat(self._fd).st_mode):
                raise OSError(errno.EINVAL, "the trace file is not a regular file", path)
            _fchmod(self._fd, 0o600)  # a file made earlier, or under another umask, is tightened
            self._day = day
        return self._fd

    def _drop_fd(self) -> None:
        descriptor, self._fd = self._fd, None
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass

    def _persist_if_due(self) -> None:
        if self._dirty and self._clock() - self._saved_at >= _SAVE_EVERY:
            self._persist()

    def _persist(self) -> None:
        """Save the counts by atomic rename; a failure leaves them dirty for the next try."""
        if not self._dirty or self._retired:
            return
        self._saved_at = self._clock()
        with self._lock:
            self._dirty = False
            payload: Dict[str, Any] = {
                "directory": self._dir,
                "started": self._started,
                "updated": self._saved_at,
                "expired": self.expired,
                "sdk_versions": self.sdk_versions,
                "counts": dict(self.counts),
            }
        path = os.path.join(self._dir, STATUS_FILE)
        temp = f"{path}.{os.getpid()}.tmp"
        try:
            _ensure_dir(self._dir)
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _NOFOLLOW
            with os.fdopen(os.open(temp, flags, 0o600), "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(temp, path)
        except OSError:
            self._dirty = True
            try:
                os.unlink(temp)
            except OSError:
                pass


# Several threads cannot be trusted across a fork, so each child rebuilds its writers.
getattr(os, "register_at_fork", lambda **hooks: None)(after_in_child=reinit_all_after_fork)
