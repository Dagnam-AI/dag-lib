"""The writer: a bounded queue, one daemon thread and a daily, private JSON-lines file.

``submit`` never blocks and never raises: a line that cannot be kept is counted and dropped, so
the live call it describes is never touched. Counts are saved to ``.status.json`` in the
directory so that ``python -m dagnam_record status`` can read them without the process.
"""

from __future__ import annotations

import atexit
from datetime import datetime, timezone
import errno
import json
import logging
import os
import queue
import threading
import time
from typing import Any, Callable, Dict, Optional

LOGGER = logging.getLogger("dagnam_record")
STATUS_FILE = ".status.json"
COUNT_KEYS = ("written", "dropped_full", "dropped_large", "dropped_error", "dropped_expired")
_SAVE_EVERY = 30.0
_DAY = 86400


def _ensure_dir(directory: str) -> None:
    """Create the directory ``0700`` (only a directory this call makes is changed)."""
    os.makedirs(os.path.dirname(directory) or ".", exist_ok=True)
    try:
        os.mkdir(directory, 0o700)
    except FileExistsError:
        pass


def _read_status(directory: str) -> Dict[str, Any]:
    try:
        with open(os.path.join(directory, STATUS_FILE), encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


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
        self._clock = clock
        self._queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=max_queue)
        self._lock = threading.Lock()
        previous = _read_status(self._dir)
        saved = previous.get("counts")
        saved = saved if isinstance(saved, dict) else {}
        # ponytail: counts carry over one process at a time; several processes sharing a
        # directory overwrite each other's file. Add per-process files if that matters.
        self.counts: Dict[str, int] = {key: _count(saved.get(key)) for key in COUNT_KEYS}
        started = previous.get("started")
        self._dirty = True  # a first run has a start time to save
        self._started = clock()
        if isinstance(started, (int, float)) and not isinstance(started, bool):
            self._started, self._dirty = float(started), False
        self.expired = False
        self._saved_at = clock()
        self._fd: Optional[int] = None
        self._day = ""
        self._thread: Optional[threading.Thread] = None
        atexit.register(self.close)
        if start:
            self._thread = threading.Thread(target=self._run, name="dagnam-record", daemon=True)
            self._thread.start()

    def submit(self, line: str) -> None:
        """Queue one line. Never blocks, never raises: a full queue drops and counts it."""
        try:
            self._queue.put_nowait(line)
        except queue.Full:
            self._bump("dropped_full")

    def close(self, timeout: float = 2.0) -> None:
        """Write what is queued (waiting up to ``timeout``), save the counts and stop."""
        atexit.unregister(self.close)
        thread = self._thread
        if thread is not None:
            try:
                self._queue.put(None, timeout=timeout)
            except queue.Full:
                pass
            thread.join(timeout)
        if thread is None or not thread.is_alive():
            self._drain_for_test()
            self._drop_fd()
        self._persist()

    def _drain_for_test(self) -> None:
        """Run the loop body for everything queued, in the calling thread."""
        while True:
            try:
                line = self._queue.get_nowait()
            except queue.Empty:
                return
            if line is not None:
                self._write(line)
                self._persist_if_due()

    def _run(self) -> None:
        while True:
            try:
                line = self._queue.get(timeout=_SAVE_EVERY)
            except queue.Empty:
                self._persist_if_due()
                continue
            if line is None:
                return
            self._write(line)
            self._persist_if_due()

    def _bump(self, key: str) -> None:
        with self._lock:
            self.counts[key] += 1
            self._dirty = True

    def _is_expired(self, now: float) -> bool:
        if not self.expired and now - self._started >= self._window:
            self.expired = True
            self._dirty = True
            LOGGER.info("dagnam_record: the recording window has ended; no more calls are recorded")
        return self.expired

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
        if self._fd is None or day != self._day:
            self._drop_fd()
            _ensure_dir(self._dir)
            path = os.path.join(self._dir, day + ".jsonl")
            self._fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
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
        if not self._dirty:
            return
        self._saved_at = self._clock()
        with self._lock:
            self._dirty = False
            payload: Dict[str, Any] = {
                "directory": self._dir,
                "started": self._started,
                "updated": self._saved_at,
                "expired": self.expired,
                "counts": dict(self.counts),
            }
        path = os.path.join(self._dir, STATUS_FILE)
        temp = f"{path}.{os.getpid()}.tmp"
        try:
            _ensure_dir(self._dir)
            descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(temp, path)
        except OSError:
            self._dirty = True
