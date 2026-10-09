"""A file or a directory as a body that is sized up front and never held in memory.

``requests`` sends a body that has a ``__len__`` with a ``Content-Length`` (not chunked), so the
platform can refuse an oversized checkpoint before it reads a byte. The size is exact because
the tar is laid out here: one header per file, the file's bytes padded to a 512-byte block,
and the end-of-archive marker. Files are read in 1 MiB slices as the body is sent.

Nothing is padded or cut to keep a promise: the platform stores whatever it receives as a
checkpoint a run may later resume from, so a file that is not exactly as planned when it is
read (shorter, longer, rewritten, swapped for a link or a pipe) aborts the stream with an
``OSError`` and the push is abandoned.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import os
from pathlib import Path
import stat
import tarfile
from typing import BinaryIO

_BLOCK = tarfile.BLOCKSIZE
_CHUNK = 1024 * 1024
_END_OF_ARCHIVE = bytes(2 * _BLOCK)
# Never follow a link that replaced the file since it was planned, and never block opening a pipe
# (the type is checked on the open descriptor). The flags that do not exist on Windows are 0.
_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
    | getattr(os, "O_BINARY", 0)
)


class FileChangedError(OSError):
    """A file was not as planned when it was read, so what would be sent is not a checkpoint."""


@dataclass(frozen=True)
class _Entry:
    arcname: str
    path: str
    size: int
    mtime: int


def _header(entry: _Entry) -> bytes:
    info = tarfile.TarInfo(entry.arcname)
    info.size = entry.size
    info.mtime = entry.mtime
    info.mode = 0o644
    return info.tobuf(tarfile.PAX_FORMAT, "utf-8", "surrogateescape")


def _padding(size: int) -> int:
    return -size % _BLOCK


def _scan(directory: str, prefix: str, found: list[_Entry]) -> int:
    """Collect the regular files under ``directory``; return how many entries were skipped.

    Names are built from the entries of each directory, so they are relative by construction
    (no ``..``, no leading ``/``). A symlink is skipped whatever it points at: it would otherwise
    carry a file from outside the directory into the archive.
    """
    skipped = 0
    with os.scandir(directory) as scan:
        children = sorted(scan, key=lambda child: child.name)
    for child in children:
        arcname = prefix + child.name
        if child.is_symlink():
            skipped += 1
        elif child.is_dir(follow_symlinks=False):
            skipped += _scan(child.path, arcname + "/", found)
        elif child.is_file(follow_symlinks=False):
            status = child.stat(follow_symlinks=False)
            found.append(_Entry(arcname, child.path, status.st_size, int(status.st_mtime)))
        else:
            skipped += 1
    return skipped


def _open_regular(path: str, size: int) -> BinaryIO:
    """Open ``path`` if it is still a regular file of ``size`` bytes (judged on the open descriptor)."""
    descriptor = os.open(path, _OPEN_FLAGS)
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise OSError("not a regular file")
        if status.st_size != size:
            raise FileChangedError(f"a file changed size from {size} to {status.st_size} bytes")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _exact(path: str, size: int) -> Iterator[bytes]:
    """Exactly ``size`` bytes of the file, or ``FileChangedError``: never padded, never cut."""
    with _open_regular(path, size) as source:
        opened = os.fstat(source.fileno()).st_mtime_ns
        remaining = size
        while remaining:
            chunk = source.read(min(_CHUNK, remaining))
            if not chunk:
                raise FileChangedError("a file shrank while it was being sent")
            remaining -= len(chunk)
            yield chunk
        if source.read(1):
            raise FileChangedError("a file grew while it was being sent")
        if os.fstat(source.fileno()).st_mtime_ns != opened:
            raise FileChangedError("a file was changed while it was being sent")


def _content(entry: _Entry) -> Iterator[bytes]:
    """The file's bytes, then the padding to a block."""
    yield from _exact(entry.path, entry.size)
    yield bytes(_padding(entry.size))


class FileStream:
    """One regular file as a sized body: ``len()`` is its size, iterating yields exactly that many bytes.

    The path is not followed if it is a link and must be a regular file (a pipe or a device could
    block or never end). ``failure`` holds the ``OSError`` that aborted the last pass, since an
    HTTP client wraps what a body raises.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = os.fspath(path)
        status = os.lstat(self._path)
        if not stat.S_ISREG(status.st_mode):
            raise OSError("not a regular file")
        self._size = status.st_size
        self.failure: OSError | None = None

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[bytes]:
        self.failure = None
        try:
            yield from _exact(self._path, self._size)
        except OSError as exc:
            self.failure = exc
            raise


class DirectoryTar:
    """The regular files under ``root`` as an uncompressed tar.

    ``len()`` is the exact byte length and iterating yields the bytes, so the object can be
    handed to ``requests`` as a body. It may be iterated more than once. ``skipped`` counts the
    symlinks and special files left out and ``files`` the files in it. ``failure`` holds the
    ``OSError`` that aborted the last pass, since an HTTP client wraps what a body raises.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = os.fspath(root)
        self._entries: list[_Entry] = []
        self.skipped = _scan(self._root, "", self._entries)
        self.files = len(self._entries)
        self.failure: OSError | None = None
        self._length = len(_END_OF_ARCHIVE) + sum(
            len(_header(entry)) + entry.size + _padding(entry.size) for entry in self._entries
        )

    def _names(self) -> list[str]:
        found: list[_Entry] = []
        _scan(self._root, "", found)
        return [entry.arcname for entry in found]

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[bytes]:
        self.failure = None
        try:
            for entry in self._entries:
                yield _header(entry)
                yield from _content(entry)
            # A shard created after the scan (an async save still going) would be missing from
            # the tar, and a checkpoint with a missing shard must not become the resume point.
            if self._names() != [entry.arcname for entry in self._entries]:
                raise FileChangedError("the set of files changed while it was being sent")
        except OSError as exc:
            self.failure = exc
            raise
        yield _END_OF_ARCHIVE
