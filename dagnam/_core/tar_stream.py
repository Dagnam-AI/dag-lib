"""A directory as an uncompressed tar stream: sized up front, never held in memory.

``requests`` sends a body that has a ``__len__`` with a ``Content-Length`` (not chunked), so the
platform can refuse an oversized checkpoint before it reads a byte. The size is exact because
the tar is laid out here: one header per file, the file's bytes padded to a 512-byte block,
and the end-of-archive marker. Files are read in 1 MiB slices as the body is sent.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import os
from pathlib import Path
import tarfile

_BLOCK = tarfile.BLOCKSIZE
_CHUNK = 1024 * 1024
_END_OF_ARCHIVE = bytes(2 * _BLOCK)


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


def _content(entry: _Entry) -> Iterator[bytes]:
    """The file's bytes, exactly ``entry.size`` of them, then the padding to a block.

    The header already promised the size, so a file that shrank since planning is padded with
    zeros and one that grew is cut. A file that vanished raises ``OSError``.
    """
    remaining = entry.size
    with Path(entry.path).open("rb") as source:
        while remaining:
            wanted = min(_CHUNK, remaining)
            chunk = source.read(wanted) or bytes(wanted)
            remaining -= len(chunk)
            yield chunk
    yield bytes(_padding(entry.size))


class DirectoryTar:
    """The regular files under ``root`` as an uncompressed tar.

    ``len()`` is the exact byte length and iterating yields the bytes, so the object can be
    handed to ``requests`` as a body. It may be iterated more than once. ``skipped`` counts the
    symlinks and special files left out.
    """

    def __init__(self, root: str | Path) -> None:
        self._entries: list[_Entry] = []
        self.skipped = _scan(os.fspath(root), "", self._entries)
        self._length = len(_END_OF_ARCHIVE) + sum(
            len(_header(entry)) + entry.size + _padding(entry.size) for entry in self._entries
        )

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[bytes]:
        for entry in self._entries:
            yield _header(entry)
            yield from _content(entry)
        yield _END_OF_ARCHIVE
