"""The on-disk workload layout: ``workloads/<id>/{dataset.jsonl, split.json, meta.json}``.

Every file is written through :func:`write_atomic`: to a temp file in the same
folder that this process alone created, then promoted with ``os.replace``, so a
reader never sees a partial file and a link planted beforehand is never written
through. ``split.json`` is the exact body ``create_explicit_splits`` sends
(``explicit_splits_body``), so the upload step sends it verbatim. Every file read
back from an audit directory goes through :func:`read_regular`, which opens a
regular file or refuses.

The folder is opened once, without following a link (``O_NOFOLLOW | O_DIRECTORY``),
and the temp file is created, renamed over the file and unlinked relative to that
handle (``dir_fd``), so a folder swapped for a link after the check cannot redirect
a write or a removal. The package is OS independent, but those flags and ``dir_fd``
exist only on POSIX. On Windows the same checks run on the path (``is_symlink`` and
``is_junction``) just before each use, the temp file is still created exclusively
under a random name (a planted link cannot match it), and a swap between a check and
its use is narrowed, not closed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import errno
import fnmatch
import json
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Any, TextIO

from dagnam._core.client.datasets import explicit_splits_body
from dagnam._core.exceptions import DagnamError

SCHEMA = "dagnam.audit.workload/1"
WORKLOADS = "workloads"
_PLAIN_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")
_WRITTEN = ("dataset.jsonl", "split.json", "meta.json")
"""The files :func:`write_workload` writes; a folder holding only these (and replays) is a scan's."""
_LEFTOVERS = (
    *(f".{name}.*.tmp" for name in _WRITTEN),  # a write a crash cut short
    *(f"{name}.tmp" for name in _WRITTEN),  # the fixed name an older version used
    "replay-*.jsonl",
)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_TEMP_TRIES = 8
_BY_FD = (
    os.unlink in os.supports_dir_fd
    and os.rename in os.supports_dir_fd  # ``os.replace`` takes the same ``src_dir_fd``
    and _NOFOLLOW != 0
    and _DIRECTORY != 0
)
"""Whether a folder can be opened once and worked in through that handle."""


class UnsafeWorkloadsError(DagnamError):
    """A path an audit writes or removes is a symbolic link, or leads outside the audit directory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(
            f"{path} is a symbolic link or leads outside the audit directory: `dagnam audit`"
            " never writes or removes anything through one inside an audit directory. Remove"
            " the link, or use another audit directory."
        )


class NotRegularFileError(UnsafeWorkloadsError):
    """A path an audit reads or writes holds a directory, a pipe or a device, not a file."""

    def __init__(self, path: Path) -> None:
        DagnamError.__init__(
            self,
            f"{path} is not a regular file (it is a directory, a pipe or a device):"
            " `dagnam audit` reads and writes only regular files in an audit directory."
            " Remove it, or use another audit directory.",
        )
        self.path = path


def _is_link(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def refuse_links(path: Path) -> None:
    """:class:`UnsafeWorkloadsError` naming ``path``, or its folder, when it is a link."""
    if _is_link(path):
        raise UnsafeWorkloadsError(path)
    if _is_link(path.parent):
        raise UnsafeWorkloadsError(path.parent)


def check_writable(path: Path) -> None:
    """Refuse a link at ``path`` or as its folder, and anything there that is not a regular file."""
    refuse_links(path)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISREG(mode):
        raise NotRegularFileError(path)


def _open_folder(folder: Path) -> int:
    """The folder opened for use as ``dir_fd``; a link, or anything that is not a folder, is refused."""
    try:
        return os.open(folder, os.O_RDONLY | _DIRECTORY | _NOFOLLOW)
    except FileNotFoundError:
        raise
    except OSError:  # swapped for a link (or for a file) since any earlier check
        raise UnsafeWorkloadsError(folder) from None


def _regular_or_refuse(fd: int, path: Path) -> None:
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise NotRegularFileError(path)


def read_regular(path: Path) -> str:
    """The text of ``path``, which must be a regular file: anything else is refused, never read.

    A pipe would block a read for ever (holding the audit directory's lock) and a
    link could lead anywhere, so the file is opened without blocking and without
    following a link, its kind is read from the open descriptor (a check by path
    would be a race), and the text comes from that descriptor. A missing file is
    the ``FileNotFoundError`` a plain read gives.
    """
    try:
        fd = os.open(path, os.O_RDONLY | _NONBLOCK | _NOFOLLOW)
    except FileNotFoundError:
        raise
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise UnsafeWorkloadsError(path) from None
        raise
    _regular_or_refuse(fd, path)
    with os.fdopen(fd, "r", encoding="utf-8") as handle:
        return handle.read()


def write_atomic(path: Path, text: str, *, mode: int = 0o666) -> None:
    """Write ``text`` to ``path`` through a temp file nobody else could have placed.

    The temp file is created beside ``path`` under a random name,
    ``O_CREAT | O_EXCL | O_NOFOLLOW`` and relative to the folder's own handle,
    synced, then renamed over ``path`` and removed again if anything fails.
    ``mode`` is the creation mode before the umask. A link at ``path`` or as its
    folder, and anything at ``path`` that is not a regular file, is refused
    (:class:`UnsafeWorkloadsError`).
    """
    folder: int | None = _open_folder(path.parent) if _BY_FD else None
    try:
        if folder is None:
            check_writable(path)
        else:
            _check_final(path, folder)
        name, fd = _create_temp(path, mode, folder)
        try:
            _fill(fd, text)
            if folder is None:
                os.replace(path.with_name(name), path)
            else:
                os.replace(name, path.name, src_dir_fd=folder, dst_dir_fd=folder)
        except BaseException:
            _unlink_temp(path, name, folder)
            raise
    finally:
        if folder is not None:
            os.close(folder)


def _check_final(path: Path, folder: int) -> None:
    try:
        kind = os.lstat(path.name, dir_fd=folder).st_mode
    except FileNotFoundError:
        return
    if stat.S_ISLNK(kind):
        raise UnsafeWorkloadsError(path)
    if not stat.S_ISREG(kind):
        raise NotRegularFileError(path)


def _create_temp(path: Path, mode: int, folder: int | None) -> tuple[str, int]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW
    for _ in range(_TEMP_TRIES):
        name = f".{path.name}.{secrets.token_hex(8)}.tmp"
        try:
            return name, os.open(
                name if folder is not None else path.with_name(name), flags, mode, dir_fd=folder
            )
        except FileExistsError:
            continue
    raise FileExistsError(f"no free temporary name beside {path}")


def _fill(fd: int, text: str) -> None:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _unlink_temp(path: Path, name: str, folder: int | None) -> None:
    try:
        os.unlink(name if folder is not None else path.with_name(name), dir_fd=folder)
    except FileNotFoundError:
        pass


def open_append(path: Path) -> TextIO:
    """``path`` opened for appending text, never through a link, and only if it is a regular file."""
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | _NOFOLLOW | _NONBLOCK
    folder: int | None = _open_folder(path.parent) if _BY_FD else None
    try:
        if folder is None:
            check_writable(path)
        fd = os.open(path.name if folder is not None else path, flags, 0o666, dir_fd=folder)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise UnsafeWorkloadsError(path) from None
        if error.errno in (errno.ENXIO, errno.EISDIR):  # a pipe nobody reads; a directory
            raise NotRegularFileError(path) from None
        raise
    finally:
        if folder is not None:
            os.close(folder)
    _regular_or_refuse(fd, path)
    return os.fdopen(fd, "a", encoding="utf-8")


def is_plain_name(name: str) -> bool:
    """Whether ``name`` is one plain folder name: no separator, no ``..``, nothing hidden."""
    return _PLAIN_NAME.fullmatch(name) is not None


def workloads_dir(out_dir: Path) -> Path:
    """``out_dir/workloads``, refused (:class:`UnsafeWorkloadsError`) when it is a link or leads out."""
    return _checked(out_dir, out_dir / WORKLOADS)


def workload_dir(out_dir: Path, workload_id: str) -> Path:
    """``out_dir/workloads/<workload_id>``, for an id that is one plain name and not a link.

    A ``ValueError`` for an id with a separator, ``..`` or nothing in it: it
    would name a folder outside ``workloads/``.
    """
    if not is_plain_name(workload_id):
        raise ValueError(f"workload id {workload_id!r} is not a plain name")
    return _checked(out_dir, workloads_dir(out_dir) / workload_id)


def _checked(out_dir: Path, path: Path) -> Path:
    if _is_link(path) or not path.resolve().is_relative_to(out_dir.resolve()):
        raise UnsafeWorkloadsError(path)
    return path


def check_workload_for_write(out_dir: Path, workload_id: str) -> Path:
    """:func:`workload_dir`, and none of the files a scan writes into it is a link either."""
    folder = workload_dir(out_dir, workload_id)
    for name in _WRITTEN:
        check_writable(folder / name)
    return folder


def _is_scan_folder(folder: Path, workload_id: str) -> bool:
    """Whether the folder's ``meta.json`` is the one :func:`write_workload` writes for this id."""
    try:
        meta = json.loads(read_regular(folder / "meta.json"))
    except (OSError, ValueError):
        return False
    return (
        isinstance(meta, dict)
        and meta.get("schema") == SCHEMA
        and meta.get("workload_id") == workload_id
    )


def _scan_files(names: Sequence[str]) -> list[str]:
    """The names a scan wrote (``meta.json`` last: it is the proof the folder is a scan's)."""
    mine = [
        n for n in names if any(fnmatch.fnmatchcase(n, pat) for pat in (*_WRITTEN, *_LEFTOVERS))
    ]
    return sorted(mine, key=lambda n: n == "meta.json")


def remove_workload(out_dir: Path, workload_id: str) -> bool:
    """Remove the files a scan wrote into one workload folder, then the folder; whether it is gone.

    A folder is a scan's only when its ``meta.json`` says so (this schema, this
    id): any other folder, even one an edited report lists, is left whole and
    ``False`` is returned. Of a scan's folder only the files
    :func:`write_workload` writes, their temp files and a run's
    ``replay-*.jsonl`` answers are removed, never a link's target and never
    anything else; a folder that holds more stays (``False``) for its owner, a
    directory named like one of the scan's files included (it is not unlinked).
    The folder is opened once, without following a link, and its files are
    unlinked through that handle, so a swap after the check cannot redirect the
    removal. Where that is not possible (Windows) the removal is by path.
    """
    folder = workload_dir(out_dir, workload_id)
    if not folder.exists():
        return True
    if not _is_scan_folder(folder, workload_id):
        return False
    if _BY_FD:
        fd = _open_folder(folder)
        try:
            for name in _scan_files(os.listdir(fd)):
                if not stat.S_ISDIR(os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode):
                    os.unlink(name, dir_fd=fd)
            left = os.listdir(fd)
        finally:
            os.close(fd)
    else:
        if _is_link(folder):
            raise UnsafeWorkloadsError(folder)
        for name in _scan_files([p.name for p in folder.iterdir()]):
            if _is_link(folder / name) or not (folder / name).is_dir():
                (folder / name).unlink()
        left = [p.name for p in folder.iterdir()]
    if left:
        return False
    folder.rmdir()
    return True


def workload_names(out_dir: Path) -> set[str]:
    """Every name under ``workloads/``, whoever made it (empty when there is no such folder)."""
    root = workloads_dir(out_dir)
    return {p.name for p in root.iterdir()} if root.exists() else set()


def write_workload(
    out_dir: Path,
    workload_id: str,
    rows: Sequence[Mapping[str, Any]],
    split: Mapping[str, Sequence[int]],
    stats: Mapping[str, Any],
) -> Path:
    """Write one workload's files under ``out_dir/workloads/<workload_id>`` and return that directory.

    ``stats`` is recorded verbatim under ``meta.json``'s ``stats`` key
    (``build_dataset`` produces the expected shape). The workload id is a
    plain string so this composes with discovery's ``Workload`` without importing it.
    """
    members = {name: list(indices) for name, indices in split.items()}
    out_of_range = sorted(
        {i for indices in members.values() for i in indices if not 0 <= i < len(rows)}
    )
    if out_of_range:
        raise ValueError(f"split names row indices outside 0..{len(rows) - 1}: {out_of_range[:5]}")

    folder = workload_dir(out_dir, workload_id)
    folder.mkdir(parents=True, exist_ok=True)
    write_atomic(
        folder / "dataset.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )
    write_atomic(folder / "split.json", json.dumps(explicit_splits_body(members), indent=2))
    meta = {
        "schema": SCHEMA,
        "workload_id": workload_id,
        "rows": len(rows),
        "splits": {name: len(indices) for name, indices in members.items()},
        "stats": dict(stats),
    }
    write_atomic(folder / "meta.json", json.dumps(meta, indent=2, ensure_ascii=False))
    return folder


__all__ = [
    "SCHEMA",
    "WORKLOADS",
    "NotRegularFileError",
    "UnsafeWorkloadsError",
    "check_workload_for_write",
    "check_writable",
    "is_plain_name",
    "open_append",
    "read_regular",
    "refuse_links",
    "remove_workload",
    "workload_dir",
    "workload_names",
    "workloads_dir",
    "write_atomic",
    "write_workload",
]
