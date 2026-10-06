"""The on-disk workload layout: dataset.jsonl, split.json, meta.json, written atomically."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat

import pytest
from tests.audit._fifo import make_fifo, refuse_a_blocking_open_of_a_pipe

from dagnam.audit import workspace
from dagnam.audit.workspace import (
    SCHEMA,
    NotRegularFileError,
    UnsafeWorkloadsError,
    check_workload_for_write,
    check_writable,
    is_plain_name,
    open_append,
    read_regular,
    remove_workload,
    workload_names,
    write_atomic,
    write_workload,
)

ROWS = [{"input": "a", "label": "x"}, {"input": "b", "label": "y"}, {"input": "c", "label": "x"}]
SPLIT = {"train": [0, 1], "eval_holdout": [2]}
STATS = {"format_key": "labeled-example", "boundary_ts": "2026-08-01T00:00:00+00:00"}


def test_workspace_layout_and_meta(tmp_path: Path) -> None:
    out = write_workload(tmp_path, "wl-1", ROWS, SPLIT, STATS)

    assert out == tmp_path / "workloads" / "wl-1"
    assert sorted(p.name for p in out.iterdir()) == ["dataset.jsonl", "meta.json", "split.json"]

    lines = (out / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == ROWS
    assert json.loads((out / "split.json").read_text(encoding="utf-8")) == {
        "member_row_indices": SPLIT
    }
    assert json.loads((out / "meta.json").read_text(encoding="utf-8")) == {
        "schema": SCHEMA,
        "workload_id": "wl-1",
        "rows": 3,
        "splits": {"train": 2, "eval_holdout": 1},
        "stats": STATS,
    }


def test_rewrite_replaces_the_files_and_leaves_no_temp_files(tmp_path: Path) -> None:
    write_workload(tmp_path, "wl-1", ROWS, SPLIT, STATS)
    out = write_workload(tmp_path, "wl-1", ROWS[:1], {"train": [0], "eval_holdout": []}, {})

    assert (out / "dataset.jsonl").read_text(encoding="utf-8") == json.dumps(ROWS[0]) + "\n"
    assert json.loads((out / "meta.json").read_text(encoding="utf-8"))["rows"] == 1
    assert not list(out.glob("*.tmp"))


def test_split_indices_must_address_the_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="row indices"):
        write_workload(tmp_path, "wl-1", ROWS, {"train": [0, 3], "eval_holdout": []}, {})


def test_non_ascii_rows_round_trip(tmp_path: Path) -> None:
    rows = [{"messages": [{"role": "user", "content": "héllo — 日本"}]}]
    out = write_workload(tmp_path, "wl-2", rows, {"train": [0], "eval_holdout": []}, {})
    assert json.loads((out / "dataset.jsonl").read_text(encoding="utf-8")) == rows[0]


def test_remove_workload_removes_what_a_scan_wrote_and_the_folder(tmp_path: Path) -> None:
    out = write_workload(tmp_path, "wl-1", ROWS, SPLIT, STATS)
    (out / "replay-head_tune.jsonl").write_text("{}\n")
    (out / "dataset.jsonl.tmp").write_text("half")

    assert remove_workload(tmp_path, "wl-1") is True
    assert not out.exists()
    assert workload_names(tmp_path) == set()
    assert remove_workload(tmp_path, "wl-1") is True  # already gone


def test_remove_workload_keeps_a_folder_that_holds_anything_else(tmp_path: Path) -> None:
    out = write_workload(tmp_path, "wl-1", ROWS, SPLIT, STATS)
    (out / "sub").mkdir()
    (out / "sub" / "kept.txt").write_text("mine")

    assert remove_workload(tmp_path, "wl-1") is False
    assert [p.name for p in out.iterdir()] == ["sub"]


def test_remove_workload_never_follows_a_link(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "dataset.jsonl").write_text("mine")
    (tmp_path / "workloads").mkdir()
    (tmp_path / "workloads" / "wl-1").symlink_to(target, target_is_directory=True)

    with pytest.raises(UnsafeWorkloadsError, match="symbolic link"):
        remove_workload(tmp_path, "wl-1")
    assert (target / "dataset.jsonl").read_text() == "mine"


def test_a_link_inside_a_workload_folder_is_removed_not_followed(tmp_path: Path) -> None:
    out = write_workload(tmp_path, "wl-1", ROWS, SPLIT, STATS)
    precious = tmp_path / "precious.jsonl"
    precious.write_text("mine")
    (out / "dataset.jsonl").unlink()
    (out / "dataset.jsonl").symlink_to(precious)

    assert remove_workload(tmp_path, "wl-1") is True
    assert precious.read_text() == "mine"


@pytest.fixture(params=[True, False], ids=["by-handle", "by-path"])
def by_fd(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> bool:
    """Both routes: the POSIX one (a folder handle) and the one Windows takes (the path)."""
    monkeypatch.setattr(workspace, "_BY_FD", request.param)
    return bool(request.param)


def test_write_atomic_never_writes_through_a_link_planted_at_any_temp_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    # Even a guessed name is no use: the temp file is created, never opened over.
    monkeypatch.setattr(workspace.secrets, "token_hex", lambda _n: "guessed")
    (tmp_path / ".out.json.guessed.tmp").symlink_to(victim)

    with pytest.raises(FileExistsError, match="no free temporary name"):
        write_atomic(tmp_path / "out.json", "{}")

    assert victim.read_text() == "precious"
    assert not (tmp_path / "out.json").exists()


def test_write_atomic_retries_a_name_that_is_taken_and_leaves_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = iter(["taken", "free"])
    monkeypatch.setattr(workspace.secrets, "token_hex", lambda _n: next(names))
    (tmp_path / ".out.json.taken.tmp").write_text("someone else's")

    write_atomic(tmp_path / "out.json", "{}")

    assert (tmp_path / "out.json").read_text() == "{}"
    assert sorted(p.name for p in tmp_path.iterdir()) == [".out.json.taken.tmp", "out.json"]


def test_write_atomic_removes_its_temp_file_when_it_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "out.json").write_text("old")

    def full(_fd: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", full)
    with pytest.raises(OSError, match="disk full"):
        write_atomic(tmp_path / "out.json", "new")

    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]
    assert (tmp_path / "out.json").read_text() == "old"


def test_write_atomic_honours_the_creation_mode(tmp_path: Path) -> None:
    write_atomic(tmp_path / "secret.json", "{}", mode=0o600)
    assert stat.S_IMODE((tmp_path / "secret.json").stat().st_mode) == 0o600


@pytest.mark.parametrize("what", ["final", "folder"])
def test_write_atomic_and_open_append_refuse_a_link_at_the_file_or_its_folder(
    tmp_path: Path, what: str
) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    real = tmp_path / "real"
    real.mkdir()
    if what == "final":
        target = tmp_path / "out.json"
        target.symlink_to(victim)
    else:
        (tmp_path / "linked").symlink_to(real, target_is_directory=True)
        target = tmp_path / "linked" / "out.json"

    with pytest.raises(UnsafeWorkloadsError, match="symbolic link"):
        write_atomic(target, "x")
    with pytest.raises(UnsafeWorkloadsError, match="symbolic link"):
        open_append(target)

    assert victim.read_text() == "precious"
    assert list(real.iterdir()) == []


@pytest.mark.usefixtures("by_fd")
def test_open_append_appends_and_creates(tmp_path: Path) -> None:
    with open_append(tmp_path / "log.jsonl") as sink:
        sink.write("a\n")
    with open_append(tmp_path / "log.jsonl") as sink:
        sink.write("b\n")
    assert (tmp_path / "log.jsonl").read_text() == "a\nb\n"


def test_a_stale_temp_file_of_a_cut_short_write_is_removed_with_its_folder(tmp_path: Path) -> None:
    out = write_workload(tmp_path, "wl-1", ROWS, SPLIT, STATS)
    (out / ".dataset.jsonl.0123456789abcdef.tmp").write_text("half")

    assert remove_workload(tmp_path, "wl-1") is True
    assert not out.exists()


def test_a_folder_swapped_for_a_link_after_the_check_is_not_followed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The folder was checked, then globbed by path: a swap in between sent the unlinks
    # into whatever the link pointed at.
    out = write_workload(tmp_path, "wl-1", ROWS, SPLIT, STATS)
    target = tmp_path / "elsewhere"
    target.mkdir()
    for name in ("dataset.jsonl", "split.json", "meta.json"):
        (target / name).write_text("precious")
    real_dir = workspace.workload_dir

    def swapping(out_dir: Path, workload_id: str) -> Path:
        folder = real_dir(out_dir, workload_id)
        for child in folder.iterdir():
            child.unlink()
        folder.rmdir()
        folder.symlink_to(target, target_is_directory=True)  # checked clean, then swapped
        return folder

    monkeypatch.setattr(workspace, "workload_dir", swapping)
    assert (out / "meta.json").exists()  # the folder is a scan's when it is checked
    monkeypatch.setattr(workspace, "_is_scan_folder", lambda *_a: True)

    with pytest.raises(UnsafeWorkloadsError):
        remove_workload(tmp_path, "wl-1")

    assert sorted(p.name for p in target.iterdir()) == ["dataset.jsonl", "meta.json", "split.json"]
    assert all((target / n).read_text() == "precious" for n in ("dataset.jsonl", "meta.json"))


def test_removal_by_path_where_the_folder_cannot_be_opened_by_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The Windows route: no O_NOFOLLOW or dir_fd, so a path check just before the unlinks.
    monkeypatch.setattr(workspace, "_BY_FD", False)
    out = write_workload(tmp_path, "wl-1", ROWS, SPLIT, STATS)
    (out / "notes.txt").write_text("mine")

    assert remove_workload(tmp_path, "wl-1") is False
    assert [p.name for p in out.iterdir()] == ["notes.txt"]
    (out / "notes.txt").unlink()
    write_workload(tmp_path, "wl-2", ROWS, SPLIT, STATS)
    assert remove_workload(tmp_path, "wl-2") is True

    target = tmp_path / "elsewhere"
    target.mkdir()
    (tmp_path / "workloads" / "wl-3").symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(workspace, "workload_dir", lambda o, i: o / "workloads" / i)
    monkeypatch.setattr(workspace, "_is_scan_folder", lambda *_a: True)
    with pytest.raises(UnsafeWorkloadsError):
        remove_workload(tmp_path, "wl-3")


def test_a_folder_is_a_scans_only_when_its_meta_says_so(tmp_path: Path) -> None:
    mine = tmp_path / "workloads" / "prod-api"
    mine.mkdir(parents=True)
    for name in ("dataset.jsonl", "split.json", "meta.json", "deploy.yaml"):
        (mine / name).write_text("not a scan's")
    other = tmp_path / "workloads" / "other"
    other.mkdir()
    (other / "dataset.jsonl").write_text("x")
    (other / "meta.json").write_text(json.dumps({"schema": SCHEMA, "workload_id": "somebody-else"}))
    unreadable = tmp_path / "workloads" / "unreadable"
    unreadable.mkdir()
    (unreadable / "meta.json").write_text("{not json")
    scalar = tmp_path / "workloads" / "scalar"
    scalar.mkdir()
    (scalar / "meta.json").write_text("[]")

    for name in ("prod-api", "other", "unreadable", "scalar"):
        assert remove_workload(tmp_path, name) is False
    assert sorted(p.name for p in mine.iterdir()) == [
        "dataset.jsonl",
        "deploy.yaml",
        "meta.json",
        "split.json",
    ]
    assert (other / "dataset.jsonl").read_text() == "x"


def test_plain_names_and_the_write_check(tmp_path: Path) -> None:
    assert is_plain_name("wl-1")
    assert is_plain_name("a.b_c")
    assert not any(is_plain_name(n) for n in ("", ".", "..", ".hidden", "a b", "a/b", "é"))
    out = write_workload(tmp_path, "wl-1", ROWS, SPLIT, STATS)
    assert check_workload_for_write(tmp_path, "wl-1") == out
    (out / "split.json").unlink()
    (out / "split.json").symlink_to(tmp_path / "x")
    with pytest.raises(UnsafeWorkloadsError):
        check_workload_for_write(tmp_path, "wl-1")


def test_read_regular_reads_a_file_and_leaves_a_missing_one_missing(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.json").write_text("héllo")
    assert read_regular(tmp_path / "a.json") == "héllo"
    with pytest.raises(FileNotFoundError):
        read_regular(tmp_path / "missing.json")


def test_read_regular_refuses_a_pipe_without_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refuse_a_blocking_open_of_a_pipe(monkeypatch)
    pipe = make_fifo(tmp_path / "scan-report.json")
    with pytest.raises(NotRegularFileError, match=r"scan-report\.json is not a regular file"):
        read_regular(pipe)


def test_read_regular_refuses_a_directory_and_a_link(tmp_path: Path) -> None:
    (tmp_path / "dir.json").mkdir()
    (tmp_path / "real.json").write_text("{}")
    (tmp_path / "link.json").symlink_to(tmp_path / "real.json")
    with pytest.raises(NotRegularFileError, match=r"dir\.json"):
        read_regular(tmp_path / "dir.json")
    with pytest.raises(UnsafeWorkloadsError, match=r"link\.json"):
        read_regular(tmp_path / "link.json")


def test_read_regular_lets_a_real_permission_error_through(tmp_path: Path) -> None:
    locked = tmp_path / "locked.json"
    locked.write_text("{}")
    locked.chmod(0)
    try:
        with pytest.raises(PermissionError):
            read_regular(locked)
    finally:
        locked.chmod(0o600)


@pytest.mark.usefixtures("by_fd")
def test_write_atomic_refuses_a_directory_or_a_pipe_where_the_file_belongs(
    tmp_path: Path,
) -> None:
    (tmp_path / "dataset.jsonl").mkdir()
    pipe = make_fifo(tmp_path / "split.json")

    for target in (tmp_path / "dataset.jsonl", pipe):
        with pytest.raises(NotRegularFileError, match="not a regular file"):
            write_atomic(target, "x")

    assert sorted(p.name for p in tmp_path.iterdir()) == ["dataset.jsonl", "split.json"]  # no temp


@pytest.mark.usefixtures("by_fd")
def test_check_writable_names_what_is_wrong(tmp_path: Path) -> None:
    check_writable(tmp_path / "new.json")
    (tmp_path / "ok.json").write_text("{}")
    check_writable(tmp_path / "ok.json")
    (tmp_path / "dir.json").mkdir()
    with pytest.raises(NotRegularFileError, match=r"dir\.json"):
        check_writable(tmp_path / "dir.json")


def test_open_append_refuses_a_pipe_and_a_directory(tmp_path: Path, by_fd: bool) -> None:
    pipe = make_fifo(tmp_path / "replay-a.jsonl")
    (tmp_path / "replay-b.jsonl").mkdir()
    for target in (pipe, tmp_path / "replay-b.jsonl"):
        with pytest.raises(NotRegularFileError):
            open_append(target)
    assert by_fd in (True, False)


@pytest.mark.parametrize("writer", ["atomic", "append"])
def test_a_folder_swapped_for_a_link_after_the_check_cannot_redirect_a_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer: str
) -> None:
    # The writer checked its folder by path and then wrote into whatever the path led to.
    # It now opens the folder once and works relative to that handle.
    folder = tmp_path / "folder"
    folder.mkdir()
    target = tmp_path / "elsewhere"
    target.mkdir()
    real_open = workspace._open_folder

    def swapping(path: Path) -> int:
        fd = real_open(path)  # the folder, opened and checked...
        folder.rename(tmp_path / "moved")  # ...then swapped for a link
        folder.symlink_to(target, target_is_directory=True)
        return fd

    monkeypatch.setattr(workspace, "_open_folder", swapping)

    if writer == "atomic":
        write_atomic(folder / "dataset.jsonl", "rows")
    else:
        with open_append(folder / "dataset.jsonl") as sink:
            sink.write("rows")

    assert list(target.iterdir()) == []
    assert (tmp_path / "moved" / "dataset.jsonl").read_text() == "rows"


def test_a_folder_that_is_a_link_is_refused_by_name(tmp_path: Path, by_fd: bool) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "linked").symlink_to(real, target_is_directory=True)

    with pytest.raises(UnsafeWorkloadsError) as info:
        write_atomic(tmp_path / "linked" / "state.json", "{}")

    assert info.value.path == tmp_path / "linked"  # the directory, not the file in it
    assert str(tmp_path / "linked") in str(info.value)
    assert list(real.iterdir()) == []
    assert by_fd in (True, False)


def test_a_folder_that_is_missing_stays_a_missing_folder(tmp_path: Path, by_fd: bool) -> None:
    with pytest.raises(FileNotFoundError):
        write_atomic(tmp_path / "nope" / "a.json", "{}")
    assert by_fd in (True, False)


def test_a_directory_named_like_a_scan_file_is_left_in_place(tmp_path: Path, by_fd: bool) -> None:
    out = write_workload(tmp_path, "wl-1", ROWS, SPLIT, STATS)
    (out / "replay-x.jsonl").mkdir()
    (out / "replay-x.jsonl" / "kept.txt").write_text("mine")

    assert remove_workload(tmp_path, "wl-1") is False

    assert [p.name for p in out.iterdir()] == ["replay-x.jsonl"]
    assert (out / "replay-x.jsonl" / "kept.txt").read_text() == "mine"
    assert by_fd in (True, False)


@pytest.mark.usefixtures("by_fd")
def test_a_temp_file_that_has_gone_when_a_write_fails_is_not_an_error_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def vanish_then_fail(_fd: int) -> None:
        for leftover in tmp_path.glob(".out.json.*.tmp"):
            leftover.unlink()
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", vanish_then_fail)

    with pytest.raises(OSError, match="disk full"):
        write_atomic(tmp_path / "out.json", "x")


@pytest.mark.usefixtures("by_fd")
def test_open_append_lets_a_real_permission_error_through(tmp_path: Path) -> None:
    locked = tmp_path / "replay.jsonl"
    locked.write_text("")
    locked.chmod(0o400)
    try:
        with pytest.raises(PermissionError):
            open_append(locked)
    finally:
        locked.chmod(0o600)
