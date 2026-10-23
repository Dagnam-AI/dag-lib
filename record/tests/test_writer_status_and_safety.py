"""The writer's saved counts, file permissions, fork safety and shutdown behaviour."""

from __future__ import annotations

import atexit
import errno
import json
import os
from pathlib import Path
import queue
import signal
import stat
import sys
import threading
from typing import Any, Callable, List

import pytest

from dagnam_record._writer import reinit_all_after_fork

T0 = 1_791_797_400.0  # 2026-10-12T09:30:00Z
DAY = 86_400
POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes and symlinks")
FORK_ONLY = pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")


def lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def status(directory: Path) -> Any:
    return json.loads((directory / ".status.json").read_text(encoding="utf-8"))


def mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def run_forked(child: Callable[[], int]) -> int:
    """Run ``child`` in a forked process and return its exit code.

    An alarm ends a child that hangs, so the wait for it is bounded by 10 seconds.
    """
    pid = os.fork()
    if pid == 0:
        code = 99
        try:
            signal.alarm(10)
            code = child()
        finally:
            os._exit(code)
    _, wait_status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(wait_status)


# --- saved counts ---------------------------------------------------------------------------


def test_counts_are_saved_every_30_seconds_when_changed_and_at_close(
    make: Any, new_clock: Any, tmp_path: Path
) -> None:
    clock = new_clock(T0)
    directory = tmp_path / "traces"
    writer = make(clock=clock)
    writer.submit("a")
    writer.drain()
    assert not (directory / ".status.json").exists()  # under 30 s: not yet
    clock.now = T0 + 31
    writer.submit("b")
    writer.drain()
    saved = status(directory)
    assert saved["counts"]["written"] == 2
    assert saved["started"] == T0
    assert saved["expired"] is False
    assert saved["directory"] == str(directory)
    clock.now += 5
    writer.submit("c")
    writer.drain()
    assert status(directory)["counts"]["written"] == 2  # changed, but under 30 s since the save
    writer.close()
    assert status(directory)["counts"]["written"] == 3
    assert [p for p in directory.iterdir() if p.name.endswith(".tmp")] == []


@POSIX_ONLY
def test_the_status_file_is_private(make: Any, tmp_path: Path) -> None:
    writer = make()
    writer.submit("a")
    writer.close()
    assert mode(tmp_path / "traces" / ".status.json") == 0o600


def test_an_unchanged_writer_saves_nothing_more(make: Any, new_clock: Any, tmp_path: Path) -> None:
    clock = new_clock(T0)
    writer = make(clock=clock)
    writer.close()  # the start time is the one thing to save
    saved = (tmp_path / "traces" / ".status.json").read_bytes()
    clock.now += 100
    writer.close()
    assert (tmp_path / "traces" / ".status.json").read_bytes() == saved


def test_a_save_that_failed_is_tried_again(
    make: Any, new_clock: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = new_clock(T0)
    directory = tmp_path / "traces"
    writer = make(clock=clock)
    real = os.replace
    failures = [1]

    def flaky(source: str, target: str) -> None:
        if failures[0]:
            failures[0] -= 1
            raise OSError(errno.EIO, "disk hiccup")
        real(source, target)

    monkeypatch.setattr(os, "replace", flaky)
    writer.submit("a")
    clock.now = T0 + 31
    writer.drain()  # the save is due and fails
    assert not (directory / ".status.json").exists()
    assert [p for p in directory.iterdir() if p.name.endswith(".tmp")] == []  # no litter
    writer.close()  # nothing changed since, yet the counts are still owed
    assert status(directory)["counts"]["written"] == 1


def test_the_idle_loop_saves_changed_counts_and_stops_on_the_signal(
    make: Any, new_clock: Any, scripted_queue: Any, tmp_path: Path
) -> None:
    clock = new_clock(T0)
    writer = make(clock=clock)
    script = scripted_queue([queue.Empty, "line", None])
    writer.swap_queue(script)
    clock.now = T0 + 31
    writer.loop(script)
    assert status(tmp_path / "traces")["counts"]["written"] == 0  # saved while idle
    assert lines(tmp_path / "traces" / "2026-10-12.jsonl") == ["line"]


def test_the_expiry_is_saved_with_the_counts(make: Any, new_clock: Any, tmp_path: Path) -> None:
    clock = new_clock(T0)
    writer = make(clock=clock, window_days=1)
    clock.now = T0 + DAY
    writer.submit("a")
    writer.drain()
    writer.close()
    saved = status(tmp_path / "traces")
    assert saved["expired"] is True
    assert saved["counts"]["dropped_expired"] == 1


def test_a_status_file_that_cannot_be_saved_is_not_an_error(make: Any, tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    writer = make(blocker / "traces")
    writer.close()
    assert not (blocker / "traces").exists()


def test_close_is_registered_for_exit_and_unregistered_once_run(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    registered: List[Any] = []
    unregistered: List[Any] = []
    monkeypatch.setattr(atexit, "register", registered.append)
    monkeypatch.setattr(atexit, "unregister", unregistered.append)
    writer = make()
    assert registered == [writer.close]
    writer.close()
    assert unregistered == [writer.close]


# --- permissions ----------------------------------------------------------------------------


@POSIX_ONLY
def test_the_directory_is_0700_and_the_file_0600(make: Any, tmp_path: Path) -> None:
    writer = make(tmp_path / "a" / "b" / "traces")
    writer.submit("{}")
    writer.drain()
    directory = tmp_path / "a" / "b" / "traces"
    assert mode(directory) == 0o700
    assert mode(directory / "2026-10-12.jsonl") == 0o600


@POSIX_ONLY
def test_an_existing_directory_we_own_is_tightened_to_0700(make: Any, tmp_path: Path) -> None:
    directory = tmp_path / "traces"
    directory.mkdir()
    os.chmod(directory, int("750", 8))  # group-readable: looser than 0700
    writer = make()
    writer.submit("{}")
    writer.drain()
    assert mode(directory) == 0o700
    assert writer.counts["written"] == 1


@POSIX_ONLY
def test_an_existing_private_directory_is_left_alone(make: Any, tmp_path: Path) -> None:
    directory = tmp_path / "traces"
    directory.mkdir()
    os.chmod(directory, 0o700)
    writer = make()
    writer.submit("{}")
    writer.drain()
    assert mode(directory) == 0o700
    assert writer.counts["written"] == 1


@POSIX_ONLY
def test_a_file_that_is_too_open_is_made_private(make: Any, tmp_path: Path) -> None:
    directory = tmp_path / "traces"
    directory.mkdir(mode=0o700)
    existing = directory / "2026-10-12.jsonl"
    existing.write_text("old\n", encoding="utf-8")
    os.chmod(existing, 0o644)
    writer = make()
    writer.submit("new")
    writer.drain()
    assert mode(existing) == 0o600
    assert lines(existing) == ["old", "new"]


@POSIX_ONLY
def test_a_directory_owned_by_someone_else_is_refused(
    make: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "traces").mkdir()
    monkeypatch.setattr("dagnam_record._writer._OWNER", os.getuid() + 1)
    writer = make()
    writer.submit("secret")
    writer.drain()
    assert writer.counts["dropped_error"] == 1
    assert list((tmp_path / "traces").iterdir()) == []


@POSIX_ONLY
def test_a_symlinked_directory_is_refused(make: Any, tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    target.mkdir()
    (tmp_path / "traces").symlink_to(target, target_is_directory=True)
    writer = make()
    writer.submit("secret")
    writer.drain()
    assert writer.counts["dropped_error"] == 1
    assert list(target.iterdir()) == []


def test_a_directory_path_that_is_a_file_is_refused(make: Any, tmp_path: Path) -> None:
    (tmp_path / "traces").write_text("not a directory", encoding="utf-8")
    writer = make()
    writer.submit("x")
    writer.drain()
    assert writer.counts["dropped_error"] == 1


@POSIX_ONLY
def test_a_symlinked_data_file_is_not_followed(make: Any, tmp_path: Path) -> None:
    directory = tmp_path / "traces"
    directory.mkdir(mode=0o700)
    victim = tmp_path / "victim.txt"
    victim.write_text("keep\n", encoding="utf-8")
    (directory / "2026-10-12.jsonl").symlink_to(victim)
    writer = make()
    writer.submit("planted")
    writer.drain()
    assert writer.counts["dropped_error"] == 1
    assert victim.read_text(encoding="utf-8") == "keep\n"


# --- fork safety ----------------------------------------------------------------------------


def test_reinit_gives_a_fresh_queue_thread_and_counts(make: Any, tmp_path: Path) -> None:
    writer = make(start=True)
    old_queue, old_thread = writer.pending(), writer.thread()
    writer.reinit_after_fork()
    assert writer.pending() is not old_queue
    assert writer.thread() is not old_thread
    assert writer.alive()
    assert all(count == 0 for count in writer.counts.values())
    writer.submit("fresh")
    old_queue.put(None)  # let the old thread end
    assert old_thread is not None
    old_thread.join(5)
    assert not old_thread.is_alive()
    writer.close()
    assert lines(tmp_path / "traces" / "2026-10-12.jsonl") == ["fresh"]
    assert writer.counts["written"] == 1


def test_reinit_drops_what_the_parent_still_has_pending(make: Any, tmp_path: Path) -> None:
    writer = make()
    writer.submit("pending")
    writer.reinit_after_fork()
    assert writer.pending().qsize() == 0
    assert writer.thread() is None
    writer.drain()
    assert not (tmp_path / "traces").exists()


def test_reinit_does_not_restart_a_closed_writer(make: Any) -> None:
    writer = make(start=True)
    writer.close()
    writer.reinit_after_fork()
    assert writer.thread() is None
    assert not writer.alive()


def test_the_fork_hook_reinitialises_every_live_writer(make: Any) -> None:
    first, second = make(), make()
    first.submit("a")
    second.submit("b")
    reinit_all_after_fork()
    assert first.pending().qsize() == 0
    assert second.pending().qsize() == 0


@FORK_ONLY
def test_a_forked_child_writes_only_its_own_lines_and_never_the_parents_pending_ones(
    make: Any, tmp_path: Path
) -> None:
    writer = make()
    writer.submit("one")
    writer.submit("two")
    writer.drain()  # the parent has written two lines
    writer.submit("pending")  # and holds one more

    def child() -> int:
        if any(writer.counts.values()) or writer.pending().qsize():
            return 2  # the child must not inherit the parent's counts or queue
        writer.submit("child")
        writer.close(timeout=5)
        return 0 if writer.counts["written"] == 1 else 3

    assert run_forked(child) == 0
    writer.drain()
    found = lines(tmp_path / "traces" / "2026-10-12.jsonl")
    assert sorted(found) == ["child", "one", "pending", "two"]


@FORK_ONLY
def test_a_forked_child_gets_a_working_thread_even_if_the_parent_held_the_queue_lock(
    make: Any, tmp_path: Path
) -> None:
    writer = make(start=True)
    held, release = threading.Event(), threading.Event()

    def hold() -> None:
        with writer.pending().mutex:  # another thread is inside the queue at the fork
            held.set()
            release.wait(10)

    holder = threading.Thread(target=hold)
    holder.start()
    assert held.wait(5)

    def child() -> int:
        if not writer.alive():
            return 2  # the thread did not survive the fork: it must be restarted
        writer.submit("child")  # would block for good on the inherited, locked queue
        writer.close(timeout=5)
        return 0 if writer.counts["written"] == 1 else 3

    try:
        code = run_forked(child)
    finally:
        release.set()
        holder.join(5)
    assert code == 0
    writer.close()
    assert lines(tmp_path / "traces" / "2026-10-12.jsonl") == ["child"]
