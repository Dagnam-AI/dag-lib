"""``install``, ``uninstall``, the kill switch and the options, against the real SDKs."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import os
from pathlib import Path
import stat
import sys
import threading
from typing import Any, Dict, List

from _api_support import finish_lines, record, run_forked, saved_counts
from _mock_server import Reply
from _openai_bodies import CHAT_PATH, MODEL, chat_body
from _openai_calls import API_KEY, unpatched
from anthropic.resources.messages import Messages
import httpx
import openai
from openai.resources.chat.completions import Completions
import pytest

import dagnam_record
from dagnam_record import _state

pytestmark = pytest.mark.usefixtures("fresh_state")


def writer_threads() -> int:
    return sum(t.name == "dagnam-record" and t.is_alive() for t in threading.enumerate())


def hide_sdks(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Make ``import`` of the named SDKs (both when none is named) fail."""
    for name in names or ("openai", "anthropic"):
        for module in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
            monkeypatch.setitem(sys.modules, module, None)
        monkeypatch.setitem(sys.modules, name, None)


def test_the_public_names() -> None:
    assert dagnam_record.__all__ == ["__version__", "install", "record_call", "uninstall"]
    assert dagnam_record.install is _state.install
    assert dagnam_record.record_call is _state.record_call
    assert dagnam_record.uninstall is _state.uninstall


def test_install_patches_the_sdks_and_a_real_call_is_recorded(
    tmp_path: Path, mock_server: Any
) -> None:
    directory = tmp_path / "traces"
    assert dagnam_record.install(dir=str(directory)) is True
    mock_server.on(CHAT_PATH, Reply(body=chat_body()))
    client = openai.OpenAI(
        api_key=API_KEY,
        base_url=mock_server.url,
        http_client=httpx.Client(trust_env=False),
        max_retries=0,
    )
    result = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "Hi"}]
    )
    assert result.choices[0].message.content == "Hello there"
    (line,) = finish_lines(directory)
    assert (line["provider"], line["endpoint"], line["status"]) == (
        "openai",
        "chat.completions",
        200,
    )
    assert API_KEY not in str(line)


def test_the_kill_switch_makes_install_a_no_op_that_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "traces"
    monkeypatch.setenv("DAGNAM_RECORD", "0")
    assert dagnam_record.install(dir=str(directory)) is False
    assert not directory.exists()
    assert _state.STATE.installed is None
    assert not hasattr(Completions.create, "__dagnam_record__")
    record()
    assert not directory.exists()
    # Bad options never break a process that has switched the recorder off.
    options: Any = {"sample": 7, "redact": "nope"}
    assert dagnam_record.install(**options) is False


@pytest.mark.parametrize("value", ["", "1", "00", " 0", "0 ", "false", "off", "no", "yes"])
def test_only_the_exact_value_zero_switches_it_off(
    value: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_RECORD", value)
    assert dagnam_record.install(dir=str(tmp_path / "t")) is True
    assert _state.STATE.installed is not None


@pytest.mark.parametrize("value", ["false", "off", "no", "FALSE"])
def test_a_value_that_looks_like_off_but_is_not_zero_warns(
    value: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("DAGNAM_RECORD", value)
    with caplog.at_level(logging.WARNING, logger="dagnam_record"):
        dagnam_record.install(dir=str(tmp_path / "t"))
    assert "does not switch the recorder off" in caplog.text
    assert "set it to 0" in caplog.text


def test_no_warning_for_unset_or_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="dagnam_record"):
        monkeypatch.delenv("DAGNAM_RECORD", raising=False)
        dagnam_record.install(dir=str(tmp_path / "a"))
        monkeypatch.setenv("DAGNAM_RECORD", "1")
        dagnam_record.install(dir=str(tmp_path / "b"))
    assert caplog.text == ""


def test_a_missing_sdk_is_not_an_error_and_record_call_still_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hide_sdks(monkeypatch)
    directory = tmp_path / "traces"
    assert dagnam_record.install(dir=str(directory)) is False
    record()
    (line,) = finish_lines(directory)
    assert line["provider"] == "other"


@pytest.mark.parametrize(("missing", "patched"), [("anthropic", Completions), ("openai", Messages)])
def test_either_sdk_alone_is_patched(
    missing: str, patched: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hide_sdks(monkeypatch, missing)
    assert dagnam_record.install(dir=str(tmp_path / "t")) is True
    assert hasattr(patched.create, "__dagnam_record__")


def test_install_before_the_sdk_exists_and_again_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = str(tmp_path / "traces")
    with monkeypatch.context() as hidden:
        hide_sdks(hidden)
        assert dagnam_record.install(dir=directory) is False
    assert dagnam_record.install(dir=directory) is True
    assert hasattr(Completions.create, "__dagnam_record__")


def test_install_prunes_the_recorders_own_files_older_than_the_window(tmp_path: Path) -> None:
    directory = tmp_path / "t"
    directory.mkdir()
    today = datetime.now(timezone.utc).date()
    old = (today - timedelta(days=15)).isoformat()
    edge = (today - timedelta(days=3)).isoformat()
    for name in (f"{old}.jsonl", f"{edge}.jsonl", f"{old}.txt", "notes.jsonl"):
        (directory / name).write_text("x\n", encoding="utf-8")
    dagnam_record.install(dir=str(directory), window_days=14)
    assert sorted(p.name for p in directory.iterdir() if p.name != ".status.json") == sorted(
        [f"{edge}.jsonl", f"{old}.txt", "notes.jsonl"]
    )
    (directory / f"{edge}.jsonl").write_text("x\n", encoding="utf-8")
    dagnam_record.install(dir=str(directory), window_days=2)  # new options: set up again
    assert not (directory / f"{edge}.jsonl").exists()


def test_install_is_idempotent(tmp_path: Path) -> None:
    before = writer_threads()
    directory = str(tmp_path / "traces")
    assert dagnam_record.install(dir=directory) is True
    first = _state.STATE.installed
    assert dagnam_record.install(dir=directory) is True
    assert _state.STATE.installed is first
    assert writer_threads() == before + 1
    assert not hasattr(unpatched(Completions.create), "__dagnam_record__")  # wrapped once only


def test_install_is_thread_safe(tmp_path: Path) -> None:
    before = writer_threads()
    directory = str(tmp_path / "traces")
    barrier = threading.Barrier(8)
    results: List[bool] = []

    def work() -> None:
        barrier.wait(timeout=10)
        results.append(dagnam_record.install(dir=directory))

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert results == [True] * 8
    assert writer_threads() == before + 1
    assert not hasattr(unpatched(Completions.create), "__dagnam_record__")


def test_a_second_install_with_other_options_reconfigures(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    before = writer_threads()
    dagnam_record.install(dir=str(first))
    record(request={"prompt": "one"})
    old = _state.STATE.installed
    assert dagnam_record.install(dir=str(second), no_raw=False, sample=1.0, window_days=3) is True
    assert _state.STATE.installed is not old
    assert writer_threads() == before + 1  # the old writer was closed, not leaked
    record(request={"prompt": "two"})
    lines = finish_lines(second)
    assert [line["request"]["prompt"] for line in lines] == ["two"]
    first_lines = list(first.glob("*.jsonl"))
    assert len(first_lines) == 1
    assert "one" in first_lines[0].read_text(encoding="utf-8")
    # The old writer drained its queue and saved its counts before it stopped.
    assert saved_counts(first)["written"] == 1


@pytest.mark.parametrize(
    "change",
    [{"window_days": 3}, {"sample": 0.5}, {"no_raw": True}, {"redact": lambda text: text}],
)
def test_any_changed_option_reconfigures(change: Any, tmp_path: Path) -> None:
    directory = str(tmp_path / "traces")
    dagnam_record.install(dir=directory)
    first = _state.STATE.installed
    dagnam_record.install(dir=directory, **change)
    assert _state.STATE.installed is not first


def test_the_same_options_again_keep_the_recorder(tmp_path: Path) -> None:
    directory = str(tmp_path / "traces")

    def mask(text: str) -> str:
        return text

    dagnam_record.install(dir=directory, sample=0.5, redact=mask)
    first = _state.STATE.installed
    dagnam_record.install(dir=directory, sample=0.5, redact=mask)
    assert _state.STATE.installed is first
    dagnam_record.install(dir=directory, sample=0.5, redact=lambda text: text)
    assert _state.STATE.installed is not first  # another callable is another option


def test_the_options_are_checked(tmp_path: Path) -> None:
    good: Dict[str, Any] = {"dir": str(tmp_path / "t")}
    bad: List[Any] = [
        {"window_days": 0},
        {"window_days": 366},
        {"window_days": True},
        {"window_days": 1.5},
        {"window_days": "14"},
        {"sample": 0},
        {"sample": -0.1},
        {"sample": 1.0001},
        {"sample": float("nan")},
        {"sample": float("inf")},
        {"sample": True},
        {"sample": "1"},
        {"no_raw": 1},
        {"no_raw": "yes"},
        {"redact": "yes"},
        {"redact": "AUTO"},
        {"redact": 3},
        {"redact": ""},
        {"dir": ""},
        {"dir": 5},
        {"dir": "a\0b"},
    ]
    for override in bad:
        options: Dict[str, Any] = {**good, **override}
        with pytest.raises(ValueError, match=r"must be|non-empty"):
            dagnam_record.install(**options)
        assert _state.STATE.installed is None
        assert not (tmp_path / "t").exists()
    assert dagnam_record.install(window_days=365, sample=1, dir=Path(good["dir"])) is True


def test_the_default_directory_is_private_and_under_the_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    dagnam_record.install()
    record()
    directory = tmp_path / ".dagnam" / "traces"
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    (line,) = finish_lines(directory)
    assert line["request"] == {"prompt": "hello", "model": "m"}
    (path,) = directory.glob("*.jsonl")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_uninstall_restores_the_sdk_stops_recording_and_closes_the_writer(tmp_path: Path) -> None:
    before = writer_threads()
    original = Completions.create
    directory = tmp_path / "traces"
    dagnam_record.install(dir=str(directory))
    assert Completions.create is not original
    record()
    dagnam_record.uninstall()
    assert Completions.create is original
    assert writer_threads() == before
    assert _state.STATE.installed is None
    record()  # a no-op now
    dagnam_record.uninstall()  # twice is fine
    assert saved_counts(directory)["written"] == 1
    assert len(list(directory.glob("*.jsonl"))) == 1


def test_the_package_import_loads_no_sdk() -> None:
    code = (
        "import sys, dagnam_record\n"
        "bad = [m for m in ('openai', 'anthropic', 'dagnam', 'dagnam_contracts') if m in sys.modules]\n"
        "print(bad)\n"
    )
    import subprocess

    result = subprocess.run(  # noqa: S603 - fixed interpreter and inline code, no untrusted input
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env={
            "PYTHONPATH": str(Path(dagnam_record.__file__).parent.parent),
            "PATH": "/usr/bin:/bin",
        },
    )
    assert result.stdout.strip() == "[]", result.stderr


def test_the_install_lock_is_replaced_in_a_forked_child(tmp_path: Path) -> None:
    held = _state.STATE.lock
    held.acquire()
    try:
        _state.after_fork_in_child()
        assert _state.STATE.lock is not held
        assert _state.STATE.lock.acquire(blocking=False)
        _state.STATE.lock.release()
    finally:
        held.release()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_a_child_forked_while_install_was_running_can_still_install(tmp_path: Path) -> None:
    _state.STATE.lock.acquire()

    def child() -> int:
        return 0 if dagnam_record.install(dir=str(tmp_path / "child")) else 2

    try:
        assert run_forked(child) == 0
    finally:
        _state.STATE.lock.release()
