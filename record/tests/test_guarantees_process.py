"""Guarantees that only a whole process can show: no network, no side effects, no hang at exit.

Each case runs a child interpreter (see ``_guarantee_child``) with a temporary HOME and checks
what it did. The child prints nothing of its own, so any output on its stdout or stderr came from
the recorder.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

import dagnam_record

TESTS = Path(__file__).parent
RECORD = Path(dagnam_record.__file__).parent.parent
NETWORK_MODULES = {
    "_socket",
    "_ssl",
    "asyncio",
    "ftplib",
    "http",
    "select",
    "selectors",
    "smtplib",
    "socket",
    "socketserver",
    "ssl",
    "urllib.error",
    "urllib.request",
    "xmlrpc",
}
"""Modules that open or speak on a network connection. The SDKs load them; the recorder must not."""
CLOSE_TIMEOUT = 2.0
"""The documented wait of ``Writer.close`` at exit, in seconds."""


def run_python(
    arguments: List[str], home: Path, timeout: float = 120
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed interpreter and our own scripts, no untrusted input
        [sys.executable, *arguments],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env={
            "PYTHONPATH": f"{RECORD}:{TESTS}",
            "PATH": "/usr/bin:/bin",
            # Apple's Python writes its bytecode cache under $HOME/Library; the tests assert that
            # nothing at all is made there.
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": str(home),
            "USERPROFILE": str(home),
        },
    )


def run_child(scenario: str, work: Path, stderr: str = "") -> Tuple[Dict[str, Any], float]:
    """The child's report and the time it ended, after checking it said only ``stderr``."""
    work.mkdir()
    started = time.time()
    result = run_python([str(TESTS / "_guarantee_child.py"), scenario, str(work)], work / "home")
    ended = time.time()
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""  # the recorder says nothing on stdout, whatever happens
    assert result.stderr == stderr  # ... or on stderr, apart from a documented warning
    assert not (work / "home").exists()  # the default ~/.dagnam was never made
    report: Dict[str, Any] = json.loads((work / "report.json").read_text(encoding="utf-8"))
    assert started <= ended
    return report, ended


def test_the_recorder_changes_no_global_state_and_registers_only_its_exit_hooks(
    tmp_path: Path,
) -> None:
    report, _ = run_child("state", tmp_path / "w")
    assert report["after"] == report["before"]  # logging, signals, hooks, warnings, env, threads...
    hooks = {tuple(hook) for hook in report["atexit"]}
    assert {hook for hook in hooks if hook[0].startswith("dagnam_record")} == {
        ("dagnam_record._writer", "Writer.close"),
        ("dagnam_record._stream", "finish_live"),
    }
    # the rest is not the recorder's own: ``weakref.finalize`` (used with ``atexit = False``, so
    # it runs nothing) and the HTTP client's certificate bundle, made by the program's SDK client
    assert {hook for hook in hooks if not hook[0].startswith("dagnam_record")} <= {
        ("weakref", "finalize._exitfunc"),
        ("certifi.core", "exit_cacert_ctx"),
    }
    assert report["during_threads"] == [["dagnam-record", True]]  # a daemon, with a stable name


def test_importing_the_package_loads_no_network_module_and_registers_nothing(
    tmp_path: Path,
) -> None:
    code = (
        "import atexit, json, logging, sys, threading, uuid\n"  # (logging has its own hook)
        "before, hooks = set(sys.modules), atexit._ncallbacks()\n"
        "import dagnam_record\n"
        "imported, on_import = set(sys.modules) - before, atexit._ncallbacks() - hooks\n"
        "for name in ('openai', 'anthropic'):\n"
        "    sys.modules[name] = None\n"
        "dagnam_record.install(dir=sys.argv[1])\n"
        "dagnam_record.record_call(provider='other', request={'a': 1}, response={'b': 2})\n"
        "dagnam_record.uninstall()\n"
        "used = set(sys.modules) - before\n"
        "print(json.dumps([sorted(imported), sorted(used), on_import]))\n"
    )
    result = run_python(["-c", code, str(tmp_path / "traces")], tmp_path / "home")
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    imported, used, on_import = json.loads(result.stdout)
    assert "dagnam_record" in imported
    assert NETWORK_MODULES.isdisjoint(imported)
    assert NETWORK_MODULES.isdisjoint(used)  # nor does install() or a recorded call load one
    assert on_import == 0  # importing registers nothing; install() asks for its hooks (see above)


def test_the_recorder_opens_no_connection_and_the_programs_own_go_only_to_the_server(
    tmp_path: Path,
) -> None:
    report, _ = run_child("network", tmp_path / "w")
    assert report["alone"] == [[], []]  # install, 50 record_calls, uninstall, status, purge
    destinations, names = report["with_calls"]
    assert destinations  # the calls did connect, so the watch was working
    assert {address for address, _ in destinations} == {f"('127.0.0.1', {report['port']})"}
    assert "dagnam-record" not in {thread for _, thread in destinations}
    assert set(names) == {"'127.0.0.1'"}


def test_a_stuck_writer_cannot_hold_up_the_end_of_the_process(tmp_path: Path) -> None:
    report, ended = run_child("stuck", tmp_path / "w")  # the child's stdout and stderr are empty
    assert ended - report["ended"] < CLOSE_TIMEOUT + 4  # the close waits at most its timeout


def test_drops_and_errors_are_silent(tmp_path: Path) -> None:
    counts, _ = run_child("failing", tmp_path / "w")
    assert counts["disk"]["dropped_error"] >= 20  # the full disk and the redactor were each hit,
    assert counts["redact"]["dropped_error"] >= 20  # and every call they spoiled was counted
    assert (
        sum(counts["gone"].values()) > 0
    )  # ... (the vanished folder: written or dropped, counted)


def test_an_unusable_directory_costs_one_warning_line_and_then_silence(tmp_path: Path) -> None:
    warning = (
        f"dagnam_record: {tmp_path / 'w' / 'traces'} cannot be used;"
        " recorded calls will be dropped and counted\n"
    )
    run_child("unusable", tmp_path / "w", stderr=warning)
