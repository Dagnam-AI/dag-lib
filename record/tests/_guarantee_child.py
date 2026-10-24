"""A child interpreter for test_guarantees_process: one scenario, a JSON report in a file.

Usage: ``python _guarantee_child.py <scenario> <work directory>``. The child's own stdout and
stderr carry nothing of its own, so whatever reaches them came from the recorder; the parent
asserts they are empty. HOME is the work directory's ``home``.
"""

from __future__ import annotations

import atexit
import contextlib
import errno
import faulthandler
import gc
import io
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import socket
import sys
import threading
import time
from typing import Any, Callable, Dict, Generator, List, Tuple
from unittest import mock
import warnings

from _anthropic_bodies import PATH, error_body, text_events
from _guarantee_support import CHAT_ARGS, MESSAGE_ARGS, anthropic_client, openai_client
from _mock_server import MockServer, Reply
from _openai_bodies import CHAT_PATH, chat_body

import dagnam_record
from dagnam_record import _cli, _state
from dagnam_record._writer import Writer

WORK = Path(sys.argv[2])
REGISTERED: List[Tuple[str, str]] = []
REAL_REGISTER = atexit.register
DESTINATIONS: List[List[Any]] = []
RESOLVED: List[str] = []


def snapshot() -> Dict[str, Any]:
    """Every piece of process-global state a library could change behind the program's back."""
    logger = logging.getLogger("dagnam_record")
    return {
        "root": [repr(h) for h in logging.getLogger().handlers],
        "root_level": logging.getLogger().level,
        "logging_disable": logging.root.manager.disable,
        "logger": [repr(logger.handlers), logger.level, logger.propagate, logger.disabled],
        "signals": {int(n): repr(signal.getsignal(n)) for n in signal.valid_signals()},
        "hooks": [repr(sys.excepthook), repr(threading.excepthook), repr(sys.unraisablehook)],
        "warnings": [
            repr(sys.modules["warnings"].showwarning),
            [repr(f) for f in warnings.filters],
        ],
        "stdio": [id(sys.stdout), id(sys.stderr), id(sys.stdin)],
        "environ": dict(os.environ),
        "cwd": os.getcwd(),
        "path": list(sys.path),
        "meta_path": [repr(f) for f in sys.meta_path],
        "trace": [repr(sys.gettrace()), repr(sys.getprofile())],
        "gc": [gc.isenabled(), gc.get_threshold(), len(gc.callbacks)],
        "faulthandler": faulthandler.is_enabled(),
        "interpreter": [sys.getrecursionlimit(), sys.getswitchinterval()],
        "timeout": socket.getdefaulttimeout(),
        "threads": sorted(
            (t.name, t.daemon) for t in threading.enumerate() if t.name == "dagnam-record"
        ),
    }


def make_calls(server: MockServer) -> None:
    """Calls that succeed, fail with an HTTP error, stream, and are dropped by the recorder."""
    server.on(CHAT_PATH, Reply(body=chat_body()), Reply(429, {"error": {"message": "slow"}}))
    server.on(
        PATH, Reply(events=text_events("Hel", "lo")), Reply(429, error_body("rate_limit_error"))
    )
    with openai_client(server) as client:
        client.chat.completions.create(**CHAT_ARGS)
        with contextlib.suppress(Exception):
            client.chat.completions.create(**CHAT_ARGS)
    with anthropic_client(server) as other:
        for _ in other.messages.create(stream=True, **MESSAGE_ARGS):
            pass
        with contextlib.suppress(Exception):
            other.messages.create(**MESSAGE_ARGS)
    loop: Dict[str, Any] = {}
    loop["me"] = loop
    dagnam_record.record_call(provider="other", request=loop, response={})  # cannot be converted
    dagnam_record.record_call(
        provider="other", request={"a": "x" * 2_000_000}, response={}
    )  # too big
    dagnam_record.record_call(provider="other", request={"a": 1}, response={"b": 2})


def noting_register(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """``atexit.register``, noting who asked."""
    REGISTERED.append(
        (getattr(function, "__module__", "?"), getattr(function, "__qualname__", "?"))
    )
    return REAL_REGISTER(function, *args, **kwargs)


def state(directory: Path) -> Dict[str, Any]:
    server = MockServer()
    before = snapshot()
    with mock.patch.object(atexit, "register", noting_register):
        dagnam_record.install(dir=str(directory))
        during = snapshot()
        make_calls(server)
        dagnam_record.uninstall()
    after = snapshot()
    server.close()
    return {
        "before": before,
        "during_threads": during["threads"],
        "after": after,
        "atexit": REGISTERED,
    }


@contextlib.contextmanager
def watching(*, allow: bool, port: int) -> Generator[None]:
    """Note every connection attempt; refuse it unless ``allow`` and it goes to the mock server."""
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
    real_info, real_create = socket.getaddrinfo, socket.create_connection

    def note(address: Any) -> None:
        DESTINATIONS.append([repr(address), threading.current_thread().name])
        if not allow or tuple(address[:2]) != ("127.0.0.1", port):
            raise OSError("no network in this test")

    def connect(self: socket.socket, address: Any) -> None:
        note(address)
        real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        note(address)
        return real_connect_ex(self, address)

    def info(host: Any, *args: Any, **kwargs: Any) -> Any:
        RESOLVED.append(repr(host))
        if not allow or host != "127.0.0.1":
            raise OSError("no name lookup in this test")
        return real_info(host, *args, **kwargs)

    def create(address: Any, *args: Any, **kwargs: Any) -> Any:
        note(address)
        return real_create(address, *args, **kwargs)

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(socket.socket, "connect", connect))
        stack.enter_context(mock.patch.object(socket.socket, "connect_ex", connect_ex))
        stack.enter_context(mock.patch.object(socket, "getaddrinfo", info))
        stack.enter_context(mock.patch.object(socket, "create_connection", create))
        yield


def network(directory: Path) -> Dict[str, Any]:
    server = MockServer()
    port = int(server.url.split(":")[2].split("/")[0])
    with watching(
        allow=False, port=port
    ):  # the recorder on its own: every attempt is refused and noted
        dagnam_record.install(dir=str(directory))
        for number in range(50):
            dagnam_record.record_call(provider="other", request={"n": number}, response={"r": 1})
        dagnam_record.uninstall()
        with contextlib.redirect_stdout(io.StringIO()):
            _cli.main(["status", "--dir", str(directory)])
            _cli.main(["purge", "--dir", str(directory), "--dry-run"])
    alone = [list(DESTINATIONS), list(RESOLVED)]
    DESTINATIONS.clear()
    RESOLVED.clear()
    with watching(allow=True, port=port):  # the program's own calls, to the mock server only
        dagnam_record.install(dir=str(directory))
        make_calls(server)
        dagnam_record.uninstall()
    server.close()
    return {"port": port, "alone": alone, "with_calls": [list(DESTINATIONS), list(RESOLVED)]}


def stuck(directory: Path) -> Dict[str, Any]:
    """A writer that never returns, a flood of calls, and then the interpreter exits."""
    forever = threading.Event()

    def never(self: Writer, line: str) -> None:
        forever.wait()

    mock.patch.object(Writer, "_write", never).start()  # (never stopped: the process is ending)
    dagnam_record.install(dir=str(directory))
    for number in range(1500):
        dagnam_record.record_call(provider="other", request={"n": number}, response={"r": 1})
    return {"ended": time.time()}


def failing(directory: Path) -> Dict[str, Any]:
    """A full disk, a redactor that raises and a directory that is gone: counted, never printed."""

    def full(descriptor: int, data: Any) -> int:
        raise OSError(errno.ENOSPC, "No space left on device")

    def redactor(text: str) -> str:
        raise RuntimeError("the redactor has a bug")

    counts: Dict[str, Dict[str, int]] = {}
    for name in ("disk", "redact", "gone"):
        where = directory.parent / name
        dagnam_record.install(dir=str(where), redact=redactor if name == "redact" else None)
        installed = _state.STATE.installed
        assert installed is not None
        with mock.patch.object(os, "write", full if name == "disk" else os.write):
            for number in range(20):
                dagnam_record.record_call(
                    provider="other", request={"n": number}, response={"r": 1}
                )
            if name == "gone":
                shutil.rmtree(where, ignore_errors=True)
            dagnam_record.uninstall()
        counts[name] = dict(installed.writer.counts)
    return counts


def unusable(directory: Path) -> Dict[str, Any]:
    """A first install into a directory that cannot be used: one warning, then silence."""
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.write_text("a file where the directory should be", encoding="utf-8")
    dagnam_record.install(dir=str(directory))
    for number in range(20):
        dagnam_record.record_call(provider="other", request={"n": number}, response={"r": 1})
    dagnam_record.uninstall()
    return {}


SCENARIOS: Dict[str, Callable[[Path], Dict[str, Any]]] = {
    "state": state,
    "network": network,
    "stuck": stuck,
    "failing": failing,
    "unusable": unusable,
}


def main() -> None:
    report = SCENARIOS[sys.argv[1]](WORK / "traces")
    (WORK / "report.json").write_text(json.dumps(report), encoding="utf-8")


if __name__ == "__main__":
    main()
