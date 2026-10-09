"""``push_checkpoint`` against a real local HTTP server: what actually goes over the wire.

The mocked tests show what is handed to ``requests``; these show what a server sees. The server
drains the body in small slices and keeps only counts and a digest, like the platform's route,
so a checkpoint is never held whole on either side. Large files are sparse, so they cost no disk.
"""

from __future__ import annotations

from collections.abc import Iterator
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import tracemalloc
from typing import TYPE_CHECKING, override

import pytest

from dagnam._core import checkpoint_push
from dagnam._core.checkpoint_push import push_checkpoint
from dagnam._core.tar_stream import DirectoryTar

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch

SPARSE_BYTES = 256 * 1024 * 1024
PEAK_LIMIT = 16 * 1024 * 1024


class _Received:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.path = ""
        self.length = 0
        self.sha256 = ""


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.received: list[_Received] = []
        self.status = 201


class _Handler(BaseHTTPRequestHandler):
    server: _Server  # pyright: ignore[reportIncompatibleVariableOverride]

    def do_POST(self) -> None:
        got = _Received()
        got.headers = {key.lower(): value for key, value in self.headers.items()}
        got.path = self.path
        digest = hashlib.sha256()
        remaining = int(self.headers.get("Content-Length", "0"))
        while remaining:
            chunk = self.rfile.read(min(64 * 1024, remaining))
            if not chunk:
                break
            digest.update(chunk)
            got.length += len(chunk)
            remaining -= len(chunk)
        got.sha256 = digest.hexdigest()
        self.server.received.append(got)
        body = json.dumps({"checkpoint_id": "ck-live", "size_bytes": got.length}).encode()
        self.send_response(self.server.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @override
    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def server(monkeypatch: PytestMonkeyPatch) -> Iterator[_Server]:
    srv = _Server()
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    monkeypatch.delenv("DAGNAM_INTERNAL", raising=False)
    monkeypatch.setenv("DAGNAM_JOB_ID", "job-1")
    monkeypatch.setenv("DAGNAM_API_KEY", "run-token")
    monkeypatch.setenv("DAGNAM_API_URL", f"http://127.0.0.1:{srv.server_address[1]}")
    monkeypatch.setattr(checkpoint_push, "_stopped", False)
    yield srv
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def _noop_log(level: str, message: str) -> None:
    raise AssertionError(f"unexpected {level} log: {message}")


def _sparse(path: Path, size: int = SPARSE_BYTES) -> Path:
    with path.open("wb") as fh:
        fh.truncate(size)
    return path


def _zeros_digest(size: int) -> str:
    digest = hashlib.sha256()
    block = bytes(1024 * 1024)
    for _ in range(size // len(block)):
        digest.update(block)
    return digest.hexdigest()


def _digest_of(stream: DirectoryTar) -> str:
    digest = hashlib.sha256()
    for part in stream:
        digest.update(part)
    return digest.hexdigest()


def _peak_while(run: object) -> int:
    assert callable(run)
    tracemalloc.start()
    try:
        run()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_a_large_file_goes_out_with_a_length_and_without_being_loaded(
    tmp_path: Path, server: _Server
) -> None:
    big = _sparse(tmp_path / "huge.pth")
    saved: list[str | None] = []

    peak = _peak_while(lambda: saved.append(push_checkpoint(big, epoch=2, step=70, log=_noop_log)))

    assert saved == ["ck-live"]
    [got] = server.received
    assert got.headers["content-length"] == str(SPARSE_BYTES)
    assert "transfer-encoding" not in got.headers
    assert got.length == SPARSE_BYTES
    assert got.headers["authorization"] == "Bearer run-token"
    assert got.headers["content-type"] == "application/octet-stream"
    assert got.path == "/api/v1/training/jobs/job-1/checkpoints/push?epoch=2&step=70&name=huge.pth"
    assert got.sha256 == _zeros_digest(SPARSE_BYTES)
    assert peak < PEAK_LIMIT, f"peak {peak} bytes for a {SPARSE_BYTES} byte file"


def test_a_directory_with_a_large_file_streams_as_a_tar_without_being_loaded(
    tmp_path: Path, server: _Server
) -> None:
    root = tmp_path / "ckpt"
    (root / "shards").mkdir(parents=True)
    _sparse(root / "shards" / "w-00001.bin")
    (root / "config.json").write_text('{"a": 1}')
    expected = DirectoryTar(root)
    saved: list[str | None] = []

    peak = _peak_while(lambda: saved.append(push_checkpoint(root, epoch=1, step=5, log=_noop_log)))

    assert saved == ["ck-live"]
    [got] = server.received
    assert got.headers["content-length"] == str(len(expected))
    assert "transfer-encoding" not in got.headers
    assert got.length == len(expected)
    assert got.sha256 == _digest_of(expected)
    assert got.path.endswith("&name=ckpt.tar")
    assert peak < PEAK_LIMIT, f"peak {peak} bytes for a {SPARSE_BYTES} byte tree"


def test_a_server_refusal_is_skipped_after_the_body_was_sent(
    tmp_path: Path, server: _Server
) -> None:
    server.status = 413
    lines: list[str] = []

    saved = push_checkpoint(
        _sparse(tmp_path / "w.pth", 4096),
        epoch=1,
        step=1,
        log=lambda level, message: lines.append(f"{level}: {message}"),
    )

    assert saved is None
    assert len(server.received) == 1
    assert len(lines) == 1
    assert "HTTP 413" in lines[0]


def test_a_server_that_is_gone_is_a_logged_skip(tmp_path: Path, server: _Server) -> None:
    server.shutdown()
    server.server_close()
    lines: list[str] = []

    saved = push_checkpoint(
        _sparse(tmp_path / "w.pth", 4096),
        epoch=1,
        step=1,
        log=lambda level, message: lines.append(message),
    )

    assert saved is None
    assert len(lines) == 1
    assert "network error" in lines[0]
