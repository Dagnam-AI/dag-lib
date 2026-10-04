"""Shared fixtures for the CLI subcommand tests."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from tests.audit._platform import Clock, FakePlatform, serve_chat, teacher
from tests.cli._audit_run import SDK_CONTRACT, Build, write_prepared

from dagnam.audit.orchestrate import run_audit_held
from dagnam.cli import main as cli_main

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch, RequestsMocker


@pytest.fixture
def run_cli(monkeypatch: PytestMonkeyPatch):
    """Set sys.argv and invoke main(); returns its exit code - use capsys for output."""

    def _run(argv: list[str]) -> int:
        monkeypatch.setattr("sys.argv", ["dagnam", *argv])
        return cli_main()

    return _run


@pytest.fixture
def prepared_dir(tmp_path: Path) -> Path:
    """A scanned audit directory for the ``audit run`` tests: two derived workloads."""
    return write_prepared(tmp_path / "audit")


@pytest.fixture
def build(monkeypatch: PytestMonkeyPatch) -> Build:
    """What the platform's ``/health/build`` answers; the SDK's own contract is pinned."""
    monkeypatch.setattr("dagnam.audit.preflight.version", lambda _distribution: SDK_CONTRACT)
    return Build()


@pytest.fixture
def platform(
    monkeypatch: PytestMonkeyPatch, requests_mock: RequestsMocker, build: Build
) -> FakePlatform:
    """The fake platform behind ``client_from_env``, with the waits on a fake clock."""
    fake = FakePlatform()
    clock = Clock()
    monkeypatch.setattr(fake, "get_platform_build", build)
    monkeypatch.setattr("dagnam.cli.audit_run.client_from_env", lambda: fake)
    monkeypatch.setattr(
        "dagnam.audit.orchestrate.run_audit_held",
        partial(run_audit_held, sleep=clock.sleep, now=clock.now),
    )
    serve_chat(requests_mock, teacher)
    return fake


@pytest.fixture
def tty(monkeypatch: PytestMonkeyPatch) -> None:
    """Stdin is a terminal, so a run asks instead of refusing."""
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
