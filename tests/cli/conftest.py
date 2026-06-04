"""Shared fixtures for the CLI subcommand tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dagnam.cli import main as cli_main

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch


@pytest.fixture
def run_cli(monkeypatch: PytestMonkeyPatch):
    """Set sys.argv and invoke main(); returns nothing - use capsys for output."""

    def _run(argv: list[str]) -> None:
        monkeypatch.setattr("sys.argv", ["dagnam", *argv])
        cli_main()

    return _run
