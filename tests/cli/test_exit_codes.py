"""Subprocess-level CLI exit-code regression tests."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize("args", [["dataset"], ["cache"], ["projects"], ["training"]])
def test_missing_subcommand_exits_two(args: list[str]) -> None:
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "dagnam.cli.main", *args],
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 2
    assert "required" in completed.stderr
