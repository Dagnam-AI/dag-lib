"""CLI import smoke tests for environments with broken optional runtimes."""

from __future__ import annotations

import subprocess
import sys


def test_cli_help_import_does_not_require_polars_runtime() -> None:
    script = r"""
import importlib.abc
import sys


class BlockPolars(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "polars" or fullname.startswith("_polars_runtime"):
            raise ImportError(f"blocked {fullname}")
        return None


sys.meta_path.insert(0, BlockPolars())
from dagnam.cli import main

try:
    main(["--help"])
except SystemExit as exc:
    raise SystemExit(exc.code)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Official CLI for Dagnam.AI" in result.stdout
