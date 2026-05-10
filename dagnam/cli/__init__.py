"""CLI entry point facade."""

from __future__ import annotations

from dagnam.cli.common import _dir_size, _error, _human_size
from dagnam.cli.main import _build_parser, main

__all__ = ["_build_parser", "_dir_size", "_error", "_human_size", "main"]
