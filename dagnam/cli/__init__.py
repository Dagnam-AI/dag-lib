"""CLI entry point facade."""

from __future__ import annotations

from dagnam.cli.common import dir_size, error, human_size
from dagnam.cli.main import build_parser, main

__all__ = ["build_parser", "dir_size", "error", "human_size", "main"]
