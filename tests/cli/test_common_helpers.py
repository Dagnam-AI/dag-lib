"""Tests for the shared CLI helpers added for version/account commands."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from dagnam.cli.common import format_ascii_art, mask_key, resolve_version

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


def test_ascii_art_stems_lines_to_available_width() -> None:
    art = format_ascii_art(columns=80)

    assert art
    assert max(len(line) for line in art.splitlines()) <= 80


def test_ascii_art_prefers_live_terminal_over_stale_columns_env(
    monkeypatch: MonkeyPatch,
) -> None:
    """Resizing the window must widen the banner even if COLUMNS is stale.

    Shells frequently export a COLUMNS value that is not updated on resize.
    The banner should track the live terminal width, not the stale env var.
    """
    # Simulate a shell that exported a narrow COLUMNS before the user resized.
    monkeypatch.setenv("COLUMNS", "20")
    # Simulate the live terminal now being wide.
    monkeypatch.setattr(os, "get_terminal_size", lambda *_: os.terminal_size((100, 24)))

    art = format_ascii_art()

    assert max(len(line) for line in art.splitlines()) > 20


def test_ascii_art_falls_back_to_shutil_without_a_tty(monkeypatch: MonkeyPatch) -> None:
    """When no terminal is attached, fall back to shutil's width detection."""

    def _no_tty(*_: object) -> os.terminal_size:
        raise OSError("not a tty")

    monkeypatch.setattr(os, "get_terminal_size", _no_tty)
    monkeypatch.setenv("COLUMNS", "57")

    art = format_ascii_art()

    assert max(len(line) for line in art.splitlines()) <= 57


class TestMaskKey:
    def test_long_key_shows_prefix_and_suffix(self) -> None:
        assert mask_key("sk_abcdefghijklmnop") == "sk_abc...mnop"

    def test_short_key_fully_masked(self) -> None:
        assert mask_key("sk_123") == "******"

    def test_boundary_ten_chars_fully_masked(self) -> None:
        assert mask_key("1234567890") == "*" * 10


class TestResolveVersion:
    def test_returns_nonempty_string(self) -> None:
        version = resolve_version()
        assert isinstance(version, str)
        assert version

    def test_falls_back_to_dunder_version_when_not_installed(self) -> None:
        import dagnam.cli.common as common_mod

        def _raise(_name: str) -> str:
            from importlib.metadata import PackageNotFoundError

            raise PackageNotFoundError(_name)

        import importlib.metadata as md

        original = md.version
        md.version = _raise  # type: ignore[assignment]
        try:
            from dagnam import __version__

            assert common_mod.resolve_version() == __version__
        finally:
            md.version = original  # type: ignore[assignment]
