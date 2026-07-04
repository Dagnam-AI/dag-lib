"""Tests for the shared CLI helpers added for version/account commands."""

from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING

import pytest

from dagnam.cli import common
from dagnam.cli.common import (
    format_ascii_art,
    format_local,
    mask_key,
    parse_api_datetime,
    resolve_version,
)

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch
    from tests.typing_helpers import PytestMonkeyPatch, StrCapture


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


def test_ascii_art_skips_stream_that_raises_then_uses_next(monkeypatch: MonkeyPatch) -> None:
    """The first stream raising must `continue` to the next attached stream."""

    def _only_stderr_has_size(fileno: int) -> os.terminal_size:
        # stdout's fileno raises; stderr's fileno succeeds.
        if fileno == 1:
            raise OSError("stdout not a tty")
        return os.terminal_size((100, 24))

    class _Stream:
        def __init__(self, fd: int) -> None:
            self._fd = fd

        def fileno(self) -> int:
            return self._fd

    monkeypatch.setattr("sys.__stdout__", _Stream(1))
    monkeypatch.setattr("sys.__stderr__", _Stream(2))
    monkeypatch.setattr(os, "get_terminal_size", _only_stderr_has_size)

    art = format_ascii_art()

    assert max(len(line) for line in art.splitlines()) > 20


def test_ascii_art_skips_none_stream(monkeypatch: MonkeyPatch) -> None:
    """A ``None`` stream (detached stdout) must be skipped without error."""
    monkeypatch.setattr("sys.__stdout__", None)
    monkeypatch.setattr(os, "get_terminal_size", lambda *_: os.terminal_size((90, 24)))

    art = format_ascii_art()

    assert art


class TestPrintNextStep:
    def test_writes_hint_to_stderr(self, capsys: StrCapture) -> None:
        common.print_next_step("dagnam projects list")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "Next: dagnam projects list" in captured.err


class TestRunCommand:
    def test_returns_zero_on_success(self) -> None:
        args = argparse.Namespace(func=lambda _a: None, debug=False)
        assert common.run_command(args) == 0

    def test_dagnam_error_prints_clean_message(self, capsys: StrCapture) -> None:
        from dagnam._core.exceptions import DagnamError

        def boom(_a: argparse.Namespace) -> None:
            raise DagnamError("token expired")

        args = argparse.Namespace(func=boom, debug=False)
        assert common.run_command(args) == 1
        err = capsys.readouterr().err
        assert "Error: token expired" in err
        assert common.DOCS_URL in err
        assert "Traceback" not in err

    def test_unexpected_error_is_generic(self, capsys: StrCapture) -> None:
        def boom(_a: argparse.Namespace) -> None:
            raise ValueError("internal detail")

        args = argparse.Namespace(func=boom, debug=False)
        assert common.run_command(args) == 1
        err = capsys.readouterr().err
        assert "unexpected error" in err
        assert "--debug" in err
        assert "internal detail" not in err

    def test_debug_flag_reraises(self) -> None:
        def boom(_a: argparse.Namespace) -> None:
            raise ValueError("internal detail")

        args = argparse.Namespace(func=boom, debug=True)
        with pytest.raises(ValueError, match="internal detail"):
            common.run_command(args)

    def test_debug_env_reraises(self, monkeypatch: PytestMonkeyPatch) -> None:
        monkeypatch.setenv("DAGNAM_DEBUG", "1")

        def boom(_a: argparse.Namespace) -> None:
            raise ValueError("internal detail")

        args = argparse.Namespace(func=boom, debug=False)
        with pytest.raises(ValueError, match="internal detail"):
            common.run_command(args)

    def test_keyboard_interrupt_returns_130(self) -> None:
        def boom(_a: argparse.Namespace) -> None:
            raise KeyboardInterrupt

        args = argparse.Namespace(func=boom, debug=False)
        assert common.run_command(args) == 130

    def test_broken_pipe_exits_quietly(self, capsys: StrCapture) -> None:
        # Under pytest capture, stdout has no real fileno; the suppress() arm
        # swallows that and the command still exits 1 with no error spew.
        def boom(_a: argparse.Namespace) -> None:
            raise BrokenPipeError

        args = argparse.Namespace(func=boom, debug=False)
        assert common.run_command(args) == 1
        assert capsys.readouterr().err == ""

    def test_broken_pipe_redirects_stdout_to_devnull(self, monkeypatch: PytestMonkeyPatch) -> None:
        # With a fileno-capable stdout, the dead pipe's fd is re-pointed at
        # devnull so interpreter shutdown does not raise while flushing.
        sink_fd = os.open(os.devnull, os.O_WRONLY)

        class _FakeStdout:
            def fileno(self) -> int:
                return sink_fd

        monkeypatch.setattr("sys.stdout", _FakeStdout())

        def boom(_a: argparse.Namespace) -> None:
            raise BrokenPipeError

        args = argparse.Namespace(func=boom, debug=False)
        try:
            assert common.run_command(args) == 1
        finally:
            os.close(sink_fd)


class TestErrorHelper:
    def test_prints_unified_frame_and_exits(self, capsys: StrCapture) -> None:
        with pytest.raises(SystemExit) as excinfo:
            common.error("bad thing happened")
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "Error: bad thing happened" in err
        assert common.DOCS_URL in err

    def test_hint_renders_try_section(self, capsys: StrCapture) -> None:
        with pytest.raises(SystemExit):
            common.error("bad thing happened", hint="Fix the input and retry.")
        err = capsys.readouterr().err
        assert "Try:" in err
        assert "Fix the input and retry." in err


class TestParseApiDatetime:
    def test_naive_string_is_assumed_utc(self) -> None:
        parsed = parse_api_datetime("2026-05-11T03:01:26")
        assert parsed.tzinfo is not None
        offset = parsed.utcoffset()
        assert offset is not None
        assert offset.total_seconds() == 0

    def test_aware_string_offset_is_preserved(self) -> None:
        parsed = parse_api_datetime("2026-05-11T03:01:26+05:30")
        offset = parsed.utcoffset()
        assert offset is not None
        assert offset.total_seconds() == 5.5 * 3600

    def test_zulu_suffix_is_utc(self) -> None:
        parsed = parse_api_datetime("2026-05-11T03:01:26Z")
        offset = parsed.utcoffset()
        assert offset is not None
        assert offset.total_seconds() == 0


class TestFormatLocal:
    def test_none_renders_dash(self) -> None:
        assert format_local(None) == "-"

    def test_empty_string_renders_dash(self) -> None:
        assert format_local("") == "-"

    def test_non_string_truthy_renders_dash(self) -> None:
        assert format_local(123) == "-"

    def test_unparseable_string_falls_back_to_date_portion(self) -> None:
        assert format_local("not-a-dateThh:mm") == "not-a-date"

    def test_utc_timestamp_renders_as_local_date(self) -> None:
        # Compare against the same UTC->local conversion the helper performs so
        # the assertion holds regardless of the test machine's timezone.
        expected = parse_api_datetime("2026-05-11T03:01:26").astimezone().strftime("%Y-%m-%d")
        assert format_local("2026-05-11T03:01:26") == expected

    def test_local_conversion_matches_utc_to_local(self) -> None:
        # Verify the helper converts UTC->local rather than truncating the raw
        # UTC date. Compared against the helper's own conversion to stay
        # timezone-portable across CI machines.
        raw = "2026-05-11T23:30:00"
        as_utc = parse_api_datetime(raw)
        offset = as_utc.utcoffset()
        assert offset is not None
        assert offset.total_seconds() == 0
        assert format_local(raw) == as_utc.astimezone().strftime("%Y-%m-%d")
