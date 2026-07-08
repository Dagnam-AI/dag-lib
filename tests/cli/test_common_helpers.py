"""Tests for the shared CLI helpers added for version/account commands."""

from __future__ import annotations

import argparse
import io
import os
import sys
from typing import TYPE_CHECKING, override

import pytest

from dagnam.cli import common
from dagnam.cli.common import (
    confirm_or_abort,
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


def test_ascii_art_color_paints_body_and_shade_runs() -> None:
    art = format_ascii_art(columns=200, color=True)
    # Letter bodies use the light-mode brand red, shading the dark-mode red.
    assert "\x1b[38;2;255;79;79m█" in art
    assert "\x1b[38;2;202;3;3m░" in art
    for line in art.splitlines():
        assert line.endswith("\x1b[0m")


def test_ascii_art_color_defaults_off_without_a_tty(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    art = format_ascii_art(columns=200)
    assert "\x1b[" not in art


def test_ascii_art_color_defaults_on_when_forced(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("FORCE_COLOR", "1")
    art = format_ascii_art(columns=200)
    assert "\x1b[38;2;255;79;79m" in art


def test_colorize_banner_paints_fallback_glyphs_in_body_red() -> None:
    # The cp1252 fallback banner is plain ASCII; its glyphs get the body red.
    painted = common._colorize_banner(common.DAGNAM_ASCII_FALLBACK_ART)  # pyright: ignore[reportPrivateUsage]
    assert "\x1b[38;2;255;79;79m" in painted
    assert "\x1b[38;2;202;3;3m" not in painted  # no shade blocks in the fallback
    assert "█" not in painted  # stays pure ASCII apart from the escapes


def test_blend_color_reaches_target_at_the_band_center() -> None:
    blend = common._blend_color  # pyright: ignore[reportPrivateUsage]
    far = blend((202, 3, 3), (255, 79, 79), 0, band=60.0)
    near = blend((202, 3, 3), (255, 79, 79), 60, band=60.0)
    static = blend((202, 3, 3), (255, 79, 79), 0, band=None)
    assert far == "\x1b[38;2;202;3;3m"  # beyond the falloff: flat base color
    assert near == "\x1b[38;2;255;79;79m"  # at the center: full target color
    assert static == far  # no band means the flat base color


class TestPrintVersionBanner:
    def test_static_when_animation_disabled(self, capsys: StrCapture) -> None:
        common.print_version_banner(animate=False)
        out = capsys.readouterr().out
        assert format_ascii_art() in out
        assert f"dagnam {resolve_version()}" in out
        assert "\x1b[?25l" not in out

    def test_default_gate_is_static_without_a_tty(
        self, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        monkeypatch.delenv("FORCE_COLOR", raising=False)
        common.print_version_banner()
        assert "\x1b[?25l" not in capsys.readouterr().out

    def test_forced_color_without_a_tty_stays_static_but_colored(
        self, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        # FORCE_COLOR turns styling on, but in-place redraw still needs a TTY.
        monkeypatch.setenv("FORCE_COLOR", "1")
        common.print_version_banner()
        out = capsys.readouterr().out
        assert "\x1b[?25l" not in out
        assert "\x1b[38;2;255;79;79m" in out

    def test_default_gate_animates_on_a_color_tty(self, monkeypatch: PytestMonkeyPatch) -> None:
        class _Tty(io.StringIO):
            encoding = "utf-8"  # keep the glyph banner (StringIO declares none)

            @override
            def isatty(self) -> bool:
                return True

        stream = _Tty()
        monkeypatch.setenv("FORCE_COLOR", "1")
        monkeypatch.setattr(sys, "stdout", stream)
        common.print_version_banner(sleep=lambda _: None)
        assert "\x1b[?25l" in stream.getvalue()

    def test_sweep_loops_redraw_in_place_then_settle_static(self, capsys: StrCapture) -> None:
        delays: list[float] = []
        common.print_version_banner(animate=True, sleep=delays.append)
        out = capsys.readouterr().out
        frames = common._BANNER_SWEEP_LOOPS * common._BANNER_SWEEP_FRAMES  # pyright: ignore[reportPrivateUsage]
        lines = format_ascii_art(color=False).count("\n") + 1
        assert out.count("\x1b[?25l") == 1  # cursor hidden once...
        assert out.count("\x1b[?25h") == 1  # ...and restored once
        assert len(delays) == frames
        assert out.count(f"\x1b[{lines}F") == frames  # every frame redraws in place
        assert format_ascii_art(color=True) in out  # ends settled on the static banner
        assert out.rstrip().endswith(f"dagnam {resolve_version()}")

    def test_fallback_banner_animates_in_brand_color_too(
        self, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        # A legacy console gets the plain-ASCII art; it sweeps the same way.
        monkeypatch.setattr(common, "_stream_can_encode", lambda *_: False)
        common.print_version_banner(animate=True, sleep=lambda _: None)
        out = capsys.readouterr().out
        assert "\x1b[?25l" in out
        assert "\x1b[38;2;255;79;79m" in out
        assert "█" not in out


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


class TestConfirmOrAbort:
    def test_assume_yes_skips_prompt_entirely(self, monkeypatch: PytestMonkeyPatch) -> None:
        def _boom(_prompt: str = "") -> str:
            raise AssertionError("input() must not be called when assume_yes is set")

        monkeypatch.setattr("builtins.input", _boom)
        confirm_or_abort("Delete everything?", assume_yes=True)  # must not raise

    def test_typed_yes_confirms(self, monkeypatch: PytestMonkeyPatch, capsys: StrCapture) -> None:
        monkeypatch.setattr("builtins.input", lambda _prompt="": "yes")
        confirm_or_abort("Delete everything?", assume_yes=False)  # must not raise
        assert "Delete everything?" in capsys.readouterr().out

    def test_anything_else_aborts_with_exit_1(
        self, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
    ) -> None:
        monkeypatch.setattr("builtins.input", lambda _prompt="": "no")
        with pytest.raises(SystemExit) as exc_info:
            confirm_or_abort("Delete everything?", assume_yes=False)
        assert exc_info.value.code == 1
        assert "confirmation not received" in capsys.readouterr().err

    def test_blank_input_aborts(self, monkeypatch: PytestMonkeyPatch, capsys: StrCapture) -> None:
        monkeypatch.setattr("builtins.input", lambda _prompt="": "")
        with pytest.raises(SystemExit) as exc_info:
            confirm_or_abort("Delete everything?", assume_yes=False)
        assert exc_info.value.code == 1


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
