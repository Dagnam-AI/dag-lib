"""Encoding-safe CLI banner.

``dagnam -v`` / ``dagnam -h`` print the branded ASCII-art banner to stdout via
argparse. On a legacy code page (cp1252 — the default Windows console, many CI
shells, Git Bash) the box-drawing glyphs (``█``/``░``) cannot be encoded, so a
naive write raises ``UnicodeEncodeError`` and crashes a command that must never
fail. The banner now upgrades the console to UTF-8 best-effort and otherwise
degrades to a plain-ASCII form, so these commands always succeed.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

from dagnam.cli.common import (
    DAGNAM_ASCII_ART,
    _stream_can_encode,
    configure_console_encoding,
    format_ascii_art,
    format_version_banner,
    resolve_version,
)

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Stream:
    """A minimal stdout-like object with a fixed declared ``encoding``."""

    def __init__(self, encoding: str | None) -> None:
        self.encoding = encoding


class _RecordingStream:
    """A stream that supports ``reconfigure`` and records the calls."""

    def __init__(self) -> None:
        self.encoding = "cp1252"
        self.calls: list[dict[str, object]] = []

    def reconfigure(self, **kwargs: object) -> None:
        self.calls.append(kwargs)
        encoding = kwargs.get("encoding")
        if isinstance(encoding, str):
            self.encoding = encoding


class _RaisingStream:
    """A stream whose ``reconfigure`` raises a given exception."""

    encoding = "cp1252"

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def reconfigure(self, **_kwargs: object) -> None:
        raise self._exc


class _Cp1252Console:
    """A writable console that enforces cp1252 on write, like a real one.

    Has no ``reconfigure`` method, so ``configure_console_encoding`` cannot
    upgrade it — exercising the ASCII-fallback path end to end.
    """

    encoding = "cp1252"

    def __init__(self) -> None:
        self.written: list[str] = []

    def write(self, text: str) -> int:
        text.encode(self.encoding)  # mimic a real cp1252 console: raises on bad glyphs
        self.written.append(text)
        return len(text)

    def flush(self) -> None:  # pragma: no cover - argparse may or may not flush
        pass


# ---------------------------------------------------------------------------
# _stream_can_encode
# ---------------------------------------------------------------------------


def test_stream_can_encode_true_for_utf8() -> None:
    assert _stream_can_encode(DAGNAM_ASCII_ART, _Stream("utf-8")) is True


def test_stream_can_encode_false_for_cp1252_glyphs() -> None:
    assert _stream_can_encode(DAGNAM_ASCII_ART, _Stream("cp1252")) is False


def test_stream_can_encode_none_encoding_defaults_to_ascii() -> None:
    # A missing/empty encoding is treated as ascii, which cannot hold the glyphs.
    assert _stream_can_encode(DAGNAM_ASCII_ART, _Stream(None)) is False
    # ...but plain ASCII text still encodes under the ascii default.
    assert _stream_can_encode("plain ascii", _Stream(None)) is True


def test_stream_can_encode_false_for_unknown_codec() -> None:
    # An unknown codec name raises LookupError, treated as not-encodable.
    assert _stream_can_encode("anything", _Stream("not-a-real-codec")) is False


# ---------------------------------------------------------------------------
# format_ascii_art / format_version_banner
# ---------------------------------------------------------------------------


def test_format_ascii_art_uses_glyphs_when_encodable(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdout", _Stream("utf-8"))
    art = format_ascii_art(columns=200)
    assert "█" in art


def test_format_ascii_art_falls_back_to_ascii_when_not_encodable(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdout", _Stream("cp1252"))
    art = format_ascii_art(columns=200)
    assert "█" not in art
    assert "|__/" in art  # the stick-letters fallback wordmark
    art.encode("cp1252")  # must not raise on a legacy code page


def test_format_version_banner_is_encoding_safe_on_cp1252(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdout", _Stream("cp1252"))
    banner = format_version_banner()
    assert f"dagnam {resolve_version()}" in banner
    banner.encode("cp1252")  # must not raise


# ---------------------------------------------------------------------------
# configure_console_encoding
# ---------------------------------------------------------------------------


def test_configure_console_encoding_reconfigures_and_skips_unsupported(
    monkeypatch: MonkeyPatch,
) -> None:
    recording = _RecordingStream()
    no_reconfigure = _Stream("cp1252")  # has no reconfigure attribute
    monkeypatch.setattr(sys, "stdout", recording)
    monkeypatch.setattr(sys, "stderr", no_reconfigure)

    configure_console_encoding()

    # Supported stream upgraded to UTF-8; unsupported one skipped without error.
    assert recording.calls == [{"encoding": "utf-8"}]
    assert recording.encoding == "utf-8"


def test_configure_console_encoding_swallows_reconfigure_errors(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdout", _RaisingStream(OSError("locked")))
    monkeypatch.setattr(sys, "stderr", _RaisingStream(ValueError("bad mode")))

    # Both exception types are swallowed; no crash.
    configure_console_encoding()


# ---------------------------------------------------------------------------
# End-to-end: -v / -h must never crash on a cp1252 console
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["-v", "--version", "-h", "--help"])
def test_cli_flag_does_not_crash_on_cp1252_console(flag: str, monkeypatch: MonkeyPatch) -> None:
    from dagnam.cli.main import main

    console = _Cp1252Console()
    monkeypatch.setattr(sys, "stdout", console)
    monkeypatch.setattr(sys, "stderr", _Cp1252Console())

    with pytest.raises(SystemExit) as exc:
        main([flag])

    assert exc.value.code == 0
    output = "".join(console.written)
    # -v/--version show the banner (stick-letters wordmark in fallback art);
    # -h/--help show compact help which mentions Dagnam.AI in the description.
    if flag in ("-v", "--version"):
        assert "|__/" in output
    else:
        assert "Dagnam.AI" in output
    output.encode("cp1252")  # the whole emitted output must survive a cp1252 write
