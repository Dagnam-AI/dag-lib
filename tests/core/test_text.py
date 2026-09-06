"""The shared terminal sanitizer: control and escape bytes out, printable text intact."""

from __future__ import annotations

from dagnam._core.text import sanitize_terminal_text


def test_strips_escape_sequences() -> None:
    # OSC 8 hyperlink spoof + a CSI sequence + a bare ESC, all neutralised.
    hostile = "\x1b]8;;http://evil.example\x07Model\x1b]8;;\x07\x1b[31m\x9b2J"
    cleaned = sanitize_terminal_text(hostile)
    assert "\x1b" not in cleaned
    assert "\x9b" not in cleaned
    assert "\x07" not in cleaned
    assert "Model" in cleaned  # printable content preserved


def test_keeps_tabs_and_newlines_and_stringifies() -> None:
    assert sanitize_terminal_text("a\tb\nc") == "a\tb\nc"
    assert sanitize_terminal_text(12) == "12"
