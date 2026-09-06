"""Neutralise terminal control characters in untrusted text.

Lives in ``_core`` so every layer that handles server- or export-supplied
strings (the CLI renderers, the audit trace readers) shares one sanitizer.
"""

from __future__ import annotations

import re

# C0 controls (except tab/newline), DEL, and C1 controls. This range includes
# ESC (0x1b) and CSI (0x9b), so stripping it neutralises every ANSI/OSC escape
# sequence — the mechanism behind terminal-title spoofing, OSC 8 hyperlink
# forgery, and OSC 52 clipboard writes from a hostile server-supplied string.
_TERMINAL_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def sanitize_terminal_text(value: object) -> str:
    """Strip terminal control/escape characters from untrusted text.

    Every human-mode renderer prints server-returned strings (dataset/hub names
    and descriptions, project titles, error bodies) straight to the TTY, and a
    trace export is untrusted input in the same way. Left raw, an embedded
    escape sequence could spoof output, retitle the terminal, forge a
    hyperlink, or write the user's clipboard. Removing the control chars
    (which include ESC and CSI) makes the text inert while leaving all
    printable content — and tabs/newlines — intact.
    """
    return _TERMINAL_CONTROL_CHARS.sub("", str(value))
