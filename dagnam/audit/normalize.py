"""System-prompt template normalization: mask the variable parts, hash what is left.

Two calls of the same workload differ only in the values interpolated into the
system prompt (ids, numbers, dates, quoted context, ``{{placeholders}}``).
Masking those and collapsing whitespace yields the *template*; its blake2b
digest is the workload id. Everything here is a pure function of its input,
so the id is identical across processes and machines.
"""

from __future__ import annotations

import hashlib
import re

UNSTRUCTURED = "unstructured"
"""The template hash of a trace without a system prompt."""

_HASH_HEX_CHARS = 16
_QUOTED_MIN_CHARS = 25  # a quoted span "over 24 characters" is context, not template

# Order matters within a pass: placeholders are masked whole before their
# contents could match anything else; URLs before emails (a URL may contain
# ``@``); UUIDs and dates before bare numbers; quoted spans after every rule
# that can change their length; whitespace last.
_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\{\{[^{}]*\}\}"), "{{VAR}}"),
    (re.compile(r"\{[^{}\n]*\}"), "{VAR}"),
    (re.compile(r"https?://\S+"), "<URL>"),
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "<EMAIL>"),
    (
        re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I),
        "<UUID>",
    ),
    (
        re.compile(
            r"\b\d{4}-\d{2}-\d{2}"
            r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?\b"
        ),
        "<DATE>",
    ),
    (re.compile(r"\d+(?:[.,]\d+)*"), "<NUM>"),
    (re.compile(rf'"[^"\n]{{{_QUOTED_MIN_CHARS},}}"'), "<QUOTED>"),
    (re.compile(r"\s+"), " "),
)


def _mask_once(text: str) -> str:
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text.strip()


def normalize_template(system: str) -> str:
    """Mask UUIDs, numbers, emails, URLs, ISO dates, long quoted spans and placeholders.

    Runs the rules to a fixed point: one pass can leave a new match behind
    (two short quoted spans that a masked number between them joins into one
    long span), and the template must be idempotent to be a stable id. Every
    change consumes a digit, quote, ``@`` or brace, so the loop ends.
    """
    masked = _mask_once(system)
    while (again := _mask_once(masked)) != masked:
        masked = again
    return masked


def template_hash(system: str | None) -> str:
    """The workload id: 16 hex chars of blake2b over the normalized template.

    ``None`` or a prompt that normalizes to nothing hashes to
    :data:`UNSTRUCTURED`, the bucket for traces without a system prompt.
    """
    template = normalize_template(system or "")
    if not template:
        return UNSTRUCTURED
    return hashlib.blake2b(template.encode("utf-8")).hexdigest()[:_HASH_HEX_CHARS]
