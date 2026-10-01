"""System-prompt template normalization: mask the variable parts, hash what is left.

Two calls of the same workload differ only in the values interpolated into the
system prompt (ids, numbers, dates and times -- ISO or "Sunday, September 27"
--, quoted context, ``{{placeholders}}``, the values of a one-line ``{...}``).
Masking those and collapsing whitespace yields the *template*; its blake2b
digest is the workload id. Everything here is a pure function of its input,
so the id is identical across processes and machines.
"""

from __future__ import annotations

from collections.abc import Callable
import functools
import hashlib
import re

UNSTRUCTURED = "unstructured"
"""The template hash of a trace without a system prompt."""

_HASH_HEX_CHARS = 16
_QUOTED_MIN_CHARS = 25  # a quoted span "over 24 characters" is context, not template

_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?"
    r"|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b\.?"
)
_BRACE_KEY = re.compile(r"""(?:^|,)\s*["']?([A-Za-z_]\w*)["']?\s*:""")
_FIELD_LIST = re.compile(r"\s*[A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*\s*")


def _brace(match: re.Match[str]) -> str:
    """A one-line ``{...}``: a schema or field list keeps its keys, anything else is a variable.

    ``{"product": str, "sentiment": str}`` and ``{invoice_total, currency}`` name
    different tasks and must hash apart; an interpolated ``{'name': 'Maria'}``
    keeps only its key, so the value never splits the workload.
    """
    body = match.group(1)
    keys = _BRACE_KEY.findall(body) or (body.split(",") if _FIELD_LIST.fullmatch(body) else [])
    return "{" + ",".join(key.strip() for key in keys) + "}" if keys else "{VAR}"


# Order matters within a pass: placeholders are masked whole before their
# contents could match anything else; URLs before emails (a URL may contain
# ``@``); UUIDs and dates before bare numbers; a month name only beside a
# number, so the prose "may" survives; quoted spans after every rule that can
# change their length; whitespace last.
_RULES: tuple[tuple[re.Pattern[str], str | Callable[[re.Match[str]], str]], ...] = (
    (re.compile(r"\{\{[^{}]*\}\}"), "{{VAR}}"),
    (re.compile(r"\{([^{}\n]*)\}"), _brace),
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
    (
        re.compile(
            r"\b(?:(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day\b|(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)(?=,))",
            re.I,
        ),
        "<DAY>",
    ),
    (re.compile(rf"\b{_MONTH}(?=,?\s*\d)", re.I), "<MONTH>"),
    (re.compile(rf"(\d(?:st|nd|rd|th)?\s+(?:of\s+)?){_MONTH}", re.I), r"\1<MONTH>"),
    (
        re.compile(r"\b\d{1,2}(?:(?::\d{2}){1,2}(?:\s*[ap]\.?m\b)?|\s*[ap]\.?m\b)", re.I),
        "<TIME>",
    ),
    (re.compile(r"\d+(?:[.,]\d+)*(?:(?:st|nd|rd|th)\b)?"), "<NUM>"),
    (re.compile(rf'"[^"\n]{{{_QUOTED_MIN_CHARS},}}"'), "<QUOTED>"),
    (re.compile(r"\s+"), " "),
)


def _mask_once(text: str) -> str:
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text.strip()


@functools.lru_cache(maxsize=4_096)
def normalize_template(system: str) -> str:
    """Mask UUIDs, numbers, emails, URLs, dates and times, long quoted spans and placeholders.

    Runs the rules to a fixed point: one pass can leave a new match behind
    (two short quoted spans that a masked number between them joins into one
    long span), and the template must be idempotent to be a stable id. Every
    change consumes a digit, quote, ``@``, brace, name of a day or month, or
    whitespace, and no rule puts one back, so the loop ends. An agent sends the
    same system prompt on every call, so results are cached: the regexes run
    once per distinct prompt, not once per call.
    """
    masked = _mask_once(system)
    while (again := _mask_once(masked)) != masked:
        masked = again
    return masked


def template_digest(template: str) -> str:
    """16 hex chars of blake2b over an already-normalized template; :data:`UNSTRUCTURED` for none."""
    if not template:
        return UNSTRUCTURED
    return hashlib.blake2b(template.encode("utf-8")).hexdigest()[:_HASH_HEX_CHARS]


def template_hash(system: str | None) -> str:
    """The workload id: 16 hex chars of blake2b over the normalized template.

    ``None`` or a prompt that normalizes to nothing hashes to
    :data:`UNSTRUCTURED`, the bucket for traces without a system prompt.
    """
    return template_digest(normalize_template(system or ""))
