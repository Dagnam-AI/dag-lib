"""Output-structure classes: what shape a workload's responses take.

The class decides which student can replace the teacher (a label classifier,
a JSON extractor, ...) and which quality floor applies. It is computed over a
sample of responses with the rules of the design, every number coming from
:mod:`dagnam.audit.thresholds`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from enum import StrEnum
import json
import math
import re
import statistics
from typing import Any, Literal

from dagnam.audit.readers.messages import effective_response
from dagnam.audit.thresholds import (
    ENUM_MAX_DISTINCT,
    ENUM_MAX_MEDIAN_TOKENS,
    JSON_KEY_STABILITY,
    JSON_OBJECT_SHARE,
    SPAN_MAX_MEDIAN_TOKENS,
    SPAN_MIN_DISTINCT_RATIO,
)

_WHITESPACE = re.compile(r"\s+")
_NO_SPACE = (
    "\u0e00-\u0eff\u1000-\u109f\u1780-\u17ff\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
)
"""Thai, Lao, Myanmar, Khmer, kana and CJK ideographs: scripts written without spaces."""
_NO_SPACE_RUN = re.compile(f"[{_NO_SPACE}]+")
_SCRIPT_RATES = {
    "cjk": ("\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff", 0.6),
    "kana": ("\u3040-\u30ff", 0.7),
    "thai": ("\u0e00-\u0e7f", 0.6),
    "hangul": ("\uac00-\ud7af", 0.9),
    "devanagari": ("\u0900-\u097f", 1.2),
}
"""Measured Qwen2.5 token rates per script character, including vowel/combining marks."""
_SCRIPT_CHARS = "".join(chars for chars, _ in _SCRIPT_RATES.values())
_SCRIPT_PIECES = "|".join(f" ?(?P<{name}>[{chars}]+)" for name, (chars, _) in _SCRIPT_RATES.items())
_STUDENT_PIECE = re.compile(
    rf"{_SCRIPT_PIECES}|[{_NO_SPACE}]"
    rf"|(?:[^\r\n\w]|_)?(?P<word>[^\W\d_{_NO_SPACE}{_SCRIPT_CHARS}]+)"
    r"|\d"
    r"| ?(?P<symbols>(?:[^\s\w]|_)+)[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)
"""Qwen2's pre-tokenizer, with ``[^\\W\\d_]`` for its letter class: a character of a no-space
script (counted alone, not in runs), a word with the space or symbol before it, a digit, a
symbol run with its newlines, a newline run, or a whitespace run."""
_WORD_LETTERS = 8
"""An ASCII word up to this long is one token: English's common words are one entry each."""
_LETTERS_PER_EXTRA_TOKEN = 4
"""Each further four letters of a longer ASCII word (an identifier, a rare word) is a token."""
_NON_ASCII_LETTERS_PER_TOKEN = 3
"""A word with accented or non-Latin letters (German, Russian) takes a token per three."""
_SYMBOLS_PER_TOKEN = 2
"""A symbol run is a token per two: the vocabulary merges ``{"``, ``":``, ``),`` and the like."""
_CALL_KEYS = frozenset({"arguments", "name"})


class StructureClass(StrEnum):
    """The four response shapes the audit tells apart."""

    ENUM_LABEL = "enum_label"
    JSON_OBJECT = "json_object"
    SHORT_SPAN = "short_span"
    FREE_TEXT = "free_text"


def normalize_response(response: str) -> str:
    """Case-fold and collapse whitespace so ``Billing`` and ``billing `` are one output."""
    return _WHITESPACE.sub(" ", response).strip().casefold()


def token_count(text: str) -> int:
    """Words, with a run of Chinese, Japanese or Thai characters counted per two characters.

    Those scripts write no spaces, so a whitespace split reads a 60-character
    Chinese reply as one "token" and free text as a short span (N8). Two
    characters a token keeps a two-character Chinese label at one token and
    that reply at thirty.
    """
    count = 0
    for word in text.split():
        runs = _NO_SPACE_RUN.findall(word)
        count += len(_NO_SPACE_RUN.sub(" ", word).split())
        count += sum((len(run) + 1) // 2 for run in runs)
    return count


def student_tokens(text: str) -> int:
    """Estimate Qwen2.5 tokens without installing or downloading its tokenizer.

    Use Qwen's pre-tokenizer pieces with measured per-script character rates,
    common ASCII words counted once, longer words per four extra letters,
    other words per three letters, and symbol runs per two characters.
    Round the total up once, preserving fractional script rates.

    This is a heuristic, not a token limit guarantee. Review-corpus estimates
    for English and structured text are about 0.9-1.15x real tokens. Script
    calibration corrects large CJK/Thai over-counts and Hangul/Devanagari
    under-counts. Latin-script languages still differ: Italian, Dutch and
    Indonesian measured about 0.7x; base64 and emoji can be about 0.5-0.6x.
    Vietnamese measured 1.22x. Script weights cannot distinguish languages
    sharing an alphabet, and unusual vocabulary can differ in any script.

    The recipe drops over-budget rows using its real tokenizer. Under-counting
    can therefore admit a workload whose every row is dropped at train time,
    wasting model startup time and credits. Over-counting can reject rows that
    fit. Recalibrate if the student tokenizer changes.
    """
    count = 0.0
    for piece in _STUDENT_PIECE.finditer(text):
        word, symbols = piece.group("word"), piece.group("symbols")
        if piece.lastgroup in _SCRIPT_RATES:
            count += len(piece.group(piece.lastgroup)) * _SCRIPT_RATES[piece.lastgroup][1]
        elif word is not None and not word.isascii():
            count += math.ceil(len(word) / _NON_ASCII_LETTERS_PER_TOKEN)
        elif word is not None:
            count += 1 + max(0, math.ceil((len(word) - _WORD_LETTERS) / _LETTERS_PER_EXTRA_TOKEN))
        elif symbols is not None:
            count += math.ceil(len(symbols) / _SYMBOLS_PER_TOKEN)
        else:
            count += 1
    return math.ceil(count)


def output_shape(
    response: str, tool_calls: Sequence[dict[str, Any]]
) -> Literal["json", "short", "long"]:
    """One response's shape: a JSON object, a short answer, or longer prose.

    Traces without a system prompt have no template to group them by, so the
    shape of each answer does it (spec D5's per-class unstructured buckets): a
    label, an extraction and a drafted reply never share a workload. Whether a
    short bucket is labels or spans is still the group's :func:`classify_outputs`.
    """
    text = effective_response(response, tool_calls)
    if _json_object(text) is not None:
        return "json"
    return "short" if token_count(text) <= SPAN_MAX_MEDIAN_TOKENS else "long"


def _json_object(text: str) -> dict[str, Any] | None:
    """The JSON object ``text`` holds; a list of tool calls reads as one call's keys (N3)."""
    try:
        value = json.loads(text)
    except ValueError:
        return None
    if isinstance(value, list) and value and all(_is_call(item) for item in value):
        return dict.fromkeys(_CALL_KEYS)
    return value if isinstance(value, dict) else None


def _is_call(value: object) -> bool:
    return isinstance(value, dict) and frozenset(value) == _CALL_KEYS


def _json_share(texts: Sequence[str]) -> float:
    """Share of ``texts`` that are JSON objects whose key set is close to the modal key set."""
    key_sets = [frozenset(obj) for obj in map(_json_object, texts) if obj is not None]
    if not key_sets:
        return 0.0
    modal = Counter(key_sets).most_common(1)[0][0]
    stable = sum(_jaccard(keys, modal) >= JSON_KEY_STABILITY for keys in key_sets)
    return stable / len(texts)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def classify_outputs(
    responses: Sequence[str], tool_calls: Sequence[tuple[dict[str, Any], ...]]
) -> StructureClass:
    """Classify a workload's responses (``tool_calls[i]`` belongs to ``responses[i]``).

    In order: ``json_object`` when at least :data:`JSON_OBJECT_SHARE` parse as
    objects with a key set within :data:`JSON_KEY_STABILITY` of the modal one;
    ``enum_label`` when at most :data:`ENUM_MAX_DISTINCT` distinct normalized
    outputs with a median length of at most :data:`ENUM_MAX_MEDIAN_TOKENS`
    tokens; ``short_span`` for a median of at most :data:`SPAN_MAX_MEDIAN_TOKENS`
    tokens and a distinct ratio above :data:`SPAN_MIN_DISTINCT_RATIO`; else
    ``free_text``, which is also the answer for no responses at all.
    """
    if len(responses) != len(tool_calls):
        raise ValueError(f"{len(responses)} responses but {len(tool_calls)} tool-call tuples")
    texts = [effective_response(r, c) for r, c in zip(responses, tool_calls, strict=True)]
    if not texts:
        return StructureClass.FREE_TEXT
    if _json_share(texts) >= JSON_OBJECT_SHARE:
        return StructureClass.JSON_OBJECT
    normalized = [normalize_response(t) for t in texts]
    distinct = len(set(normalized))
    median_tokens = statistics.median(token_count(t) for t in normalized)
    if distinct <= ENUM_MAX_DISTINCT and median_tokens <= ENUM_MAX_MEDIAN_TOKENS:
        return StructureClass.ENUM_LABEL
    if median_tokens <= SPAN_MAX_MEDIAN_TOKENS and distinct / len(texts) > SPAN_MIN_DISTINCT_RATIO:
        return StructureClass.SHORT_SPAN
    return StructureClass.FREE_TEXT
