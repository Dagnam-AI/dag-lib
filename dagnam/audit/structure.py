"""Output-structure classes: what shape a workload's responses take.

The class decides which student can replace the teacher (a label classifier,
a JSON extractor, ...) and which quality floor applies. It is computed over a
sample of responses by the rules of :func:`classify_outputs`, every number coming
from :mod:`dagnam.audit.thresholds`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from enum import StrEnum
import functools
import json
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
from dagnam.audit.token_estimate import estimate

_WHITESPACE = re.compile(r"\s+")
_NO_SPACE = (
    "\u0e00-\u0eff\u1000-\u109f\u1780-\u17ff\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
)
"""Thai, Lao, Myanmar, Khmer, kana and CJK ideographs: scripts written without spaces."""
_NO_SPACE_RUN = re.compile(f"[{_NO_SPACE}]+")
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
    Chinese reply as one "token" and free text as a short span. Two
    characters a token keeps a two-character Chinese label at one token and
    that reply at thirty.
    """
    count = 0
    for word in text.split():
        runs = _NO_SPACE_RUN.findall(word)
        count += len(_NO_SPACE_RUN.sub(" ", word).split())
        count += sum((len(run) + 1) // 2 for run in runs)
    return count


@functools.lru_cache(maxsize=256)
def student_tokens(text: str) -> int:
    """Estimate the student's (Qwen2.5) tokens in ``text`` without its tokenizer.

    The count is :func:`dagnam.audit.token_estimate.estimate`: the text is brought to NFC as the
    tokenizer does, every character starts at its UTF-8 bytes, the most a byte-level tokenizer can
    spend, and is discounted only where the student's vocabulary shows a merge exists, each
    discount with a limit. It guesses no language, so mixed-language text, a list of labels and a
    terse prompt are counted like any other. Cached, because every row of a workload carries the
    same system prompt. Each message of a row is counted on its own.

    It is an estimate, fitted to the real tokenizer, not a limit guarantee. What was measured:

    - Natural text: at least 0.95 of the real count on 2,498 held-out rows of 500 or more tokens
      (0.954 at the lowest, 0.971 at the 1st percentile), 1.04 at the median and 1.22 at the 95th
      percentile. By kind (median, 95th percentile): English 1.05 and 1.10; Spanish, French,
      German and the like 1.20 and 1.26; Cyrillic 1.07 and 1.21; Arabic script 1.08 and 1.14;
      Indic 1.01 and 1.05; Tibetan 1.02 and 1.04; Simplified Chinese 1.06 and 1.08; Traditional
      Chinese 1.05 and 1.07; Japanese 1.03 and 1.10; Korean 1.05 and 1.12; JSON 1.04 and 1.07;
      code 1.12 and 1.26. Measured scripts are listed in :mod:`dagnam.audit.token_estimate`;
      scripts with no natural text in the corpus (Syriac, Thaana, Cherokee, Gothic and the like)
      stay at the safe byte price for every letter that is not a token.
    - Random or crafted text: no lower bound. It can count about half the real tokens (lowest
      measured 0.486 on random Cyrillic words, 0.504 on random Thai, 0.547 to 0.62 on a repeated
      number sign or random Tibetan, 0.75 on licence keys, 0.88 to 0.90 on base32 over plausible
      text) and 0.36 on crafted alternating han units, so a row estimated to fit the 2,048-token
      cap can exceed it by up to about 2.8 times on crafted text. Pairs of units that break each
      other and cycles of 9 or more fragile han units, which its 8-unit window does not see, are a
      known limit.
    - Never above the UTF-8 bytes of the NFC form of the text.
    - It leans high where it cannot be right: code with long identifiers up to 3 times, a symbol
      repeated thousands of times up to 21 times, a repeated fragile 4-character han unit 4
      times; text of a few tokens rounds up, so a one-word text can count two.
    - Memory is flat in the size of the text, apart from the three vocabulary tables (about 13 MB
      once in use), the normalised copy of text that is not already NFC, and a few copies of one
      unbroken word.

    The recipe drops over-budget rows with its real tokenizer, so an under-count admits a
    workload whose rows are all dropped at train time, and an over-count leaves out rows that
    fit. Rebuild the tables and refit if the student tokenizer changes.
    """
    return estimate(text)


def output_shape(
    response: str, tool_calls: Sequence[dict[str, Any]]
) -> Literal["json", "short", "long"]:
    """One response's shape: a JSON object, a short answer, or longer prose.

    Traces without a system prompt have no template to group them by, so the
    shape of each answer does it, one bucket per shape: a
    label, an extraction and a drafted reply never share a workload. Whether a
    short bucket is labels or spans is still the group's :func:`classify_outputs`.
    """
    text = effective_response(response, tool_calls)
    if _json_object(text) is not None:
        return "json"
    return "short" if token_count(text) <= SPAN_MAX_MEDIAN_TOKENS else "long"


def _json_object(text: str) -> dict[str, Any] | None:
    """The JSON object ``text`` holds; a list of tool calls reads as one call's keys."""
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):  # not JSON, or nested too deeply to read as JSON
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
