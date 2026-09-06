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
import re
import statistics
from typing import Any

from dagnam.audit.readers.base import get
from dagnam.audit.thresholds import (
    ENUM_MAX_DISTINCT,
    ENUM_MAX_MEDIAN_TOKENS,
    JSON_KEY_STABILITY,
    JSON_OBJECT_SHARE,
    SPAN_MAX_MEDIAN_TOKENS,
    SPAN_MIN_DISTINCT_RATIO,
)

_WHITESPACE = re.compile(r"\s+")


class StructureClass(StrEnum):
    """The four response shapes the audit tells apart."""

    ENUM_LABEL = "enum_label"
    JSON_OBJECT = "json_object"
    SHORT_SPAN = "short_span"
    FREE_TEXT = "free_text"


def normalize_response(response: str) -> str:
    """Case-fold and collapse whitespace so ``Billing`` and ``billing `` are one output."""
    return _WHITESPACE.sub(" ", response).strip().casefold()


def effective_response(response: str, tool_calls: Sequence[dict[str, Any]]) -> str:
    """The text a response is judged on: its first tool call's arguments as JSON, else the text.

    A tool call is a JSON object over its arguments whatever the export's
    shape (OpenAI ``function.arguments`` as a JSON string, LangChain ``args``,
    a bare ``arguments`` object); arguments that are not an object are
    wrapped so the call still reads as one.
    """
    if not tool_calls:
        return response
    arguments = get(tool_calls[0], "function.arguments", "args", "arguments")
    if isinstance(arguments, str):
        arguments = _json_object(arguments) or arguments
    if not isinstance(arguments, dict):
        arguments = {} if arguments is None else {"arguments": arguments}
    return json.dumps(arguments, sort_keys=True)


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


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
    median_tokens = statistics.median(len(t.split()) for t in normalized)
    if distinct <= ENUM_MAX_DISTINCT and median_tokens <= ENUM_MAX_MEDIAN_TOKENS:
        return StructureClass.ENUM_LABEL
    if median_tokens <= SPAN_MAX_MEDIAN_TOKENS and distinct / len(texts) > SPAN_MIN_DISTINCT_RATIO:
        return StructureClass.SHORT_SPAN
    return StructureClass.FREE_TEXT
