"""Client-side agreement between a candidate and the teacher on the holdout (spec D3).

Pure Python. Labels and short spans score as normalized exact match (with
macro-F1 alongside); JSON objects score per field, micro-averaged. Every
score carries ``n`` and a 95% Wilson interval on the headline value, and the
frontier decides on the interval's lower bound, never the point estimate.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
from typing import Any

from dagnam._types import JsonValue
from dagnam.audit.derive import normalize_label

Z95 = 1.959963984540054
"""Two-sided 95% normal quantile."""


@dataclass(frozen=True, slots=True)
class Agreement:
    """A score with its sample size and interval; the extra fields depend on ``metric``."""

    metric: str
    value: float
    ci95: tuple[float, float]
    n: int
    exact: float | None = None
    macro_f1: float | None = None
    field_precision: float | None = None
    field_recall: float | None = None
    field_f1: float | None = None

    def to_json(self) -> dict[str, JsonValue]:
        """The ``agreement`` object of ``audit-report.json`` (spec section 7)."""
        extras: dict[str, JsonValue] = {
            key: value
            for key, value in (
                ("exact", self.exact),
                ("macro_f1", self.macro_f1),
                ("field_precision", self.field_precision),
                ("field_recall", self.field_recall),
                ("field_f1", self.field_f1),
            )
            if value is not None
        }
        return {
            "metric": self.metric,
            "value": self.value,
            "ci95": [self.ci95[0], self.ci95[1]],
            "n": self.n,
            **extras,
        }


def wilson_interval(successes: int, n: int, *, z: float = Z95) -> tuple[float, float]:
    """The Wilson score interval for ``successes`` out of ``n``; ``(0, 1)`` when ``n`` is 0."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, center - half), min(1.0, center + half))


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _paired(pred: Sequence[str], truth: Sequence[str]) -> None:
    if len(pred) != len(truth):
        raise ValueError(f"{len(pred)} predictions but {len(truth)} truths")


def score_labels(pred: Sequence[str], truth: Sequence[str]) -> Agreement:
    """Normalized exact match (the headline, with its Wilson interval) plus macro-F1.

    Both sides go through :func:`dagnam.audit.derive.normalize_label`, so
    ``"Billing."`` agrees with ``"billing"``. Macro-F1 averages the per-class
    F1 over every class either side produced.
    """
    _paired(pred, truth)
    pairs = [(normalize_label(p), normalize_label(t)) for p, t in zip(pred, truth, strict=True)]
    hits = sum(p == t for p, t in pairs)
    classes = {label for pair in pairs for label in pair}
    f1s = [
        _f1(
            sum(p == t == c for p, t in pairs),
            sum(p == c != t for p, t in pairs),
            sum(t == c != p for p, t in pairs),
        )[2]
        for c in classes
    ]
    n = len(pairs)
    exact = hits / n if n else 0.0
    return Agreement(
        metric="exact",
        value=exact,
        ci95=wilson_interval(hits, n),
        n=n,
        exact=exact,
        macro_f1=sum(f1s) / len(f1s) if f1s else 0.0,
    )


def _fields(text: str) -> dict[str, str]:
    """A JSON object's fields as canonical strings; anything else is no fields at all."""
    try:
        value: Any = json.loads(text)
    except ValueError:
        return {}
    if not isinstance(value, dict):
        return {}
    return {str(k): json.dumps(v, sort_keys=True, ensure_ascii=False) for k, v in value.items()}


def modal_keys(truth: Sequence[str]) -> list[str]:
    """The most common key set among the truths that parse as objects, sorted."""
    key_sets = [frozenset(fields) for fields in map(_fields, truth) if fields]
    if not key_sets:
        return []
    return sorted(Counter(key_sets).most_common(1)[0][0])


def score_json(pred: Sequence[str], truth: Sequence[str], keys: Sequence[str]) -> Agreement:
    """Per-field exact precision/recall/F1 over ``keys``, micro-averaged across rows.

    A field counts as a hit when both objects carry it with the same
    (canonical JSON) value; a wrong value is both a false positive and a false
    negative. The headline is the micro-F1, which equals ``2tp / (2tp+fp+fn)``
    and so takes a Wilson interval over that many trials. ``n`` is the row count.
    """
    _paired(pred, truth)
    tp = fp = fn = 0
    for p_text, t_text in zip(pred, truth, strict=True):
        p, t = _fields(p_text), _fields(t_text)
        for key in keys:
            if key in p and key in t and p[key] == t[key]:
                tp += 1
                continue
            fp += key in p
            fn += key in t
    precision, recall, f1 = _f1(tp, fp, fn)
    return Agreement(
        metric="field_f1",
        value=f1,
        ci95=wilson_interval(2 * tp, 2 * tp + fp + fn),
        n=len(pred),
        field_precision=precision,
        field_recall=recall,
        field_f1=f1,
    )


__all__ = ["Z95", "Agreement", "modal_keys", "score_json", "score_labels", "wilson_interval"]
