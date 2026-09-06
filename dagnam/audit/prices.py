"""Vendor price tables and the platform's estimated serving rates.

A price table is a versioned JSON file under ``prices/`` copied from the
vendors' public price pages on its ``as_of`` date (URLs in its ``_sources``
key); the newest bundled one is the default. A model without a row prices to
``None`` — never a guess. ``serving.json`` holds the two student serving rates,
labelled *estimated* until the platform bills for real.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Any

from dagnam._core.exceptions import DagnamError

PRICES_DIR = Path(__file__).parent / "prices"
_ONE_MILLION = 1_000_000
_ROW_FIELDS = ("input_per_m", "output_per_m", "cached_input_per_m", "cheaper_variant")


class PriceTableError(DagnamError):
    """A price table file is missing or does not have the documented shape."""

    def __init__(self, path: Path, why: str) -> None:
        self.path = path
        super().__init__(f"price table {path}: {why}")


@dataclass(frozen=True, slots=True)
class PriceRow:
    """One model's list prices in USD per million tokens."""

    model: str
    input_per_m: float
    output_per_m: float
    cached_input_per_m: float | None
    cheaper_variant: str | None


@dataclass(frozen=True, slots=True)
class PriceTable:
    """A versioned set of :class:`PriceRow` values keyed by model id."""

    version: str
    as_of: date
    rows: Mapping[str, PriceRow]

    @classmethod
    def load(cls, path: Path | None) -> PriceTable:
        """Read ``path``, or the newest bundled ``<yyyy-mm>.json`` when ``None``."""
        if path is None:
            path = sorted(PRICES_DIR.glob("????-??.json"))[-1]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise PriceTableError(path, "no such file") from None
        except ValueError as exc:
            raise PriceTableError(path, f"not valid JSON ({exc})") from None
        if not isinstance(data, dict):
            raise PriceTableError(path, "the top level must be a JSON object")
        version, as_of, rows = data.get("version"), data.get("as_of"), data.get("rows")
        if not isinstance(version, str):
            raise PriceTableError(path, "'version' must be a string")
        try:
            as_of_date = date.fromisoformat(as_of if isinstance(as_of, str) else "")
        except ValueError:
            raise PriceTableError(path, "'as_of' must be an ISO date") from None
        if not isinstance(rows, dict):
            raise PriceTableError(path, "'rows' must be an object keyed by model id")
        return cls(version, as_of_date, {m: _row(path, m, raw) for m, raw in rows.items()})

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
        """USD for the given token counts, or ``None`` when ``model`` has no row."""
        row = self.rows.get(model)
        if row is None:
            return None
        return (
            prompt_tokens * row.input_per_m + completion_tokens * row.output_per_m
        ) / _ONE_MILLION

    def age_days(self, today: date) -> int:
        """Days since the table's ``as_of`` date."""
        return (today - self.as_of).days


def _row(path: Path, model: str, raw: Any) -> PriceRow:
    if not isinstance(raw, dict) or set(raw) != set(_ROW_FIELDS):
        raise PriceTableError(path, f"row {model} must have exactly the fields {list(_ROW_FIELDS)}")
    values = [raw[f] for f in _ROW_FIELDS]
    prices_ok = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values[:2])
    cached_ok = values[2] is None or isinstance(values[2], (int, float))
    variant_ok = values[3] is None or isinstance(values[3], str)
    if not (prices_ok and cached_ok and variant_ok):
        raise PriceTableError(path, f"row {model} has a mistyped field")
    cached = None if values[2] is None else float(values[2])
    return PriceRow(model, float(values[0]), float(values[1]), cached, values[3])


def _serving_rates() -> Mapping[str, Mapping[str, Any]]:
    data = json.loads((PRICES_DIR / "serving.json").read_text(encoding="utf-8"))
    return data["rates"]


SERVING_RATES: Mapping[str, Mapping[str, Any]] = _serving_rates()
"""``cpu-classifier`` ($ per 1K requests) and ``gpu-small-llm`` ($ per 1M output tokens), each labelled ``basis: estimated`` with its assumptions."""
