"""The thresholds module carries every audit constant, named as the design lists them."""

from __future__ import annotations

from dagnam.audit import thresholds

SPEC_TABLE = {
    "WINDOW_DAYS": 30,
    "MIN_TRACES_PER_WORKLOAD": 1_000,
    "MIN_HOLDOUT": 200,
    "MAINTENANCE_USD_MONTH": 50.0,
    "RATIO_NOT_WORTH_IT": 3.0,
    "RATIO_CANDIDATE": 10.0,
    "FLOOR_LABEL": 0.97,
    "FLOOR_JSON": 0.95,
    "ENUM_MAX_DISTINCT": 50,
    "JSON_KEY_STABILITY": 0.8,
    "MAX_UNSTRUCTURED_SHARE": 0.2,
    "STRUCTURE_SAMPLE": 200,
}


def test_design_table_values_verbatim() -> None:
    for name, value in SPEC_TABLE.items():
        assert getattr(thresholds, name) == value, name


def test_structure_rule_constants_are_named() -> None:
    assert thresholds.ENUM_MAX_MEDIAN_TOKENS == 5
    assert thresholds.JSON_OBJECT_SHARE == 0.95
    assert thresholds.SPAN_MAX_MEDIAN_TOKENS == 8
    assert thresholds.SPAN_MIN_DISTINCT_RATIO == 0.5
    assert thresholds.EXCERPT_CHARS == 200
    assert thresholds.DAYS_PER_MONTH == 30


def test_module_holds_only_upper_case_constants() -> None:
    public = [name for name in vars(thresholds) if not name.startswith("_")]
    assert public
    assert all(name.isupper() for name in public)
    assert all(isinstance(getattr(thresholds, name), (int, float)) for name in public)
