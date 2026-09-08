"""Every constant the workload audit decides with, in one place.

The economics table -- the two ratio bands, the maintenance charge, the two
quality floors, the sample minimums and the month length -- is
``dagnam_contracts.audit.verdict``'s, because the platform decides the same
workload by the same numbers; this module is the SDK's name for them. The
second block names the output-structure and rendering rules the SDK owns
alone, so no module carries a bare number. Nothing else lives here: the module
is data, not code.
"""

from dagnam_contracts.audit.verdict import (
    DAYS_PER_MONTH,
    FLOOR_JSON,
    FLOOR_LABEL,
    MAINTENANCE_USD_MONTH,
    MIN_HOLDOUT,
    MIN_TRACES_PER_WORKLOAD,
    RATIO_CANDIDATE,
    RATIO_NOT_WORTH_IT,
)

WINDOW_DAYS = 30
"""Export window (days) assumed when the timestamps do not span it."""
ENUM_MAX_DISTINCT = 50
"""At most this many distinct normalized outputs for an ``enum_label`` workload."""
JSON_KEY_STABILITY = 0.8
"""Minimum Jaccard similarity of an object's key set to the modal key set."""
MAX_UNSTRUCTURED_SHARE = 0.2
"""Above this share of traces without a system prompt, discovery is flagged low-confidence."""
STRUCTURE_SAMPLE = 200
"""Responses (first in time order) the structure class is decided on."""

ENUM_MAX_MEDIAN_TOKENS = 5
"""``enum_label`` also needs a median response length at or under this many tokens."""
JSON_OBJECT_SHARE = 0.95
"""Share of responses that must parse as key-stable JSON objects for ``json_object``."""
SPAN_MAX_MEDIAN_TOKENS = 8
"""``short_span`` needs a median response length at or under this many tokens."""
SPAN_MIN_DISTINCT_RATIO = 0.5
"""``short_span`` needs a distinct-output ratio strictly above this."""
EXCERPT_CHARS = 200
"""Length cap of the masked template excerpt a report may show."""
PRICE_TABLE_STALE_DAYS = 60
"""A price table older than this (days since its ``as_of`` date) puts a warning in the report."""
MAX_SEQ_LENGTH = 2048
"""Characters of rendered prompt a derived row keeps (the recipes' ``max_seq_length`` default)."""

__all__ = [
    "DAYS_PER_MONTH",
    "ENUM_MAX_DISTINCT",
    "ENUM_MAX_MEDIAN_TOKENS",
    "EXCERPT_CHARS",
    "FLOOR_JSON",
    "FLOOR_LABEL",
    "JSON_KEY_STABILITY",
    "JSON_OBJECT_SHARE",
    "MAINTENANCE_USD_MONTH",
    "MAX_SEQ_LENGTH",
    "MAX_UNSTRUCTURED_SHARE",
    "MIN_HOLDOUT",
    "MIN_TRACES_PER_WORKLOAD",
    "PRICE_TABLE_STALE_DAYS",
    "RATIO_CANDIDATE",
    "RATIO_NOT_WORTH_IT",
    "SPAN_MAX_MEDIAN_TOKENS",
    "SPAN_MIN_DISTINCT_RATIO",
    "STRUCTURE_SAMPLE",
    "WINDOW_DAYS",
]
