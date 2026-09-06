"""Every constant the workload audit decides with, in one place.

The first block is the design's economics table; the second names the
output-structure rules so no module carries a bare number. Nothing else lives
here: the module is data, not code.
"""

WINDOW_DAYS = 30
"""Export window (days) assumed when the timestamps do not span it."""
MIN_TRACES_PER_WORKLOAD = 1_000
"""Below this a workload is ``too_few_samples`` (still reported)."""
MIN_HOLDOUT = 200
"""Below this after the split a workload is ``too_few_samples``."""
MAINTENANCE_USD_MONTH = 50.0
"""Per replaced workload, added to the student's monthly cost."""
RATIO_NOT_WORTH_IT = 3.0
"""Teacher $/month over (student $/month + maintenance) below this is ``not_worth_it``."""
RATIO_CANDIDATE = 10.0
"""At or above this the workload is a ``candidate``; between the two ratios, ``marginal``."""
FLOOR_LABEL = 0.97
"""Quality floor on the agreement lower bound for label workloads."""
FLOOR_JSON = 0.95
"""Quality floor on the agreement lower bound for JSON workloads."""
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
DAYS_PER_MONTH = 30
"""Monthly figures are per-day rates times this."""
PRICE_TABLE_STALE_DAYS = 60
"""A price table older than this (days since its ``as_of`` date) puts a warning in the report."""
MAX_SEQ_LENGTH = 2048
"""Characters of rendered prompt a derived row keeps (the recipes' ``max_seq_length`` default)."""
