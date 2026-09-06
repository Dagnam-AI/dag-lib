"""Workload audit: read LLM traces from the tools that already hold them.

Everything the audit consumes is a :class:`TraceRecord`; :func:`read_traces`
produces them from a Langfuse, LangSmith or OpenAI export, or from any
JSONL/CSV file through a column map. :func:`build_dataset` turns one
workload's records into redacted, deduplicated, time-split training rows and
:func:`write_workload` puts them on disk.
"""

from __future__ import annotations

from dagnam.audit.derive import (
    FORMAT_BY_STRUCTURE,
    DedupStats,
    DeriveStats,
    WorkloadDataset,
    build_dataset,
    dedup_rows,
    derive_rows,
    normalize_label,
)
from dagnam.audit.readers import (
    MALFORMED_FATAL_SHARE,
    MalformedExportError,
    ReadStats,
    UnsupportedExportError,
    read_traces,
)
from dagnam.audit.record import Message, TraceRecord
from dagnam.audit.redact import PII_POLICY, RedactStats, redact_rows
from dagnam.audit.split import HOLDOUT_SHARE, split_boundary, time_split
from dagnam.audit.workspace import SCHEMA, write_workload

__all__ = [
    "FORMAT_BY_STRUCTURE",
    "HOLDOUT_SHARE",
    "MALFORMED_FATAL_SHARE",
    "PII_POLICY",
    "SCHEMA",
    "DedupStats",
    "DeriveStats",
    "MalformedExportError",
    "Message",
    "ReadStats",
    "RedactStats",
    "TraceRecord",
    "UnsupportedExportError",
    "WorkloadDataset",
    "build_dataset",
    "dedup_rows",
    "derive_rows",
    "normalize_label",
    "read_traces",
    "redact_rows",
    "split_boundary",
    "time_split",
    "write_workload",
]
