"""Workload audit: read LLM traces from the tools that already hold them.

Everything the audit consumes is a :class:`TraceRecord`; :func:`read_traces`
produces them from a Langfuse, LangSmith or OpenAI export, or from any
JSONL/CSV file through a column map. :func:`discover_workloads` groups the
records by system-prompt template and output structure into :class:`Workload`
summaries, most monthly spend first. :func:`build_dataset` turns one
workload's records into redacted, deduplicated, time-split training rows and
:func:`write_workload` puts them on disk.
summaries, most monthly spend first; :func:`replaceability` judges each one
by the design's economics and :func:`build_scan_report` assembles the report.
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
from dagnam.audit.discover import Workload, discover_workloads
from dagnam.audit.economics import Verdict, replaceability
from dagnam.audit.normalize import normalize_template, template_hash
from dagnam.audit.prices import PriceTable, PriceTableError
from dagnam.audit.readers import (
    MALFORMED_FATAL_SHARE,
    MalformedExportError,
    ReadStats,
    UnsupportedExportError,
    read_traces,
)
from dagnam.audit.record import Message, TraceRecord
from dagnam.audit.redact import PII_POLICY, RedactStats, redact_rows
from dagnam.audit.scan_report import ScanReport, Window, build_scan_report, write_scan_report
from dagnam.audit.split import HOLDOUT_SHARE, split_boundary, time_split
from dagnam.audit.structure import StructureClass, classify_outputs
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
    "PriceTable",
    "PriceTableError",
    "ReadStats",
    "RedactStats",
    "ScanReport",
    "StructureClass",
    "TraceRecord",
    "UnsupportedExportError",
    "Verdict",
    "Window",
    "Workload",
    "WorkloadDataset",
    "build_dataset",
    "build_scan_report",
    "classify_outputs",
    "dedup_rows",
    "derive_rows",
    "discover_workloads",
    "normalize_label",
    "normalize_template",
    "read_traces",
    "redact_rows",
    "replaceability",
    "split_boundary",
    "template_hash",
    "time_split",
    "write_scan_report",
    "write_workload",
]
