"""Workload audit: read LLM traces from the tools that already hold them.

Everything the audit consumes is a :class:`TraceRecord`; :func:`read_traces`
produces them from a Langfuse, LangSmith or OpenAI export, or from any
JSONL/CSV file through a column map. :func:`discover_workloads` groups the
records by system-prompt template and output structure into :class:`Workload`
summaries, most monthly spend first; :func:`replaceability` judges each one
by the design's economics and :func:`build_scan_report` assembles the report.
"""

from __future__ import annotations

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
from dagnam.audit.scan_report import ScanReport, Window, build_scan_report, write_scan_report
from dagnam.audit.structure import StructureClass, classify_outputs

__all__ = [
    "MALFORMED_FATAL_SHARE",
    "MalformedExportError",
    "Message",
    "PriceTable",
    "PriceTableError",
    "ReadStats",
    "ScanReport",
    "StructureClass",
    "TraceRecord",
    "UnsupportedExportError",
    "Verdict",
    "Window",
    "Workload",
    "build_scan_report",
    "classify_outputs",
    "discover_workloads",
    "normalize_template",
    "read_traces",
    "replaceability",
    "template_hash",
    "write_scan_report",
]
