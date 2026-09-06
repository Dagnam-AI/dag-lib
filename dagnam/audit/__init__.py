"""Workload audit: read LLM traces from the tools that already hold them.

Everything the audit consumes is a :class:`TraceRecord`; :func:`read_traces`
produces them from a Langfuse, LangSmith or OpenAI export, or from any
JSONL/CSV file through a column map. :func:`discover_workloads` groups the
records by system-prompt template and output structure into :class:`Workload`
summaries, most monthly spend first.
"""

from __future__ import annotations

from dagnam.audit.discover import Workload, discover_workloads
from dagnam.audit.normalize import normalize_template, template_hash
from dagnam.audit.readers import (
    MALFORMED_FATAL_SHARE,
    MalformedExportError,
    ReadStats,
    UnsupportedExportError,
    read_traces,
)
from dagnam.audit.record import Message, TraceRecord
from dagnam.audit.structure import StructureClass, classify_outputs

__all__ = [
    "MALFORMED_FATAL_SHARE",
    "MalformedExportError",
    "Message",
    "ReadStats",
    "StructureClass",
    "TraceRecord",
    "UnsupportedExportError",
    "Workload",
    "classify_outputs",
    "discover_workloads",
    "normalize_template",
    "read_traces",
    "template_hash",
]
