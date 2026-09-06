"""Workload audit: read LLM traces from the tools that already hold them.

Everything the audit consumes is a :class:`TraceRecord`; :func:`read_traces`
produces them from a Langfuse, LangSmith or OpenAI export, or from any
JSONL/CSV file through a column map. :func:`discover_workloads` groups the
records by system-prompt template and output structure into :class:`Workload`
summaries, most monthly spend first. :func:`build_dataset` turns one
workload's records into redacted, deduplicated, time-split training rows and
:func:`write_workload` puts them on disk. :func:`run_audit` then drives the
platform through the :data:`CANDIDATES` per workload -- upload, split, train,
serve, replay the holdout -- keeping a resumable :class:`AuditState`, and
:func:`frontier` names the cheapest candidate whose agreement clears the
floor. :func:`replaceability` judges each workload by the design's economics
and :func:`build_scan_report` assembles the report.
"""

from __future__ import annotations

from dagnam.audit.candidates import CANDIDATES, CandidateKind, CandidateSpec
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
from dagnam.audit.frontier import (
    CandidateResult,
    Endpoint,
    Latency,
    Winner,
    frontier,
    replay_holdout,
)
from dagnam.audit.normalize import normalize_template, template_hash
from dagnam.audit.orchestrate import run_audit
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
from dagnam.audit.scoring import Agreement, score_json, score_labels
from dagnam.audit.secrets import SecretStore
from dagnam.audit.split import HOLDOUT_SHARE, split_boundary, time_split
from dagnam.audit.state import AuditState, StepState, load_state, save_state
from dagnam.audit.structure import StructureClass, classify_outputs
from dagnam.audit.workspace import SCHEMA, write_workload

__all__ = [
    "CANDIDATES",
    "FORMAT_BY_STRUCTURE",
    "HOLDOUT_SHARE",
    "MALFORMED_FATAL_SHARE",
    "PII_POLICY",
    "SCHEMA",
    "Agreement",
    "AuditState",
    "CandidateKind",
    "CandidateResult",
    "CandidateSpec",
    "DedupStats",
    "DeriveStats",
    "Endpoint",
    "Latency",
    "MalformedExportError",
    "Message",
    "PriceTable",
    "PriceTableError",
    "ReadStats",
    "RedactStats",
    "ScanReport",
    "SecretStore",
    "StepState",
    "StructureClass",
    "TraceRecord",
    "UnsupportedExportError",
    "Verdict",
    "Window",
    "Winner",
    "Workload",
    "WorkloadDataset",
    "build_dataset",
    "build_scan_report",
    "classify_outputs",
    "dedup_rows",
    "derive_rows",
    "discover_workloads",
    "frontier",
    "load_state",
    "normalize_label",
    "normalize_template",
    "read_traces",
    "redact_rows",
    "replaceability",
    "replay_holdout",
    "run_audit",
    "save_state",
    "score_json",
    "score_labels",
    "split_boundary",
    "template_hash",
    "time_split",
    "write_scan_report",
    "write_workload",
]
