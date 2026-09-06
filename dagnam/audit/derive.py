"""Turn a workload's trace records into training rows in the recipe's row format.

The workload itself (Task 4's ``Workload``) is not imported here so the two
branches compose without a shared module: callers pass the records the
workload owns (``[records[i] for i in w.record_indices]``) and its
``structure_class`` value as a plain string.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import string
from typing import Any

from dagnam_contracts.hygiene import compute_exact_duplicates
from dagnam_contracts.prompts import render_chat_prompt

from dagnam.audit.record import TraceRecord
from dagnam.audit.redact import RedactStats, redact_rows
from dagnam.audit.split import HOLDOUT_SHARE, split_boundary, time_split

# Structure class (Task 4's ``StructureClass`` values) -> platform row format.
# A new class plugs in here, never as an ``if structure_class ==`` branch.
FORMAT_BY_STRUCTURE: dict[str, str] = {
    "enum_label": "labeled-example",
    "json_object": "chat-messages",
    "short_span": "chat-messages",
    "free_text": "chat-messages",
}

_TRAILING = string.punctuation + string.whitespace


@dataclass(frozen=True, slots=True)
class DeriveStats:
    """What derivation produced; ``record_indices`` maps each row back to its record."""

    rows: int
    truncated: int
    truncation_rate: float
    format_key: str
    skipped: int
    record_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DedupStats:
    """Exact duplicates removed; ``kept_indices`` maps each surviving row to its input row."""

    removed: int
    kept_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class WorkloadDataset:
    """The derived, redacted, deduplicated rows with their split and the stats for ``meta.json``."""

    rows: list[dict[str, Any]]
    split: dict[str, list[int]]
    stats: dict[str, Any]


def normalize_label(response: str) -> str:
    """Strip, casefold and drop trailing punctuation, so ``"Returns."`` and ``"returns"`` agree."""
    return response.strip().casefold().rstrip(_TRAILING)


def _prompt_turns(record: TraceRecord) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in record.messages]


def _labeled_example(record: TraceRecord, max_seq_length: int) -> tuple[dict[str, Any], bool]:
    rendered = render_chat_prompt(_prompt_turns(record), system=record.system)
    truncated = len(rendered) > max_seq_length
    # From the front: the latest turn is what the label answers.
    return {
        "input": rendered[-max_seq_length:],
        "label": normalize_label(record.response),
    }, truncated


def _chat_messages(record: TraceRecord, max_seq_length: int) -> tuple[dict[str, Any], bool]:
    turns = _prompt_turns(record)
    budget = max_seq_length - (len(record.system) if record.system else 0)
    truncated = sum(len(t["content"]) for t in turns) > budget
    # Drop the oldest turns first; the last turn always survives, even over budget.
    while len(turns) > 1 and sum(len(t["content"]) for t in turns) > budget:
        del turns[0]
    if not turns:
        return {"messages": []}, truncated
    system = [{"role": "system", "content": record.system}] if record.system else []
    assistant = {"role": "assistant", "content": record.response.strip()}
    return {"messages": [*system, *turns, assistant]}, truncated


_BUILDERS = {"labeled-example": _labeled_example, "chat-messages": _chat_messages}


def _usable(row: dict[str, Any]) -> bool:
    if "label" in row:
        return bool(row["input"]) and bool(row["label"])
    messages: list[dict[str, str]] = row["messages"]
    return len(messages) > 1 and bool(messages[-1]["content"])


def derive_rows(
    records: Sequence[TraceRecord], *, structure_class: str, max_seq_length: int
) -> tuple[list[dict[str, Any]], DeriveStats]:
    """Build one training row per usable record in the format ``structure_class`` maps to.

    ``max_seq_length`` is a character budget on the prompt side (the SDK has no
    tokenizer): a ``labeled-example`` input is cut from the front so the last
    turn survives, a ``chat-messages`` row drops its oldest turns. Records with
    an empty prompt or an empty (normalized) response are skipped and counted.
    """
    format_key = FORMAT_BY_STRUCTURE.get(structure_class)
    if format_key is None:
        raise ValueError(
            f"unknown structure class {structure_class!r}; "
            f"expected one of {', '.join(FORMAT_BY_STRUCTURE)}"
        )
    build = _BUILDERS[format_key]
    rows: list[dict[str, Any]] = []
    kept: list[int] = []
    truncated = 0
    for index, record in enumerate(records):
        row, was_truncated = build(record, max_seq_length)
        if not _usable(row):
            continue
        rows.append(row)
        kept.append(index)
        truncated += was_truncated
    stats = DeriveStats(
        rows=len(rows),
        truncated=truncated,
        truncation_rate=truncated / len(rows) if rows else 0.0,
        format_key=format_key,
        skipped=len(records) - len(rows),
        record_indices=tuple(kept),
    )
    return rows, stats


def dedup_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], DedupStats]:
    """Drop exact duplicates (contracts' ``canonical_row_hash``), keeping the first occurrence."""
    duplicates = set(compute_exact_duplicates(rows).duplicate_indices)
    kept = tuple(i for i in range(len(rows)) if i not in duplicates)
    return [rows[i] for i in kept], DedupStats(removed=len(duplicates), kept_indices=kept)


def build_dataset(
    records: Sequence[TraceRecord],
    *,
    structure_class: str,
    max_seq_length: int,
    holdout_share: float = HOLDOUT_SHARE,
) -> WorkloadDataset:
    """Derive, redact, dedup and time-split one workload's records, keeping rows and records aligned."""
    derived, derive_stats = derive_rows(
        records, structure_class=structure_class, max_seq_length=max_seq_length
    )
    redacted, redact_stats = redact_rows(derived)
    rows, dedup_stats = dedup_rows(redacted)
    kept_records = [records[derive_stats.record_indices[i]] for i in dedup_stats.kept_indices]
    split = time_split(kept_records, holdout_share=holdout_share)
    boundary = split_boundary(kept_records, split)
    return WorkloadDataset(
        rows=rows,
        split=split,
        stats={
            "format_key": derive_stats.format_key,
            "derive": {
                "rows": derive_stats.rows,
                "skipped": derive_stats.skipped,
                "truncated": derive_stats.truncated,
                "truncation_rate": derive_stats.truncation_rate,
            },
            "redact": _redact_dict(redact_stats),
            "dedup": {"removed": dedup_stats.removed},
            "boundary_ts": boundary.isoformat() if boundary else None,
        },
    )


def _redact_dict(stats: RedactStats) -> dict[str, Any]:
    return {
        "counts": dict(stats.counts),
        "pass_list": list(stats.pass_list),
        "rows_changed": stats.rows_changed,
    }


__all__ = [
    "FORMAT_BY_STRUCTURE",
    "DedupStats",
    "DeriveStats",
    "WorkloadDataset",
    "build_dataset",
    "dedup_rows",
    "derive_rows",
    "normalize_label",
]
