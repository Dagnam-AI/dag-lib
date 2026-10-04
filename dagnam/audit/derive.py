"""Turn a workload's trace records into training rows in the recipe's row format.

:func:`build_dataset` takes the records one workload owns
(``[records[i] for i in w.record_indices]``) and its ``structure_class`` value
as a plain string; :func:`derive_workloads` builds the same datasets for
several workloads from two more passes over the export, holding only the rows
they keep.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from typing import Any

from dagnam_contracts.audit.scoring import normalize_label
from dagnam_contracts.hygiene import canonical_row_hash, compute_exact_duplicates
from dagnam_contracts.prompts import render_chat_prompt

from dagnam.audit.discover import Workload
from dagnam.audit.readers.messages import effective_response
from dagnam.audit.record import TraceRecord
from dagnam.audit.redact import PII_POLICY, RedactStats, redact_records
from dagnam.audit.split import HOLDOUT_SHARE, cap_train, split_boundary, time_split
from dagnam.audit.structure import student_tokens
from dagnam.audit.thresholds import ENUM_MAX_DISTINCT, MAX_TRAIN_ROWS, SFT_MAX_TOKENS

# Structure class (``StructureClass`` values) -> platform row format.
# A new class plugs in here, never as an ``if structure_class ==`` branch.
FORMAT_BY_STRUCTURE: dict[str, str] = {
    "enum_label": "labeled-example",
    "json_object": "chat-messages",
    "short_span": "chat-messages",
    "free_text": "chat-messages",
}
MAX_STRATA: dict[str, int] = {"chat-messages": ENUM_MAX_DISTINCT}
"""Formats whose targets may not be classes (an extraction's answers): past this many distinct
targets the cap samples them as one stratum. A label is always a class, so a
``labeled-example`` cap keeps every one."""
TOKEN_BUDGET: dict[str, int] = {"chat-messages": SFT_MAX_TOKENS}
"""Formats whose recipe drops, never cuts, a training row over this many tokens.

A ``labeled-example`` classifier truncates its input by design, so it has none.
"""
_MESSAGE_TOKENS = 5
"""Chat-template tokens around each message: Qwen2.5's ``<|im_start|>role\\n ... <|im_end|>\\n``."""
_DEFAULT_SYSTEM_TOKENS = 21
"""The system turn Qwen2.5's template adds to a row that has none, with its markers."""


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


def _prompt_turns(record: TraceRecord) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in record.messages]


def _answer(record: TraceRecord) -> str:
    """What the teacher answered: its first tool call's name and arguments when it made one."""
    return effective_response(record.response, record.response_tool_calls)


def _labeled_example(record: TraceRecord, max_seq_length: int) -> tuple[dict[str, Any], bool]:
    rendered = render_chat_prompt(_prompt_turns(record), system=record.system)
    truncated = len(rendered) > max_seq_length
    # From the front: the latest turn is what the label answers.
    return {
        "input": rendered[-max_seq_length:],
        "label": normalize_label(_answer(record)),
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
    assistant = {"role": "assistant", "content": _answer(record).strip()}
    return {"messages": [*system, *turns, assistant]}, truncated


_BUILDERS = {"labeled-example": _labeled_example, "chat-messages": _chat_messages}


def _target(row: Mapping[str, Any]) -> str:
    """What a row teaches: its label, or its final assistant reply."""
    return str(row["label"]) if "label" in row else str(row["messages"][-1]["content"])


def _row_tokens(row: Mapping[str, Any]) -> int:
    """A ``chat-messages`` row's length as the student reads it, by :func:`student_tokens`.

    The template's own tokens are counted exactly, so a row is as close to
    Qwen2.5's count as its text is: see :func:`student_tokens` for how close
    that is in each language.
    """
    messages: list[dict[str, str]] = row["messages"]
    default = 0 if messages[0]["role"] == "system" else _DEFAULT_SYSTEM_TOKENS
    return default + sum(_MESSAGE_TOKENS + student_tokens(m["content"]) for m in messages)


def _usable(row: dict[str, Any]) -> bool:
    if "label" in row:
        return bool(row["input"]) and bool(row["label"])
    messages: list[dict[str, str]] = row["messages"]
    return len(messages) > 1 and bool(messages[-1]["content"])


def derive_rows(
    records: Sequence[TraceRecord], *, structure_class: str, max_seq_length: int
) -> tuple[list[dict[str, Any]], DeriveStats]:
    """Build one training row per usable record in the format ``structure_class`` maps to.

    The target is the record's answer: its text, or its first tool call's name
    and arguments as one JSON object when it made one -- the same text the
    structure class was judged on. ``max_seq_length`` is a character budget on the prompt
    side (the SDK has no tokenizer): a ``labeled-example`` input is cut from the
    front so the last turn survives, a ``chat-messages`` row drops its oldest
    turns. Records with an empty prompt or an empty (normalized) answer are
    skipped and counted.
    """
    format_key = _format_key(structure_class)
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


@dataclass(frozen=True, slots=True)
class _Example:
    """One derived row as a plan keeps it: where it came from and its digest, never its text."""

    position: int
    ts: datetime
    session_id: str | None
    digest: str
    stratum: str
    truth_changed: bool
    over_budget: bool


@dataclass(frozen=True, slots=True)
class _Selection:
    """The rows a dataset keeps (their examples, in row order), their split and the stats."""

    examples: list[_Example]
    split: dict[str, list[int]]
    stats: dict[str, Any]


def _format_key(structure_class: str) -> str:
    format_key = FORMAT_BY_STRUCTURE.get(structure_class)
    if format_key is None:
        raise ValueError(
            f"unknown structure class {structure_class!r}; "
            f"expected one of {', '.join(FORMAT_BY_STRUCTURE)}"
        )
    return format_key


@dataclass(slots=True)
class _Plan:
    """One workload's rows, decided a record at a time without keeping any of their text.

    Each record is redacted and derived as it arrives (redaction first, so the
    character budget can never cut an identifier in half); only the row's
    digest, label, time and session are kept. :meth:`select` then dedups,
    splits and caps those, and the chosen rows are derived again from their
    records -- so a workload of any size costs memory for its kept rows only.
    """

    format_key: str
    max_seq_length: int
    counts: Counter[str] = field(default_factory=lambda: Counter(dict.fromkeys(PII_POLICY, 0)))
    rows_changed: int = 0
    skipped: int = 0
    truncated: int = 0
    examples: list[_Example] = field(default_factory=list[_Example])

    def row(self, record: TraceRecord) -> tuple[dict[str, Any] | None, bool, RedactStats, bool]:
        """``(the row, or None when unusable; truncated; redaction stats; truth changed)``."""
        (redacted,), stats = redact_records([record])
        row, truncated = _BUILDERS[self.format_key](redacted, self.max_seq_length)
        changed = redacted.response != _answer(record)
        return (row if _usable(row) else None), truncated, stats, changed

    def add(self, position: int, record: TraceRecord) -> None:
        row, truncated, stats, changed = self.row(record)
        self.counts.update(stats.counts)
        self.rows_changed += stats.rows_changed
        if row is None:
            self.skipped += 1
            return
        self.truncated += truncated
        digest = canonical_row_hash(row)
        # The cap's stratum is the target, so a router's rare route keeps its share.
        target = hashlib.blake2b(_target(row).encode("utf-8"), digest_size=8).hexdigest()
        budget = TOKEN_BUDGET.get(self.format_key)
        over = budget is not None and _row_tokens(row) > budget
        self.examples.append(
            _Example(position, record.ts, record.session_id, digest, target, changed, over)
        )

    def select(self, *, holdout_share: float, max_train_rows: int) -> _Selection:
        """Dedup (first in time order), split by time, then thin the training rows.

        A training row over the student's token budget is dropped, and
        the rest are capped.
        """
        derived = sorted(self.examples, key=lambda e: e.position)
        seen: set[str] = set()
        kept = [e for e in derived if not (e.digest in seen or seen.add(e.digest))]
        whole = time_split(kept, holdout_share=holdout_share)
        fits = [i for i in whole["train"] if not kept[i].over_budget]
        train = cap_train(
            fits,
            [e.stratum for e in kept],
            [e.digest for e in kept],
            limit=max_train_rows,
            max_strata=MAX_STRATA.get(self.format_key),
        )
        keep = sorted(train + whole["eval_holdout"])
        renumber = {old: new for new, old in enumerate(keep)}
        split = {
            name: [renumber[i] for i in picked]
            for name, picked in (("train", train), ("eval_holdout", whole["eval_holdout"]))
        }
        final = [kept[i] for i in keep]
        boundary = split_boundary(final, split)
        stats = {
            "format_key": self.format_key,
            "derive": {
                "rows": len(derived),
                "skipped": self.skipped,
                "truncated": self.truncated,
                "truncation_rate": self.truncated / len(derived) if derived else 0.0,
            },
            "redact": {
                "counts": dict(self.counts),
                "pass_list": list(PII_POLICY),
                "rows_changed": self.rows_changed,
                "truths_changed": sum(e.truth_changed for e in final),
            },
            "dedup": {"removed": len(derived) - len(kept)},
            "budget": {
                "max_tokens": TOKEN_BUDGET.get(self.format_key),
                "dropped": len(whole["train"]) - len(fits),
            },
            "cap": {
                "limit": max_train_rows,
                "train_rows": len(fits),
                "dropped": len(fits) - len(train),
            },
            "boundary_ts": boundary.isoformat() if boundary else None,
        }
        return _Selection(final, split, stats)

    def dataset(self, rows: Mapping[int, Any], selection: _Selection) -> WorkloadDataset:
        """The dataset of the selected rows (``rows`` holds them by position), in row order."""
        return WorkloadDataset(
            rows=[rows[e.position] for e in selection.examples],
            split=selection.split,
            stats=selection.stats,
        )


def build_dataset(
    records: Sequence[TraceRecord],
    *,
    structure_class: str,
    max_seq_length: int,
    holdout_share: float = HOLDOUT_SHARE,
    max_train_rows: int = MAX_TRAIN_ROWS,
) -> WorkloadDataset:
    """Redact, derive, dedup and time-split one workload's records, keeping rows and records aligned.

    Redaction comes first, so the character budget can never cut an identifier
    in half and leave the part a detector no longer recognises. Dedup runs on
    the redacted rows: two calls that differ only in an email address are one
    example once it is masked, and keeping both would put the same row on both
    sides of the split. A ``chat-messages`` training row the student's token
    budget cannot hold is dropped (the recipe would drop it after the
    credits are spent), and the rest are capped at ``max_train_rows`` (a
    proportional sample by target, so a candidate trains inside its recipe's
    hard ceiling); the holdout is kept whole, as serving will see it.
    ``stats["redact"]["truths_changed"]`` counts the kept rows whose target
    redaction rewrote, since those are scored against the placeholder rather
    than the value; ``stats["budget"]`` and ``stats["cap"]`` say how many
    training rows the budget and the cap dropped.
    """
    plan = _Plan(_format_key(structure_class), max_seq_length)
    for position, record in enumerate(records):
        plan.add(position, record)
    selection = plan.select(holdout_share=holdout_share, max_train_rows=max_train_rows)
    rows = {e.position: plan.row(records[e.position])[0] for e in selection.examples}
    return plan.dataset(rows, selection)


def derive_workloads(
    records: Callable[[], Iterable[TraceRecord]],
    workloads: Sequence[Workload],
    *,
    max_seq_length: int,
) -> dict[str, WorkloadDataset]:
    """Each workload's :func:`build_dataset`, from two more passes over the export.

    ``records()`` streams the export as :func:`~dagnam.audit.discover.discover_workloads`
    numbered it. The first pass plans every workload a record at a time; the
    second derives only the rows each keeps. Nothing else of the export is ever
    held, so memory follows the kept rows, not the export.
    """
    if not workloads:
        return {}
    place = {i: (w.id, at) for w in workloads for at, i in enumerate(w.record_indices)}
    plans = {w.id: _Plan(_format_key(w.structure_class.value), max_seq_length) for w in workloads}
    for index, record in enumerate(records()):
        if (spot := place.get(index)) is not None:
            plans[spot[0]].add(spot[1], record)
    selections = {
        workload_id: plan.select(holdout_share=HOLDOUT_SHARE, max_train_rows=MAX_TRAIN_ROWS)
        for workload_id, plan in plans.items()
    }
    wanted = {(wid, e.position) for wid, sel in selections.items() for e in sel.examples}
    rows: dict[str, dict[int, Any]] = {w.id: {} for w in workloads}
    for index, record in enumerate(records()):
        if (spot := place.get(index)) in wanted:
            rows[spot[0]][spot[1]] = plans[spot[0]].row(record)[0]
    return {wid: plans[wid].dataset(rows[wid], sel) for wid, sel in selections.items()}


__all__ = [
    "FORMAT_BY_STRUCTURE",
    "MAX_STRATA",
    "TOKEN_BUDGET",
    "DedupStats",
    "DeriveStats",
    "WorkloadDataset",
    "build_dataset",
    "dedup_rows",
    "derive_rows",
    "derive_workloads",
    "normalize_label",
]
