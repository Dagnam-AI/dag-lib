"""``scan_export``: read a trace export, discover and price its workloads, derive their rows.

This is the pure half of ``dagnam audit scan``: it reads the export and returns what it found.
It writes no file, opens no socket and prints nothing, so a caller that wants the report on
disk passes the result to :func:`dagnam.audit.scan_report.write_scan`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from dagnam.audit.derive import WorkloadDataset, derive_workloads
from dagnam.audit.discover import Workload, by_spend, discover_workloads
from dagnam.audit.economics import price_workload, replaceability
from dagnam.audit.orchestrate import AUDITED_VERDICTS
from dagnam.audit.prices import PriceTable
from dagnam.audit.readers import ReadStats, Source, read_traces
from dagnam.audit.record import TraceRecord
from dagnam.audit.scan_report import Window, scan_window
from dagnam.audit.thresholds import MAX_SEQ_LENGTH, WINDOW_DAYS


class NoTracesError(ValueError):
    """The export holds no trace to audit."""


@dataclass(frozen=True, slots=True)
class ScanResult:
    """What a scan found: the priced workloads and the rows derived for the ones worth auditing."""

    workloads: tuple[Workload, ...]
    """Every workload, most monthly spend first."""
    datasets: dict[str, WorkloadDataset]
    """The derived dataset of each audited workload, by workload id."""
    table: PriceTable
    """The price table the workloads were priced with."""
    window: Window
    """The window the per-day rates were computed over."""
    source: str
    """The export format that was read."""
    stats: ReadStats
    """Row accounting of the read; final, the export having been read to its end."""


def scan_export(
    path: Path,
    *,
    source: Source,
    column_map: Mapping[str, str] | None = None,
    window_days: int | None = None,
    sample_rate: float = 1.0,
    price_table: Path | None = None,
) -> ScanResult:
    """Read the export at ``path``, discover and price its workloads, derive the rows to audit.

    The export is streamed, never held whole: once to discover the workloads, then twice more
    to plan and derive the rows of the ones worth auditing. ``window_days`` is the window the
    per-day rates assume when the timestamps span less (30 when ``None``); ``sample_rate`` is
    the share of the traffic a sampled export holds, and volume and spend are scaled up by its
    inverse. ``price_table`` replaces the bundled vendor price table.

    Raises :class:`NoTracesError` when the export holds no trace, and :class:`ValueError` for
    a ``column_map`` on a source that takes none, a ``window_days`` under 1, or a
    ``sample_rate`` outside ``(0, 1]``. Reading can also raise
    :class:`~dagnam.audit.readers.UnsupportedExportError` and
    :class:`~dagnam.audit.readers.MalformedExportError`.
    """
    days = WINDOW_DAYS if window_days is None else window_days
    if days < 1:
        raise ValueError(f"window_days must be at least 1; got {days}")

    def records() -> Iterator[TraceRecord]:
        return read_traces(path, source=source, column_map=column_map)[0]

    records_first, stats = read_traces(path, source=source, column_map=column_map)
    found = discover_workloads(records_first, window_days=days, sample_rate=sample_rate)
    if not found:
        raise NoTracesError(f"{path}: no traces to audit")
    table = PriceTable.load(price_table)
    workloads = by_spend(price_workload(w, table) for w in found)  # sorted once priced
    audited = [w for w in workloads if replaceability(w).status in AUDITED_VERDICTS]
    datasets = derive_workloads(records, audited, max_seq_length=MAX_SEQ_LENGTH)
    window = scan_window(workloads, window_days=days, sample_rate=sample_rate)
    return ScanResult(workloads, datasets, table, window, source, stats)
