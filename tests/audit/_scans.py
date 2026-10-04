"""Scans written into a directory, for the tests of what a rescan keeps, replaces and removes."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tests.audit._records import make_record, make_workload

from dagnam.audit.derive import build_dataset
from dagnam.audit.prices import PriceRow, PriceTable
from dagnam.audit.scan_report import Window, write_scan

WINDOW = Window(
    start=datetime(2026, 8, 1, tzinfo=UTC), end=datetime(2026, 8, 31, tzinfo=UTC), days=30.0
)
FRESH = PriceTable(
    version="2026-09",
    as_of=datetime.now(UTC).date(),
    rows={"m": PriceRow("m", 1.0, 2.0, None, None)},
)


def labels(count: int) -> Any:
    records = [make_record(system=f"Label {i}.", response="ab"[i % 2]) for i in range(count)]
    return build_dataset(records, structure_class="enum_label", max_seq_length=2_048)


def scan(out_dir: Path, datasets: dict[str, Any], **kwargs: Any) -> None:
    workloads = [replace(make_workload(), id=key) for key in datasets]
    write_scan(
        out_dir, workloads, datasets, source="jsonl", window=WINDOW, price_table=FRESH, **kwargs
    )
