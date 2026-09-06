"""The on-disk workload layout: ``workloads/<id>/{dataset.jsonl, split.json, meta.json}``.

Every file is written to a same-directory ``.tmp`` and promoted with
``os.replace``, so a reader never sees a partial file. ``split.json`` is the
exact body ``create_explicit_splits`` sends (``explicit_splits_body``), so
Task 8 uploads it verbatim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any

from dagnam._core.client.datasets import explicit_splits_body

SCHEMA = "dagnam.audit.workload/1"


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a same-directory ``.tmp`` promoted with ``os.replace``."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_workload(
    out_dir: Path,
    workload_id: str,
    rows: Sequence[Mapping[str, Any]],
    split: Mapping[str, Sequence[int]],
    stats: Mapping[str, Any],
) -> Path:
    """Write one workload's files under ``out_dir/workloads/<workload_id>`` and return that directory.

    ``stats`` is recorded verbatim under ``meta.json``'s ``stats`` key
    (``build_dataset`` produces the expected shape). The workload id is a
    plain string so this composes with Task 4's ``Workload`` without importing it.
    """
    members = {name: list(indices) for name, indices in split.items()}
    out_of_range = sorted(
        {i for indices in members.values() for i in indices if not 0 <= i < len(rows)}
    )
    if out_of_range:
        raise ValueError(f"split names row indices outside 0..{len(rows) - 1}: {out_of_range[:5]}")

    workload_dir = out_dir / "workloads" / workload_id
    workload_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(
        workload_dir / "dataset.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )
    write_atomic(workload_dir / "split.json", json.dumps(explicit_splits_body(members), indent=2))
    meta = {
        "schema": SCHEMA,
        "workload_id": workload_id,
        "rows": len(rows),
        "splits": {name: len(indices) for name, indices in members.items()},
        "stats": dict(stats),
    }
    write_atomic(workload_dir / "meta.json", json.dumps(meta, indent=2, ensure_ascii=False))
    return workload_dir


__all__ = ["SCHEMA", "write_atomic", "write_workload"]
