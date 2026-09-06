"""The on-disk workload layout: dataset.jsonl, split.json, meta.json, written atomically."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dagnam.audit.workspace import SCHEMA, write_workload

ROWS = [{"input": "a", "label": "x"}, {"input": "b", "label": "y"}, {"input": "c", "label": "x"}]
SPLIT = {"train": [0, 1], "eval_holdout": [2]}
STATS = {"format_key": "labeled-example", "boundary_ts": "2026-08-01T00:00:00+00:00"}


def test_workspace_layout_and_meta(tmp_path: Path) -> None:
    out = write_workload(tmp_path, "wl-1", ROWS, SPLIT, STATS)

    assert out == tmp_path / "workloads" / "wl-1"
    assert sorted(p.name for p in out.iterdir()) == ["dataset.jsonl", "meta.json", "split.json"]

    lines = (out / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == ROWS
    assert json.loads((out / "split.json").read_text(encoding="utf-8")) == {
        "member_row_indices": SPLIT
    }
    assert json.loads((out / "meta.json").read_text(encoding="utf-8")) == {
        "schema": SCHEMA,
        "workload_id": "wl-1",
        "rows": 3,
        "splits": {"train": 2, "eval_holdout": 1},
        "stats": STATS,
    }


def test_rewrite_replaces_the_files_and_leaves_no_temp_files(tmp_path: Path) -> None:
    write_workload(tmp_path, "wl-1", ROWS, SPLIT, STATS)
    out = write_workload(tmp_path, "wl-1", ROWS[:1], {"train": [0], "eval_holdout": []}, {})

    assert (out / "dataset.jsonl").read_text(encoding="utf-8") == json.dumps(ROWS[0]) + "\n"
    assert json.loads((out / "meta.json").read_text(encoding="utf-8"))["rows"] == 1
    assert not list(out.glob("*.tmp"))


def test_split_indices_must_address_the_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="row indices"):
        write_workload(tmp_path, "wl-1", ROWS, {"train": [0, 3], "eval_holdout": []}, {})


def test_non_ascii_rows_round_trip(tmp_path: Path) -> None:
    rows = [{"messages": [{"role": "user", "content": "héllo — 日本"}]}]
    out = write_workload(tmp_path, "wl-2", rows, {"train": [0], "eval_holdout": []}, {})
    assert json.loads((out / "dataset.jsonl").read_text(encoding="utf-8")) == rows[0]
