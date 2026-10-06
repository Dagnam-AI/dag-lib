"""``read_traces`` dispatch through the source registry."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from dagnam.audit import UnsupportedExportError, read_traces
from dagnam.audit.discover import discover_workloads
from dagnam.audit.readers import READERS, Source, generic, langfuse, langsmith, openai_jsonl


def test_registry_covers_every_documented_source() -> None:
    assert set(READERS) == {"langfuse", "langsmith", "openai", "jsonl", "csv"}
    assert READERS["langfuse"](None) is langfuse.READER
    assert READERS["langsmith"](None) is langsmith.READER
    assert READERS["openai"](None) is openai_jsonl.READER
    assert READERS["jsonl"] is generic.bind
    assert READERS["csv"] is generic.bind


def test_unknown_source_is_rejected_before_reading(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown source 'helicone'; expected one of csv, jsonl"):
        read_traces(tmp_path / "missing.jsonl", source=cast("Source", "helicone"))


def test_column_map_is_rejected_for_vendor_sources(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="column_map applies to the jsonl and csv sources only"):
        read_traces(tmp_path / "missing.jsonl", source="langfuse", column_map={"system": "s"})


def test_vendor_export_missing_required_fields(fixtures_dir: Path) -> None:
    records, _ = read_traces(fixtures_dir / "generic_sample.jsonl", source="langfuse")
    with pytest.raises(UnsupportedExportError) as info:
        next(records)
    assert info.value.source == "langfuse"
    assert info.value.missing == langfuse.REQUIRED_FIELDS
    assert str(info.value).startswith("langfuse export is missing required field(s) ")


NAMED_IDS = ["983692dea8aecce9", "be7a84cfa8530536", "ee55f97cc0aad676", "ee60e4f2c7340b71"]
TEMPLATE_IDS = ["41615658691038a0", "47cf5214326d057a", "59cf4830241f021d", "d199932a5e7455b2"]
GENERIC_MAP = {
    "trace_id": "call_id",
    "ts": "at",
    "system": "sys",
    "messages": "prompt",
    "response": "out",
    "workload_hint": "kind",
}
UNNAMED_MAP = {**GENERIC_MAP, "workload_hint": "no_such_column"}  # keyed by template instead


@pytest.mark.parametrize(
    ("name", "source", "column_map", "ids"),
    [
        ("langfuse_sample.jsonl", "langfuse", None, NAMED_IDS),
        ("langsmith_sample.jsonl", "langsmith", None, NAMED_IDS),
        ("openai_sample.jsonl", "openai", None, NAMED_IDS),
        ("generic_sample.jsonl", "jsonl", GENERIC_MAP, NAMED_IDS),
        ("generic_sample.csv", "csv", GENERIC_MAP, NAMED_IDS),
        ("generic_sample.jsonl", "jsonl", UNNAMED_MAP, TEMPLATE_IDS),
        ("generic_sample.csv", "csv", UNNAMED_MAP, TEMPLATE_IDS),
    ],
)
def test_the_sample_exports_keep_their_workload_ids(
    fixtures_dir: Path,
    name: str,
    source: Source,
    column_map: dict[str, str] | None,
    ids: list[str],
) -> None:
    # A workload id is the digest of the normalized system prompt (or the step's name): a
    # change to the readers, the reasoning rules or the normalizer must never move one.
    records, _ = read_traces(fixtures_dir / name, source=source, column_map=column_map)
    assert sorted(w.id for w in discover_workloads(list(records), window_days=30)) == ids
