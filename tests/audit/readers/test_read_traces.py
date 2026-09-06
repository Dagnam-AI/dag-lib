"""``read_traces`` dispatch through the source registry."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from dagnam.audit import UnsupportedExportError, read_traces
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
