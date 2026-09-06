"""Shared fixtures for the audit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from dagnam.audit import TraceRecord, read_traces

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def langfuse_records() -> list[TraceRecord]:
    records, _ = read_traces(FIXTURES / "langfuse_sample.jsonl", source="langfuse")
    return list(records)
