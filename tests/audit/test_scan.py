"""``scan_export``: the pure part of ``dagnam audit scan``, as a library call."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import dataclasses
import json
from pathlib import Path
import socket
from typing import Any

import pytest
from tests.audit.test_scan_cli_characterization import (
    CALLS,
    FIXTURES,
    GOLDEN,
    _generated_export,
    _prices,
)
from tests.typing_helpers import PytestMonkeyPatch, StrCapture

from dagnam.audit import NoTracesError, PriceTable, ScanResult, TraceRecord, scan_export
from dagnam.audit.readers import Source
import dagnam.audit.scan as scan_module


def assign(target: object, name: str, value: object) -> None:
    setattr(target, name, value)


def _golden(case: str) -> dict[str, Any]:
    golden: dict[str, Any] = json.loads(GOLDEN.read_text(encoding="utf-8"))[case]
    return golden["report"]


def _ids(report: Mapping[str, Any]) -> list[str]:
    return [w["id"] for w in report["workloads"]]


@pytest.fixture
def generated(tmp_path: Path) -> Path:
    (tmp_path / "export").mkdir()
    return _generated_export(tmp_path / "export")


@pytest.mark.parametrize(("case", "source"), [("openai", "openai"), ("dagnam", "dagnam")])
def test_a_fixture_scans_to_the_workloads_the_command_reported(
    case: str, source: Source, tmp_path: Path
) -> None:
    name = "recorder_openai_chat.jsonl" if case == "dagnam" else f"{case}_sample.jsonl"
    result = scan_export(FIXTURES / name, source=source, price_table=_prices(tmp_path))

    golden = _golden(case)
    assert [w.id for w in result.workloads] == _ids(golden)
    assert result.datasets == {}
    assert result.source == source
    assert result.table.version == golden["price_table_version"]


def test_a_generated_export_derives_the_rows_the_command_wrote(
    generated: Path, tmp_path: Path
) -> None:
    result = scan_export(
        generated,
        source="jsonl",
        column_map={"response": "out"},
        window_days=60,
        sample_rate=0.5,
        price_table=_prices(tmp_path),
    )

    golden = _golden("generated")
    assert [w.id for w in result.workloads] == _ids(golden)
    derived = {w["id"]: w["dataset"]["rows"] for w in golden["workloads"] if w["dataset"]}
    assert derived
    assert {key: len(ds.rows) for key, ds in result.datasets.items()} == derived
    assert (result.window.days, result.window.sample_rate) == (
        golden["window"]["days"],
        golden["window"]["sample_rate"],
    )
    assert result.window.start.isoformat() == golden["window"]["start"]
    assert result.window.end.isoformat() == golden["window"]["end"]


def test_the_stats_are_final_when_scan_export_returns(generated: Path) -> None:
    result = scan_export(generated, source="jsonl", column_map={"response": "out"})

    assert result.stats.rows_seen == CALLS + 3
    assert result.stats.rows_kept == CALLS + 3
    assert result.stats.rows_malformed == 0


def test_an_export_with_no_traces_raises_no_traces_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(NoTracesError, match="no traces to audit") as raised:
        scan_export(empty, source="jsonl")

    assert isinstance(raised.value, ValueError)
    assert not isinstance(raised.value, SystemExit)
    assert str(empty) in str(raised.value)


def test_a_price_table_overrides_the_bundled_one(tmp_path: Path) -> None:
    sample = FIXTURES / "openai_sample.jsonl"
    bundled = scan_export(sample, source="openai")
    custom = scan_export(sample, source="openai", price_table=_prices(tmp_path))

    assert bundled.table.version == PriceTable.load(None).version
    assert custom.table.version == "golden-1"
    assert bundled.table.version != custom.table.version


def test_a_sample_rate_scales_the_volume_as_the_flag_does() -> None:
    sample = FIXTURES / "openai_sample.jsonl"
    full = scan_export(sample, source="openai")
    sampled = scan_export(sample, source="openai", sample_rate=0.25)

    for whole, scaled in zip(full.workloads, sampled.workloads, strict=True):
        assert scaled.calls == whole.calls
        assert scaled.calls_per_day == pytest.approx(4 * whole.calls_per_day)
    assert (full.window.sample_rate, sampled.window.sample_rate) == (1.0, 0.25)


@pytest.mark.parametrize("sample_rate", [0.0, 1.5])
def test_a_sample_rate_outside_the_unit_interval_is_a_value_error(sample_rate: float) -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        scan_export(FIXTURES / "openai_sample.jsonl", source="openai", sample_rate=sample_rate)


@pytest.mark.parametrize("window_days", [0, -3])
def test_a_window_under_a_day_is_a_value_error(window_days: int) -> None:
    with pytest.raises(ValueError, match="window_days"):
        scan_export(FIXTURES / "openai_sample.jsonl", source="openai", window_days=window_days)


def test_the_window_defaults_to_thirty_days() -> None:
    result = scan_export(FIXTURES / "openai_sample.jsonl", source="openai")

    assert result.window.days == 30.0


def test_a_column_map_works_for_jsonl_and_is_refused_for_a_fixed_source() -> None:
    with pytest.raises(ValueError, match="column_map applies to the jsonl and csv sources only"):
        scan_export(FIXTURES / "openai_sample.jsonl", source="openai", column_map={"response": "x"})
    with pytest.raises(ValueError, match="column_map applies"):
        scan_export(FIXTURES / "recorder_openai_chat.jsonl", source="dagnam", column_map={"a": "b"})


def test_scan_export_is_pure(
    generated: Path,
    tmp_path: Path,
    monkeypatch: PytestMonkeyPatch,
    capsys: StrCapture,
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network")

    monkeypatch.setattr(socket, "socket", refuse)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    before = sorted(tmp_path.rglob("*"))

    result = scan_export(generated, source="jsonl", column_map={"response": "out"})

    assert result.datasets
    assert sorted(tmp_path.rglob("*")) == before
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")


def test_the_export_is_streamed_in_three_passes_and_never_held(
    generated: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    opened: list[Iterator[TraceRecord]] = []
    real = scan_module.read_traces

    def counting(*args: Any, **kwargs: Any) -> Any:
        records, stats = real(*args, **kwargs)
        opened.append(records)
        return records, stats

    monkeypatch.setattr(scan_module, "read_traces", counting)
    result = scan_export(generated, source="jsonl", column_map={"response": "out"})

    assert result.datasets
    assert len(opened) == 3  # discover, plan, derive
    assert all(not isinstance(records, list | tuple) for records in opened)


def test_an_export_with_nothing_worth_auditing_is_read_once(
    monkeypatch: PytestMonkeyPatch,
) -> None:
    calls: list[int] = []
    real = scan_module.read_traces

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(scan_module, "read_traces", counting)
    result = scan_export(FIXTURES / "openai_sample.jsonl", source="openai")

    assert result.datasets == {}
    assert len(calls) == 1


def test_a_scan_result_is_frozen_and_slotted() -> None:
    result = scan_export(FIXTURES / "openai_sample.jsonl", source="openai")

    assert dataclasses.is_dataclass(result)
    assert [f.name for f in dataclasses.fields(ScanResult)] == [
        "workloads",
        "datasets",
        "table",
        "window",
        "source",
        "stats",
    ]
    assert isinstance(result.workloads, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        assign(result, "source", "other")
    assert not hasattr(result, "__dict__")
