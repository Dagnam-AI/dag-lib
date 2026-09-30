"""CLI ``audit scan``: what it reads, what it writes, and the flags it parses."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import socket
import sys
from typing import TYPE_CHECKING, Any

import pytest

from dagnam.audit.candidates import CandidateKind
from dagnam.audit.normalize import template_hash
from dagnam.audit.state import AuditState, StepState, lock_audit, save_state
from dagnam.cli.audit import parse_credits, parse_map, parse_sample_rate, parse_window

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture

HEAD, SFT, HOSTED = CandidateKind.HEAD_TUNE, CandidateKind.SFT_SMALL, CandidateKind.HOSTED_FLOOR
CALLS = 1_200
"""Enough traces for one workload to clear ``MIN_TRACES_PER_WORKLOAD`` and ``MIN_HOLDOUT``."""


@pytest.fixture(autouse=True)
def no_real_keyring(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)


def _row(i: int, *, response_key: str = "response") -> dict[str, object]:
    return {
        "trace_id": f"t{i}",
        "ts": f"2026-08-{1 + i % 28:02d}T{i % 24:02d}:{i % 60:02d}:00Z",
        "model": "gpt-4o-mini",
        "system": "Label the ticket.",
        "messages": [{"role": "user", "content": f"ticket {i} from customer@example.com"}],
        response_key: "ab"[i % 2],
        "prompt_tokens": 100,
        "completion_tokens": 2,
        "latency_ms": 500,
        "cost_usd": 0.6,  # $720/month against a $50 maintenance floor: a candidate
    }


@pytest.fixture
def export(tmp_path: Path) -> Path:
    """A generic JSONL export: one label workload worth auditing plus one tiny free-text one."""
    rows = [_row(i) for i in range(CALLS)]
    rows += [
        {**_row(CALLS + i), "system": "Write a reply.", "response": f"Dear customer {i}, " * 20}
        for i in range(3)
    ]
    path = tmp_path / "traces.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_scan_opens_no_network(
    run_cli: CliRunner,
    export: Path,
    tmp_path: Path,
    monkeypatch: PytestMonkeyPatch,
    capsys: StrCapture,
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network")

    monkeypatch.setattr(socket, "socket", refuse)
    out = tmp_path / "audit"
    assert run_cli(["audit", "scan", str(export), "--source", "jsonl", "--out", str(out)]) == 0

    assert (out / "scan-report.md").exists()
    assert (out / "workloads").is_dir()
    report = json.loads((out / "scan-report.json").read_text(encoding="utf-8"))
    label, free_text = report["workloads"]
    assert label["verdict"]["status"] == "candidate"
    assert label["dataset"] == {
        "rows": CALLS,
        "dedup_removed": 0,
        "redactions": CALLS,
        "truncated": 0,
        "truths_redacted": 0,
        "train_rows_capped": 0,
        "train_rows_over_budget": 0,
        "split": {"train": CALLS - CALLS // 5, "eval_holdout": CALLS // 5},
    }
    assert report["pii"]["counts"]["PII_EMAIL"] == CALLS
    assert report["pii"]["pass_list"] == list(report["pii"]["counts"])
    assert report["window"]["days"] == 30.0
    workload_dir = out / "workloads" / label["id"]
    assert {p.name for p in workload_dir.iterdir()} == {"dataset.jsonl", "split.json", "meta.json"}
    assert "customer@example.com" not in workload_dir.joinpath("dataset.jsonl").read_text()
    assert free_text["verdict"]["status"] == "not_audited"
    assert free_text["dataset"] is None
    captured = capsys.readouterr()
    assert "1 candidate, 1 not_audited" in captured.out
    assert f"Next: dagnam audit run {out}" in captured.err


def test_scan_with_nothing_worth_auditing_writes_no_workloads(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    sample = Path(__file__).parents[1] / "audit" / "fixtures" / "langfuse_sample.jsonl"
    out = tmp_path / "audit"
    assert run_cli(["audit", "scan", str(sample), "--source", "langfuse", "--out", str(out)]) == 0
    captured = capsys.readouterr()
    assert "Next:" not in captured.err
    assert not (out / "workloads").exists()
    report = json.loads((out / "scan-report.json").read_text(encoding="utf-8"))
    assert {w["verdict"]["status"] for w in report["workloads"]} <= {
        "too_few_samples",
        "not_audited",
    }
    assert all(w["dataset"] is None for w in report["workloads"])


def test_scan_json_map_window_and_price_table(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text(
        "".join(json.dumps(_row(i, response_key="out")) + "\n" for i in range(CALLS)),
        encoding="utf-8",
    )
    table = tmp_path / "prices.json"
    table.write_text(
        json.dumps({"version": "custom-1", "as_of": "2026-09-01", "rows": {}}), encoding="utf-8"
    )
    out = tmp_path / "audit"
    argv = [
        "audit",
        "scan",
        str(path),
        "--source",
        "jsonl",
        "--map",
        "response=out",
        "--window",
        "60d",
        "--price-table",
        str(table),
        "--out",
        str(out),
        "--json",
    ]
    assert run_cli(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads((out / "scan-report.json").read_text(encoding="utf-8"))
    assert printed["window"]["days"] == 60.0
    assert printed["price_table_version"] == "custom-1"
    assert printed["workloads"][0]["calls"] == CALLS


def test_scan_empty_export_fails_with_a_reason(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "scan", str(empty), "--source", "jsonl", "--out", str(tmp_path)])
    assert exc.value.code == 1
    assert "no traces to audit" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "scan", str(empty), "--source", "jsonl", "--json"])
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["error"].endswith("no traces to audit")


def test_scan_flag_parsers() -> None:
    assert parse_window("30d") == 30
    assert parse_window("7") == 7
    # "\u00b2" is a digit to `str.isdigit` and not a number to `int`.
    for bad in ("0d", "d", "-3", "month", "\u00b2", "\u00b2d"):
        with pytest.raises(argparse.ArgumentTypeError, match="--window expects"):
            parse_window(bad)
    assert parse_credits("0") == 0
    assert parse_credits("500") == 500
    for bad in ("-5", "5.5", "\u00b2", "many"):
        with pytest.raises(argparse.ArgumentTypeError, match="--max-credits expects"):
            parse_credits(bad)
    assert parse_map(None) is None
    assert parse_map([]) is None
    assert parse_map(["response=out", "system=sys"]) == {"response": "out", "system": "sys"}
    for bad in ("response", "=out", "response="):
        with pytest.raises(argparse.ArgumentTypeError, match="--map expects"):
            parse_map([bad])


def test_scan_argparse_rejects_bad_window(run_cli: CliRunner, capsys: StrCapture) -> None:
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "scan", "x.jsonl", "--source", "jsonl", "--window", "soon"])
    assert exc.value.code == 2
    assert "--window expects" in capsys.readouterr().err


T0 = 1_788_220_800  # 2026-09-01T00:00:00Z


def _oai(
    i: int,
    system: str,
    content: str | None,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    prompt_tokens: int = 2_000,
) -> dict[str, Any]:
    """One OpenAI ``{request, response}`` line, a minute after the previous one."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "request": {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"t{i}"},
            ],
        },
        "response": {
            "id": f"chatcmpl-{i}",
            "created": T0 + i * 60,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": message}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 20},
        },
    }


def _export(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = tmp_path / "export.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _dear_prices(tmp_path: Path) -> Path:
    """A table that makes gpt-4o-mini dear enough for 1,200 calls to be worth auditing."""
    row = {"input_per_m": 400.0, "output_per_m": 1_000.0, "cached_input_per_m": None}
    table = {
        "version": "dear",
        "as_of": datetime.now(UTC).date().isoformat(),
        "rows": {"gpt-4o-mini": {**row, "cheaper_variant": None}},
    }
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    return path


def _scan(run_cli: CliRunner, export: Path, out: Path, *extra: str) -> dict[str, Any]:
    argv = ["audit", "scan", str(export), "--source", "openai", "--out", str(out), *extra]
    assert run_cli(argv) == 0
    return json.loads((out / "scan-report.json").read_text(encoding="utf-8"))


def test_a_tool_call_extraction_workload_is_audited(run_cli: CliRunner, tmp_path: Path) -> None:
    # 1,200 function-calling extractions with ``content: null`` were json_object at
    # $998/month, then "too_few_samples: 0 holdout rows" because none derived.
    rows = []
    for i in range(CALLS):
        arguments = json.dumps({"product": f"p{i % 9}", "sentiment": ["pos", "neg"][i % 2]})
        function = {"name": "record_ticket", "arguments": arguments}
        call = {"id": f"c{i}", "type": "function", "function": function}
        rows.append(_oai(i, "Extract fields by calling record_ticket.", None, tool_calls=[call]))
    out = tmp_path / "audit"

    report = _scan(
        run_cli, _export(tmp_path, rows), out, "--price-table", str(_dear_prices(tmp_path))
    )

    (workload,) = report["workloads"]
    assert (workload["structure_class"], workload["response_mode"]) == ("json_object", "tool_call")
    assert workload["verdict"]["status"] == "candidate"
    assert workload["dataset"]["split"]["eval_holdout"] == CALLS // 5
    assert any("answers with tool calls" in w for w in report["warnings"])
    first = json.loads(
        (out / "workloads" / workload["id"] / "dataset.jsonl").read_text().splitlines()[0]
    )
    assert json.loads(first["messages"][-1]["content"]) == {
        "arguments": {"product": "p0", "sentiment": "pos"},
        "name": "record_ticket",
    }


def test_redacted_json_targets_stay_json_and_are_counted(
    run_cli: CliRunner, tmp_path: Path
) -> None:
    # A contact extraction whose targets carry an email and a Luhn-valid card number.
    rows = [
        _oai(i, "Extract the contact as JSON.", json.dumps({"card": 4111111111111111, "id": i}))
        for i in range(CALLS)
    ]
    out = tmp_path / "audit"

    report = _scan(
        run_cli, _export(tmp_path, rows), out, "--price-table", str(_dear_prices(tmp_path))
    )

    (workload,) = report["workloads"]
    assert workload["dataset"]["truths_redacted"] == CALLS
    lines = (out / "workloads" / workload["id"] / "dataset.jsonl").read_text().splitlines()
    targets = [json.loads(json.loads(line)["messages"][-1]["content"]) for line in lines]
    assert {t["card"] for t in targets} == {"[REDACTED:PII_PAYMENT_CARD]"}
    warning = f"{workload['id']}: redaction rewrote 1,200 of 1,200 training targets"
    assert warning in (out / "scan-report.md").read_text()


def test_one_epoch_timestamp_does_not_stretch_the_window(
    run_cli: CliRunner, tmp_path: Path, export: Path
) -> None:
    # One ``created: 0`` among a day of calls made the window 20,697 days and the
    # $998 candidate a $1.45 not_worth_it.
    lines = export.read_text(encoding="utf-8").splitlines()
    lines[5] = json.dumps({**json.loads(lines[5]), "ts": 0})
    export.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = tmp_path / "audit"

    assert run_cli(["audit", "scan", str(export), "--source", "jsonl", "--out", str(out)]) == 0

    report = json.loads((out / "scan-report.json").read_text(encoding="utf-8"))
    assert report["window"]["days"] == 30.0
    assert report["workloads"][0]["verdict"]["status"] == "candidate"


def test_the_most_spend_comes_first_when_the_price_table_priced_it(
    run_cli: CliRunner, tmp_path: Path
) -> None:
    # Discovery sorts while table-priced costs are still unknown, i.e. by hash.
    systems = sorted(("Classify the ticket A.", "Classify the ticket B."), key=template_hash)
    dear = systems[-1]  # the larger hash: last before the fix, first after it
    rows = [
        _oai(i, system, "yes", prompt_tokens=9_000 if system == dear else 1_000)
        for i in range(20)
        for system in systems
    ]

    report = _scan(run_cli, _export(tmp_path, rows), tmp_path / "audit")

    costs = [w["cost_usd_month"] for w in report["workloads"]]
    assert [w["id"] for w in report["workloads"]] == [
        template_hash(dear),
        template_hash(systems[0]),
    ]
    assert costs == sorted(costs, reverse=True)


def test_rescanning_into_a_run_directory_refuses_to_swap_its_data(
    run_cli: CliRunner, tmp_path: Path, export: Path, capsys: StrCapture
) -> None:
    # scan A, run, scan B into the same --out, run: the resumed run replayed B's
    # holdout against the model trained on A and paired B's spend with A's agreement.
    out = tmp_path / "audit"
    assert run_cli(["audit", "scan", str(export), "--source", "jsonl", "--out", str(out)]) == 0
    label = json.loads((out / "scan-report.json").read_text(encoding="utf-8"))["workloads"][0]
    save_state(out, AuditState(workloads={label["id"]: {HEAD: StepState(dataset_id="ds-1")}}))
    dataset = (out / "workloads" / label["id"] / "dataset.jsonl").read_text(encoding="utf-8")

    # The same export again is the same scan: nothing the run trained on changes.
    assert run_cli(["audit", "scan", str(export), "--source", "jsonl", "--out", str(out)]) == 0

    other = tmp_path / "other.jsonl"
    other.write_text(
        "".join(json.dumps({**_row(i), "response": "cd"[i % 2]}) + "\n" for i in range(CALLS)),
        encoding="utf-8",
    )
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "scan", str(other), "--source", "jsonl", "--out", str(out)])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert label["id"] in err
    assert "--force" in err
    assert (out / "workloads" / label["id"] / "dataset.jsonl").read_text() == dataset

    # M6: the replay answers of the rows --force rewrites belong to the old rows.
    stale = out / "workloads" / label["id"] / "replay-head_tune.jsonl"
    stale.write_text('{"deployment_id": "dep-1", "balance_before": 1}\n', encoding="utf-8")
    argv = ["audit", "scan", str(other), "--source", "jsonl", "--out", str(out), "--force"]
    assert run_cli(argv) == 0
    assert (out / "workloads" / label["id"] / "dataset.jsonl").read_text() != dataset
    assert not stale.exists()


def test_a_scan_does_not_rewrite_rows_under_a_live_run(
    run_cli: CliRunner, tmp_path: Path, export: Path, capsys: StrCapture
) -> None:
    """M6: a workload the live run has not reached yet is not in `state.json` to protect it."""
    out = tmp_path / "audit"
    out.mkdir()
    with lock_audit(out), pytest.raises(SystemExit) as exc:
        run_cli(["audit", "scan", str(export), "--source", "jsonl", "--out", str(out)])
    assert exc.value.code == 1
    assert "is in use by another `dagnam audit` command" in capsys.readouterr().err
    assert not (out / "scan-report.json").exists()


def test_a_sampled_export_is_scaled_back_to_the_traffic(run_cli: CliRunner, tmp_path: Path) -> None:
    # N18: a Langfuse/LangSmith export sampled at 25% understated volume and spend 4x.
    rows = [_oai(i, "Label the ticket.", "ab"[i % 2]) for i in range(40)]
    export = _export(tmp_path, rows)
    full = _scan(run_cli, export, tmp_path / "full")
    sampled = _scan(run_cli, export, tmp_path / "sampled", "--sample-rate", "0.25")

    (w_full,), (w_sampled,) = full["workloads"], sampled["workloads"]
    assert w_sampled["calls"] == w_full["calls"] == 40
    assert w_sampled["calls_per_day"] == pytest.approx(4 * w_full["calls_per_day"])
    assert w_sampled["cost_usd_month"] == pytest.approx(4 * w_full["cost_usd_month"])
    assert (full["window"]["sample_rate"], sampled["window"]["sample_rate"]) == (1.0, 0.25)
    assert "sampled at 25%" in (tmp_path / "sampled" / "scan-report.md").read_text()


def test_sample_rate_parser() -> None:
    assert parse_sample_rate("0.1") == 0.1
    assert parse_sample_rate("1") == 1.0
    assert parse_sample_rate("10%") == 0.1
    for bad in ("0", "1.5", "-0.2", "most", "nan", "0%"):
        with pytest.raises(argparse.ArgumentTypeError, match="--sample-rate expects"):
            parse_sample_rate(bad)
