"""The scan report: the JSON contract and the markdown rendered from it."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
import json
from pathlib import Path
from typing import Any

import pytest
from tests.audit._records import make_record, make_workload

from dagnam.audit.candidates import TRAINING_CREDITS_MAX, CandidateKind
from dagnam.audit.derive import build_dataset
from dagnam.audit.discover import Workload
from dagnam.audit.prices import PriceRow, PriceTable
from dagnam.audit.readers.messages import TOOL_CALL_NOTE
from dagnam.audit.scan_report import (
    ScanReport,
    SupersededRunError,
    Window,
    build_scan_report,
    dataset_entry,
    superseded_workloads,
    write_scan,
    write_scan_report,
)
from dagnam.audit.state import AuditState, StepState, load_state, save_state
from dagnam.audit.steps_train import credits_spent
from dagnam.audit.structure import StructureClass
from dagnam.audit.thresholds import (
    MAX_UNSTRUCTURED_SHARE,
    PRICE_TABLE_STALE_DAYS,
    SFT_MIN_TRAIN_ROWS,
)
from dagnam.audit.workspace import write_workload

WINDOW = Window(
    start=datetime(2026, 8, 1, tzinfo=UTC), end=datetime(2026, 8, 31, tzinfo=UTC), days=30.0
)
FRESH = PriceTable(
    version="2026-09",
    as_of=datetime.now(UTC).date(),
    rows={"m": PriceRow("m", 1.0, 2.0, None, None)},
)
PASS_LIST = ["PII_EMAIL", "PII_PHONE"]
COUNTS = {"PII_EMAIL": 2}


def _workloads() -> list[Workload]:
    return [
        make_workload(calls_per_day=5_000, cost_month=4_000.0),  # candidate
        make_workload(calls_per_day=200, cost_month=120.0),  # not worth it
        make_workload(
            calls_per_day=10_000, cost_month=5_000.0, cls=StructureClass.FREE_TEXT
        ),  # not audited
        make_workload(calls_per_day=10, cost_month=5_000.0, n=999),  # too few samples
        make_workload(
            calls_per_day=100, cost_month=None, n=3_000, models=("m",)
        ),  # priced from the table
        make_workload(calls_per_day=100, cost_month=None, models=("other",)),  # unknown cost
    ]


def _report(workloads: list[Workload] | None = None, table: PriceTable = FRESH) -> ScanReport:
    return build_scan_report(
        workloads if workloads is not None else _workloads(),
        source="langfuse",
        window=WINDOW,
        price_table=table,
        pii_pass_list=PASS_LIST,
        pii_counts=COUNTS,
    )


def test_report_follows_the_contract() -> None:
    report = _report()
    js = report.to_json()

    assert js["schema"] == ScanReport.schema == "dagnam.audit.scan/1"
    assert datetime.fromisoformat(js["generated_at"]).tzinfo is not None
    assert js["source"] == "langfuse"
    assert js["window"] == {
        "start": "2026-08-01T00:00:00+00:00",
        "end": "2026-08-31T00:00:00+00:00",
        "days": 30.0,
        "sample_rate": 1.0,
    }
    assert js["price_table_version"] == "2026-09"
    assert js["pii"] == {"pass_list": PASS_LIST, "counts": COUNTS}
    assert js["totals"] == {
        "calls": 5_000 * 4 + 999 + 3_000,
        "cost_usd_month": pytest.approx(4_000 + 120 + 5_000 + 5_000 + 0.312),
        "prompt_tokens": sum(w.prompt_tokens for w in _workloads()),
        "completion_tokens": sum(w.completion_tokens for w in _workloads()),
    }
    assert js["warnings"] == []
    assert [w["verdict"]["status"] for w in js["workloads"]] == [
        "candidate",
        "not_worth_it",
        "not_audited",
        "too_few_samples",
        "not_worth_it",
        "unknown_cost",
    ]
    priced = js["workloads"][4]  # priced from the table, then judged like any other
    assert priced["cost_source"] == "price_table"
    assert priced["cost_usd_month"] == pytest.approx(0.312)
    first = js["workloads"][0]
    assert set(first) == set(_workloads()[0].to_json()) | {"verdict", "dataset"}
    assert set(first["verdict"]) == {"status", "ratio", "savings_usd_month", "reason"}
    assert first["dataset"] is None


def test_price_table_stamps_version_and_warns_when_stale() -> None:
    stale = PriceTable(version="2025-01", as_of=date(2025, 1, 6), rows={})

    report = _report(table=stale)

    assert report.price_table_version == "2025-01"
    (warning,) = report.warnings
    assert "2025-01" in warning
    assert str(PRICE_TABLE_STALE_DAYS) in warning
    assert "stale" in warning


def test_unstructured_share_warns_low_confidence() -> None:
    low = make_workload(calls_per_day=100, n=3_000, confidence="low")
    high = make_workload(calls_per_day=100, n=3_000)

    assert _report([low, high, high, high, high]).warnings == ()  # exactly the boundary share
    (warning,) = _report([low, high, high]).warnings
    assert f"{MAX_UNSTRUCTURED_SHARE:.0%}" in warning
    assert "low-confidence" in warning


def test_empty_scan_is_a_valid_report() -> None:
    js = _report([]).to_json()

    assert js["workloads"] == []
    assert js["totals"]["cost_usd_month"] == 0.0


def test_markdown_is_a_view_of_the_json(tmp_path: Path) -> None:
    report = _report()
    write_scan_report(report, tmp_path)
    md = (tmp_path / "scan-report.md").read_text()
    js = json.loads((tmp_path / "scan-report.json").read_text())

    assert js == report.to_json()
    for w in js["workloads"]:
        if w["cost_usd_month"] is not None:
            assert f"{w['cost_usd_month']:.2f}" in md  # every number in md comes from js
        assert w["verdict"]["reason"] in md
    assert f"{js['totals']['cost_usd_month']:.2f}" in md
    assert js["price_table_version"] in md
    assert js["generated_at"] in md
    assert md.index("## Not worth replacing") < md.index("## Candidates")
    assert md.index("## Candidates") < md.index("## PII")


def test_markdown_shows_customer_verdicts_and_never_replace_without_a_winner(
    tmp_path: Path,
) -> None:
    write_scan_report(_report(), tmp_path)
    md = (tmp_path / "scan-report.md").read_text()

    keep, candidates = md.split("## Candidates")
    assert keep.count("| KEEP |") == 5
    assert "NOT YET" not in keep
    assert candidates.count("| NOT YET |") == 1
    assert "KEEP" not in candidates
    assert "REPLACE" not in md  # no winner exists at scan time


def test_markdown_carries_the_warnings_and_unknown_costs(tmp_path: Path) -> None:
    stale = PriceTable(version="2025-01", as_of=date(2025, 1, 6), rows={})
    write_scan_report(_report(table=stale), tmp_path)
    md = (tmp_path / "scan-report.md").read_text()

    assert "## Warnings" in md
    assert "stale" in md
    assert "| unknown |" in md


def test_dataset_numbers_ride_along_and_a_thin_holdout_is_too_few_samples() -> None:
    dataset = {
        "rows": 4_000,
        "dedup_removed": 10,
        "redactions": 3,
        "truncated": 0,
        "split": {"train": 3_200, "eval_holdout": 800},
    }
    thin = {**dataset, "split": {"train": 3_900, "eval_holdout": 100}}
    workloads = [make_workload(calls_per_day=5_000, cost_month=4_000.0)] * 2
    js = build_scan_report(
        workloads,
        source="langfuse",
        window=WINDOW,
        price_table=FRESH,
        pii_pass_list=PASS_LIST,
        pii_counts=COUNTS,
        datasets={"w1": dataset},
    ).to_json()
    assert js["workloads"][0]["dataset"] == dataset
    assert js["workloads"][0]["verdict"]["status"] == "candidate"

    js = build_scan_report(
        workloads,
        source="langfuse",
        window=WINDOW,
        price_table=FRESH,
        pii_pass_list=PASS_LIST,
        pii_counts=COUNTS,
        datasets={"w1": thin},
    ).to_json()
    assert js["workloads"][0]["verdict"] == {
        "status": "too_few_samples",
        "ratio": None,
        "savings_usd_month": None,
        "reason": "100 holdout rows after the split; 200 needed",
    }


def test_a_tool_call_workload_is_named_in_a_warning() -> None:
    w = replace(make_workload(calls_per_day=5_000, cost_month=4_000.0), response_mode="tool_call")
    (warning,) = _report([w]).warnings
    assert warning == f"workload w1: {TOOL_CALL_NOTE}"


def test_targets_rewritten_by_redaction_are_counted_and_warned(tmp_path: Path) -> None:
    dataset = {
        "rows": 1_200,
        "dedup_removed": 0,
        "redactions": 2_400,
        "truncated": 0,
        "truths_redacted": 1_200,
        "split": {"train": 960, "eval_holdout": 240},
    }
    clean = {**dataset, "truths_redacted": 0}
    workloads = [
        make_workload(calls_per_day=5_000, cost_month=4_000.0),
        replace(make_workload(calls_per_day=5_000, cost_month=4_000.0), id="w2"),
    ]
    report = build_scan_report(
        workloads,
        source="openai",
        window=WINDOW,
        price_table=FRESH,
        pii_pass_list=PASS_LIST,
        pii_counts=COUNTS,
        datasets={"w1": dataset, "w2": clean},
    )
    (warning,) = report.warnings
    assert "w1: redaction rewrote 1,200 of 1,200 training targets" in warning
    write_scan_report(report, tmp_path)
    assert warning in (tmp_path / "scan-report.md").read_text()


def test_dataset_entry_carries_the_scan_numbers() -> None:
    records = [
        make_record(system=f"Label ticket {i}.", response=f"{i}@x.io", trace_id=str(i))
        for i in range(5)
    ]
    built = build_dataset(records, structure_class="enum_label", max_seq_length=2_048)
    assert dataset_entry(built) == {
        "rows": 5,
        "dedup_removed": 0,
        "redactions": 5,
        "truncated": 0,
        "truths_redacted": 5,
        "train_rows_capped": 0,
        "train_rows_over_budget": 0,
        "split": {"train": 4, "eval_holdout": 1},
    }
    capped = build_dataset(
        records, structure_class="enum_label", max_seq_length=2_048, max_train_rows=2
    )
    assert dataset_entry(capped)["train_rows_capped"] == 2


def test_a_capped_training_set_is_warned() -> None:
    # R3-15: counted in the report, so a reader knows the candidate saw a sample.
    dataset = {
        "rows": 5_800,
        "dedup_removed": 0,
        "redactions": 0,
        "truncated": 0,
        "truths_redacted": 0,
        "train_rows_capped": 18_465,
        "split": {"train": 5_000, "eval_holdout": 800},
    }
    report = build_scan_report(
        [make_workload(calls_per_day=5_000, cost_month=4_000.0)],
        source="openai",
        window=WINDOW,
        price_table=FRESH,
        pii_pass_list=PASS_LIST,
        pii_counts=COUNTS,
        datasets={"w1": dataset},
    )
    (warning,) = report.warnings
    assert warning.startswith("workload w1 trains on a 5,000-row sample of its 23,465")


def _over_budget(over: int, train: int) -> ScanReport:
    dataset = {
        "rows": train + 1_000,
        "dedup_removed": 0,
        "redactions": 0,
        "truncated": 0,
        "truths_redacted": 0,
        "train_rows_capped": 0,
        "train_rows_over_budget": over,
        "split": {"train": train, "eval_holdout": 1_000},
    }
    return build_scan_report(
        [make_workload(calls_per_day=5_000, cost_month=4_000.0, cls=StructureClass.JSON_OBJECT)],
        source="langfuse",
        window=WINDOW,
        price_table=FRESH,
        pii_pass_list=PASS_LIST,
        pii_counts=COUNTS,
        datasets={"w1": dataset},
    )


def test_training_rows_over_the_student_s_context_are_counted_and_can_leave_too_few() -> None:
    # B2-3: long-document extraction trained on few or no rows once the recipe dropped
    # its over-budget rows, and the scan had called it a candidate.
    (entry,) = _over_budget(3_990, SFT_MIN_TRAIN_ROWS - 1).to_json()["workloads"]
    assert entry["dataset"]["train_rows_over_budget"] == 3_990
    assert entry["verdict"] == {
        "status": "too_few_samples",
        "ratio": None,
        "savings_usd_month": None,
        "reason": "3,990 rows exceed the student's 2,048-token context",
    }
    report = _over_budget(40, SFT_MIN_TRAIN_ROWS)
    (entry,) = report.to_json()["workloads"]
    assert entry["verdict"]["status"] == "candidate"
    assert report.warnings == (
        "workload w1: 40 training rows exceed the student's 2,048-token context (by an"
        " estimate) and are left out of its training; the holdout keeps them",
    )
    assert _over_budget(0, 10).to_json()["workloads"][0]["verdict"]["status"] == "candidate"


POLICY_SENTENCES = (
    "Refunds are issued to the original payment method within five business days of approval.",
    "A customer may return any unopened item within thirty days of delivery for a full refund.",
    "Opened items can be exchanged for store credit when they arrive damaged or defective.",
    "Shipping fees are not refundable unless the order arrived late or in the wrong condition.",
    "Escalate every complaint that mentions a safety hazard to the quality team immediately.",
    "Loyalty members receive free returns and priority handling on every support request.",
    "Never promise a delivery date that the carrier has not confirmed in its tracking data.",
    "When a customer asks to cancel, confirm whether the order has already left the warehouse.",
)


def _policy(chars: int) -> str:
    """A support-triage system prompt: ``chars`` characters of plain English policy."""
    text, i = "You triage support messages for an online store. Company policy follows.\n\n", 0
    while len(text) < chars:
        sentence = POLICY_SENTENCES[i % len(POLICY_SENTENCES)]
        text, i = text + sentence + (" " if i % 4 != 3 else "\n\n"), i + 1
    return text


def _scanned(system: str) -> dict[str, Any]:
    """1,000 extraction calls under ``system``, derived and judged the way the scan does."""
    records = [
        make_record(
            system=system,
            response=json.dumps({"intent": ["refund", "cancel"][i % 2], "ticket": i}),
            ts=datetime(2026, 8, 1, tzinfo=UTC) + timedelta(minutes=i),
            trace_id=f"t{i}",
        )
        for i in range(1_000)
    ]
    dataset = build_dataset(records, structure_class="json_object", max_seq_length=2_048)
    workload = make_workload(
        calls_per_day=5_000, cost_month=4_000.0, cls=StructureClass.JSON_OBJECT
    )
    report = build_scan_report(
        [workload],
        source="langfuse",
        window=WINDOW,
        price_table=FRESH,
        pii_pass_list=PASS_LIST,
        pii_counts=COUNTS,
        datasets={"w1": dataset_entry(dataset)},
    )
    (entry,) = report.to_json()["workloads"]
    return entry


def test_a_long_system_prompt_that_fits_is_trained_and_a_row_that_does_not_is_dropped() -> None:
    # RR-1: the estimate counted English at 1.6x, so a support-triage workload whose
    # 7.5k-character policy made rows of ~1,550 real tokens read too_few_samples with
    # every training row "over" 2,048. This 8,207-character policy measures 1,552
    # Qwen2.5 tokens a row (template included); the estimate says 1,562.
    fits = _scanned(_policy(8_200))
    assert fits["verdict"]["status"] == "candidate"
    assert fits["dataset"]["train_rows_over_budget"] == 0
    assert fits["dataset"]["split"] == {"train": 800, "eval_holdout": 200}
    # A 12k-character invoice is ~6,500 real tokens: every training row is really over.
    over = _scanned(
        "Extract the invoice total as JSON.\n" + "Line 17: widget, 12 x 3.50 EUR\n" * 400
    )
    assert over["verdict"]["status"] == "too_few_samples"
    assert over["verdict"]["reason"] == "800 rows exceed the student's 2,048-token context"


def test_superseded_workloads_names_every_run_workload_this_scan_would_change(
    tmp_path: Path,
) -> None:
    records = [
        make_record(system=f"Label ticket {i}.", response="ab"[i % 2], trace_id=str(i))
        for i in range(5)
    ]
    built = build_dataset(records, structure_class="enum_label", max_seq_length=2_048)
    assert superseded_workloads(tmp_path, {"w1": built}) == []  # no run, nothing to protect

    write_workload(tmp_path, "w1", built.rows, built.split, built.stats)
    write_workload(tmp_path, "w2", built.rows, built.split, built.stats)
    steps = {CandidateKind.HEAD_TUNE: StepState(dataset_id="ds")}
    save_state(tmp_path, AuditState(workloads={"w1": steps, "w2": steps, "gone": steps}))
    fewer = build_dataset(records[:4], structure_class="enum_label", max_seq_length=2_048)

    # w1 is unchanged; w2's rows would change; "gone" has no files and no rows any more.
    assert superseded_workloads(tmp_path, {"w1": built, "w2": fewer}) == ["w2", "gone"]
    assert superseded_workloads(tmp_path, {"w1": built, "w2": built, "gone": built}) == ["gone"]


def test_chinese_policy_that_fits_keeps_its_training_rows() -> None:
    # RR2-1: the reviewer's 2.6k-character policy yielded ~1,549 Qwen tokens per row.
    policy = "客户服务部门负责处理所有与订单、配送和退款相关的问题。如果包裹在运输过程中损坏\uff0c客户可以在收到后七天内申请全额退款。对于缺少的商品\uff0c我们需要客户提供订单号和收货照片作为凭证。会员用户享有优先处理的权利\uff0c一般会在二十四小时内得到回复。请注意\uff0c定制商品和已拆封的电子产品不支持无理由退货。如果客户要求更换地址\uff0c必须在商品发货之前提交申请。所有退款将原路返回\uff0c到账时间通常为三到五个工作日。当客户情绪激动时\uff0c请保持礼貌并先表示理解\uff0c再说明处理方案。涉及金额超过一千元的赔偿\uff0c需要主管审批后才能执行。请不要向客户透露其他用户的任何个人信息或订单信息。如遇系统故障\uff0c应告知客户预计恢复时间并记录工单编号。每次对话结束前\uff0c请确认客户的问题是否已经完全解决。"
    system = (policy * 10)[:2600]
    fits = _scanned(system)
    assert fits["verdict"]["status"] == "candidate"
    assert fits["dataset"]["train_rows_over_budget"] == 0
    assert fits["dataset"]["split"] == {"train": 800, "eval_holdout": 200}


@pytest.mark.parametrize("published", [False, True])
def test_forced_scan_never_resumes_superseded_candidates(tmp_path: Path, published: bool) -> None:
    records = [make_record(system=f"Label {i}.", response="ab"[i % 2]) for i in range(5)]
    old = build_dataset(records, structure_class="enum_label", max_seq_length=2_048)
    new = build_dataset(records[:4], structure_class="enum_label", max_seq_length=2_048)
    write_workload(tmp_path, "w1", old.rows, old.split, old.stats)
    step = StepState(
        dataset_id="ds",
        version_id="v",
        run_id="run",
        run_status="completed",
        training_job_id="job",
        model_version_id="model",
        deployment_id="dep",
        key_ref="key",
        scored=True,
        training_cost_credits=12.0,
        replay_cost_credits=7.0,
    )
    save_state(
        tmp_path,
        AuditState(
            audit_id="audit" if published else None,
            workloads={"w1": {CandidateKind.HEAD_TUNE: step}},
        ),
    )
    replay = tmp_path / "workloads/w1/replay-head_tune.jsonl"
    replay.write_text("old reply")
    for suffix in ("json", "md"):
        (tmp_path / f"audit-report.{suffix}").write_text("old score")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    if published:
        with pytest.raises(SupersededRunError, match="new --out"):
            write_scan(
                tmp_path,
                [],
                {"w1": new},
                source="jsonl",
                window=WINDOW,
                price_table=FRESH,
                force=True,
            )
        assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    else:
        write_scan(
            tmp_path, [], {"w1": new}, source="jsonl", window=WINDOW, price_table=FRESH, force=True
        )
        state = load_state(tmp_path)
        assert state.candidate("w1", CandidateKind.HEAD_TUNE) == StepState()
        assert not replay.exists()
        assert not (tmp_path / "audit-report.json").exists()
        assert not (tmp_path / "audit-report.md").exists()
        assert state.retired == [step]
        assert credits_spent(state) == 19.0
        write_scan(
            tmp_path, [], {"w1": new}, source="jsonl", window=WINDOW, price_table=FRESH, force=True
        )
        assert credits_spent(load_state(tmp_path)) == 19.0  # unchanged scans never add it twice
        second = state.candidate("w1", CandidateKind.HEAD_TUNE)
        second.run_id, second.run_status = "new-run", "running"
        save_state(tmp_path, state)
        replay.write_text(
            '{"deployment_id": "dep", "balance_before": 100}\n{ "row": 0, "answer": "yes", "ms": 1.0}\n'
        )
        write_scan(
            tmp_path, [], {"w1": old}, source="jsonl", window=WINDOW, price_table=FRESH, force=True
        )
        assert credits_spent(load_state(tmp_path)) == 19.0 + TRAINING_CREDITS_MAX + 2.0
        assert len(load_state(tmp_path).retired) == 2


@pytest.mark.parametrize("change", ["rows", "added", "added-retired", "removed", "unchanged"])
@pytest.mark.parametrize("force", [False, True])
def test_published_scan_is_protected_before_any_candidate_starts(
    tmp_path: Path, change: str, force: bool
) -> None:
    records = [make_record(system=f"Label {i}.", response="ab"[i % 2]) for i in range(5)]
    dataset = build_dataset(records, structure_class="enum_label", max_seq_length=2_048)
    write_scan(
        tmp_path,
        [make_workload(), replace(make_workload(), id="without-dataset")],
        {"w1": dataset},
        source="jsonl",
        window=WINDOW,
        price_table=FRESH,
    )
    # Publisher.start persists this association before the first candidate is instantiated.
    save_state(tmp_path, AuditState(audit_id="published", workloads={}))
    # A retired local dataset folder outside the published scan is not an active workload.
    write_workload(tmp_path, "retired", dataset.rows, dataset.split, dataset.stats)
    proposed = {"w1": dataset}
    if change == "rows":
        proposed["w1"] = build_dataset(
            records[:4], structure_class="enum_label", max_seq_length=2_048
        )
    elif change == "added":
        proposed["w2"] = dataset
    elif change == "added-retired":
        proposed["retired"] = dataset
    elif change == "removed":
        proposed.clear()
    workloads = [replace(make_workload(), id=key) for key in proposed]
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    if change == "unchanged":
        write_scan(
            tmp_path,
            workloads,
            proposed,
            source="jsonl",
            window=WINDOW,
            price_table=FRESH,
            force=force,
        )
        assert (tmp_path / "state.json").read_bytes() == before[tmp_path / "state.json"]
        for path, contents in before.items():
            if "workloads" in path.parts:
                assert path.read_bytes() == contents
    else:
        with pytest.raises(SupersededRunError, match="new --out"):
            write_scan(
                tmp_path,
                workloads,
                proposed,
                source="jsonl",
                window=WINDOW,
                price_table=FRESH,
                force=force,
            )
        assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
