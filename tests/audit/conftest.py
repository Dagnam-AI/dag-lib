"""Shared fixtures for the audit tests."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import sys
from typing import Any

import pytest
from tests.audit._platform import Clock, FakePlatform, json_row, label_row

from dagnam.audit import TraceRecord, read_traces, write_workload
from dagnam.audit.candidates import HEAD_TUNE, CandidateSpec
from dagnam.audit.secrets import SecretStore
from dagnam.audit.steps import StepContext
from dagnam.audit.structure import StructureClass

FIXTURES = Path(__file__).parent / "fixtures"
PASS_LIST = ["PII_EMAIL", "PII_PHONE", "PII_PAYMENT_CARD", "PII_NATIONAL_ID"]
TRAIN = list(range(16))
HOLDOUT = [16, 17, 18, 19]
SCAN_REPORT: dict[str, Any] = {
    "schema": "dagnam.audit.scan/1",
    "price_table_version": "2026-09",
    "workloads": [
        {"id": "w1", "structure_class": "enum_label", "verdict": {"status": "candidate"}},
        {"id": "w2", "structure_class": "json_object", "verdict": {"status": "marginal"}},
        {"id": "w3", "structure_class": "free_text", "verdict": {"status": "not_audited"}},
        {"id": "w4", "structure_class": "enum_label", "verdict": {"status": "not_worth_it"}},
        {"id": "w5", "structure_class": "enum_label", "verdict": {"status": "candidate"}},
    ],
}


@pytest.fixture(autouse=True)
def no_real_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never touch the developer's real keychain: the audit tests run in file mode."""
    monkeypatch.setitem(sys.modules, "keyring", None)


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def langfuse_records() -> list[TraceRecord]:
    records, _ = read_traces(FIXTURES / "langfuse_sample.jsonl", source="langfuse")
    return list(records)


def stats(format_key: str) -> dict[str, Any]:
    return {
        "format_key": format_key,
        "redact": {"counts": {}, "pass_list": PASS_LIST, "rows_changed": 0},
    }


@pytest.fixture
def audit_dir(tmp_path: Path) -> Path:
    """A scanned audit dir: a label workload, a JSON workload, and two that are not audited."""
    root = tmp_path / "audit"
    root.mkdir()
    split = {"train": TRAIN, "eval_holdout": HOLDOUT}
    write_workload(root, "w1", [label_row(i) for i in range(20)], split, stats("labeled-example"))
    write_workload(root, "w2", [json_row(i) for i in range(20)], split, stats("chat-messages"))
    (root / "scan-report.json").write_text(json.dumps(SCAN_REPORT), encoding="utf-8")
    return root


@pytest.fixture
def platform() -> FakePlatform:
    return FakePlatform()


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def make_ctx(audit_dir: Path, platform: FakePlatform, clock: Clock) -> Callable[..., StepContext]:
    def build(
        workload_id: str = "w1",
        spec: CandidateSpec = HEAD_TUNE,
        structure_class: StructureClass = StructureClass.ENUM_LABEL,
        **overrides: Any,
    ) -> StepContext:
        settings: dict[str, Any] = {
            "audit_dir": audit_dir,
            "workload_id": workload_id,
            "structure_class": structure_class,
            "spec": spec,
            "client": platform,
            "secrets": SecretStore(audit_dir),
            "floor": 0.97,
            "max_credits": 500,
            "sleep": clock.sleep,
            "now": clock.now,
        }
        settings.update(overrides)
        return StepContext(**settings)

    return build
