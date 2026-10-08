"""A published run that cannot be published halts before it creates anything; ``--local-only`` runs on."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

import pytest
from tests.audit._chat import Clock, label_row
from tests.audit._platform import FakePlatform
from tests.audit.conftest import DERIVED, HOLDOUT, TRAIN, stats

from dagnam.audit import write_workload
from dagnam.audit.orchestrate import run_audit
from dagnam.audit.publish import MAX_WORKLOADS, Publisher
from dagnam.audit.state import AuditState, load_state

CREATES = ("upload_dataset", "create_foundation_run", "create_deployment", "create_audit")


@pytest.fixture
def wide_dir(tmp_path: Path) -> Callable[[int], Path]:
    """An audit directory whose scan derived ``count`` runnable label workloads."""

    def build(count: int) -> Path:
        root = tmp_path / f"wide-{count}"
        root.mkdir()
        split = {"train": TRAIN, "eval_holdout": HOLDOUT}
        ids = [f"w{i}" for i in range(count)]
        for workload_id in ids:
            write_workload(
                root,
                workload_id,
                [label_row(i) for i in range(20)],
                split,
                stats("labeled-example"),
            )
        report: dict[str, Any] = {
            "schema": "dagnam.audit.scan/1",
            "price_table_version": "2026-09",
            "workloads": [
                {
                    "id": workload_id,
                    "structure_class": "enum_label",
                    "verdict": {"status": "candidate"},
                    "dataset": DERIVED,
                }
                for workload_id in ids
            ],
        }
        (root / "scan-report.json").write_text(json.dumps(report), encoding="utf-8")
        return root

    return build


def run(audit_dir: Path, platform: FakePlatform, clock: Clock, *, published: bool) -> AuditState:
    return run_audit(
        audit_dir,
        floor=None,
        workloads=None,
        max_credits=500,
        wait=False,
        client=platform,
        publisher=Publisher(platform, AuditState()) if published else None,
        sleep=clock.sleep,
        now=clock.now,
    )


def test_a_published_run_over_the_cap_halts_before_it_creates_anything(
    wide_dir: Callable[[int], Path], platform: FakePlatform, clock: Clock
) -> None:
    audit_dir = wide_dir(MAX_WORKLOADS + 1)

    state = run(audit_dir, platform, clock, published=True)

    assert state.halted is not None
    assert state.halted["reason"] == "publish_failed"
    detail = str(state.halted["detail"])
    assert "selected 201 workloads" in detail
    assert "at most 200" in detail
    assert "--workloads" in detail
    assert "--local-only" in detail
    assert load_state(audit_dir).halted == state.halted
    assert state.audit_id is None
    assert platform.audits == []
    assert [call for call in platform.call_log if call in CREATES] == []


def test_a_published_run_at_the_cap_is_published(
    wide_dir: Callable[[int], Path], platform: FakePlatform, clock: Clock
) -> None:
    state = run(wide_dir(MAX_WORKLOADS), platform, clock, published=True)

    assert state.audit_id == "audit-1"
    assert len(platform.audits[0]["workloads"]) == MAX_WORKLOADS
    assert state.halted is None or state.halted["reason"] != "publish_failed"


def test_a_local_only_run_over_the_cap_is_unaffected(
    wide_dir: Callable[[int], Path], platform: FakePlatform, clock: Clock
) -> None:
    state = run(wide_dir(MAX_WORKLOADS + 1), platform, clock, published=False)

    assert state.audit_id is None
    assert platform.audits == []
    assert state.halted is None or state.halted["reason"] != "publish_failed"
    assert "upload_dataset" in platform.call_log  # it went on to create what it needs, locally
