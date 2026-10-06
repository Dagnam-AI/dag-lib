"""A call that ended inside the model's reasoning is a call: counted, priced, never an answer."""

from __future__ import annotations

from dataclasses import replace

from tests.audit._records import make_record

from dagnam.audit.derive import derive_rows
from dagnam.audit.discover import discover_workloads
from dagnam.audit.economics import replaceability
from dagnam.audit.structure import StructureClass


def _records(n: int, *, every: int, cost: float | None = 0.5, tokens: int = 2) -> list:
    return [
        replace(
            make_record(
                system="Label the ticket.",
                response="" if i % every == 0 else "ab"[i % 2],
                trace_id=f"t{i}",
                cost_usd=cost,
                completion_tokens=tokens,
            ),
            reasoning_only=i % every == 0,
        )
        for i in range(n)
    ]


def test_answerless_calls_stay_in_the_counts_the_cost_and_the_verdict_but_not_the_answers() -> None:
    (w,) = discover_workloads(_records(1_200, every=4), window_days=30)

    assert w.calls == 1_200
    assert w.answerless_calls == 300
    assert w.answerless_spend_share == 0.25  # by the export's own cost
    assert w.cost_usd_month == 1_200 * 0.5
    assert w.distinct_outputs == 2  # the empty reply is no output
    assert w.structure_class is StructureClass.ENUM_LABEL  # and no part of the structure sample
    assert replaceability(w).status == "candidate"


def test_the_share_of_spend_falls_back_to_completion_tokens_then_to_calls() -> None:
    (by_tokens,) = discover_workloads(_records(8, every=4, cost=None, tokens=10), window_days=30)
    (by_calls,) = discover_workloads(_records(8, every=4, cost=None, tokens=0), window_days=30)

    assert by_tokens.answerless_spend_share == 0.25
    assert by_calls.answerless_spend_share == 0.25
    assert by_tokens.cost_usd_month is None


def test_a_workload_with_no_answerless_call_says_so() -> None:
    answered = [r for r in _records(9, every=3) if not r.reasoning_only]

    (w,) = discover_workloads(answered, window_days=30)

    assert (w.answerless_calls, w.answerless_spend_share) == (0, 0.0)


def test_a_workload_of_nothing_but_answerless_calls_is_still_one_workload() -> None:
    (w,) = discover_workloads(_records(5, every=1), window_days=30)

    assert (w.calls, w.answerless_calls, w.answerless_spend_share) == (5, 5, 1.0)
    assert (w.distinct_outputs, w.structure_class) == (0, StructureClass.FREE_TEXT)


def test_an_answerless_call_gives_no_training_row() -> None:
    records = _records(10, every=2)

    rows, stats = derive_rows(records, structure_class="enum_label", max_seq_length=2_048)

    assert len(rows) == 5
    assert stats.skipped == 5
