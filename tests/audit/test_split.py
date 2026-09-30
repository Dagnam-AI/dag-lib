"""Time-ordered holdout split, snapped so no session straddles the boundary."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from hypothesis import given, strategies as st
import pytest

from dagnam.audit import TraceRecord
from dagnam.audit.split import HOLDOUT_SHARE, cap_train, split_boundary, time_split
from dagnam.audit.thresholds import ENUM_MAX_DISTINCT


def _rec(ts: int, session: str | None) -> TraceRecord:
    return TraceRecord(
        trace_id=f"t{ts}",
        ts=datetime(2026, 8, 1, tzinfo=UTC) + timedelta(minutes=ts),
        model="m",
        system=None,
        messages=(),
        response="r",
        response_tool_calls=(),
        prompt_tokens=1,
        completion_tokens=1,
        latency_ms=1.0,
        cost_usd=None,
        session_id=session,
        outcome=None,
        workload_hint=None,
    )


def test_spec_ratio() -> None:
    assert HOLDOUT_SHARE == 0.2


def test_time_split_never_straddles_a_session() -> None:
    recs = [_rec(ts=i, session=f"s{i // 4}") for i in range(400)]
    split = time_split(recs)

    def sessions(indices: list[int]) -> set[str | None]:
        return {recs[i].session_id for i in indices}

    assert set(split) == {"train", "eval_holdout"}
    assert sessions(split["train"]).isdisjoint(sessions(split["eval_holdout"]))
    assert max(recs[i].ts for i in split["train"]) < min(recs[i].ts for i in split["eval_holdout"])
    assert 0.18 <= len(split["eval_holdout"]) / 400 <= 0.25
    assert sorted(split["train"] + split["eval_holdout"]) == list(range(400))


def test_a_session_at_the_boundary_is_pulled_into_train() -> None:
    # Ten rows, nominal cut after row 8; rows 7..9 share a session.
    recs = [_rec(ts=i, session="late" if i >= 7 else f"s{i}") for i in range(10)]
    split = time_split(recs)
    assert split == {"train": list(range(10)), "eval_holdout": []}


def test_rows_are_ordered_by_time_not_by_position() -> None:
    recs = [_rec(ts=9 - i, session=None) for i in range(10)]
    split = time_split(recs)
    assert split == {"train": list(range(2, 10)), "eval_holdout": [0, 1]}
    assert split_boundary(recs, split) == recs[1].ts


def test_sessionless_rows_split_individually() -> None:
    recs = [_rec(ts=i, session=None) for i in range(10)]
    assert time_split(recs)["eval_holdout"] == [8, 9]
    assert time_split(recs, holdout_share=0.5)["eval_holdout"] == [5, 6, 7, 8, 9]


def test_empty_input_and_empty_holdout() -> None:
    assert time_split([]) == {"train": [], "eval_holdout": []}
    assert split_boundary([], {"train": [], "eval_holdout": []}) is None


@pytest.mark.parametrize("share", [0.0, 1.0, -0.1, 1.5])
def test_holdout_share_must_be_a_proper_fraction(share: float) -> None:
    with pytest.raises(ValueError, match="holdout_share"):
        time_split([_rec(0, None)], holdout_share=share)


@given(
    st.lists(st.tuples(st.integers(0, 50), st.sampled_from(["a", "b", "c", None])), max_size=60),
    st.floats(0.05, 0.95),
)
def test_split_is_a_partition_and_sessions_never_straddle(
    spec: list[tuple[int, str | None]], share: float
) -> None:
    recs = [_rec(ts, session) for ts, session in spec]
    split = time_split(recs, holdout_share=share)

    assert sorted(split["train"] + split["eval_holdout"]) == list(range(len(recs)))
    train = {recs[i].session_id for i in split["train"]} - {None}
    holdout = {recs[i].session_id for i in split["eval_holdout"]} - {None}
    assert train.isdisjoint(holdout)
    assert time_split(recs, holdout_share=share) == split


def test_cap_train_keeps_a_proportional_stratified_sample() -> None:
    # R3-15: 23,465 train rows ran past the recipe's 1-hour ceiling; a capped sample
    # keeps every class, in proportion, chosen by content so any input order agrees.
    strata = ["billing"] * 80 + ["refund"] * 18 + ["rare"] * 2
    keys = [f"{i:03d}" for i in range(100)]
    kept = cap_train(list(range(100)), strata, keys, limit=10)

    assert [strata[i] for i in kept].count("billing") == 8
    assert [strata[i] for i in kept].count("refund") == 2  # ceil(10 * 18 / 100)
    assert [strata[i] for i in kept].count("rare") == 1
    assert kept == sorted(kept)
    assert cap_train(list(range(100)), strata, keys[::-1], limit=10) != kept  # by content
    assert cap_train([3, 5], ["a", "a"], ["x", "y"], limit=10) == [3, 5]  # under the cap


def test_cap_train_samples_targets_that_are_not_classes_as_one_stratum() -> None:
    # m4: a stratum per distinct answer keeps a router's rare routes; an extraction's
    # answers are all distinct, and a stratum each would keep every row past the cap.
    keys = [f"{i:03d}" for i in range(100)]
    rows = list(range(100))
    classes = [f"route{i % ENUM_MAX_DISTINCT}" for i in range(100)]
    capped = cap_train(rows, classes, keys, limit=10, max_strata=ENUM_MAX_DISTINCT)
    assert len(capped) == ENUM_MAX_DISTINCT
    answers = [f"answer {i}" for i in range(100)]
    assert cap_train(rows, answers, keys, limit=10, max_strata=ENUM_MAX_DISTINCT) == rows[:10]
    # N1: without a bound (labels), every stratum keeps its ceiling, however many.
    assert cap_train(rows, answers, keys, limit=10) == rows
