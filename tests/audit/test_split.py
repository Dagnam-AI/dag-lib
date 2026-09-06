"""Time-ordered holdout split, snapped so no session straddles the boundary."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from hypothesis import given, strategies as st
import pytest

from dagnam.audit import TraceRecord
from dagnam.audit.split import HOLDOUT_SHARE, split_boundary, time_split


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
