"""Time-ordered train / eval_holdout split, snapped so no session straddles it (spec U3)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import math
from typing import Protocol


class Stamped(Protocol):
    """A row's record, as far as the split reads it: when it happened, and its session."""

    @property
    def ts(self) -> datetime: ...

    @property
    def session_id(self) -> str | None: ...


# Spec U3: the last 20% of rows by time is ``eval_holdout``, the earlier 80% ``train``.
HOLDOUT_SHARE = 0.2


def time_split(
    records_in_row_order: Sequence[Stamped], *, holdout_share: float = HOLDOUT_SHARE
) -> dict[str, list[int]]:
    """Split row indices into ``train`` and ``eval_holdout`` by ``ts``.

    ``records_in_row_order[i]`` is the record row ``i`` was derived from. The
    nominal cut leaves the latest ``holdout_share`` of rows in the holdout;
    any session that already has a row before the cut is pulled whole into
    ``train``, so a ``session_id`` never straddles the boundary. Rows without
    a session split individually. The minimum holdout size (spec §8
    ``MIN_HOLDOUT``) is the verdict's concern, not the split's.
    """
    if not 0 < holdout_share < 1:
        raise ValueError(f"holdout_share must be in (0, 1); got {holdout_share}")
    records = records_in_row_order
    order = sorted(range(len(records)), key=lambda i: (records[i].ts, i))
    cut = len(records) - round(len(records) * holdout_share)

    def key(i: int) -> str | int:
        return records[i].session_id or i

    # ponytail: a session spanning the whole window drags every later row into
    # train; cap the drag (or split such sessions by time) if a real export shows it.
    train_sessions = {key(i) for i in order[:cut]}
    holdout = [i for i in order[cut:] if key(i) not in train_sessions]
    train = sorted(set(range(len(records))) - set(holdout))
    return {"train": train, "eval_holdout": sorted(holdout)}


def cap_train(
    train: Sequence[int],
    strata: Sequence[str],
    order_keys: Sequence[str],
    *,
    limit: int,
    max_strata: int | None = None,
) -> list[int]:
    """At most about ``limit`` of the ``train`` row indices, a proportional stratified sample.

    ``strata[i]`` / ``order_keys[i]`` belong to row ``i`` (its target: a
    label, a router's route; a digest of the row). Each stratum keeps
    ``ceil(limit * its share)`` rows, so every class survives and the classes
    keep their proportions; within a stratum, rows are taken in ``order_keys``
    order, so the sample depends on the content and never on the input order.
    The ceilings can exceed ``limit`` by fewer rows than there are strata, so
    past ``max_strata`` of them -- targets that are not classes, as an
    extraction's answers are not -- they are sampled as one stratum (m4);
    ``None`` keeps every stratum, as labels need (N1).
    """
    if len(train) <= limit:
        return list(train)
    by_stratum: dict[str, list[int]] = {}
    for i in train:
        by_stratum.setdefault(strata[i], []).append(i)
    if max_strata is not None and len(by_stratum) > max_strata:
        by_stratum = {"": list(train)}
    kept: list[int] = []
    for rows in by_stratum.values():
        quota = math.ceil(limit * len(rows) / len(train))
        kept += sorted(rows, key=lambda i: order_keys[i])[:quota]
    return sorted(kept)


def split_boundary(
    records_in_row_order: Sequence[Stamped], split: Mapping[str, Sequence[int]]
) -> datetime | None:
    """The earliest holdout timestamp, or ``None`` when the holdout is empty."""
    holdout = split["eval_holdout"]
    return min(records_in_row_order[i].ts for i in holdout) if holdout else None


__all__ = ["HOLDOUT_SHARE", "cap_train", "split_boundary", "time_split"]
