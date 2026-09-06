"""The candidate set per output-structure class (spec U4, D6).

A registry, never an ``if structure_class ==`` branch: the orchestrator asks
:data:`CANDIDATES` what to run for a workload and the report asks it what to
price. ``hosted_floor`` is the cheapest hosted variant from the price table
(no run, no deployment); ``head_tune`` and ``sft_small`` are trained on the
platform and scored through their own endpoint.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from dagnam.audit.structure import StructureClass


class CandidateKind(StrEnum):
    """The three ways a workload can be replaced."""

    HOSTED_FLOOR = "hosted_floor"
    HEAD_TUNE = "head_tune"
    SFT_SMALL = "sft_small"


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """What one candidate is trained with and priced at.

    ``recipe_key``/``base_family`` are ``None`` for a candidate that needs no
    run; ``max_params`` caps the base chosen from the catalog (``None`` means
    the family's smallest, whatever its size). ``serving_rate_key`` names the
    row in Task 5's ``SERVING_RATES`` the report prices serving with; it is
    ``None`` for the hosted floor, whose cost comes from the price table.
    """

    kind: CandidateKind
    recipe_key: str | None
    base_family: str | None
    max_params: int | None
    serving_rate_key: str | None


HOSTED_FLOOR = CandidateSpec(CandidateKind.HOSTED_FLOOR, None, None, None, None)
HEAD_TUNE = CandidateSpec(
    CandidateKind.HEAD_TUNE, "head-tune-text-classification@1.1", "bert", None, "cpu-classifier"
)
SFT_SMALL = CandidateSpec(
    CandidateKind.SFT_SMALL, "qlora-sft-chat@1.1", "qwen2", 3_000_000_000, "gpu-small-llm"
)

CANDIDATES: Mapping[StructureClass, tuple[CandidateSpec, ...]] = {
    StructureClass.ENUM_LABEL: (HOSTED_FLOOR, HEAD_TUNE),
    StructureClass.JSON_OBJECT: (HOSTED_FLOOR, SFT_SMALL),
    StructureClass.SHORT_SPAN: (HOSTED_FLOOR, SFT_SMALL),
    StructureClass.FREE_TEXT: (),
}
"""Candidates per structure class, in the order the frontier runs them."""

__all__ = ["CANDIDATES", "HEAD_TUNE", "HOSTED_FLOOR", "SFT_SMALL", "CandidateKind", "CandidateSpec"]
