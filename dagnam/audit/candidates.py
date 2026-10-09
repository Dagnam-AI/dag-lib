"""The candidate set per output-structure class.

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
from typing import Literal

from dagnam.audit.structure import StructureClass

StudentKind = Literal["cpu-classifier", "gpu-small-llm"]
"""The two serving shapes a student is priced as (``dagnam_contracts.audit.serving``)."""

RECIPE_MAX_MINUTES = 60
"""Both recipes' own time bound: the platform stops a run at ``max_duration_seconds=3_600``."""
PROVIDER_MARGIN_MINUTES = 15
"""The platform bills a run up to its time bound PLUS this provider-side margin."""
CREDITS_PER_GPU_MINUTE = 2
"""The A10G rate every base the audit picks is priced on (the smallest tier that fits it)."""
TRAINING_CREDITS_MAX = (RECIPE_MAX_MINUTES + PROVIDER_MARGIN_MINUTES) * CREDITS_PER_GPU_MINUTE
"""What one audit training run can cost at most, known before it is submitted.

This is the same quantity the platform reserves when it admits the run: the whole
span the run can be billed (its time bound plus the provider margin) at the tier's
rate, 75 minutes at 2 credits. The budget holds each run to it, so ``--max-credits``
never admits a run the account could not start, and the platform returns whatever the
run does not use.

The platform offers no pre-submit estimate for a foundation run -- its
``credits_estimate_max`` arrives in the submit's response, and that is a
smaller, display-only band -- so the SDK mirrors the reservation's inputs.

ponytail: mirrors three platform constants (the recipe bound, the provider margin and
the tier rate); a server-side pre-submit estimate replaces it when one exists.
"""


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
    row in the contract's ``SERVING_RATES`` the report prices serving with; it is
    ``None`` for the hosted floor, whose cost comes from the price table.
    ``training_credits_max`` is what its run may charge at most (``None``: no run).
    """

    kind: CandidateKind
    recipe_key: str | None
    base_family: str | None
    max_params: int | None
    serving_rate_key: StudentKind | None
    training_credits_max: int | None


HOSTED_FLOOR = CandidateSpec(CandidateKind.HOSTED_FLOOR, None, None, None, None, None)
HEAD_TUNE = CandidateSpec(
    CandidateKind.HEAD_TUNE,
    "head-tune-text-classification@1.2",
    "bert",
    None,
    "cpu-classifier",
    TRAINING_CREDITS_MAX,
)
SFT_SMALL = CandidateSpec(
    CandidateKind.SFT_SMALL,
    "qlora-sft-chat@1.2",
    "qwen2",
    3_000_000_000,
    "gpu-small-llm",
    TRAINING_CREDITS_MAX,
)

CANDIDATES: Mapping[StructureClass, tuple[CandidateSpec, ...]] = {
    StructureClass.ENUM_LABEL: (HOSTED_FLOOR, HEAD_TUNE),
    StructureClass.JSON_OBJECT: (HOSTED_FLOOR, SFT_SMALL),
    StructureClass.SHORT_SPAN: (HOSTED_FLOOR, SFT_SMALL),
    StructureClass.FREE_TEXT: (),
}
"""Candidates per structure class, in the order the frontier runs them."""

__all__ = [
    "CANDIDATES",
    "CREDITS_PER_GPU_MINUTE",
    "HEAD_TUNE",
    "HOSTED_FLOOR",
    "PROVIDER_MARGIN_MINUTES",
    "RECIPE_MAX_MINUTES",
    "SFT_SMALL",
    "TRAINING_CREDITS_MAX",
    "CandidateKind",
    "CandidateSpec",
    "StudentKind",
]
