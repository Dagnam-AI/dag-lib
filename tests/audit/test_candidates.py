"""The candidate registry is the design's table, verbatim."""

from __future__ import annotations

from dagnam.audit.candidates import (
    CANDIDATES,
    HEAD_TUNE,
    HOSTED_FLOOR,
    PROVIDER_MARGIN_MINUTES,
    RECIPE_MAX_MINUTES,
    SFT_SMALL,
    TRAINING_CREDITS_MAX,
    CandidateKind,
    CandidateSpec,
)
from dagnam.audit.structure import StructureClass


def test_registry_matches_the_design_table() -> None:
    kinds = {cls: tuple(spec.kind for spec in specs) for cls, specs in CANDIDATES.items()}
    assert kinds == {
        StructureClass.ENUM_LABEL: (CandidateKind.HOSTED_FLOOR, CandidateKind.HEAD_TUNE),
        StructureClass.JSON_OBJECT: (CandidateKind.HOSTED_FLOOR, CandidateKind.SFT_SMALL),
        StructureClass.SHORT_SPAN: (CandidateKind.HOSTED_FLOOR, CandidateKind.SFT_SMALL),
        StructureClass.FREE_TEXT: (),
    }
    assert set(CANDIDATES) == set(StructureClass)


def test_trained_candidates_name_their_recipe_base_serving_rate_and_ceiling() -> None:
    assert (
        CandidateSpec(
            CandidateKind.HEAD_TUNE,
            "head-tune-text-classification@1.2",
            "bert",
            None,
            "cpu-classifier",
            TRAINING_CREDITS_MAX,
        )
        == HEAD_TUNE
    )
    assert (
        CandidateSpec(
            CandidateKind.SFT_SMALL,
            "qlora-sft-chat@1.2",
            "qwen2",
            3_000_000_000,
            "gpu-small-llm",
            TRAINING_CREDITS_MAX,
        )
        == SFT_SMALL
    )


def test_a_run_is_budgeted_at_everything_the_platform_can_bill_it() -> None:
    """The recipes' hour plus the provider margin, at the A10G's 2 credits a minute.

    The platform can bill a run for exactly this span, so a smaller figure here
    would let spending pass ``--max-credits``.
    """
    assert RECIPE_MAX_MINUTES + PROVIDER_MARGIN_MINUTES == 75
    assert TRAINING_CREDITS_MAX == 150


def test_hosted_floor_needs_no_run() -> None:
    assert HOSTED_FLOOR.recipe_key is None
    assert HOSTED_FLOOR.base_family is None
    assert HOSTED_FLOOR.serving_rate_key is None
    assert HOSTED_FLOOR.training_credits_max is None
    assert CandidateKind("head_tune") is CandidateKind.HEAD_TUNE
