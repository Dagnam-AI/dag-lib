"""The candidate registry is the design's table, verbatim."""

from __future__ import annotations

from dagnam.audit.candidates import (
    CANDIDATES,
    HEAD_TUNE,
    HOSTED_FLOOR,
    SFT_SMALL,
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


def test_trained_candidates_name_their_recipe_base_and_serving_rate() -> None:
    assert (
        CandidateSpec(
            CandidateKind.HEAD_TUNE,
            "head-tune-text-classification@1.1",
            "bert",
            None,
            "cpu-classifier",
        )
        == HEAD_TUNE
    )
    assert (
        CandidateSpec(
            CandidateKind.SFT_SMALL, "qlora-sft-chat@1.1", "qwen2", 3_000_000_000, "gpu-small-llm"
        )
        == SFT_SMALL
    )


def test_hosted_floor_needs_no_run() -> None:
    assert HOSTED_FLOOR.recipe_key is None
    assert HOSTED_FLOOR.base_family is None
    assert HOSTED_FLOOR.serving_rate_key is None
    assert CandidateKind("head_tune") is CandidateKind.HEAD_TUNE
