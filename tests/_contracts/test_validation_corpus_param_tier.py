"""Cross-validator parity corpus — public SDK (param-tier) side.

The bundled ``validation-corpus.json`` (byte-identical to
``info/contracts/validation-corpus.json``) pins the reconciled cross-validator
verdict for each diagram. The SDK is param-tier only — it has no graph or shape
engine — so it cannot reproduce the structural ``is_valid`` flip or the
shape/graph codes. What it CAN (and must) reproduce is the *parameter* slice of
the verdict: after normalizing the diagram exactly as the studio does, the SDK's
``validate_architecture`` must emit exactly the corpus's ``PARAMETER_ERROR``
multiset for every case — proving the public SDK's param validation agrees with
the reconciled corpus on the dimension it owns. (The full structural/shape
verdict is guarded by the backend and frontend twins.)
"""

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import pytest

from dagnam._contracts import normalize_diagram_state, validate_architecture

_CORPUS = json.loads(
    (
        Path(__file__).resolve().parents[2] / "dagnam" / "_contracts" / "validation-corpus.json"
    ).read_text(encoding="utf-8")
)
_CASES: list[dict[str, Any]] = _CORPUS["cases"]


def _expected_param_errors(case: dict[str, Any]) -> int:
    return sum(1 for code in case["expected"]["normalized_codes"] if code == "PARAMETER_ERROR")


def test_corpus_exposes_cases() -> None:
    assert len(_CASES) > 0


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c["name"])
def test_sdk_reproduces_param_tier_verdict(case: dict[str, Any]) -> None:
    diagram = normalize_diagram_state(case["diagram"])
    assert isinstance(diagram, Mapping)
    # The corpus PARAMETER_ERROR multiset is the blocking (error-severity) slice;
    # non-blocking advisories (warning/info) are not part of the reconciled verdict.
    errors = [e for e in validate_architecture(diagram) if e.severity == "error"]
    assert len(errors) == _expected_param_errors(case), case["name"]
