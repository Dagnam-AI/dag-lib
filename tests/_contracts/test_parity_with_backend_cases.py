"""Pin the closed e2e-06 hole from the SDK side using the REAL generated schema.

The SDK interpreter must flag the exact e2e-06-unet-p2 condition (a conv node
with bare-int padding) the backend authority flags, and the SDK persist path
must raise before the wire — so an SDK-built model can never reach a state the
Studio would reject.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from dagnam import projects
from dagnam._contracts import validate_params
from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import ArchitectureValidationError
from dagnam._types import JsonValue

# The exact e2e-06-unet-p2 failure: a conv node with bare-int padding.
_E2E06_CONFIG: dict[str, Any] = {"filters": 64, "kernelSize": 3, "padding": 1}
_E2E06_DIAGRAM: JsonValue = {
    "nodes": [
        {
            "id": "u-conv",
            "data": {"componentId": "convolution-layer", "config": _E2E06_CONFIG},
        }
    ]
}


def test_sdk_flags_the_e2e06_padding_like_the_backend() -> None:
    errs = validate_params("convolution-layer", _E2E06_CONFIG, "u-conv")
    assert len(errs) == 1
    assert "explicit" in errs[0].message.lower()


def test_sdk_persist_is_rejected_for_the_e2e06_architecture() -> None:
    c = MagicMock(spec=DagnamClient)
    c.save_architecture = MagicMock(return_value={"version_id": "v1"})
    with pytest.raises(ArchitectureValidationError):
        projects.save_architecture("p1", _E2E06_DIAGRAM, {"layers": []}, client=c)
    c.save_architecture.assert_not_called()
