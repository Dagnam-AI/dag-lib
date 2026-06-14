"""Unit tests for the Studio diagram-state helper."""

from __future__ import annotations

from typing import cast

import pytest

from dagnam import studio
from dagnam._types import JsonObject


def _nodes(state: JsonObject) -> list[JsonObject]:
    value = state["nodes"]
    assert isinstance(value, list)
    return [cast("JsonObject", node) for node in value]


class TestBuildDiagramState:
    def test_emits_null_positions_for_every_node(self) -> None:
        state = studio.build_diagram_state(
            nodes=[
                {"id": "a", "type": "denseLayer", "data": {"label": "Dense"}},
                {"id": "b", "type": "outputLayer", "data": {"label": "Out"}},
            ],
            edges=[{"id": "a-b", "source": "a", "target": "b"}],
        )
        assert [node["position"] for node in _nodes(state)] == [None, None]

    def test_preserves_node_fields_and_edges(self) -> None:
        state = studio.build_diagram_state(
            nodes=[{"id": "a", "type": "denseLayer", "data": {"label": "Dense"}}],
            edges=[{"id": "a-b", "source": "a", "target": "b"}],
        )
        assert _nodes(state)[0] == {
            "id": "a",
            "type": "denseLayer",
            "data": {"label": "Dense"},
            "position": None,
        }
        assert state["edges"] == [{"id": "a-b", "source": "a", "target": "b"}]

    def test_defaults_viewport_when_omitted(self) -> None:
        state = studio.build_diagram_state(nodes=[], edges=[])
        assert state["viewport"] == {"x": 0, "y": 0, "zoom": 1}

    def test_uses_supplied_viewport(self) -> None:
        state = studio.build_diagram_state(
            nodes=[],
            edges=[],
            viewport={"x": 5, "y": 6, "zoom": 2},
        )
        assert state["viewport"] == {"x": 5, "y": 6, "zoom": 2}

    def test_existing_position_is_overwritten_with_null(self) -> None:
        state = studio.build_diagram_state(
            nodes=[
                {
                    "id": "a",
                    "type": "denseLayer",
                    "data": {},
                    "position": {"x": 9, "y": 9},
                }
            ],
            edges=[],
        )
        assert _nodes(state)[0]["position"] is None

    def test_missing_required_node_key_raises(self) -> None:
        with pytest.raises(KeyError):
            studio.build_diagram_state(
                nodes=[{"type": "denseLayer", "data": {}}],
                edges=[],
            )
