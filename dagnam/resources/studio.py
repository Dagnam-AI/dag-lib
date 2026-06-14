"""Helpers for authoring Studio diagram state without coordinates."""

from __future__ import annotations

from collections.abc import Sequence

from dagnam._types import JsonObject, JsonValue

_DEFAULT_VIEWPORT: JsonObject = {"x": 0, "y": 0, "zoom": 1}


def build_diagram_state(
    nodes: Sequence[JsonObject],
    edges: Sequence[JsonValue],
    *,
    viewport: JsonObject | None = None,
) -> JsonObject:
    """Build a diagram-state payload that the Studio lays out on first open."""
    diagram_nodes: list[JsonValue] = []
    for node in nodes:
        diagram_node: JsonObject = {
            "id": node["id"],
            "type": node["type"],
            "data": node["data"],
            "position": None,
        }
        diagram_nodes.append(diagram_node)

    return {
        "nodes": diagram_nodes,
        "edges": list(edges),
        "viewport": dict(viewport) if viewport is not None else dict(_DEFAULT_VIEWPORT),
    }


__all__ = ["build_diagram_state"]
