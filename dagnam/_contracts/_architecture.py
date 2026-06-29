"""Walk a diagram_state and validate every node's params against the schema."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dagnam._contracts._interpret import ParamError, validate_params
from dagnam._contracts._schema import LAYER_TYPE_TO_COMPONENT


def _resolve_component_id(data: Mapping[str, Any], config: Mapping[str, Any]) -> str | None:
    component_id = data.get("componentId") or config.get("componentId")
    if component_id:
        return str(component_id)
    layer_type = str(data.get("layer_type", "")).lower()
    return LAYER_TYPE_TO_COMPONENT.get(layer_type)


def validate_architecture(diagram_state: Mapping[str, Any]) -> list[ParamError]:
    """Return all declarative-param errors for the nodes in *diagram_state*."""
    errors: list[ParamError] = []
    nodes = diagram_state.get("nodes") or []
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        data = node.get("data")
        data = data if isinstance(data, Mapping) else {}
        raw_config = data.get("config")
        config = raw_config if isinstance(raw_config, Mapping) else data
        component_id = _resolve_component_id(data, config)
        if component_id is None:
            continue
        node_id = str(node.get("id", ""))
        errors.extend(validate_params(component_id, config, node_id))
    return errors
