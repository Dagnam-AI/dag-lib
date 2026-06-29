"""The diagram walker resolves each node's component and validates its params."""

from __future__ import annotations

from dagnam._contracts import validate_architecture


def _node(component_id: str, config: dict[str, object], node_id: str = "c1") -> dict[str, object]:
    return {"id": node_id, "type": "layer", "data": {"componentId": component_id, "config": config}}


def test_walker_flags_bare_int_padding_on_a_conv_node() -> None:
    state = {"nodes": [_node("convolution-layer", {"filters": 8, "kernelSize": 3, "padding": 2})]}
    errs = validate_architecture(state)
    assert len(errs) == 1
    assert errs[0].node_id == "c1"


def test_walker_resolves_legacy_layer_type_nodes() -> None:
    # Node carrying only data.layer_type (no componentId) still resolves via the map.
    state = {
        "nodes": [{"id": "c2", "data": {"layer_type": "conv2d", "filters": 0, "kernelSize": 3}}]
    }
    errs = validate_architecture(state)
    assert any(e.node_id == "c2" and "filters" in e.message for e in errs)


def test_walker_ignores_unknown_and_unkeyed_nodes() -> None:
    state = {
        "nodes": [
            {"id": "x", "data": {"componentId": "mystery", "config": {}}},
            {"id": "y"},
        ]
    }
    assert validate_architecture(state) == []


def test_walker_handles_missing_nodes_key() -> None:
    assert validate_architecture({}) == []


def test_walker_skips_non_mapping_nodes() -> None:
    assert validate_architecture({"nodes": ["not-a-node", 42, None]}) == []


def test_walker_reads_component_id_from_config_when_absent_on_data() -> None:
    state = {
        "nodes": [
            {
                "id": "c3",
                "data": {"config": {"componentId": "convolution-layer", "kernelSize": 3}},
            }
        ]
    }
    errs = validate_architecture(state)
    assert any(e.node_id == "c3" and "filters" in e.message for e in errs)
