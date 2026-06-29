"""SDK-local canonical normalizer — the public twin of the backend normalizer.
Schema-driven off the bundled component-schema.json: only kind=="padding" params
are upgraded (the pooling enum is left untouched), idempotent, lossless on
unrecognized input. These cases also drive full branch coverage of normalize.py.
"""

from __future__ import annotations

from typing import Any, cast

from dagnam._contracts.normalize import (
    _candidate_keys,
    normalize_architecture_config,
    normalize_diagram_state,
)
from dagnam._types import JsonValue


def _obj(value: JsonValue) -> dict[str, Any]:
    """Narrow a normalizer's ``JsonValue`` result to an indexable mapping.

    The normalizers are typed ``JsonValue -> JsonValue`` (any JSON node); these
    tests feed object inputs and assert on object outputs, so casting keeps the
    assertions readable without weakening production typing."""
    return cast("dict[str, Any]", value)


def test_candidate_keys_dedupes_overlapping_alias_variants() -> None:
    # An alias whose casing variant collides with the key's is de-duplicated, so
    # the same config key is never visited twice.
    assert _candidate_keys({"key": "rate", "aliases": ["rate"]}) == ["rate"]


def test_diagram_state_conv_bare_int_padding_upgrades() -> None:
    out = normalize_diagram_state(
        {
            "nodes": [
                {
                    "id": "c1",
                    "data": {
                        "componentId": "convolution-layer",
                        "config": {"filters": 8, "kernelSize": 3, "padding": 2},
                    },
                }
            ]
        }
    )
    assert _obj(out)["nodes"][0]["data"]["config"]["padding"] == {"mode": "explicit", "value": 2}


def test_diagram_state_legacy_string_and_layer_type_resolution() -> None:
    out = normalize_diagram_state(
        {"nodes": [{"id": "c1", "data": {"layer_type": "conv2d", "config": {"padding": "same"}}}]}
    )
    assert _obj(out)["nodes"][0]["data"]["config"]["padding"] == {"mode": "same"}


def test_architecture_config_layers_normalize_by_layer_type() -> None:
    out = normalize_architecture_config(
        {
            "layers": [
                {
                    "id": "c1",
                    "type": "conv2d",
                    "config": {"filters": 8, "kernelSize": 3, "padding": 2},
                }
            ],
            "connections": [],
        }
    )
    assert _obj(out)["layers"][0]["config"]["padding"] == {"mode": "explicit", "value": 2}


def test_pooling_enum_padding_is_left_untouched() -> None:
    out = normalize_architecture_config(
        {
            "layers": [
                {
                    "id": "p1",
                    "type": "pooling",
                    "config": {"poolingType": "max", "padding": "valid"},
                }
            ]
        }
    )
    assert _obj(out)["layers"][0]["config"]["padding"] == "valid"


def test_negative_and_garbage_padding_pass_through() -> None:
    out = normalize_diagram_state(
        {
            "nodes": [
                {
                    "id": "c1",
                    "data": {"componentId": "convolution-layer", "config": {"padding": -1}},
                }
            ]
        }
    )
    assert _obj(out)["nodes"][0]["data"]["config"]["padding"] == -1


def test_bool_padding_is_never_wrapped() -> None:
    out = normalize_diagram_state(
        {
            "nodes": [
                {
                    "id": "c1",
                    "data": {"componentId": "convolution-layer", "config": {"padding": True}},
                }
            ]
        }
    )
    assert _obj(out)["nodes"][0]["data"]["config"]["padding"] is True


def test_unknown_component_and_unidentified_nodes_pass_through() -> None:
    # Unknown componentId -> config untouched; node without data -> untouched.
    out = normalize_diagram_state(
        {
            "nodes": [
                {"id": "x", "data": {"componentId": "mystery", "config": {"padding": 2}}},
                {"id": "y"},
                "not-a-node",
            ]
        }
    )
    assert _obj(out)["nodes"][0]["data"]["config"]["padding"] == 2
    assert _obj(out)["nodes"][1] == {"id": "y"}
    assert _obj(out)["nodes"][2] == "not-a-node"


def test_node_with_no_config_or_non_dict_config_passes_through() -> None:
    out = normalize_diagram_state(
        {"nodes": [{"id": "c1", "data": {"componentId": "convolution-layer", "config": "nope"}}]}
    )
    assert _obj(out)["nodes"][0]["data"]["config"] == "nope"


def test_layers_non_dict_entries_pass_through() -> None:
    out = normalize_architecture_config({"layers": ["x", {"type": None, "config": {"padding": 2}}]})
    assert _obj(out)["layers"][0] == "x"
    # type None -> no component -> config untouched
    assert _obj(out)["layers"][1]["config"] == {"padding": 2}


def test_non_dict_and_missing_nodes_inputs_pass_through() -> None:
    assert normalize_diagram_state(None) is None
    assert normalize_diagram_state({"viewport": {}}) == {"viewport": {}}
    assert normalize_diagram_state({"nodes": "not-a-list"}) == {"nodes": "not-a-list"}
    assert normalize_architecture_config([]) == []
    assert normalize_architecture_config({"connections": []}) == {"connections": []}
    assert normalize_architecture_config({"layers": "not-a-list"}) == {"layers": "not-a-list"}


def test_padding_param_absent_from_config_is_a_noop() -> None:
    # conv declares a padding param, but this config omits it: the candidate-key
    # loop finds nothing and leaves the config unchanged.
    out = normalize_diagram_state(
        {
            "nodes": [
                {"id": "c1", "data": {"componentId": "convolution-layer", "config": {"filters": 8}}}
            ]
        }
    )
    assert _obj(out)["nodes"][0]["data"]["config"] == {"filters": 8}


def test_empty_string_identifier_resolves_to_nothing() -> None:
    out = normalize_architecture_config(
        {"layers": [{"id": "x", "type": "", "config": {"padding": 2}}]}
    )
    assert _obj(out)["layers"][0]["config"] == {"padding": 2}


def test_idempotent() -> None:
    once = normalize_diagram_state(
        {
            "nodes": [
                {"id": "c1", "data": {"componentId": "convolution-layer", "config": {"padding": 2}}}
            ]
        }
    )
    assert normalize_diagram_state(once) == once
