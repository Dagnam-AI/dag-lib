"""The bundled component schema loads and indexes by component id + layer type."""

from __future__ import annotations

from dagnam._contracts import COMPONENT_REGISTRY, LAYER_TYPE_TO_COMPONENT, SCHEMA_VERSION


def test_schema_loads_and_indexes_padding_components() -> None:
    assert SCHEMA_VERSION == 1
    conv = COMPONENT_REGISTRY["convolution-layer"]
    assert conv["layer_type"] == "conv2d"
    padding = next(p for p in conv["params"] if p["key"] == "padding")
    assert padding["kind"] == "padding"
    assert LAYER_TYPE_TO_COMPONENT["conv2d"] == "convolution-layer"
