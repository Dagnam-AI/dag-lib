"""Backwards-compatibility shim — the contract now ships as ``dagnam-contracts``.

The interpreter, normalizers and generated ``component-schema.json`` that used
to live here are the canonical :mod:`dagnam_contracts` distribution, shared
verbatim with the platform backend and the Studio so all three reach identical
verdicts. Every name below is re-exported unchanged; import
``dagnam_contracts`` directly instead.

REMOVE IN 0.10.0 (shipped deprecated for one minor release, 0.9.0).
"""

from dagnam_contracts import (
    COMPONENT_REGISTRY,
    LAYER_TYPE_TO_COMPONENT,
    SCHEMA_VERSION,
    ParamError,
    normalize_architecture_config,
    normalize_diagram_state,
    validate_architecture,
    validate_params,
)

__all__ = [
    "COMPONENT_REGISTRY",
    "LAYER_TYPE_TO_COMPONENT",
    "SCHEMA_VERSION",
    "ParamError",
    "normalize_architecture_config",
    "normalize_diagram_state",
    "validate_architecture",
    "validate_params",
]
