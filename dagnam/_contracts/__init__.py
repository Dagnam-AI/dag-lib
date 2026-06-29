"""Generated validation contract + SDK-local interpreter/normalizers.

Ships GENERATED DATA (component-schema.json) plus a dependency-free interpreter
that mirrors the backend's declarative param validator, and the SDK-local
padding normalizers. No backend imports (spec §11).
"""

from dagnam._contracts._architecture import validate_architecture
from dagnam._contracts._interpret import ParamError, validate_params
from dagnam._contracts._schema import (
    COMPONENT_REGISTRY,
    LAYER_TYPE_TO_COMPONENT,
    SCHEMA_VERSION,
)
from dagnam._contracts.normalize import (
    normalize_architecture_config,
    normalize_diagram_state,
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
