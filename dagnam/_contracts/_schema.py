"""Load the generated component schema (DATA) shipped with the SDK.

This JSON is generated from the backend's canonical Pydantic registry and
regenerated-and-diffed in CI (info/scripts/check_contracts.py). The SDK ships
the DATA only — it never imports backend code (spec §11).
"""

from __future__ import annotations

from importlib import resources
import json
from typing import Any, Final

_RAW: Final[str] = (
    resources.files(__package__).joinpath("component-schema.json").read_text(encoding="utf-8")
)
_SCHEMA: Final[dict[str, Any]] = json.loads(_RAW)

SCHEMA_VERSION: Final[int] = int(_SCHEMA["version"])
COMPONENTS: Final[list[dict[str, Any]]] = list(_SCHEMA["components"])
COMPONENT_REGISTRY: Final[dict[str, dict[str, Any]]] = {c["component_id"]: c for c in COMPONENTS}
LAYER_TYPE_TO_COMPONENT: Final[dict[str, str]] = {
    c["layer_type"]: c["component_id"] for c in COMPONENTS
}
