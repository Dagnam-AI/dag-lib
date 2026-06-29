"""SDK-local mirror of the backend's declarative parameter interpreter.

Verdicts and messages match mvp-backend/src/validation/contracts/interpret.py
exactly (the test cases are shared). Scope is the declarative param tier only:
numeric bounds, enums, typed padding, required-ness, conditional applicability.
Cross-parameter / shape rules are out of scope (a later plan).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dagnam._contracts._schema import COMPONENT_REGISTRY

_PADDING_MODES = ("valid", "same", "explicit")
_PADDING_SHAPE = "{mode:'same'|'valid'|'explicit', value?}"


@dataclass(frozen=True)
class ParamError:
    """A single declarative-parameter validation failure."""

    type: str
    message: str
    node_id: str | None
    severity: str


def _err(message: str, node_id: str) -> ParamError:
    return ParamError(type="parameter_error", message=message, node_id=node_id, severity="error")


def _snake_to_camel(key: str) -> str:
    head, *rest = key.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in rest)


def _camel_to_snake(key: str) -> str:
    out: list[str] = []
    for ch in key:
        if ch.isupper():
            out.append("_")
            out.append(ch.lower())
        else:
            out.append(ch)
    return "".join(out)


def _variants(name: str) -> tuple[str, str, str]:
    return name, _snake_to_camel(name), _camel_to_snake(name)


def _candidate_keys(param: Mapping[str, Any]) -> list[str]:
    names = [param["key"], *(param.get("aliases") or [])]
    candidates: list[str] = []
    for name in names:
        for variant in _variants(name):
            if variant not in candidates:
                candidates.append(variant)
    return candidates


def _resolve(config: Mapping[str, Any], param: Mapping[str, Any]) -> tuple[bool, Any]:
    """Return ``(present, value)``. A key whose value is ``None`` counts as absent."""
    for candidate in _candidate_keys(param):
        if candidate in config and config[candidate] is not None:
            return True, config[candidate]
    return False, None


def _control_value(config: Mapping[str, Any], control_key: str) -> Any:
    for candidate in _variants(control_key):
        if candidate in config and config[candidate] is not None:
            return config[candidate]
    return None


def _param_active(param: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    applies_when = param.get("applies_when")
    if not applies_when:
        return True
    for control_key, allowed in applies_when.items():
        allowed_lower = {str(a).lower() for a in allowed}
        if str(_control_value(config, control_key)).lower() not in allowed_lower:
            return False
    return True


def _check_padding(value: Any, key: str, cid: str, node_id: str) -> list[ParamError]:
    if isinstance(value, str):
        if value in ("valid", "same"):
            return []
        return [
            _err(
                f"{cid}: {key} must be 'valid', 'same', or a typed object {_PADDING_SHAPE}, got {value!r}",
                node_id,
            )
        ]
    if isinstance(value, dict):
        mode = value.get("mode")
        if mode not in _PADDING_MODES:
            return [
                _err(
                    f"{cid}: {key}.mode must be 'valid', 'same', or 'explicit', got {mode!r}",
                    node_id,
                )
            ]
        if mode == "explicit":
            v = value.get("value")
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                return [
                    _err(
                        f"{cid}: explicit {key} needs a non-negative integer value, got {v!r}",
                        node_id,
                    )
                ]
        return []
    return [
        _err(
            f"{cid}: {key} must be a typed object {_PADDING_SHAPE}, got {value!r} — "
            f"wrap an explicit pad as {{mode:'explicit', value:{value!r}}}",
            node_id,
        )
    ]


def _fmt(x: float) -> str:
    return f"{x:g}"


def _check_number(value: Any, param: Mapping[str, Any], cid: str, node_id: str) -> list[ParamError]:
    nc = param.get("numeric")
    if nc is None:
        return []
    # bool is an int subclass in Python; a checkbox value is never a valid number.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return [_err(f"{cid}: {param['key']} must be a number, got {value!r}", node_id)]
    lo, hi, integer = nc.get("min"), nc.get("max"), nc.get("integer", False)
    if integer and not float(value).is_integer():
        return [_err(f"{cid}: {param['key']} must be a whole number, got {value!r}", node_id)]
    if lo is not None and value < lo:
        return [_err(f"{cid}: {param['key']} must be at least {_fmt(lo)}, got {value!r}", node_id)]
    if hi is not None and value > hi:
        return [_err(f"{cid}: {param['key']} must be at most {_fmt(hi)}, got {value!r}", node_id)]
    return []


def validate_params(component_id: str, config: Mapping[str, Any], node_id: str) -> list[ParamError]:
    """Validate *config* against the canonical schema for *component_id*.

    Unknown components are out of contract scope, so they yield no errors here.
    """
    spec = COMPONENT_REGISTRY.get(component_id)
    if spec is None:
        return []
    errors: list[ParamError] = []
    for param in spec["params"]:
        if not _param_active(param, config):
            continue
        present, value = _resolve(config, param)
        if not present:
            if param.get("required"):
                errors.append(
                    _err(f"{component_id}: missing required parameter '{param['key']}'", node_id)
                )
            continue
        kind = param["kind"]
        if kind == "padding":
            errors.extend(_check_padding(value, param["key"], component_id, node_id))
        elif kind == "number":
            errors.extend(_check_number(value, param, component_id, node_id))
        elif kind == "enum":
            enum_values = param.get("enum_values")
            if enum_values is not None and str(value) not in enum_values:
                errors.append(
                    _err(
                        f"{component_id}: {param['key']} must be one of {enum_values}, got {value!r}",
                        node_id,
                    )
                )
    return errors
